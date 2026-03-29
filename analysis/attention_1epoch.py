"""
Track cross-level attention bias emergence during early training.

Probes HiT-Tiny attention every 50 steps for 5 epochs.
Runs on MPS/CPU.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn as nn
import numpy as np
import timm
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model.hit import extract_pyramid_patches, build_pyramid_coords, ContinuousPE3D


LEVELS = [1, 2, 4, 8, 14]


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_level_ranges(levels):
    ranges = {}
    offset = 1
    for lvl_idx, n in enumerate(levels):
        count = n * n
        ranges[lvl_idx] = (offset, offset + count)
        offset += count
    return ranges


def get_parent_child_set(levels):
    pairs = set()
    offset = {}
    idx = 0
    for lvl_idx, n in enumerate(levels):
        offset[lvl_idx] = idx
        idx += n * n
    for lvl_idx in range(len(levels) - 1):
        parent_n = levels[lvl_idx]
        child_n = levels[lvl_idx + 1]
        ratio = child_n // parent_n
        for pr in range(parent_n):
            for pc in range(parent_n):
                parent_idx = offset[lvl_idx] + pr * parent_n + pc
                for dr in range(ratio):
                    for dc in range(ratio):
                        cr = pr * ratio + dr
                        cc = pc * ratio + dc
                        if cr < child_n and cc < child_n:
                            child_idx = offset[lvl_idx + 1] + cr * child_n + cc
                            pairs.add((parent_idx + 1, child_idx + 1))
                            pairs.add((child_idx + 1, parent_idx + 1))
    return pairs


class HiTTiny(nn.Module):
    def __init__(self, num_classes=10, levels=None):
        super().__init__()
        self.levels = levels or LEVELS
        self.total_patches = sum(n * n for n in self.levels)
        self.patch_size = 16
        vit = timm.create_model("vit_tiny_patch16_224", pretrained=False, num_classes=num_classes)
        self.embed_dim = vit.embed_dim
        self.patch_proj = nn.Linear(16 * 16 * 3, self.embed_dim)
        self.cls_token = vit.cls_token
        self.pe = ContinuousPE3D(self.embed_dim, num_freq=32)
        coords = build_pyramid_coords(self.levels)
        self.register_buffer("pyramid_coords", coords)
        self.cls_pos = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        self.pos_drop = vit.pos_drop
        self.blocks = vit.blocks
        self.norm = vit.norm
        self.head = vit.head

    def forward(self, images):
        B = images.size(0)
        patches = extract_pyramid_patches(images, self.levels, self.patch_size)
        x = self.patch_proj(patches)
        pe = self.pe(self.pyramid_coords)
        x = x + pe.unsqueeze(0)
        cls_tokens = self.cls_token.expand(B, -1, -1) + self.cls_pos
        x = torch.cat([cls_tokens, x], dim=1)
        x = self.pos_drop(x)
        x = self.blocks(x)
        x = self.norm(x)
        return self.head(x[:, 0])


def extract_attention(model, images):
    B = images.size(0)
    patches = extract_pyramid_patches(images, model.levels, model.patch_size)
    x = model.patch_proj(patches)
    pe = model.pe(model.pyramid_coords)
    x = x + pe.unsqueeze(0)
    cls_tokens = model.cls_token.expand(B, -1, -1) + model.cls_pos
    x = torch.cat([cls_tokens, x], dim=1)
    x = model.pos_drop(x)

    block = model.blocks[0]
    y = block.norm1(x)
    attn_module = block.attn
    B, N, C = y.shape
    qkv = attn_module.qkv(y).reshape(B, N, 3, attn_module.num_heads, C // attn_module.num_heads)
    qkv = qkv.permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)
    scale = (C // attn_module.num_heads) ** -0.5
    attn_probs = (q @ k.transpose(-2, -1) * scale).softmax(dim=-1)
    return attn_probs.detach()


def compute_ratio(attn_probs, levels):
    avg = attn_probs.mean(dim=(0, 1)).cpu().numpy()
    N = avg.shape[0]
    level_ranges = get_level_ranges(levels)
    pc_set = get_parent_child_set(levels)

    intra, pc = [], []
    for i in range(1, N):
        i_level = None
        for lvl, (s, e) in level_ranges.items():
            if s <= i < e:
                i_level = lvl
                break
        for j in range(1, N):
            if i == j:
                continue
            j_level = None
            for lvl, (s, e) in level_ranges.items():
                if s <= j < e:
                    j_level = lvl
                    break
            if i_level == j_level:
                intra.append(avg[i, j])
            elif (i, j) in pc_set:
                pc.append(avg[i, j])

    return float(np.mean(pc) / np.mean(intra)), float(np.mean(intra)), float(np.mean(pc))


def main():
    device = get_device()
    print(f"Device: {device}")
    print()

    # Data
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    train_set = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
    train_loader = DataLoader(train_set, batch_size=64, shuffle=True, num_workers=0)

    # Fixed probe batch (from val set, no augmentation)
    probe_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    val_set = datasets.CIFAR10(root="./data", train=False, download=True, transform=probe_transform)
    probe_loader = DataLoader(val_set, batch_size=16, shuffle=False, num_workers=0)
    probe_images, _ = next(iter(probe_loader))

    # Model
    torch.manual_seed(42)
    model = HiTTiny(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.05)

    num_epochs = 5
    probe_interval = 50
    total_steps_per_epoch = len(train_loader)

    print("=" * 60)
    print(f"HiT-Tiny Attention Bias: 5 epochs, probe every {probe_interval} steps")
    print(f"Steps per epoch: {total_steps_per_epoch}")
    print("=" * 60)
    print()

    # Probe: true initialization
    model.eval()
    with torch.no_grad():
        probs = extract_attention(model, probe_images.to(device))
    ratio, intra, pc = compute_ratio(probs, LEVELS)
    print(f"  [init          ]  ratio={ratio:.4f}  intra={intra:.6f}  pc={pc:.6f}")

    history = [{"global_step": 0, "epoch": 0, "ratio": ratio, "intra": intra, "pc": pc}]

    global_step = 0
    for epoch in range(num_epochs):
        model.train()
        for step, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            global_step += 1

            if global_step % probe_interval == 0:
                model.eval()
                with torch.no_grad():
                    probs = extract_attention(model, probe_images.to(device))
                ratio, intra, pc = compute_ratio(probs, LEVELS)
                print(f"  [ep{epoch+1} step {step+1:4d}/{total_steps_per_epoch}  gs={global_step:5d}]  ratio={ratio:.4f}  intra={intra:.6f}  pc={pc:.6f}  loss={loss.item():.4f}")
                history.append({"global_step": global_step, "epoch": epoch + 1, "ratio": ratio, "intra": intra, "pc": pc})
                model.train()

    # Plot
    import matplotlib.pyplot as plt
    fig_dir = os.path.join(os.path.dirname(__file__), "figures")
    os.makedirs(fig_dir, exist_ok=True)

    steps = [h["global_step"] for h in history]
    ratios = [h["ratio"] for h in history]

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(steps, ratios, marker="o", markersize=3, linewidth=1.5)
    ax.axhline(y=1.0, color="red", linestyle="--", alpha=0.5, label="No bias (1.0)")
    for ep in range(1, num_epochs + 1):
        ax.axvline(x=ep * total_steps_per_epoch, color="gray", linestyle=":", alpha=0.4)
        ax.text(ep * total_steps_per_epoch, ax.get_ylim()[1] * 0.95, f"ep{ep}", ha="center", fontsize=8, color="gray")
    ax.set_xlabel("Global Step")
    ax.set_ylabel("Parent-Child / Intra-Level Ratio")
    ax.set_title("HiT-Tiny: Cross-Level Attention Bias Over 5 Epochs")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "attention_bias_5ep.png"), dpi=150)
    print(f"\nSaved: {fig_dir}/attention_bias_5ep.png")

    # Find peak
    peak = max(history, key=lambda h: h["ratio"])
    print(f"\nPeak ratio: {peak['ratio']:.4f} at global_step={peak['global_step']} (epoch {peak['epoch']})")
    print("=" * 60)


if __name__ == "__main__":
    main()
