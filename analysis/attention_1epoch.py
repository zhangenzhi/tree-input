"""
Verify: does 1 epoch of training create cross-level attention bias?

Probes HiT-Tiny attention at:
  - Step 0: before any training (true initialization)
  - Step 0: after first batch (1 gradient update)
  - End of epoch 1: after full epoch

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

    print("=" * 60)
    print("HiT-Tiny Attention Bias: Init -> 1 Epoch")
    print("=" * 60)
    print()

    # Probe: true initialization (before any training)
    model.eval()
    with torch.no_grad():
        probs = extract_attention(model, probe_images.to(device))
    ratio, intra, pc = compute_ratio(probs, LEVELS)
    print(f"  [Before training]  ratio={ratio:.4f}  intra={intra:.6f}  pc={pc:.6f}")

    # Train step by step for first few batches, then rest of epoch
    model.train()
    steps_to_probe = [1, 5, 10, 50, 100]
    probe_idx = 0
    total_steps = len(train_loader)

    for step, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        if probe_idx < len(steps_to_probe) and (step + 1) == steps_to_probe[probe_idx]:
            model.eval()
            with torch.no_grad():
                probs = extract_attention(model, probe_images.to(device))
            ratio, intra, pc = compute_ratio(probs, LEVELS)
            print(f"  [After step {step+1:4d}/{total_steps}]  ratio={ratio:.4f}  intra={intra:.6f}  pc={pc:.6f}  loss={loss.item():.4f}")
            model.train()
            probe_idx += 1

    # Probe: end of epoch 1
    model.eval()
    with torch.no_grad():
        probs = extract_attention(model, probe_images.to(device))
    ratio, intra, pc = compute_ratio(probs, LEVELS)
    print(f"  [After epoch 1]    ratio={ratio:.4f}  intra={intra:.6f}  pc={pc:.6f}")

    print()
    print("=" * 60)


if __name__ == "__main__":
    main()
