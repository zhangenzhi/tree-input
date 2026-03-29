"""
Convergence comparison: HiT vs ViT on CIFAR-10 (resized to 224).

Uses ViT-Tiny scale (embed_dim=192, depth=6, heads=3) for speed.
100 epochs, with attention score tracking for HiT every 5 epochs.

Tracks three categories of attention in HiT layer 0:
  - Intra-level: between patches at the same pyramid level
  - Parent-child: between adjacent-level patches with spatial containment
  - Cross-other: cross-level pairs without direct containment
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import time
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import timm
from model.hit import extract_pyramid_patches, build_pyramid_coords, ContinuousPE3D


LEVELS = [1, 2, 4, 8, 14]


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_cifar10(batch_size=64):
    transform_train = transforms.Compose([
        transforms.Resize(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    transform_val = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    train_set = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform_train)
    val_set = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform_val)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, val_loader


# ---- Attention analysis helpers ----

def get_level_ranges(levels):
    """Return (start, end) token index for each level. Index 0 = CLS."""
    ranges = {}
    offset = 1
    for lvl_idx, n in enumerate(levels):
        count = n * n
        ranges[lvl_idx] = (offset, offset + count)
        offset += count
    return ranges


def get_parent_child_set(levels):
    """Return set of (token_i, token_j) pairs that are parent-child (with CLS offset)."""
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
                            # +1 for CLS offset, symmetric
                            pairs.add((parent_idx + 1, child_idx + 1))
                            pairs.add((child_idx + 1, parent_idx + 1))
    return pairs


def extract_hit_attention(model, images):
    """Extract layer 0 attention probs from HiT model. Returns [B, heads, N, N]."""
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


def compute_attention_stats(attn_probs, levels):
    """Compute mean attention for intra-level, parent-child, cross-other.

    Args:
        attn_probs: [B, heads, N, N] tensor
    Returns:
        dict with mean attention probs for each category
    """
    avg = attn_probs.mean(dim=(0, 1)).cpu().numpy()  # [N, N]
    N = avg.shape[0]
    level_ranges = get_level_ranges(levels)
    pc_set = get_parent_child_set(levels)

    intra = []
    parent_child = []
    cross_other = []

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
            val = avg[i, j]
            if i_level == j_level:
                intra.append(val)
            elif (i, j) in pc_set:
                parent_child.append(val)
            else:
                cross_other.append(val)

    return {
        "intra_level": float(np.mean(intra)),
        "parent_child": float(np.mean(parent_child)),
        "cross_other": float(np.mean(cross_other)),
        "pc_intra_ratio": float(np.mean(parent_child) / np.mean(intra)),
    }


# ---- Models ----

class ViTTiny(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.vit = timm.create_model("vit_tiny_patch16_224", pretrained=False, num_classes=num_classes)

    def forward(self, x):
        return self.vit(x)


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


# ---- Training ----

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * labels.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)
    return total_loss / total, 100.0 * correct / total


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        total_loss += loss.item() * labels.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)
    return total_loss / total, 100.0 * correct / total


def run_experiment(num_epochs=100, batch_size=64, lr=1e-3, attn_probe_interval=5, skip_vit=False):
    device = get_device()
    print(f"Device: {device}")
    print(f"Epochs: {num_epochs}, Batch size: {batch_size}, LR: {lr}")
    print(f"Attention probe every {attn_probe_interval} epochs")
    if skip_vit:
        print("Skipping ViT-Tiny (--skip_vit)")
    print()

    train_loader, val_loader = get_cifar10(batch_size)
    criterion = nn.CrossEntropyLoss()

    # Grab a fixed batch for attention probing (use val set for consistency)
    probe_images, _ = next(iter(val_loader))
    probe_images = probe_images[:16]  # 16 images is enough

    results = {}

    models_to_run = [("ViT-Tiny", ViTTiny), ("HiT-Tiny", HiTTiny)]
    if skip_vit:
        models_to_run = [("HiT-Tiny", HiTTiny)]

    for name, model_fn in models_to_run:
        print(f"{'='*60}")
        print(f"Training {name}...")
        print(f"{'='*60}")

        torch.manual_seed(42)
        model = model_fn(num_classes=10).to(device)
        param_count = sum(p.numel() for p in model.parameters())
        print(f"Parameters: {param_count:,}")

        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)
        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

        history = {
            "train_loss": [], "train_acc": [],
            "val_loss": [], "val_acc": [],
            "epoch_time": [],
        }

        # Attention tracking (HiT only)
        attn_history = [] if name == "HiT-Tiny" else None

        for epoch in range(num_epochs):
            start = time.time()
            train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
            cosine_scheduler.step()
            val_loss, val_acc = validate(model, val_loader, criterion, device)
            elapsed = time.time() - start

            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)
            history["epoch_time"].append(elapsed)

            # Probe attention for HiT
            if attn_history is not None and (epoch % attn_probe_interval == 0 or epoch == num_epochs - 1):
                model.eval()
                with torch.no_grad():
                    probe_dev = probe_images.to(device)
                    attn_probs = extract_hit_attention(model, probe_dev)
                    stats = compute_attention_stats(attn_probs, LEVELS)
                    stats["epoch"] = epoch
                    attn_history.append(stats)
                    print(f"  [Attn@{epoch}] intra={stats['intra_level']:.6f} "
                          f"pc={stats['parent_child']:.6f} "
                          f"ratio={stats['pc_intra_ratio']:.4f}")

            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"  Epoch {epoch+1:3d}/{num_epochs} | "
                      f"train_loss={train_loss:.4f} train_acc={train_acc:.1f}% | "
                      f"val_loss={val_loss:.4f} val_acc={val_acc:.1f}% | "
                      f"time={elapsed:.1f}s lr={optimizer.param_groups[0]['lr']:.6f}")

        results[name] = history
        if attn_history is not None:
            results["attn_history"] = attn_history
        print()

    # ---- Save raw results ----
    fig_dir = os.path.join(os.path.dirname(__file__), "figures")
    os.makedirs(fig_dir, exist_ok=True)

    # Save attention history as JSON
    if "attn_history" in results:
        with open(os.path.join(fig_dir, "attn_history.json"), "w") as f:
            json.dump(results["attn_history"], f, indent=2)
        print(f"Saved: {fig_dir}/attn_history.json")

    # ---- Plot 1: Loss and accuracy curves ----
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    epochs = range(1, num_epochs + 1)

    for name in ["ViT-Tiny", "HiT-Tiny"]:
        if name not in results:
            continue
        hist = results[name]
        axes[0, 0].plot(epochs, hist["train_loss"], label=name, linewidth=1.5)
        axes[0, 1].plot(epochs, hist["val_loss"], label=name, linewidth=1.5)
        axes[1, 0].plot(epochs, hist["train_acc"], label=name, linewidth=1.5)
        axes[1, 1].plot(epochs, hist["val_acc"], label=name, linewidth=1.5)

    axes[0, 0].set_title("Train Loss")
    axes[0, 1].set_title("Val Loss")
    axes[1, 0].set_title("Train Acc (%)")
    axes[1, 1].set_title("Val Acc (%)")

    for ax in axes.flat:
        ax.set_xlabel("Epoch")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.suptitle(f"ViT-Tiny vs HiT-Tiny on CIFAR-10 ({num_epochs} epochs)", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "convergence_100ep.png"), dpi=150)
    print(f"Saved: {fig_dir}/convergence_100ep.png")

    # ---- Plot 2: Attention score evolution ----
    if "attn_history" in results:
        attn_h = results["attn_history"]
        attn_epochs = [s["epoch"] for s in attn_h]

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Left: raw attention values
        axes[0].plot(attn_epochs, [s["intra_level"] for s in attn_h], label="Intra-level", marker="o")
        axes[0].plot(attn_epochs, [s["parent_child"] for s in attn_h], label="Parent-child", marker="s")
        axes[0].plot(attn_epochs, [s["cross_other"] for s in attn_h], label="Cross-other", marker="^")
        uniform = 1.0 / (1 + sum(n * n for n in LEVELS))
        axes[0].axhline(y=uniform, color="red", linestyle="--", alpha=0.5, label=f"Uniform={uniform:.5f}")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Mean Attention Probability")
        axes[0].set_title("HiT-Tiny Layer 0: Attention by Category")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Right: parent-child / intra-level ratio
        axes[1].plot(attn_epochs, [s["pc_intra_ratio"] for s in attn_h], marker="o", color="darkorange", linewidth=2)
        axes[1].axhline(y=1.0, color="red", linestyle="--", alpha=0.5, label="No bias (ratio=1.0)")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Parent-Child / Intra-Level Ratio")
        axes[1].set_title("Cross-Level Structural Attention Bias Over Training")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, "attention_evolution.png"), dpi=150)
        print(f"Saved: {fig_dir}/attention_evolution.png")

    # ---- Summary ----
    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for name in ["ViT-Tiny", "HiT-Tiny"]:
        if name not in results:
            continue
        hist = results[name]
        best_val = max(hist["val_acc"])
        best_epoch = hist["val_acc"].index(best_val) + 1
        avg_time = np.mean(hist["epoch_time"])
        final_train = hist["train_acc"][-1]
        final_val = hist["val_acc"][-1]
        gap = final_train - final_val
        print(f"  {name:10s} | best_val={best_val:.1f}% (ep{best_epoch}) | "
              f"final train={final_train:.1f}% val={final_val:.1f}% gap={gap:.1f}% | "
              f"avg_time={avg_time:.1f}s/ep")

    if "attn_history" in results:
        attn_h = results["attn_history"]
        print()
        print("Attention evolution (parent-child / intra-level ratio):")
        for s in attn_h:
            print(f"  epoch {s['epoch']:3d}: ratio={s['pc_intra_ratio']:.4f}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--attn_probe_interval", type=int, default=5)
    parser.add_argument("--skip_vit", action="store_true", help="Skip ViT-Tiny, only run HiT-Tiny")
    args = parser.parse_args()
    run_experiment(
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        attn_probe_interval=args.attn_probe_interval,
        skip_vit=args.skip_vit,
    )
