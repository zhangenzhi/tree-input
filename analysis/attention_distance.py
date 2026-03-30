"""
Layer-wise attention distance analysis for ViT-Tiny and HiT-Tiny.

For each transformer layer, computes the average spatial distance (in original
image coordinates) between query tokens and their attended key tokens.

ViT: If hierarchical processing is learned, shallow layers should attend locally
(short distance) and deep layers globally (long distance).

HiT: Additionally tracks attention distribution across pyramid levels per layer.

Trains both models for 100 epochs on CIFAR-10, then analyzes the trained models.
Alternatively, pass --load_dir to skip training and load saved models.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import json
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import timm
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model.hit import extract_pyramid_patches, build_pyramid_coords, ContinuousPE3D


LEVELS = [1, 2, 4, 8, 14]
PATCH_SIZE = 16
IMAGE_SIZE = 224


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


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


# ---- Spatial coordinate helpers ----

def get_vit_patch_coords():
    """Return [196, 2] array of (cx, cy) for ViT 14x14 grid, normalized to [0,1]."""
    coords = []
    for row in range(14):
        for col in range(14):
            cx = (col + 0.5) / 14
            cy = (row + 0.5) / 14
            coords.append([cx, cy])
    return np.array(coords)


def get_hit_patch_coords(levels):
    """Return [total_patches, 2] array of (cx, cy) for HiT pyramid."""
    coords = []
    for n in levels:
        for row in range(n):
            for col in range(n):
                cx = (col + 0.5) / n
                cy = (row + 0.5) / n
                coords.append([cx, cy])
    return np.array(coords)


def get_hit_level_indices(levels):
    """Return dict: level_idx -> list of patch indices (0-based, no CLS)."""
    result = {}
    idx = 0
    for lvl_idx, n in enumerate(levels):
        result[lvl_idx] = list(range(idx, idx + n * n))
        idx += n * n
    return result


# ---- Attention extraction for all layers ----

def extract_all_layer_attention_vit(model, images):
    """Extract attention probs from all layers of ViT. Returns list of [B, heads, N, N]."""
    vit = model.vit
    x = vit.patch_embed(images)
    cls_tokens = vit.cls_token.expand(images.size(0), -1, -1)
    x = torch.cat([cls_tokens, x], dim=1)
    x = x + vit.pos_embed
    x = vit.pos_drop(x)

    all_attn = []
    for block in vit.blocks:
        # Extract attention
        y = block.norm1(x)
        attn_mod = block.attn
        B, N, C = y.shape
        qkv = attn_mod.qkv(y).reshape(B, N, 3, attn_mod.num_heads, C // attn_mod.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        scale = (C // attn_mod.num_heads) ** -0.5
        attn_probs = (q @ k.transpose(-2, -1) * scale).softmax(dim=-1)
        all_attn.append(attn_probs.detach().cpu())

        # Forward through the block for next layer
        x = x + block.attn(block.norm1(x))
        x = x + block.mlp(block.norm2(x))

    return all_attn


def extract_all_layer_attention_hit(model, images):
    """Extract attention probs from all layers of HiT. Returns list of [B, heads, N, N]."""
    B = images.size(0)
    patches = extract_pyramid_patches(images, model.levels, model.patch_size)
    x = model.patch_proj(patches)
    pe = model.pe(model.pyramid_coords)
    x = x + pe.unsqueeze(0)
    cls_tokens = model.cls_token.expand(B, -1, -1) + model.cls_pos
    x = torch.cat([cls_tokens, x], dim=1)
    x = model.pos_drop(x)

    all_attn = []
    for block in model.blocks:
        y = block.norm1(x)
        attn_mod = block.attn
        B2, N, C = y.shape
        qkv = attn_mod.qkv(y).reshape(B2, N, 3, attn_mod.num_heads, C // attn_mod.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        scale = (C // attn_mod.num_heads) ** -0.5
        attn_probs = (q @ k.transpose(-2, -1) * scale).softmax(dim=-1)
        all_attn.append(attn_probs.detach().cpu())

        x = x + block.attn(block.norm1(x))
        x = x + block.mlp(block.norm2(x))

    return all_attn


# ---- Analysis functions ----

def compute_attention_distance_vit(all_attn, patch_coords):
    """Compute mean attention distance per layer for ViT.

    For each query patch, compute weighted average spatial distance to all key patches,
    weighted by attention probability. Average over batch, heads, and query patches.

    Returns: [num_layers] array of mean distances.
    """
    num_patches = patch_coords.shape[0]
    # Pairwise distances between patches: [196, 196]
    diff = patch_coords[:, None, :] - patch_coords[None, :, :]  # [196, 196, 2]
    dist_matrix = np.sqrt((diff ** 2).sum(axis=-1))  # [196, 196]

    distances = []
    for layer_attn in all_attn:
        # layer_attn: [B, heads, N, N] where N = 197 (CLS + 196)
        # Skip CLS token (index 0), only compute for patch tokens
        attn = layer_attn[:, :, 1:, 1:].numpy()  # [B, heads, 196, 196]
        # Normalize attention over key patches (already softmaxed, but CLS column removed)
        attn = attn / (attn.sum(axis=-1, keepdims=True) + 1e-9)
        # Weighted distance: [B, heads, 196]
        weighted_dist = (attn * dist_matrix[None, None, :, :]).sum(axis=-1)
        distances.append(float(weighted_dist.mean()))

    return np.array(distances)


def compute_attention_distance_hit(all_attn, patch_coords):
    """Same as ViT version but for HiT's 281 patches."""
    num_patches = patch_coords.shape[0]
    diff = patch_coords[:, None, :] - patch_coords[None, :, :]
    dist_matrix = np.sqrt((diff ** 2).sum(axis=-1))

    distances = []
    for layer_attn in all_attn:
        attn = layer_attn[:, :, 1:, 1:].numpy()
        attn = attn / (attn.sum(axis=-1, keepdims=True) + 1e-9)
        weighted_dist = (attn * dist_matrix[None, None, :, :]).sum(axis=-1)
        distances.append(float(weighted_dist.mean()))

    return np.array(distances)


def compute_level_attention_per_layer(all_attn, levels):
    """For HiT: compute fraction of attention going to each pyramid level, per layer.

    Returns: [num_layers, num_levels] array.
    """
    level_indices = get_hit_level_indices(levels)
    num_levels = len(levels)
    total_patches = sum(n * n for n in levels)

    result = []
    for layer_attn in all_attn:
        # [B, heads, N, N] -> average over batch and heads -> [N, N]
        avg_attn = layer_attn.mean(dim=(0, 1)).numpy()
        # For all query patches (skip CLS), compute how much attention goes to each level
        level_fracs = []
        for lvl_idx in range(num_levels):
            key_indices = [idx + 1 for idx in level_indices[lvl_idx]]  # +1 for CLS offset
            # Sum attention to this level's patches, averaged over all query patches
            frac = avg_attn[1:, key_indices].sum(axis=-1).mean()
            level_fracs.append(float(frac))
        result.append(level_fracs)

    return np.array(result)


# ---- Training ----

def train_model(model, train_loader, criterion, optimizer, scheduler, device, num_epochs):
    for epoch in range(num_epochs):
        model.train()
        correct, total = 0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
        scheduler.step()
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"    Epoch {epoch+1:3d}/{num_epochs} | acc={100.*correct/total:.1f}%")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--probe_batch_size", type=int, default=32)
    args = parser.parse_args()

    device = get_device()
    print(f"Device: {device}")

    # Data
    transform_train = transforms.Compose([
        transforms.Resize(224), transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    transform_val = transforms.Compose([
        transforms.Resize(224), transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    train_set = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform_train)
    val_set = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform_val)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=0)
    probe_loader = DataLoader(val_set, batch_size=args.probe_batch_size, shuffle=False, num_workers=0)

    # Get probe images
    probe_images, _ = next(iter(probe_loader))

    criterion = nn.CrossEntropyLoss()
    fig_dir = os.path.join(os.path.dirname(__file__), "figures")
    os.makedirs(fig_dir, exist_ok=True)

    results = {}

    # ---- Train and analyze ViT-Tiny ----
    print("=" * 60)
    print("Training ViT-Tiny...")
    print("=" * 60)
    torch.manual_seed(42)
    vit_model = ViTTiny(num_classes=10).to(device)
    vit_opt = torch.optim.AdamW(vit_model.parameters(), lr=args.lr, weight_decay=0.05)
    vit_sched = torch.optim.lr_scheduler.CosineAnnealingLR(vit_opt, T_max=args.num_epochs)
    train_model(vit_model, train_loader, criterion, vit_opt, vit_sched, device, args.num_epochs)

    print("  Extracting attention from all layers...")
    vit_model.eval()
    with torch.no_grad():
        vit_all_attn = extract_all_layer_attention_vit(vit_model, probe_images.to(device))
    vit_coords = get_vit_patch_coords()
    vit_distances = compute_attention_distance_vit(vit_all_attn, vit_coords)
    results["ViT-Tiny"] = {"distances": vit_distances.tolist()}
    print(f"  Attention distances per layer: {[f'{d:.4f}' for d in vit_distances]}")

    # ---- Train and analyze HiT-Tiny ----
    print("=" * 60)
    print("Training HiT-Tiny...")
    print("=" * 60)
    torch.manual_seed(42)
    hit_model = HiTTiny(num_classes=10).to(device)
    hit_opt = torch.optim.AdamW(hit_model.parameters(), lr=args.lr, weight_decay=0.05)
    hit_sched = torch.optim.lr_scheduler.CosineAnnealingLR(hit_opt, T_max=args.num_epochs)
    train_model(hit_model, train_loader, criterion, hit_opt, hit_sched, device, args.num_epochs)

    print("  Extracting attention from all layers...")
    hit_model.eval()
    with torch.no_grad():
        hit_all_attn = extract_all_layer_attention_hit(hit_model, probe_images.to(device))
    hit_coords = get_hit_patch_coords(LEVELS)
    hit_distances = compute_attention_distance_hit(hit_all_attn, hit_coords)
    hit_level_attn = compute_level_attention_per_layer(hit_all_attn, LEVELS)
    results["HiT-Tiny"] = {
        "distances": hit_distances.tolist(),
        "level_attention": hit_level_attn.tolist(),
    }
    print(f"  Attention distances per layer: {[f'{d:.4f}' for d in hit_distances]}")

    # Save results
    with open(os.path.join(fig_dir, "attention_distance.json"), "w") as f:
        json.dump(results, f, indent=2)

    # ---- Plot 1: Attention distance per layer ----
    num_layers = len(vit_distances)
    layers = range(1, num_layers + 1)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(layers, vit_distances, marker="o", label="ViT-Tiny", linewidth=2)
    ax.plot(layers, hit_distances, marker="s", label="HiT-Tiny", linewidth=2)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Mean Attention Distance (normalized)")
    ax.set_title("Attention Distance per Layer (trained 100 epochs on CIFAR-10)")
    ax.set_xticks(list(layers))
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "attention_distance.png"), dpi=150)
    print(f"\nSaved: {fig_dir}/attention_distance.png")

    # ---- Plot 2: HiT level attention distribution per layer ----
    level_names = [f"L{i}({n}x{n})" for i, n in enumerate(LEVELS)]
    fig, ax = plt.subplots(figsize=(12, 5))
    bottom = np.zeros(num_layers)
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(LEVELS)))
    for lvl_idx in range(len(LEVELS)):
        values = hit_level_attn[:, lvl_idx]
        ax.bar(layers, values, bottom=bottom, label=level_names[lvl_idx],
               color=colors[lvl_idx], alpha=0.8)
        bottom += values
    ax.set_xlabel("Layer")
    ax.set_ylabel("Fraction of Attention")
    ax.set_title("HiT-Tiny: Attention Distribution Across Pyramid Levels per Layer")
    ax.set_xticks(list(layers))
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "hit_level_attention.png"), dpi=150)
    print(f"Saved: {fig_dir}/hit_level_attention.png")

    # ---- Summary ----
    print()
    print("=" * 60)
    print("SUMMARY: Attention Distance per Layer")
    print("=" * 60)
    print(f"  {'Layer':>5s}  {'ViT-Tiny':>10s}  {'HiT-Tiny':>10s}  {'Diff':>10s}")
    for i in range(num_layers):
        diff = hit_distances[i] - vit_distances[i]
        print(f"  {i+1:5d}  {vit_distances[i]:10.4f}  {hit_distances[i]:10.4f}  {diff:+10.4f}")
    print()

    # Check for hierarchical pattern
    vit_slope = np.polyfit(range(num_layers), vit_distances, 1)[0]
    hit_slope = np.polyfit(range(num_layers), hit_distances, 1)[0]
    print(f"  ViT distance trend (slope): {vit_slope:+.4f} ({'local→global' if vit_slope > 0.001 else 'flat/mixed'})")
    print(f"  HiT distance trend (slope): {hit_slope:+.4f} ({'local→global' if hit_slope > 0.001 else 'flat/mixed'})")
    print()

    print("HiT-Tiny Level Attention per Layer:")
    print(f"  {'Layer':>5s}  " + "  ".join(f"{name:>10s}" for name in level_names))
    for i in range(num_layers):
        vals = "  ".join(f"{hit_level_attn[i, j]:10.4f}" for j in range(len(LEVELS)))
        print(f"  {i+1:5d}  {vals}")
    print("=" * 60)


if __name__ == "__main__":
    main()
