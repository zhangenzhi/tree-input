"""
Verify whether cross-level attention bias exists at initialization (epoch 0).

Tests both HiT-Tiny (embed_dim=192) and HiT-B (embed_dim=768) to examine
how embedding dimension affects structural attention bias.

Runs on CPU, no GPU needed.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import timm
from model.hit import extract_pyramid_patches, build_pyramid_coords, ContinuousPE3D


LEVELS = [1, 2, 4, 8, 14]
PATCH_SIZE = 16
IMAGE_SIZE = 224


def get_level_ranges(levels):
    ranges = {}
    offset = 1  # skip CLS
    for lvl_idx, n in enumerate(levels):
        count = n * n
        ranges[lvl_idx] = (offset, offset + count)
        offset += count
    return ranges


def get_parent_child_pairs(levels):
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
                            pairs.add((parent_idx, child_idx))
    return pairs


# ---- HiT-Tiny model (matching convergence_test.py) ----

class HiTTiny(nn.Module):
    def __init__(self, num_classes=10, levels=None):
        super().__init__()
        self.levels = levels or LEVELS
        self.total_patches = sum(n * n for n in self.levels)
        self.patch_size = 16

        vit = timm.create_model("vit_tiny_patch16_224", pretrained=False, num_classes=num_classes)
        self.embed_dim = vit.embed_dim  # 192

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


# ---- HiT-B model (from model/hit.py) ----

from model.hit import HiTBase


# ---- Attention extraction (works for both Tiny and Base) ----

def extract_attention(model, images):
    """Extract layer 0 attention logits and probs."""
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
    attn_logits = (q @ k.transpose(-2, -1)) * scale
    attn_probs = attn_logits.softmax(dim=-1)

    return attn_logits.detach(), attn_probs.detach()


def compute_stats(avg_logits, avg_probs, levels):
    """Compute attention stats by category."""
    level_ranges = get_level_ranges(levels)
    parent_child_pairs = get_parent_child_pairs(levels)
    total_patches = sum(n * n for n in levels)
    N = 1 + total_patches

    pc_set = set()
    for (pi, ci) in parent_child_pairs:
        pc_set.add((pi + 1, ci + 1))
        pc_set.add((ci + 1, pi + 1))

    intra_logits, pc_logits, other_logits = [], [], []
    intra_probs, pc_probs, other_probs = [], [], []

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
                intra_logits.append(avg_logits[i, j])
                intra_probs.append(avg_probs[i, j])
            elif (i, j) in pc_set:
                pc_logits.append(avg_logits[i, j])
                pc_probs.append(avg_probs[i, j])
            else:
                other_logits.append(avg_logits[i, j])
                other_probs.append(avg_probs[i, j])

    return {
        "intra_logits": (np.mean(intra_logits), np.std(intra_logits), len(intra_logits)),
        "pc_logits": (np.mean(pc_logits), np.std(pc_logits), len(pc_logits)),
        "other_logits": (np.mean(other_logits), np.std(other_logits), len(other_logits)),
        "intra_probs": (np.mean(intra_probs), np.std(intra_probs)),
        "pc_probs": (np.mean(pc_probs), np.std(pc_probs)),
        "other_probs": (np.mean(other_probs), np.std(other_probs)),
        "uniform": 1.0 / N,
        "ratio": np.mean(pc_probs) / np.mean(intra_probs),
    }


def print_stats(name, stats):
    print(f"--- {name} ---")
    print(f"  Attention LOGITS:")
    print(f"    Intra-level:        mean={stats['intra_logits'][0]:.6f}  std={stats['intra_logits'][1]:.6f}  n={stats['intra_logits'][2]}")
    print(f"    Parent-child:       mean={stats['pc_logits'][0]:.6f}  std={stats['pc_logits'][1]:.6f}  n={stats['pc_logits'][2]}")
    print(f"    Cross-other:        mean={stats['other_logits'][0]:.6f}  std={stats['other_logits'][1]:.6f}  n={stats['other_logits'][2]}")
    print(f"  Attention PROBS:")
    print(f"    Uniform baseline:   {stats['uniform']:.6f}")
    print(f"    Intra-level:        mean={stats['intra_probs'][0]:.6f}  std={stats['intra_probs'][1]:.6f}")
    print(f"    Parent-child:       mean={stats['pc_probs'][0]:.6f}  std={stats['pc_probs'][1]:.6f}")
    print(f"    Cross-other:        mean={stats['other_probs'][0]:.6f}  std={stats['other_probs'][1]:.6f}")
    print(f"  PC / Intra ratio:     {stats['ratio']:.4f}")
    print()


def analyze():
    print("=" * 60)
    print("HiT Attention at Initialization: Tiny vs Base")
    print("=" * 60)
    print(f"Levels: {LEVELS}, Total patches: {sum(n*n for n in LEVELS)}")
    print()

    # Load CIFAR-10 images
    from torchvision import datasets, transforms
    transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    cifar = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)
    loader = torch.utils.data.DataLoader(cifar, batch_size=16, shuffle=False)
    images, _ = next(iter(loader))
    print(f"Input: CIFAR-10 images, batch={images.shape}")
    print()

    all_stats = {}

    # ---- HiT-Tiny (embed_dim=192) ----
    torch.manual_seed(42)
    hit_tiny = HiTTiny(num_classes=10)
    hit_tiny.eval()
    print(f"HiT-Tiny: embed_dim={hit_tiny.embed_dim}, params={sum(p.numel() for p in hit_tiny.parameters()):,}")

    with torch.no_grad():
        logits_t, probs_t = extract_attention(hit_tiny, images)
    avg_logits_t = logits_t.mean(dim=(0, 1)).numpy()
    avg_probs_t = probs_t.mean(dim=(0, 1)).numpy()
    max_probs_t = probs_t.max(dim=1).values.mean(dim=0).numpy()  # max over heads, mean over batch
    stats_t = compute_stats(avg_logits_t, avg_probs_t, LEVELS)
    print_stats("HiT-Tiny (embed_dim=192)", stats_t)
    all_stats["HiT-Tiny"] = stats_t

    # ---- HiT-B (embed_dim=768) ----
    torch.manual_seed(42)
    hit_b = HiTBase(
        image_size=IMAGE_SIZE,
        patch_size=PATCH_SIZE,
        num_classes=1000,
        levels=LEVELS,
        pretrained=False,
    )
    hit_b.eval()
    print(f"HiT-B: embed_dim={hit_b.embed_dim}, params={sum(p.numel() for p in hit_b.parameters()):,}")

    with torch.no_grad():
        logits_b, probs_b = extract_attention(hit_b, images)
    avg_logits_b = logits_b.mean(dim=(0, 1)).numpy()
    avg_probs_b = probs_b.mean(dim=(0, 1)).numpy()
    max_probs_b = probs_b.max(dim=1).values.mean(dim=0).numpy()  # max over heads, mean over batch
    stats_b = compute_stats(avg_logits_b, avg_probs_b, LEVELS)
    print_stats("HiT-B (embed_dim=768)", stats_b)
    all_stats["HiT-B"] = stats_b

    # ---- Plots ----
    fig_dir = os.path.join(os.path.dirname(__file__), "figures")
    os.makedirs(fig_dir, exist_ok=True)

    # 1. Attention heatmaps side by side
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    boundaries = [0]
    for n in LEVELS:
        boundaries.append(boundaries[-1] + n * n)
    boundaries = [b + 1 for b in boundaries]

    for ax, avg_probs, title in [
        (axes[0], avg_probs_t, f"HiT-Tiny (dim=192, ratio={stats_t['ratio']:.3f})"),
        (axes[1], avg_probs_b, f"HiT-B (dim=768, ratio={stats_b['ratio']:.3f})"),
    ]:
        im = ax.imshow(avg_probs, cmap="viridis", aspect="auto")
        ax.set_title(title)
        ax.set_xlabel("Key token")
        ax.set_ylabel("Query token")
        for b in boundaries:
            ax.axhline(y=b - 0.5, color="red", linewidth=0.5, alpha=0.7)
            ax.axvline(x=b - 0.5, color="red", linewidth=0.5, alpha=0.7)
        plt.colorbar(im, ax=ax)

    plt.suptitle("Attention at Initialization (mean): Tiny vs Base (CIFAR-10)", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "attention_heatmap_tiny_vs_base.png"), dpi=150)
    print(f"Saved: {fig_dir}/attention_heatmap_tiny_vs_base.png")

    # 1b. Max-over-heads heatmaps (amplifies per-head structural patterns)
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    for ax, max_probs, title in [
        (axes[0], max_probs_t, f"HiT-Tiny (dim=192) max-over-heads"),
        (axes[1], max_probs_b, f"HiT-B (dim=768) max-over-heads"),
    ]:
        im = ax.imshow(max_probs, cmap="hot", aspect="auto")
        ax.set_title(title)
        ax.set_xlabel("Key token")
        ax.set_ylabel("Query token")
        for b in boundaries:
            ax.axhline(y=b - 0.5, color="cyan", linewidth=0.5, alpha=0.7)
            ax.axvline(x=b - 0.5, color="cyan", linewidth=0.5, alpha=0.7)
        plt.colorbar(im, ax=ax)

    plt.suptitle("Attention at Initialization (max over heads): Tiny vs Base (CIFAR-10)", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "attention_heatmap_max_heads.png"), dpi=150)
    print(f"Saved: {fig_dir}/attention_heatmap_max_heads.png")

    # 1c. Per-head heatmaps for HiT-Tiny (show individual head patterns)
    num_heads_t = probs_t.shape[1]
    fig, axes = plt.subplots(1, num_heads_t, figsize=(4 * num_heads_t, 4))
    per_head_t = probs_t.mean(dim=0).numpy()  # [heads, N, N]
    for h in range(num_heads_t):
        im = axes[h].imshow(per_head_t[h], cmap="hot", aspect="auto")
        axes[h].set_title(f"Head {h}")
        for b in boundaries:
            axes[h].axhline(y=b - 0.5, color="cyan", linewidth=0.3, alpha=0.5)
            axes[h].axvline(x=b - 0.5, color="cyan", linewidth=0.3, alpha=0.5)
        axes[h].set_xticks([])
        axes[h].set_yticks([])
    plt.suptitle("HiT-Tiny: Per-Head Attention at Initialization", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "attention_heatmap_per_head_tiny.png"), dpi=150)
    print(f"Saved: {fig_dir}/attention_heatmap_per_head_tiny.png")

    # 2. Bar chart comparison
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(3)
    width = 0.35
    tiny_vals = [stats_t["intra_probs"][0], stats_t["pc_probs"][0], stats_t["other_probs"][0]]
    base_vals = [stats_b["intra_probs"][0], stats_b["pc_probs"][0], stats_b["other_probs"][0]]
    ax.bar(x - width/2, tiny_vals, width, label=f"HiT-Tiny (ratio={stats_t['ratio']:.3f})", color="#DD8452")
    ax.bar(x + width/2, base_vals, width, label=f"HiT-B (ratio={stats_b['ratio']:.3f})", color="#4C72B0")
    ax.axhline(y=stats_t["uniform"], color="red", linestyle="--", alpha=0.5, label="Uniform")
    ax.set_xticks(x)
    ax.set_xticklabels(["Intra-level", "Parent-child", "Cross-other"])
    ax.set_ylabel("Mean Attention Probability")
    ax.set_title("Structural Attention Bias: Effect of Embedding Dimension")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "attention_bar_tiny_vs_base.png"), dpi=150)
    print(f"Saved: {fig_dir}/attention_bar_tiny_vs_base.png")

    # ---- Summary ----
    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  {'Model':15s} | embed_dim | PC/Intra ratio")
    print(f"  {'-'*15} | {'-'*9} | {'-'*14}")
    for name, stats in all_stats.items():
        dim = 192 if "Tiny" in name else 768
        print(f"  {name:15s} | {dim:9d} | {stats['ratio']:.4f}")
    print()
    if stats_t["ratio"] > stats_b["ratio"] * 1.1:
        print("  -> Structural bias is STRONGER in smaller models.")
        print("  -> Higher embed_dim dilutes pixel-space correlations more effectively.")
    else:
        print("  -> No significant difference between model sizes.")
    print("=" * 60)


if __name__ == "__main__":
    analyze()
