"""
Verify whether cross-level attention bias exists at initialization (epoch 0).

This script:
1. Creates a randomly initialized HiT-B model
2. Feeds random images (or real images if available) through the first transformer block
3. Extracts the attention map from layer 0
4. Computes average attention weight for:
   - intra-level pairs (same pyramid level)
   - cross-level parent-child pairs (adjacent levels)
   - cross-level distant pairs (non-adjacent levels)
5. Plots the attention heatmap and prints statistics

If cross-level attention is significantly higher than intra-level at init,
the "structural similarity biases training" hypothesis has support.
If attention is roughly uniform, the hypothesis lacks foundation.

Runs on CPU, no GPU needed.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from model.hit import HiTBase, extract_pyramid_patches, build_pyramid_coords


LEVELS = [1, 2, 4, 8, 14]
PATCH_SIZE = 16
IMAGE_SIZE = 224


def get_level_ranges(levels):
    """Return (start, end) index for each level in the token sequence.
    Index 0 is CLS token, patches start at index 1.
    """
    ranges = {}
    offset = 1  # skip CLS
    for lvl_idx, n in enumerate(levels):
        count = n * n
        ranges[lvl_idx] = (offset, offset + count)
        offset += count
    return ranges


def get_parent_child_pairs(levels):
    """Return set of (parent_patch_idx, child_patch_idx) pairs.
    Indices are global patch indices (0-based, before adding CLS offset).
    """
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


def extract_attention(model, images):
    """Run forward through patch embedding + first transformer block,
    return attention weights from layer 0.
    """
    B = images.size(0)

    # Patch extraction and projection (same as HiTBase.forward)
    patches = extract_pyramid_patches(images, model.levels, model.patch_size)
    x = model.patch_proj(patches)
    pe = model.pe(model.pyramid_coords)
    x = x + pe.unsqueeze(0)
    cls_tokens = model.cls_token.expand(B, -1, -1) + model.cls_pos
    x = torch.cat([cls_tokens, x], dim=1)
    x = model.pos_drop(x)

    # Extract attention from first block
    block = model.blocks[0]
    y = block.norm1(x)

    # timm's Attention module
    attn_module = block.attn
    B, N, C = y.shape
    qkv = attn_module.qkv(y).reshape(B, N, 3, attn_module.num_heads, C // attn_module.num_heads)
    qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, heads, N, head_dim]
    q, k, v = qkv.unbind(0)

    scale = (C // attn_module.num_heads) ** -0.5
    attn_weights = (q @ k.transpose(-2, -1)) * scale  # [B, heads, N, N]
    attn_probs = attn_weights.softmax(dim=-1)

    return attn_weights.detach(), attn_probs.detach()


def analyze():
    print("=" * 60)
    print("HiT-B Attention at Initialization Analysis")
    print("=" * 60)
    print(f"Levels: {LEVELS}, Total patches: {sum(n*n for n in LEVELS)}")
    print(f"Sequence length: 1 (CLS) + {sum(n*n for n in LEVELS)} = {1 + sum(n*n for n in LEVELS)}")
    print()

    # Create model (random init)
    torch.manual_seed(42)
    model = HiTBase(
        image_size=IMAGE_SIZE,
        patch_size=PATCH_SIZE,
        num_classes=1000,
        levels=LEVELS,
        pretrained=False,
    )
    model.eval()

    # Use real CIFAR-10 images resized to 224x224
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

    with torch.no_grad():
        attn_logits, attn_probs = extract_attention(model, images)

    # Average over batch and heads: [N, N]
    avg_logits = attn_logits.mean(dim=(0, 1)).numpy()
    avg_probs = attn_probs.mean(dim=(0, 1)).numpy()

    # Compute statistics by region
    level_ranges = get_level_ranges(LEVELS)
    parent_child_pairs = get_parent_child_pairs(LEVELS)
    total_patches = sum(n * n for n in LEVELS)
    N = 1 + total_patches  # including CLS

    # Collect attention values
    intra_level_logits = []
    cross_parent_child_logits = []
    cross_other_logits = []
    intra_level_probs = []
    cross_parent_child_probs = []
    cross_other_probs = []

    # Build parent-child lookup with CLS offset (+1)
    pc_set = set()
    for (pi, ci) in parent_child_pairs:
        pc_set.add((pi + 1, ci + 1))
        pc_set.add((ci + 1, pi + 1))  # symmetric

    for i in range(1, N):  # skip CLS as query for cleaner analysis
        for j in range(1, N):
            if i == j:
                continue
            # Determine if same level
            i_level = None
            j_level = None
            for lvl, (s, e) in level_ranges.items():
                if s <= i < e:
                    i_level = lvl
                if s <= j < e:
                    j_level = lvl

            logit_val = avg_logits[i, j]
            prob_val = avg_probs[i, j]

            if i_level == j_level:
                intra_level_logits.append(logit_val)
                intra_level_probs.append(prob_val)
            elif (i, j) in pc_set:
                cross_parent_child_logits.append(logit_val)
                cross_parent_child_probs.append(prob_val)
            else:
                cross_other_logits.append(logit_val)
                cross_other_probs.append(prob_val)

    uniform_prob = 1.0 / N

    print("--- Attention LOGITS (before softmax) ---")
    print(f"  Intra-level:          mean={np.mean(intra_level_logits):.6f}  std={np.std(intra_level_logits):.6f}  n={len(intra_level_logits)}")
    print(f"  Cross parent-child:   mean={np.mean(cross_parent_child_logits):.6f}  std={np.std(cross_parent_child_logits):.6f}  n={len(cross_parent_child_logits)}")
    print(f"  Cross other:          mean={np.mean(cross_other_logits):.6f}  std={np.std(cross_other_logits):.6f}  n={len(cross_other_logits)}")
    print()
    print("--- Attention PROBABILITIES (after softmax) ---")
    print(f"  Uniform baseline:     {uniform_prob:.6f}")
    print(f"  Intra-level:          mean={np.mean(intra_level_probs):.6f}  std={np.std(intra_level_probs):.6f}")
    print(f"  Cross parent-child:   mean={np.mean(cross_parent_child_probs):.6f}  std={np.std(cross_parent_child_probs):.6f}")
    print(f"  Cross other:          mean={np.mean(cross_other_probs):.6f}  std={np.std(cross_other_probs):.6f}")
    print()

    # Ratio: how much larger is parent-child vs intra-level?
    ratio = np.mean(cross_parent_child_probs) / np.mean(intra_level_probs)
    print(f"  Parent-child / Intra-level ratio: {ratio:.4f}")
    print(f"  (ratio=1.0 means no bias; >1.0 means cross-level bias exists)")
    print()

    # Also compare with standard ViT baseline attention
    print("--- For reference: Standard ViT-B (no pyramid) ---")
    import timm
    vit = timm.create_model("vit_base_patch16_224", pretrained=False, num_classes=1000)
    vit.eval()
    with torch.no_grad():
        # Get attention from ViT layer 0
        vit_x = vit.patch_embed(images)
        vit_cls = vit.cls_token.expand(4, -1, -1)
        vit_x = torch.cat([vit_cls, vit_x], dim=1)
        vit_x = vit_x + vit.pos_embed
        vit_x = vit.pos_drop(vit_x)

        vit_block = vit.blocks[0]
        vit_y = vit_block.norm1(vit_x)
        vit_attn = vit_block.attn
        B2, N2, C2 = vit_y.shape
        qkv2 = vit_attn.qkv(vit_y).reshape(B2, N2, 3, vit_attn.num_heads, C2 // vit_attn.num_heads)
        qkv2 = qkv2.permute(2, 0, 3, 1, 4)
        q2, k2, v2 = qkv2.unbind(0)
        scale2 = (C2 // vit_attn.num_heads) ** -0.5
        vit_attn_probs = (q2 @ k2.transpose(-2, -1) * scale2).softmax(dim=-1)
        vit_avg_probs = vit_attn_probs.mean(dim=(0, 1)).numpy()

    vit_uniform = 1.0 / N2
    # Adjacent patches (horizontal neighbors) vs distant
    vit_adjacent = []
    vit_distant = []
    grid = 14
    for i in range(1, N2):
        for j in range(1, N2):
            if i == j:
                continue
            ri, ci_ = (i - 1) // grid, (i - 1) % grid
            rj, cj = (j - 1) // grid, (j - 1) % grid
            dist = abs(ri - rj) + abs(ci_ - cj)
            if dist == 1:
                vit_adjacent.append(vit_avg_probs[i, j])
            elif dist > 5:
                vit_distant.append(vit_avg_probs[i, j])

    print(f"  Uniform baseline:     {vit_uniform:.6f}")
    print(f"  Adjacent (dist=1):    mean={np.mean(vit_adjacent):.6f}  std={np.std(vit_adjacent):.6f}")
    print(f"  Distant (dist>5):     mean={np.mean(vit_distant):.6f}  std={np.std(vit_distant):.6f}")
    print(f"  Adjacent / Distant:   {np.mean(vit_adjacent) / np.mean(vit_distant):.4f}")
    print()

    # ---- Plots ----
    os.makedirs(os.path.join(os.path.dirname(__file__), "figures"), exist_ok=True)
    fig_dir = os.path.join(os.path.dirname(__file__), "figures")

    # 1. Full attention heatmap
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    im0 = axes[0].imshow(avg_probs, cmap="viridis", aspect="auto")
    axes[0].set_title("HiT-B: Attention Probs (Layer 0, avg over batch & heads)")
    axes[0].set_xlabel("Key token")
    axes[0].set_ylabel("Query token")
    # Draw level boundaries
    boundaries = [0]
    for n in LEVELS:
        boundaries.append(boundaries[-1] + n * n)
    boundaries = [b + 1 for b in boundaries]  # +1 for CLS
    for b in boundaries:
        axes[0].axhline(y=b - 0.5, color="red", linewidth=0.5, alpha=0.7)
        axes[0].axvline(x=b - 0.5, color="red", linewidth=0.5, alpha=0.7)
    plt.colorbar(im0, ax=axes[0])

    im1 = axes[1].imshow(vit_avg_probs, cmap="viridis", aspect="auto")
    axes[1].set_title("ViT-B: Attention Probs (Layer 0, avg over batch & heads)")
    axes[1].set_xlabel("Key token")
    axes[1].set_ylabel("Query token")
    plt.colorbar(im1, ax=axes[1])

    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "attention_heatmap_init.png"), dpi=150)
    print(f"Saved: {fig_dir}/attention_heatmap_init.png")

    # 2. Box plot comparison
    fig, ax = plt.subplots(figsize=(8, 5))
    data = [intra_level_probs, cross_parent_child_probs, cross_other_probs]
    labels = ["Intra-level", "Parent-Child", "Cross-other"]
    bp = ax.boxplot(data, labels=labels, showfliers=False, patch_artist=True)
    colors = ["#4C72B0", "#DD8452", "#55A868"]
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.axhline(y=uniform_prob, color="red", linestyle="--", label=f"Uniform={uniform_prob:.5f}")
    ax.set_ylabel("Attention Probability")
    ax.set_title("HiT-B Layer 0 Attention Distribution at Initialization")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "attention_boxplot_init.png"), dpi=150)
    print(f"Saved: {fig_dir}/attention_boxplot_init.png")

    print()
    print("=" * 60)
    print("CONCLUSION:")
    if ratio > 1.05:
        print(f"  Cross-level parent-child attention is {ratio:.2f}x higher than intra-level.")
        print("  -> Structural similarity bias EXISTS at initialization.")
        print("  -> The hypothesis has empirical support.")
    elif ratio > 0.95:
        print(f"  Cross-level parent-child attention is ~equal to intra-level (ratio={ratio:.2f}).")
        print("  -> Attention is roughly UNIFORM at initialization.")
        print("  -> The structural bias hypothesis LACKS support.")
    else:
        print(f"  Cross-level parent-child attention is LOWER than intra-level (ratio={ratio:.2f}).")
        print("  -> No structural bias detected.")
    print("=" * 60)


if __name__ == "__main__":
    analyze()
