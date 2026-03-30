"""
Training dynamics analysis for HiT-Tiny.

Periodically snapshots during training (every N epochs):
  1. Attention heatmap (max-over-heads) for all layers
  2. Attention entropy per layer
  3. CLS token attention distribution across pyramid levels per layer
  4. Value norm per pyramid level per layer
  5. Feature norm per pyramid level per layer (after each block)

Saves all data to JSON + generates visualizations.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import json
import time
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
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


def get_level_indices(levels):
    """Return dict: level_idx -> list of token indices (1-based, after CLS)."""
    result = {}
    offset = 1
    for lvl_idx, n in enumerate(levels):
        result[lvl_idx] = list(range(offset, offset + n * n))
        offset += n * n
    return result


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


def extract_dynamics(model, images):
    """Forward pass collecting per-layer diagnostics.

    Returns dict with:
      attn_probs: list of [B, heads, N, N] per layer
      attn_values: list of [B, heads, N, head_dim] per layer (attention-weighted values)
      features: list of [B, N, C] per layer (output of each block)
    """
    B = images.size(0)
    patches = extract_pyramid_patches(images, model.levels, model.patch_size)
    x = model.patch_proj(patches)
    pe = model.pe(model.pyramid_coords)
    x = x + pe.unsqueeze(0)
    cls_tokens = model.cls_token.expand(B, -1, -1) + model.cls_pos
    x = torch.cat([cls_tokens, x], dim=1)
    x = model.pos_drop(x)

    all_attn_probs = []
    all_value_norms = []
    all_feature_norms = []

    for block in model.blocks:
        y = block.norm1(x)
        attn_mod = block.attn
        B2, N, C = y.shape
        head_dim = C // attn_mod.num_heads

        qkv = attn_mod.qkv(y).reshape(B2, N, 3, attn_mod.num_heads, head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        scale = head_dim ** -0.5
        attn_probs = (q @ k.transpose(-2, -1) * scale).softmax(dim=-1)
        all_attn_probs.append(attn_probs.detach().cpu())

        # Value norms after attention weighting: [B, heads, N, head_dim]
        attn_out = attn_probs @ v  # [B, heads, N, head_dim]
        v_norms = attn_out.norm(dim=-1).detach().cpu()  # [B, heads, N]
        all_value_norms.append(v_norms)

        # Forward through block
        x = x + block.attn(block.norm1(x))
        x = x + block.mlp(block.norm2(x))

        # Feature norms after block: [B, N, C]
        feat_norms = x.norm(dim=-1).detach().cpu()  # [B, N]
        all_feature_norms.append(feat_norms)

    return all_attn_probs, all_value_norms, all_feature_norms


def compute_snapshot(all_attn_probs, all_value_norms, all_feature_norms, levels):
    """Compute all metrics from a single forward pass."""
    level_indices = get_level_indices(levels)
    num_layers = len(all_attn_probs)
    num_levels = len(levels)
    N = all_attn_probs[0].shape[2]

    snapshot = {
        "entropy_per_layer": [],
        "cls_level_attn": [],      # [num_layers, num_levels]
        "value_norm_per_level": [], # [num_layers, num_levels]
        "feature_norm_per_level": [],  # [num_layers, num_levels]
        "heatmaps_max": [],         # [num_layers] of [N, N] (max over heads, mean over batch)
    }

    for layer_idx in range(num_layers):
        attn = all_attn_probs[layer_idx]  # [B, heads, N, N]
        v_norms = all_value_norms[layer_idx]  # [B, heads, N]
        f_norms = all_feature_norms[layer_idx]  # [B, N]

        # 1. Entropy: -sum(p * log(p)), averaged over batch, heads, queries
        p = attn.float()
        log_p = torch.log(p + 1e-9)
        entropy = -(p * log_p).sum(dim=-1).mean().item()  # scalar
        snapshot["entropy_per_layer"].append(entropy)

        # 2. CLS (token 0) attention to each level
        cls_attn = attn[:, :, 0, :].mean(dim=(0, 1)).numpy()  # [N]
        cls_level = []
        for lvl_idx in range(num_levels):
            indices = level_indices[lvl_idx]
            cls_level.append(float(cls_attn[indices].sum()))
        snapshot["cls_level_attn"].append(cls_level)

        # 3. Value norm per level (mean over batch, heads)
        v_avg = v_norms.mean(dim=(0, 1)).numpy()  # [N]
        v_level = []
        for lvl_idx in range(num_levels):
            indices = level_indices[lvl_idx]
            v_level.append(float(v_avg[indices].mean()))
        snapshot["value_norm_per_level"].append(v_level)

        # 4. Feature norm per level (mean over batch)
        f_avg = f_norms.mean(dim=0).numpy()  # [N]
        f_level = []
        for lvl_idx in range(num_levels):
            indices = level_indices[lvl_idx]
            f_level.append(float(f_avg[indices].mean()))
        snapshot["feature_norm_per_level"].append(f_level)

        # 5. Heatmap: max over heads, mean over batch
        heatmap = attn.max(dim=1).values.mean(dim=0).numpy()  # [N, N]
        snapshot["heatmaps_max"].append(heatmap.tolist())

    return snapshot


def plot_snapshot(snapshot, epoch, levels, fig_dir):
    """Save a comprehensive figure for one snapshot."""
    num_layers = len(snapshot["entropy_per_layer"])
    num_levels = len(levels)
    level_names = [f"L{i}({n}x{n})" for i, n in enumerate(levels)]
    layers = list(range(1, num_layers + 1))

    boundaries = [0]
    for n in levels:
        boundaries.append(boundaries[-1] + n * n)
    boundaries = [b + 1 for b in boundaries]

    fig = plt.figure(figsize=(20, 16))
    gs = GridSpec(3, num_layers, figure=fig, hspace=0.35, wspace=0.3)

    # Row 1: Heatmaps per layer
    for l in range(num_layers):
        ax = fig.add_subplot(gs[0, l])
        heatmap = np.array(snapshot["heatmaps_max"][l])
        ax.imshow(heatmap, cmap="hot", aspect="auto")
        ax.set_title(f"L{l+1}", fontsize=9)
        for b in boundaries:
            ax.axhline(y=b - 0.5, color="cyan", linewidth=0.3, alpha=0.5)
            ax.axvline(x=b - 0.5, color="cyan", linewidth=0.3, alpha=0.5)
        ax.set_xticks([])
        ax.set_yticks([])

    # Row 2: Entropy + CLS level attention
    ax_ent = fig.add_subplot(gs[1, :num_layers // 2])
    ax_ent.bar(layers, snapshot["entropy_per_layer"], color="#4C72B0", alpha=0.8)
    ax_ent.set_xlabel("Layer")
    ax_ent.set_ylabel("Entropy")
    ax_ent.set_title("Attention Entropy per Layer")
    ax_ent.set_xticks(layers)
    ax_ent.grid(True, alpha=0.3, axis="y")

    ax_cls = fig.add_subplot(gs[1, num_layers // 2:])
    cls_data = np.array(snapshot["cls_level_attn"])  # [num_layers, num_levels]
    bottom = np.zeros(num_layers)
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, num_levels))
    for lvl_idx in range(num_levels):
        ax_cls.bar(layers, cls_data[:, lvl_idx], bottom=bottom,
                   label=level_names[lvl_idx], color=colors[lvl_idx], alpha=0.8)
        bottom += cls_data[:, lvl_idx]
    ax_cls.set_xlabel("Layer")
    ax_cls.set_ylabel("Attention Fraction")
    ax_cls.set_title("CLS Token Attention to Each Level")
    ax_cls.set_xticks(layers)
    ax_cls.legend(fontsize=7, loc="upper right")
    ax_cls.grid(True, alpha=0.3, axis="y")

    # Row 3: Value norm + Feature norm per level
    ax_val = fig.add_subplot(gs[2, :num_layers // 2])
    val_data = np.array(snapshot["value_norm_per_level"])  # [num_layers, num_levels]
    for lvl_idx in range(num_levels):
        ax_val.plot(layers, val_data[:, lvl_idx], marker="o", markersize=3,
                    label=level_names[lvl_idx], color=colors[lvl_idx])
    ax_val.set_xlabel("Layer")
    ax_val.set_ylabel("Mean Value Norm")
    ax_val.set_title("Attention-Weighted Value Norm per Level")
    ax_val.set_xticks(layers)
    ax_val.legend(fontsize=7)
    ax_val.grid(True, alpha=0.3)

    ax_feat = fig.add_subplot(gs[2, num_layers // 2:])
    feat_data = np.array(snapshot["feature_norm_per_level"])  # [num_layers, num_levels]
    for lvl_idx in range(num_levels):
        ax_feat.plot(layers, feat_data[:, lvl_idx], marker="o", markersize=3,
                     label=level_names[lvl_idx], color=colors[lvl_idx])
    ax_feat.set_xlabel("Layer")
    ax_feat.set_ylabel("Mean Feature Norm")
    ax_feat.set_title("Feature Norm per Level (after block)")
    ax_feat.set_xticks(layers)
    ax_feat.legend(fontsize=7)
    ax_feat.grid(True, alpha=0.3)

    fig.suptitle(f"HiT-Tiny Training Dynamics — Epoch {epoch}", fontsize=16)
    plt.savefig(os.path.join(fig_dir, f"dynamics_epoch_{epoch:03d}.png"), dpi=120)
    plt.close(fig)


def plot_evolution(all_snapshots, epochs, levels, fig_dir):
    """Plot how metrics evolve across training."""
    num_levels = len(levels)
    level_names = [f"L{i}({n}x{n})" for i, n in enumerate(levels)]
    num_layers = len(all_snapshots[0]["entropy_per_layer"])
    colors_level = plt.cm.viridis(np.linspace(0.2, 0.9, num_levels))

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Entropy evolution per layer
    for l in range(num_layers):
        vals = [s["entropy_per_layer"][l] for s in all_snapshots]
        axes[0, 0].plot(epochs, vals, label=f"Layer {l+1}", linewidth=1.5)
    axes[0, 0].set_title("Attention Entropy per Layer Over Training")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Entropy")
    axes[0, 0].legend(fontsize=7)
    axes[0, 0].grid(True, alpha=0.3)

    # 2. CLS attention to L0 (whole image) over training per layer
    for l in range(num_layers):
        vals = [s["cls_level_attn"][l][0] for s in all_snapshots]
        axes[0, 1].plot(epochs, vals, label=f"Layer {l+1}", linewidth=1.5)
    axes[0, 1].set_title("CLS Attention to Level 0 (whole image) Over Training")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Attention Fraction")
    axes[0, 1].legend(fontsize=7)
    axes[0, 1].grid(True, alpha=0.3)

    # 3. Feature norm ratio: L0 / L4 over training (layer 1 and last layer)
    for l in [0, num_layers - 1]:
        vals = [s["feature_norm_per_level"][l][0] / (s["feature_norm_per_level"][l][-1] + 1e-9)
                for s in all_snapshots]
        axes[1, 0].plot(epochs, vals, label=f"Layer {l+1}", linewidth=1.5)
    axes[1, 0].set_title("Feature Norm Ratio (L0/L4) Over Training")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("Norm Ratio")
    axes[1, 0].axhline(y=1.0, color="red", linestyle="--", alpha=0.3)
    axes[1, 0].legend(fontsize=7)
    axes[1, 0].grid(True, alpha=0.3)

    # 4. Value norm of L0 vs L4 in last layer
    for lvl_idx in [0, len(levels) - 1]:
        vals = [s["value_norm_per_level"][-1][lvl_idx] for s in all_snapshots]
        axes[1, 1].plot(epochs, vals, label=f"{level_names[lvl_idx]} (last layer)", linewidth=1.5)
    axes[1, 1].set_title("Value Norm in Last Layer Over Training")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("Value Norm")
    axes[1, 1].legend(fontsize=7)
    axes[1, 1].grid(True, alpha=0.3)

    plt.suptitle("HiT-Tiny Training Dynamics Evolution", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "dynamics_evolution.png"), dpi=150)
    print(f"Saved: {fig_dir}/dynamics_evolution.png")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--snapshot_interval", type=int, default=5)
    args = parser.parse_args()

    device = get_device()
    print(f"Device: {device}")
    print(f"Epochs: {args.num_epochs}, snapshot every {args.snapshot_interval} epochs")

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
    probe_loader = DataLoader(val_set, batch_size=32, shuffle=False, num_workers=0)
    probe_images, _ = next(iter(probe_loader))

    fig_dir = os.path.join(os.path.dirname(__file__), "figures", "dynamics")
    os.makedirs(fig_dir, exist_ok=True)

    # Model
    torch.manual_seed(42)
    model = HiTTiny(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs)

    all_snapshots = []
    snapshot_epochs = []

    # Snapshot at init
    model.eval()
    with torch.no_grad():
        attn_p, val_n, feat_n = extract_dynamics(model, probe_images.to(device))
    snap = compute_snapshot(attn_p, val_n, feat_n, LEVELS)
    all_snapshots.append(snap)
    snapshot_epochs.append(0)
    plot_snapshot(snap, 0, LEVELS, fig_dir)
    print(f"  [Snapshot epoch 0] entropy_L1={snap['entropy_per_layer'][0]:.3f}")

    for epoch in range(args.num_epochs):
        model.train()
        correct, total, epoch_loss = 0, 0, 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * labels.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
        scheduler.step()

        train_acc = 100.0 * correct / total
        train_loss = epoch_loss / total

        if (epoch + 1) % args.snapshot_interval == 0 or epoch == 0:
            model.eval()
            with torch.no_grad():
                attn_p, val_n, feat_n = extract_dynamics(model, probe_images.to(device))
            snap = compute_snapshot(attn_p, val_n, feat_n, LEVELS)
            all_snapshots.append(snap)
            snapshot_epochs.append(epoch + 1)
            plot_snapshot(snap, epoch + 1, LEVELS, fig_dir)
            print(f"  [Epoch {epoch+1:3d}] loss={train_loss:.4f} acc={train_acc:.1f}% | "
                  f"entropy_L1={snap['entropy_per_layer'][0]:.3f} "
                  f"cls_L0={snap['cls_level_attn'][0][0]:.4f}")
        elif (epoch + 1) % 10 == 0:
            print(f"  [Epoch {epoch+1:3d}] loss={train_loss:.4f} acc={train_acc:.1f}%")

    # ---- Level Ablation Test ----
    # After training, test val_acc with different level subsets removed
    print()
    print("=" * 60)
    print("Level Ablation Test (trained model)")
    print("=" * 60)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=0)

    ablation_configs = [
        ("Full (L0-L4)", None),           # all levels, no ablation
        ("L4 only (fine)", [4]),           # only finest patches, like standard ViT
        ("L0-L3 only (coarse)", [0, 1, 2, 3]),  # only coarse levels, no fine detail
        ("L3-L4", [3, 4]),                # skip global levels
        ("L0+L4", [0, 4]),                # global + fine only, skip intermediate
    ]

    level_offsets = {}
    offset = 1  # skip CLS
    for lvl_idx, n in enumerate(LEVELS):
        level_offsets[lvl_idx] = (offset, offset + n * n)
        offset += n * n

    ablation_results = []

    for config_name, keep_levels in ablation_configs:
        model.eval()
        correct, total_count = 0, 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                B = images.size(0)

                # Build full sequence
                patches = extract_pyramid_patches(images, model.levels, model.patch_size)
                x = model.patch_proj(patches)
                pe = model.pe(model.pyramid_coords)
                x = x + pe.unsqueeze(0)
                cls_tokens = model.cls_token.expand(B, -1, -1) + model.cls_pos
                x = torch.cat([cls_tokens, x], dim=1)  # [B, 1+281, C]

                if keep_levels is not None:
                    # Keep CLS (index 0) + selected levels
                    keep_indices = [0]  # CLS
                    for lvl_idx in keep_levels:
                        s, e = level_offsets[lvl_idx]
                        keep_indices.extend(range(s, e))
                    x = x[:, keep_indices, :]

                x = model.pos_drop(x)
                x = model.blocks(x)
                x = model.norm(x)
                logits = model.head(x[:, 0])

                _, predicted = logits.max(1)
                correct += predicted.eq(labels).sum().item()
                total_count += labels.size(0)

        acc = 100.0 * correct / total_count
        n_tokens = x.shape[1] if keep_levels is not None else 1 + sum(n * n for n in LEVELS)
        ablation_results.append({"config": config_name, "val_acc": acc, "tokens": n_tokens})
        print(f"  {config_name:25s} | tokens={n_tokens:4d} | val_acc={acc:.1f}%")

    # Save ablation results
    with open(os.path.join(fig_dir, "ablation_results.json"), "w") as f:
        json.dump(ablation_results, f, indent=2)
    print(f"\nSaved: {fig_dir}/ablation_results.json")

    # Ablation bar chart
    fig, ax = plt.subplots(figsize=(10, 5))
    names = [r["config"] for r in ablation_results]
    accs = [r["val_acc"] for r in ablation_results]
    tokens = [r["tokens"] for r in ablation_results]
    bars = ax.bar(range(len(names)), accs, color=["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2"])
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=15, ha="right")
    ax.set_ylabel("Val Acc (%)")
    ax.set_title("HiT-Tiny Level Ablation (trained 100 epochs)")
    for i, (bar, tok) in enumerate(zip(bars, tokens)):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f"{tok}t", ha="center", fontsize=8, color="gray")
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "level_ablation.png"), dpi=150)
    print(f"Saved: {fig_dir}/level_ablation.png")
    print()

    # Save all snapshots
    serializable = []
    for snap in all_snapshots:
        s = {k: v for k, v in snap.items() if k != "heatmaps_max"}
        serializable.append(s)
    with open(os.path.join(fig_dir, "dynamics_history.json"), "w") as f:
        json.dump({"epochs": snapshot_epochs, "snapshots": serializable}, f, indent=2)
    print(f"\nSaved: {fig_dir}/dynamics_history.json")

    # Evolution plot
    plot_evolution(all_snapshots, snapshot_epochs, LEVELS, fig_dir)

    print("Done.")


if __name__ == "__main__":
    main()
