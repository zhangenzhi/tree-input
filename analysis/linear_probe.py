"""
Linear probe analysis: what information does each layer x level contain?

For both ViT-Tiny and HiT-Tiny (trained on CIFAR-10):
1. Train each model for 100 epochs
2. Freeze the model
3. For each layer, for each level (HiT) or the full patch set (ViT):
   - Extract token representations
   - Train a linear classifier on top (frozen features)
   - Report val accuracy

This directly answers:
- At which layer does each level become informative for classification?
- Does L4 gain classification ability after interacting with L0-L3?
- How does HiT's per-layer information distribution compare to ViT?
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


def get_cifar10(batch_size=64):
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
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, val_loader


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


# ---- Feature extraction ----

def get_level_indices(levels):
    """Return dict: level_idx -> list of token indices (1-based, after CLS)."""
    result = {}
    offset = 1
    for lvl_idx, n in enumerate(levels):
        result[lvl_idx] = list(range(offset, offset + n * n))
        offset += n * n
    return result


def extract_layer_features_vit(model, images):
    """Extract per-layer features from ViT. Returns list of [B, N, C] for each layer."""
    vit = model.vit
    x = vit.patch_embed(images)
    cls_tokens = vit.cls_token.expand(images.size(0), -1, -1)
    x = torch.cat([cls_tokens, x], dim=1)
    x = x + vit.pos_embed
    x = vit.pos_drop(x)

    layer_features = []
    for block in vit.blocks:
        x = x + block.attn(block.norm1(x))
        x = x + block.mlp(block.norm2(x))
        layer_features.append(x.detach())

    return layer_features


def extract_layer_features_hit(model, images):
    """Extract per-layer features from HiT. Returns list of [B, N, C] for each layer."""
    B = images.size(0)
    patches = extract_pyramid_patches(images, model.levels, model.patch_size)
    x = model.patch_proj(patches)
    pe = model.pe(model.pyramid_coords)
    x = x + pe.unsqueeze(0)
    cls_tokens = model.cls_token.expand(B, -1, -1) + model.cls_pos
    x = torch.cat([cls_tokens, x], dim=1)
    x = model.pos_drop(x)

    layer_features = []
    for block in model.blocks:
        x = x + block.attn(block.norm1(x))
        x = x + block.mlp(block.norm2(x))
        layer_features.append(x.detach())

    return layer_features


# ---- Feature dataset ----

def collect_features(model, loader, extract_fn, device):
    """Run model on full dataset, collect per-layer features.
    Returns: list of (features_tensor, labels_tensor) per layer.
    """
    model.eval()
    num_layers = None
    all_features = None
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            layer_feats = extract_fn(model, images)

            if num_layers is None:
                num_layers = len(layer_feats)
                all_features = [[] for _ in range(num_layers)]

            for l in range(num_layers):
                all_features[l].append(layer_feats[l].cpu())
            all_labels.append(labels)

    all_labels = torch.cat(all_labels, dim=0)
    for l in range(num_layers):
        all_features[l] = torch.cat(all_features[l], dim=0)

    return all_features, all_labels


def pool_level_features(features, token_indices):
    """Average pool features over specified token indices.
    features: [N_samples, seq_len, C]
    token_indices: list of int
    Returns: [N_samples, C]
    """
    return features[:, token_indices, :].mean(dim=1)


# ---- Linear probe ----

def train_linear_probe(train_features, train_labels, val_features, val_labels,
                       embed_dim, num_classes=10, lr=1e-2, epochs=50, device="cpu"):
    """Train a linear classifier on frozen features. Returns best val accuracy."""
    probe = nn.Linear(embed_dim, num_classes).to(device)
    optimizer = torch.optim.Adam(probe.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    train_features = train_features.to(device)
    train_labels = train_labels.to(device)
    val_features = val_features.to(device)
    val_labels = val_labels.to(device)

    best_acc = 0.0
    for epoch in range(epochs):
        # Train
        probe.train()
        # Mini-batch training for memory efficiency
        perm = torch.randperm(train_features.size(0))
        batch_size = 512
        for i in range(0, train_features.size(0), batch_size):
            idx = perm[i:i+batch_size]
            logits = probe(train_features[idx])
            loss = criterion(logits, train_labels[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Eval
        probe.eval()
        with torch.no_grad():
            val_logits = probe(val_features)
            _, predicted = val_logits.max(1)
            acc = 100.0 * predicted.eq(val_labels).sum().item() / val_labels.size(0)
        if acc > best_acc:
            best_acc = acc

    return best_acc


# ---- Main experiment ----

def train_backbone(model, train_loader, device, num_epochs=100, lr=1e-3, save_path=None):
    """Train model on CIFAR-10. Saves checkpoint to save_path."""
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

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

    if save_path:
        torch.save({
            "model": model.state_dict(),
            "epoch": num_epochs,
            "train_acc": 100. * correct / total,
        }, save_path)
        print(f"    Saved checkpoint: {save_path}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone_epochs", type=int, default=100)
    parser.add_argument("--probe_epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--probe_lr", type=float, default=1e-2)
    parser.add_argument("--ckpt_dir", type=str, default="./output/cifar10_ckpt",
                        help="Directory to save/load backbone checkpoints")
    args = parser.parse_args()

    device = get_device()
    print(f"Device: {device}")

    train_loader, val_loader = get_cifar10(args.batch_size)

    fig_dir = os.path.join(os.path.dirname(__file__), "figures")
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(args.ckpt_dir, exist_ok=True)
    log_path = os.path.join(fig_dir, "linear_probe.log")
    log_file = open(log_path, "w")

    def log(msg):
        print(msg)
        log_file.write(msg + "\n")
        log_file.flush()

    results = {}

    # ============================================================
    # ViT-Tiny
    # ============================================================
    vit_ckpt = os.path.join(args.ckpt_dir, "vit_tiny_cifar10.pt")
    torch.manual_seed(42)
    vit_model = ViTTiny(num_classes=10).to(device)

    if os.path.exists(vit_ckpt):
        log("=" * 70)
        log(f"Loading ViT-Tiny from {vit_ckpt}")
        log("=" * 70)
        ckpt = torch.load(vit_ckpt, map_location=device, weights_only=True)
        vit_model.load_state_dict(ckpt["model"])
        log(f"  Loaded (epoch={ckpt['epoch']}, train_acc={ckpt['train_acc']:.1f}%)")
    else:
        log("=" * 70)
        log("Training ViT-Tiny backbone...")
        log("=" * 70)
        train_backbone(vit_model, train_loader, device, args.backbone_epochs, args.lr,
                       save_path=vit_ckpt)

    log("\nCollecting ViT-Tiny features...")
    train_feats_vit, train_labels = collect_features(
        vit_model, train_loader, extract_layer_features_vit, device)
    val_feats_vit, val_labels = collect_features(
        vit_model, val_loader, extract_layer_features_vit, device)

    num_layers = len(train_feats_vit)
    embed_dim = train_feats_vit[0].shape[-1]

    # Probe configs for ViT: CLS token, all patches (avg pool), individual spatial regions
    vit_probe_configs = [
        ("CLS", [0]),
        ("All patches (avg)", list(range(1, 197))),
    ]

    log(f"\nViT-Tiny Linear Probe (probe_epochs={args.probe_epochs}):")
    log(f"{'Config':>25s}  " + "  ".join(f"L{l+1:2d}" for l in range(num_layers)))

    vit_results = {}
    for config_name, token_indices in vit_probe_configs:
        accs = []
        for layer_idx in range(num_layers):
            train_f = pool_level_features(train_feats_vit[layer_idx], token_indices)
            val_f = pool_level_features(val_feats_vit[layer_idx], token_indices)
            acc = train_linear_probe(
                train_f, train_labels, val_f, val_labels,
                embed_dim, num_classes=10, lr=args.probe_lr,
                epochs=args.probe_epochs, device=device)
            accs.append(acc)
        vit_results[config_name] = accs
        log(f"{config_name:>25s}  " + "  ".join(f"{a:5.1f}" for a in accs))

    results["ViT-Tiny"] = vit_results

    # Free memory
    del train_feats_vit, val_feats_vit, vit_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ============================================================
    # HiT-Tiny
    # ============================================================
    hit_ckpt = os.path.join(args.ckpt_dir, "hit_tiny_cifar10.pt")
    torch.manual_seed(42)
    hit_model = HiTTiny(num_classes=10).to(device)

    if os.path.exists(hit_ckpt):
        log("\n" + "=" * 70)
        log(f"Loading HiT-Tiny from {hit_ckpt}")
        log("=" * 70)
        ckpt = torch.load(hit_ckpt, map_location=device, weights_only=True)
        hit_model.load_state_dict(ckpt["model"])
        log(f"  Loaded (epoch={ckpt['epoch']}, train_acc={ckpt['train_acc']:.1f}%)")
    else:
        log("\n" + "=" * 70)
        log("Training HiT-Tiny backbone...")
        log("=" * 70)
        train_backbone(hit_model, train_loader, device, args.backbone_epochs, args.lr,
                       save_path=hit_ckpt)

    log("\nCollecting HiT-Tiny features...")
    train_feats_hit, train_labels = collect_features(
        hit_model, train_loader, extract_layer_features_hit, device)
    val_feats_hit, val_labels = collect_features(
        hit_model, val_loader, extract_layer_features_hit, device)

    level_indices = get_level_indices(LEVELS)
    level_names = [f"L{i}({n}x{n})" for i, n in enumerate(LEVELS)]

    # Probe configs for HiT
    hit_probe_configs = [
        ("CLS", [0]),
        ("All tokens (avg)", list(range(1, 1 + sum(n*n for n in LEVELS)))),
        ("L4 only (14x14)", level_indices[4]),
        ("L0-L3 only (coarse)", level_indices[0] + level_indices[1] + level_indices[2] + level_indices[3]),
    ]
    # Add individual levels
    for lvl_idx in range(len(LEVELS)):
        hit_probe_configs.append((level_names[lvl_idx], level_indices[lvl_idx]))

    log(f"\nHiT-Tiny Linear Probe (probe_epochs={args.probe_epochs}):")
    log(f"{'Config':>25s}  " + "  ".join(f"L{l+1:2d}" for l in range(num_layers)))

    hit_results = {}
    for config_name, token_indices in hit_probe_configs:
        accs = []
        for layer_idx in range(num_layers):
            train_f = pool_level_features(train_feats_hit[layer_idx], token_indices)
            val_f = pool_level_features(val_feats_hit[layer_idx], token_indices)
            acc = train_linear_probe(
                train_f, train_labels, val_f, val_labels,
                embed_dim, num_classes=10, lr=args.probe_lr,
                epochs=args.probe_epochs, device=device)
            accs.append(acc)
        hit_results[config_name] = accs
        log(f"{config_name:>25s}  " + "  ".join(f"{a:5.1f}" for a in accs))

    results["HiT-Tiny"] = hit_results

    # Save results
    with open(os.path.join(fig_dir, "linear_probe_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    log(f"\nSaved: {fig_dir}/linear_probe_results.json")

    # ---- Plot 1: ViT vs HiT CLS probe ----
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    layers = list(range(1, num_layers + 1))

    axes[0].plot(layers, vit_results["CLS"], marker="o", label="ViT CLS", linewidth=2)
    axes[0].plot(layers, hit_results["CLS"], marker="s", label="HiT CLS", linewidth=2)
    axes[0].plot(layers, vit_results["All patches (avg)"], marker="^", label="ViT All (avg)", linewidth=1.5, linestyle="--")
    axes[0].plot(layers, hit_results["All tokens (avg)"], marker="v", label="HiT All (avg)", linewidth=1.5, linestyle="--")
    axes[0].set_xlabel("Layer")
    axes[0].set_ylabel("Probe Val Accuracy (%)")
    axes[0].set_title("CLS and All-token Probe: ViT vs HiT")
    axes[0].set_xticks(layers)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # ---- Plot 2: HiT per-level probe ----
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(LEVELS)))
    for lvl_idx in range(len(LEVELS)):
        axes[1].plot(layers, hit_results[level_names[lvl_idx]], marker="o", markersize=4,
                     label=level_names[lvl_idx], color=colors[lvl_idx], linewidth=1.5)
    axes[1].plot(layers, hit_results["CLS"], marker="s", label="CLS", color="red",
                 linewidth=2, linestyle="--")
    axes[1].set_xlabel("Layer")
    axes[1].set_ylabel("Probe Val Accuracy (%)")
    axes[1].set_title("HiT-Tiny: Per-Level Linear Probe")
    axes[1].set_xticks(layers)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "linear_probe.png"), dpi=150)
    log(f"Saved: {fig_dir}/linear_probe.png")

    # ---- Plot 3: HiT L4 vs ViT All patches (key comparison) ----
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(layers, vit_results["All patches (avg)"], marker="o", label="ViT: All patches (196)", linewidth=2)
    ax.plot(layers, hit_results["L4 only (14x14)"], marker="s", label="HiT: L4 only (196)", linewidth=2)
    ax.plot(layers, hit_results["L0-L3 only (coarse)"], marker="^", label="HiT: L0-L3 only (85)", linewidth=2)
    ax.plot(layers, hit_results["All tokens (avg)"], marker="v", label="HiT: All (281)", linewidth=2, linestyle="--")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Probe Val Accuracy (%)")
    ax.set_title("Key Comparison: Does HiT's L4 gain info from L0-L3 interaction?")
    ax.set_xticks(layers)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "linear_probe_key_comparison.png"), dpi=150)
    log(f"Saved: {fig_dir}/linear_probe_key_comparison.png")

    # ---- Summary ----
    log("\n" + "=" * 70)
    log("KEY COMPARISONS")
    log("=" * 70)

    log("\n1. CLS token probe (classification readiness at each layer):")
    log(f"   {'':>10s}  " + "  ".join(f"L{l+1:2d}" for l in range(num_layers)))
    log(f"   {'ViT':>10s}  " + "  ".join(f"{a:5.1f}" for a in vit_results["CLS"]))
    log(f"   {'HiT':>10s}  " + "  ".join(f"{a:5.1f}" for a in hit_results["CLS"]))

    log("\n2. HiT L4 vs ViT patches (does L4 benefit from L0-L3 interaction?):")
    log(f"   {'':>15s}  " + "  ".join(f"L{l+1:2d}" for l in range(num_layers)))
    log(f"   {'ViT patches':>15s}  " + "  ".join(f"{a:5.1f}" for a in vit_results["All patches (avg)"]))
    log(f"   {'HiT L4 only':>15s}  " + "  ".join(f"{a:5.1f}" for a in hit_results["L4 only (14x14)"]))
    diff = [h - v for h, v in zip(hit_results["L4 only (14x14)"], vit_results["All patches (avg)"])]
    log(f"   {'Diff (H-V)':>15s}  " + "  ".join(f"{d:+5.1f}" for d in diff))

    log("\n3. Per-level probe accuracy at final layer:")
    for lvl_idx in range(len(LEVELS)):
        name = level_names[lvl_idx]
        acc = hit_results[name][-1]
        log(f"   {name:>15s}: {acc:.1f}%")

    log(f"\nLog saved to: {log_path}")
    log("Done.")
    log_file.close()


if __name__ == "__main__":
    main()
