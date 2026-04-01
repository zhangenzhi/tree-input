"""
Micro-prefix experiment: replace macro L0-L3 with 85 random 8x8 micro-crops.

Compares three models:
- ViT-Tiny: 196 tokens (baseline)
- HiT-Tiny (macro-prefix): 85 coarse + 196 fine = 281 tokens
- HiT-micro (micro-prefix): 85 random 8x8 crops + 196 fine = 281 tokens

Tests whether fine-grained micro-patches provide a different kind of
regularization (shallow-layer) vs macro-prefix (deep-layer).

Runs training + linear probe for HiT-micro. Loads ViT and HiT checkpoints
if available from previous runs.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import timm
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model.hit import build_pyramid_coords, ContinuousPE3D, extract_pyramid_patches


LEVELS = [1, 2, 4, 8, 14]
NUM_MICRO = 85  # same as L0-L3 total tokens (1+4+16+64)
MICRO_CROP_SIZE = 8  # crop 8x8 from original image
PATCH_SIZE = 16


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


# ---- Micro crop extraction ----

def extract_micro_crops(images, num_crops=85, crop_size=8, patch_size=16, training=True):
    """Extract random 8x8 crops from images, resize to 16x16.

    Args:
        images: [B, 3, H, W]
        num_crops: number of micro crops to extract
        crop_size: size of each crop in the original image
        patch_size: resize target
        training: if True, random locations; if False, fixed grid for reproducibility

    Returns:
        crops: [B, num_crops, 3*patch_size*patch_size]
        coords: [num_crops, 3] (cx, cy, s) for PE
    """
    B, C, H, W = images.shape

    if training:
        # Random crop locations
        top = torch.randint(0, H - crop_size + 1, (num_crops,))
        left = torch.randint(0, W - crop_size + 1, (num_crops,))
    else:
        # Fixed deterministic grid for eval: spread crops evenly
        torch.manual_seed(0)
        top = torch.randint(0, H - crop_size + 1, (num_crops,))
        left = torch.randint(0, W - crop_size + 1, (num_crops,))

    crops = []
    coords = []
    for i in range(num_crops):
        t, l = top[i].item(), left[i].item()
        crop = images[:, :, t:t+crop_size, l:l+crop_size]  # [B, C, crop_size, crop_size]
        resized = F.interpolate(crop, size=(patch_size, patch_size),
                                mode='bilinear', align_corners=False)  # [B, C, P, P]
        crops.append(resized.reshape(B, -1))  # [B, C*P*P]

        # Continuous coordinates
        cx = (l + crop_size / 2) / W
        cy = (t + crop_size / 2) / H
        s = crop_size / H  # scale relative to image
        coords.append([cx, cy, s])

    crops = torch.stack(crops, dim=1)  # [B, num_crops, C*P*P]
    coords = torch.tensor(coords, dtype=torch.float32)  # [num_crops, 3]
    return crops, coords


# ---- HiT-micro model ----

class HiTMicroTiny(nn.Module):
    """HiT with micro-prefix: 85 random 8x8 crops + 196 standard patches."""

    def __init__(self, num_classes=10, num_micro=85, crop_size=8):
        super().__init__()
        self.num_micro = num_micro
        self.crop_size = crop_size
        self.patch_size = 16
        self.levels = [14]  # only finest level for L4
        self.total_fine = 14 * 14  # 196

        vit = timm.create_model("vit_tiny_patch16_224", pretrained=False, num_classes=num_classes)
        self.embed_dim = vit.embed_dim  # 192

        # Shared projection for both micro and fine patches (both are 16x16x3)
        self.patch_proj = nn.Linear(16 * 16 * 3, self.embed_dim)

        self.cls_token = vit.cls_token
        self.pe = ContinuousPE3D(self.embed_dim, num_freq=32)

        # Fine-level (L4) coordinates
        fine_coords = build_pyramid_coords([14])
        self.register_buffer("fine_coords", fine_coords)  # [196, 3]

        self.cls_pos = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        self.pos_drop = vit.pos_drop
        self.blocks = vit.blocks
        self.norm = vit.norm
        self.head = vit.head

    def forward(self, images):
        B = images.size(0)

        # 1. Extract micro crops (random during training, fixed during eval)
        micro_patches, micro_coords = extract_micro_crops(
            images, self.num_micro, self.crop_size, self.patch_size,
            training=self.training
        )
        micro_coords = micro_coords.to(images.device)

        # 2. Extract fine-level patches (standard 14x14 grid)
        fine_patches = extract_pyramid_patches(images, [14], self.patch_size)  # [B, 196, C*P*P]

        # 3. Concatenate: [micro | fine]
        all_patches = torch.cat([micro_patches, fine_patches], dim=1)  # [B, 281, C*P*P]

        # 4. Project
        x = self.patch_proj(all_patches)  # [B, 281, embed_dim]

        # 5. Positional encoding
        all_coords = torch.cat([micro_coords, self.fine_coords], dim=0)  # [281, 3]
        pe = self.pe(all_coords)
        x = x + pe.unsqueeze(0)

        # 6. CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1) + self.cls_pos
        x = torch.cat([cls_tokens, x], dim=1)  # [B, 282, embed_dim]

        # 7. Transformer
        x = self.pos_drop(x)
        x = self.blocks(x)
        x = self.norm(x)

        return self.head(x[:, 0])


# ---- Feature extraction ----

def extract_layer_features_micro(model, images):
    """Extract per-layer features from HiT-micro."""
    B = images.size(0)

    micro_patches, micro_coords = extract_micro_crops(
        images, model.num_micro, model.crop_size, model.patch_size,
        training=False  # fixed locations for feature extraction
    )
    micro_coords = micro_coords.to(images.device)
    fine_patches = extract_pyramid_patches(images, [14], model.patch_size)

    all_patches = torch.cat([micro_patches, fine_patches], dim=1)
    x = model.patch_proj(all_patches)

    all_coords = torch.cat([micro_coords, model.fine_coords], dim=0)
    pe = model.pe(all_coords)
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


def extract_layer_features_vit(model, images):
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


# ---- Shared utilities ----

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


class HiTRandomTiny(nn.Module):
    """HiT with random-prefix: 85 duplicated random fine-level patches + 196 standard patches.
    Same token count as HiT, but prefix carries redundant (not new) information.
    """

    def __init__(self, num_classes=10, num_random=85):
        super().__init__()
        self.num_random = num_random
        self.patch_size = 16
        self.levels = [14]
        self.total_fine = 14 * 14

        vit = timm.create_model("vit_tiny_patch16_224", pretrained=False, num_classes=num_classes)
        self.embed_dim = vit.embed_dim

        self.patch_proj = nn.Linear(16 * 16 * 3, self.embed_dim)
        self.cls_token = vit.cls_token
        self.pe = ContinuousPE3D(self.embed_dim, num_freq=32)

        fine_coords = build_pyramid_coords([14])
        self.register_buffer("fine_coords", fine_coords)

        self.cls_pos = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        self.pos_drop = vit.pos_drop
        self.blocks = vit.blocks
        self.norm = vit.norm
        self.head = vit.head

    def forward(self, images):
        B = images.size(0)

        # Extract standard fine patches
        fine_patches = extract_pyramid_patches(images, [14], self.patch_size)  # [B, 196, C*P*P]

        # Randomly duplicate 85 patches from fine level
        if self.training:
            indices = torch.randint(0, 196, (self.num_random,))
        else:
            torch.manual_seed(0)
            indices = torch.randint(0, 196, (self.num_random,))

        random_patches = fine_patches[:, indices, :]  # [B, 85, C*P*P]
        random_coords = self.fine_coords[indices]  # [85, 3]

        # Concatenate: [random | fine]
        all_patches = torch.cat([random_patches, fine_patches], dim=1)
        x = self.patch_proj(all_patches)

        all_coords = torch.cat([random_coords, self.fine_coords], dim=0)
        pe = self.pe(all_coords)
        x = x + pe.unsqueeze(0)

        cls_tokens = self.cls_token.expand(B, -1, -1) + self.cls_pos
        x = torch.cat([cls_tokens, x], dim=1)

        x = self.pos_drop(x)
        x = self.blocks(x)
        x = self.norm(x)
        return self.head(x[:, 0])


def extract_layer_features_random(model, images):
    """Extract per-layer features from HiT-Random."""
    B = images.size(0)
    fine_patches = extract_pyramid_patches(images, [14], model.patch_size)

    torch.manual_seed(0)
    indices = torch.randint(0, 196, (model.num_random,))
    random_patches = fine_patches[:, indices, :]
    random_coords = model.fine_coords[indices]

    all_patches = torch.cat([random_patches, fine_patches], dim=1)
    x = model.patch_proj(all_patches)

    all_coords = torch.cat([random_coords, model.fine_coords], dim=0)
    pe = model.pe(all_coords)
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


def collect_features(model, loader, extract_fn, device):
    model.eval()
    all_features = None
    all_labels = []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            layer_feats = extract_fn(model, images)
            if all_features is None:
                all_features = [[] for _ in range(len(layer_feats))]
            for l in range(len(layer_feats)):
                all_features[l].append(layer_feats[l].cpu())
            all_labels.append(labels)
    all_labels = torch.cat(all_labels, dim=0)
    for l in range(len(all_features)):
        all_features[l] = torch.cat(all_features[l], dim=0)
    return all_features, all_labels


def pool_level_features(features, token_indices):
    return features[:, token_indices, :].mean(dim=1)


def train_linear_probe(train_features, train_labels, val_features, val_labels,
                       embed_dim, num_classes=10, lr=1e-2, epochs=50, device="cpu"):
    probe = nn.Linear(embed_dim, num_classes).to(device)
    optimizer = torch.optim.Adam(probe.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    train_features = train_features.to(device)
    train_labels = train_labels.to(device)
    val_features = val_features.to(device)
    val_labels = val_labels.to(device)
    best_acc = 0.0
    for epoch in range(epochs):
        probe.train()
        perm = torch.randperm(train_features.size(0))
        for i in range(0, train_features.size(0), 512):
            idx = perm[i:i+512]
            logits = probe(train_features[idx])
            loss = criterion(logits, train_labels[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        probe.eval()
        with torch.no_grad():
            val_logits = probe(val_features)
            _, predicted = val_logits.max(1)
            acc = 100.0 * predicted.eq(val_labels).sum().item() / val_labels.size(0)
        if acc > best_acc:
            best_acc = acc
    return best_acc


def train_backbone(model, train_loader, device, num_epochs=100, lr=1e-3, save_path=None):
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
        torch.save({"model": model.state_dict(), "epoch": num_epochs,
                     "train_acc": 100. * correct / total}, save_path)
        print(f"    Saved: {save_path}")


# ---- Main ----

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone_epochs", type=int, default=100)
    parser.add_argument("--probe_epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--probe_lr", type=float, default=1e-2)
    parser.add_argument("--ckpt_dir", type=str, default="./output/cifar10_ckpt")
    args = parser.parse_args()

    device = get_device()
    print(f"Device: {device}")

    train_loader, val_loader = get_cifar10(args.batch_size)

    fig_dir = os.path.join(os.path.dirname(__file__), "figures")
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(args.ckpt_dir, exist_ok=True)
    log_path = os.path.join(fig_dir, "micro_prefix_probe.log")
    log_file = open(log_path, "w")

    def log(msg):
        print(msg)
        log_file.write(msg + "\n")
        log_file.flush()

    results = {}

    # ---- Load or train all three models ----
    models_config = [
        ("ViT-Tiny", ViTTiny, "vit_tiny_cifar10.pt", extract_layer_features_vit),
        ("HiT-Tiny (macro)", HiTTiny, "hit_tiny_cifar10.pt", extract_layer_features_hit),
        ("HiT-micro", HiTMicroTiny, "hit_micro_tiny_cifar10.pt", extract_layer_features_micro),
        ("HiT-random", HiTRandomTiny, "hit_random_tiny_cifar10.pt", extract_layer_features_random),
    ]

    for name, model_cls, ckpt_name, extract_fn in models_config:
        ckpt_path = os.path.join(args.ckpt_dir, ckpt_name)

        log(f"\n{'='*70}")
        torch.manual_seed(42)
        if name == "HiT-micro":
            model = model_cls(num_classes=10, num_micro=NUM_MICRO, crop_size=MICRO_CROP_SIZE).to(device)
        elif name == "HiT-random":
            model = model_cls(num_classes=10, num_random=NUM_MICRO).to(device)
        else:
            model = model_cls(num_classes=10).to(device)

        if os.path.exists(ckpt_path):
            log(f"Loading {name} from {ckpt_path}")
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
            model.load_state_dict(ckpt["model"])
            log(f"  Loaded (epoch={ckpt['epoch']}, train_acc={ckpt['train_acc']:.1f}%)")
        else:
            log(f"Training {name}...")
            train_backbone(model, train_loader, device, args.backbone_epochs, args.lr,
                           save_path=ckpt_path)

        # Collect features
        log(f"Collecting {name} features...")
        train_feats, train_labels = collect_features(model, train_loader, extract_fn, device)
        val_feats, val_labels = collect_features(model, val_loader, extract_fn, device)

        num_layers = len(train_feats)
        embed_dim = train_feats[0].shape[-1]

        # Define probe configs per model
        if name == "ViT-Tiny":
            probe_configs = [
                ("CLS", [0]),
                ("All patches", list(range(1, 197))),
            ]
        elif "macro" in name:
            probe_configs = [
                ("CLS", [0]),
                ("L4 only", list(range(86, 282))),
                ("L0-L3 only", list(range(1, 86))),
            ]
        elif name == "HiT-micro":
            probe_configs = [
                ("CLS", [0]),
                ("Fine only (L4)", list(range(86, 282))),
                ("Micro only", list(range(1, 86))),
                ("All tokens", list(range(1, 282))),
            ]
        else:  # HiT-random
            probe_configs = [
                ("CLS", [0]),
                ("Fine only (L4)", list(range(86, 282))),
                ("Random only", list(range(1, 86))),
                ("All tokens", list(range(1, 282))),
            ]

        log(f"\n{name} Linear Probe:")
        log(f"{'Config':>20s}  " + "  ".join(f"L{l+1:2d}" for l in range(num_layers)))

        model_results = {}
        for config_name, token_indices in probe_configs:
            accs = []
            for layer_idx in range(num_layers):
                train_f = pool_level_features(train_feats[layer_idx], token_indices)
                val_f = pool_level_features(val_feats[layer_idx], token_indices)
                acc = train_linear_probe(
                    train_f, train_labels, val_f, val_labels,
                    embed_dim, lr=args.probe_lr, epochs=args.probe_epochs, device=device)
                accs.append(acc)
            model_results[config_name] = accs
            log(f"{config_name:>20s}  " + "  ".join(f"{a:5.1f}" for a in accs))

        results[name] = model_results

        del train_feats, val_feats, model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ---- Key comparison: L4 tokens across three models ----
    log(f"\n{'='*70}")
    log("KEY COMPARISON: Fine-level (L4) probe accuracy across models")
    log(f"{'='*70}")

    vit_l4 = results["ViT-Tiny"]["All patches"]
    hit_macro_l4 = results["HiT-Tiny (macro)"]["L4 only"]
    hit_micro_l4 = results["HiT-micro"]["Fine only (L4)"]
    hit_random_l4 = results["HiT-random"]["Fine only (L4)"]
    num_layers = len(vit_l4)

    log(f"\n{'Model':>20s}  " + "  ".join(f"L{l+1:2d}" for l in range(num_layers)))
    log(f"{'ViT patches':>20s}  " + "  ".join(f"{a:5.1f}" for a in vit_l4))
    log(f"{'HiT-macro L4':>20s}  " + "  ".join(f"{a:5.1f}" for a in hit_macro_l4))
    log(f"{'HiT-micro L4':>20s}  " + "  ".join(f"{a:5.1f}" for a in hit_micro_l4))
    log(f"{'HiT-random L4':>20s}  " + "  ".join(f"{a:5.1f}" for a in hit_random_l4))

    diff_macro = [m - v for m, v in zip(hit_macro_l4, vit_l4)]
    diff_micro = [m - v for m, v in zip(hit_micro_l4, vit_l4)]
    diff_random = [m - v for m, v in zip(hit_random_l4, vit_l4)]
    log(f"{'Macro-ViT diff':>20s}  " + "  ".join(f"{d:+5.1f}" for d in diff_macro))
    log(f"{'Micro-ViT diff':>20s}  " + "  ".join(f"{d:+5.1f}" for d in diff_micro))
    log(f"{'Random-ViT diff':>20s}  " + "  ".join(f"{d:+5.1f}" for d in diff_random))

    # ---- Plot ----
    layers = list(range(1, num_layers + 1))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: L4 probe accuracy
    axes[0].plot(layers, vit_l4, marker="o", label="ViT patches (196)", linewidth=2)
    axes[0].plot(layers, hit_macro_l4, marker="s", label="HiT-macro L4 (196)", linewidth=2)
    axes[0].plot(layers, hit_micro_l4, marker="^", label="HiT-micro L4 (196)", linewidth=2)
    axes[0].plot(layers, hit_random_l4, marker="x", label="HiT-random L4 (196)", linewidth=2, linestyle=":")
    axes[0].set_xlabel("Layer")
    axes[0].set_ylabel("Probe Val Accuracy (%)")
    axes[0].set_title("Fine-level (L4) Probe: Macro vs Micro vs Random Prefix")
    axes[0].set_xticks(layers)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Right: Diff vs ViT
    axes[1].plot(layers, diff_macro, marker="s", label="Macro prefix - ViT", linewidth=2, color="tab:orange")
    axes[1].plot(layers, diff_micro, marker="^", label="Micro prefix - ViT", linewidth=2, color="tab:green")
    axes[1].plot(layers, diff_random, marker="x", label="Random prefix - ViT", linewidth=2, color="tab:gray", linestyle=":")
    axes[1].axhline(y=0, color="red", linestyle="--", alpha=0.5)
    axes[1].set_xlabel("Layer")
    axes[1].set_ylabel("Accuracy Difference vs ViT (%)")
    axes[1].set_title("Internalization Gap: Macro vs Micro vs Random Prefix")
    axes[1].set_xticks(layers)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "micro_vs_macro_probe.png"), dpi=150)
    log(f"\nSaved: {fig_dir}/micro_vs_macro_probe.png")

    # Save results
    with open(os.path.join(fig_dir, "micro_prefix_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    log(f"Saved: {fig_dir}/micro_prefix_results.json")

    log(f"\nLog saved to: {log_path}")
    log("Done.")
    log_file.close()


if __name__ == "__main__":
    main()
