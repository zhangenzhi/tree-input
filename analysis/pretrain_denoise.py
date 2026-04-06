"""
Two-stage training: pretrain with denoising + macro-prefix, then finetune classification.

Stage 1 (Pretrain): HiT with macro-prefix (L0-L3) + denoising objective.
  - Add Gaussian noise to each L4 patch token
  - Model predicts clean patch pixels from noisy input
  - L0-L3 provide global context to guide denoising
  - Forces model to internalize both macro structure and micro detail

Stage 2 (Finetune): Remove L0-L3 and denoising head, only L4 + CLS for classification.
  - Load pretrained weights (patch_proj, blocks, norm)
  - Add classification head, train on CIFAR-10 / Imagenette

Comparison:
  - ViT-Tiny (from scratch)
  - HiT-Tiny (from scratch, macro-prefix)
  - HiT-pretrained (stage1 denoise → stage2 finetune, L4 only at inference)
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import json
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import timm
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model.hit import extract_pyramid_patches, build_pyramid_coords, ContinuousPE3D


LEVELS = [1, 2, 4, 8, 14]
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


# ---- Stage 1: Pretrain model with denoising ----

class HiTDenoise(nn.Module):
    """HiT with macro-prefix + denoising head.

    During pretraining:
    - L0-L3 provide clean global context
    - L4 patches receive Gaussian noise
    - Model predicts clean L4 patch pixels from noisy input + global context
    """

    def __init__(self, levels=None, noise_std=0.3):
        super().__init__()
        self.levels = levels or LEVELS
        self.total_patches = sum(n * n for n in self.levels)
        self.patch_size = PATCH_SIZE
        self.noise_std = noise_std
        self.fine_patches = 14 * 14  # 196

        vit = timm.create_model("vit_tiny_patch16_224", pretrained=False, num_classes=10)
        self.embed_dim = vit.embed_dim  # 192

        self.patch_proj = nn.Linear(PATCH_SIZE * PATCH_SIZE * 3, self.embed_dim)
        self.cls_token = vit.cls_token
        self.pe = ContinuousPE3D(self.embed_dim, num_freq=32)

        coords = build_pyramid_coords(self.levels)
        self.register_buffer("pyramid_coords", coords)

        self.cls_pos = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        self.pos_drop = vit.pos_drop
        self.blocks = vit.blocks
        self.norm = vit.norm

        # Denoising head: predict clean patch pixels from transformer output
        # Only applied to L4 tokens
        self.denoise_head = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.GELU(),
            nn.Linear(self.embed_dim, PATCH_SIZE * PATCH_SIZE * 3),
        )

        # L4 token offset in the full sequence (after CLS)
        self.l4_start = 1 + sum(n * n for n in self.levels[:-1])  # skip CLS + L0-L3
        self.l4_end = self.l4_start + self.fine_patches

    def forward(self, images, noise_std=None):
        """
        Args:
            images: [B, 3, 224, 224]
            noise_std: override noise level (None = use default)
        Returns:
            pred_pixels: [B, 196, 3*16*16] predicted clean patch pixels
            clean_pixels: [B, 196, 3*16*16] ground truth clean patch pixels
        """
        if noise_std is None:
            noise_std = self.noise_std

        B = images.size(0)

        # Extract all pyramid patches
        all_patches = extract_pyramid_patches(images, self.levels, self.patch_size)
        # all_patches: [B, total_patches, 3*P*P]

        # Save clean L4 pixels as target
        l4_offset = sum(n * n for n in self.levels[:-1])
        clean_pixels = all_patches[:, l4_offset:, :].clone()  # [B, 196, 3*P*P]

        # Add noise only to L4 patches
        if self.training and noise_std > 0:
            noise = torch.randn_like(clean_pixels) * noise_std
            all_patches[:, l4_offset:, :] = all_patches[:, l4_offset:, :] + noise

        # Project all patches
        x = self.patch_proj(all_patches)

        # Positional encoding
        pe = self.pe(self.pyramid_coords)
        x = x + pe.unsqueeze(0)

        # CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1) + self.cls_pos
        x = torch.cat([cls_tokens, x], dim=1)

        # Transformer
        x = self.pos_drop(x)
        x = self.blocks(x)
        x = self.norm(x)

        # Denoise head on L4 tokens only
        l4_features = x[:, self.l4_start:self.l4_end, :]  # [B, 196, embed_dim]
        pred_pixels = self.denoise_head(l4_features)  # [B, 196, 3*P*P]

        return pred_pixels, clean_pixels


# ---- Stage 2: Finetune model (L4 only, no prefix) ----

class ViTFinetune(nn.Module):
    """ViT-Tiny using pretrained weights from HiTDenoise.
    Only uses L4 tokens (196 patches) — same as standard ViT at inference.
    """

    def __init__(self, num_classes=10):
        super().__init__()
        self.patch_size = PATCH_SIZE
        self.embed_dim = 192

        vit = timm.create_model("vit_tiny_patch16_224", pretrained=False, num_classes=num_classes)

        self.patch_proj = nn.Linear(PATCH_SIZE * PATCH_SIZE * 3, self.embed_dim)
        self.cls_token = vit.cls_token
        self.pe = ContinuousPE3D(self.embed_dim, num_freq=32)

        fine_coords = build_pyramid_coords([14])
        self.register_buffer("fine_coords", fine_coords)

        self.cls_pos = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        self.pos_drop = vit.pos_drop
        self.blocks = vit.blocks
        self.norm = vit.norm
        self.head = nn.Linear(self.embed_dim, num_classes)

    def forward(self, images):
        B = images.size(0)
        patches = extract_pyramid_patches(images, [14], self.patch_size)
        x = self.patch_proj(patches)
        pe = self.pe(self.fine_coords)
        x = x + pe.unsqueeze(0)
        cls_tokens = self.cls_token.expand(B, -1, -1) + self.cls_pos
        x = torch.cat([cls_tokens, x], dim=1)
        x = self.pos_drop(x)
        x = self.blocks(x)
        x = self.norm(x)
        return self.head(x[:, 0])


# ---- Baselines ----

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


# ---- Training utilities ----

def train_denoise_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    total = 0
    for images, _ in loader:
        images = images.to(device)
        optimizer.zero_grad()
        pred, target = model(images)
        loss = F.mse_loss(pred, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
        total += images.size(0)
    return total_loss / total


@torch.no_grad()
def eval_denoise(model, loader, device):
    model.eval()
    total_loss = 0.0
    total = 0
    for images, _ in loader:
        images = images.to(device)
        pred, target = model(images, noise_std=model.noise_std)
        loss = F.mse_loss(pred, target)
        total_loss += loss.item() * images.size(0)
        total += images.size(0)
    return total_loss / total


def train_cls_epoch(model, loader, criterion, optimizer, device):
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
def eval_cls(model, loader, criterion, device):
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


def transfer_weights(pretrained_model, finetune_model):
    """Transfer shared weights from pretrained HiTDenoise to ViTFinetune."""
    ft_state = finetune_model.state_dict()
    pt_state = pretrained_model.state_dict()

    transferred = []
    for key in ft_state:
        if key in pt_state and ft_state[key].shape == pt_state[key].shape:
            ft_state[key] = pt_state[key]
            transferred.append(key)

    finetune_model.load_state_dict(ft_state)
    return transferred


# ---- Main ----

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrain_epochs", type=int, default=50)
    parser.add_argument("--finetune_epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--pretrain_lr", type=float, default=1e-3)
    parser.add_argument("--finetune_lr", type=float, default=1e-3)
    parser.add_argument("--noise_std", type=float, default=0.3)
    parser.add_argument("--ckpt_dir", type=str, default="./output/cifar10_ckpt")
    args = parser.parse_args()

    device = get_device()
    print(f"Device: {device}")

    train_loader, val_loader = get_cifar10(args.batch_size)

    fig_dir = os.path.join(os.path.dirname(__file__), "figures")
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(args.ckpt_dir, exist_ok=True)

    log_path = os.path.join(fig_dir, "pretrain_denoise.log")
    log_file = open(log_path, "w")

    def log(msg):
        print(msg)
        log_file.write(msg + "\n")
        log_file.flush()

    log(f"Pretrain: {args.pretrain_epochs} epochs, noise_std={args.noise_std}")
    log(f"Finetune: {args.finetune_epochs} epochs")
    log("")

    # ================================================================
    # Stage 1: Pretrain with denoising
    # ================================================================
    log("=" * 70)
    log("Stage 1: Pretraining HiT-Denoise (macro-prefix + denoising)")
    log("=" * 70)

    pt_ckpt = os.path.join(args.ckpt_dir, "hit_denoise_pretrained.pt")

    torch.manual_seed(42)
    pretrain_model = HiTDenoise(noise_std=args.noise_std).to(device)
    param_count = sum(p.numel() for p in pretrain_model.parameters())
    log(f"Parameters: {param_count:,}")

    if os.path.exists(pt_ckpt):
        log(f"Loading pretrained from {pt_ckpt}")
        ckpt = torch.load(pt_ckpt, map_location=device, weights_only=True)
        pretrain_model.load_state_dict(ckpt["model"])
        log(f"  Loaded (epoch={ckpt['epoch']})")
    else:
        optimizer = torch.optim.AdamW(pretrain_model.parameters(), lr=args.pretrain_lr, weight_decay=0.05)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.pretrain_epochs)

        pretrain_history = []
        for epoch in range(args.pretrain_epochs):
            train_loss = train_denoise_epoch(pretrain_model, train_loader, optimizer, device)
            scheduler.step()
            val_loss = eval_denoise(pretrain_model, val_loader, device)
            pretrain_history.append({"epoch": epoch + 1, "train_loss": train_loss, "val_loss": val_loss})

            if (epoch + 1) % 10 == 0 or epoch == 0:
                log(f"  Epoch {epoch+1:3d}/{args.pretrain_epochs} | "
                    f"train_mse={train_loss:.6f} val_mse={val_loss:.6f} | "
                    f"lr={optimizer.param_groups[0]['lr']:.6f}")

        torch.save({
            "model": pretrain_model.state_dict(),
            "epoch": args.pretrain_epochs,
            "history": pretrain_history,
        }, pt_ckpt)
        log(f"  Saved: {pt_ckpt}")

    # ================================================================
    # Stage 2: Finetune for classification (L4 only)
    # ================================================================
    log("")
    log("=" * 70)
    log("Stage 2: Finetuning (L4 only, pretrained weights)")
    log("=" * 70)

    torch.manual_seed(42)
    finetune_model = ViTFinetune(num_classes=10).to(device)

    # Transfer weights from pretrained model
    transferred = transfer_weights(pretrain_model, finetune_model)
    log(f"  Transferred {len(transferred)} weight tensors from pretrained model")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(finetune_model.parameters(), lr=args.finetune_lr, weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.finetune_epochs)

    ft_history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc = 0.0

    for epoch in range(args.finetune_epochs):
        train_loss, train_acc = train_cls_epoch(finetune_model, train_loader, criterion, optimizer, device)
        scheduler.step()
        val_loss, val_acc = eval_cls(finetune_model, val_loader, criterion, device)

        ft_history["train_loss"].append(train_loss)
        ft_history["train_acc"].append(train_acc)
        ft_history["val_loss"].append(val_loss)
        ft_history["val_acc"].append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc

        if (epoch + 1) % 10 == 0 or epoch == 0:
            log(f"  Epoch {epoch+1:3d}/{args.finetune_epochs} | "
                f"train_loss={train_loss:.4f} train_acc={train_acc:.1f}% | "
                f"val_loss={val_loss:.4f} val_acc={val_acc:.1f}%")

    log(f"\n  Best val_acc: {best_val_acc:.1f}%")

    ft_ckpt = os.path.join(args.ckpt_dir, "hit_denoise_finetuned.pt")
    torch.save({
        "model": finetune_model.state_dict(),
        "epoch": args.finetune_epochs,
        "best_val_acc": best_val_acc,
    }, ft_ckpt)
    log(f"  Saved: {ft_ckpt}")

    # ================================================================
    # Baselines (load from cache or train)
    # ================================================================
    log("")
    log("=" * 70)
    log("Baselines")
    log("=" * 70)

    baseline_results = {}

    for name, model_cls, ckpt_name in [
        ("ViT-Tiny", ViTTiny, "vit_tiny_cifar10.pt"),
        ("HiT-Tiny (macro)", HiTTiny, "hit_tiny_cifar10.pt"),
    ]:
        ckpt_path = os.path.join(args.ckpt_dir, ckpt_name)
        torch.manual_seed(42)
        model = model_cls(num_classes=10).to(device)

        if os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
            model.load_state_dict(ckpt["model"])
            # Evaluate
            val_loss, val_acc = eval_cls(model, val_loader, criterion, device)
            log(f"  {name}: loaded from cache, val_acc={val_acc:.1f}%")
            baseline_results[name] = val_acc
        else:
            log(f"  {name}: no checkpoint found at {ckpt_path}, skipping")

    # ================================================================
    # Summary
    # ================================================================
    log("")
    log("=" * 70)
    log("SUMMARY (all using 196 tokens at inference)")
    log("=" * 70)

    if "ViT-Tiny" in baseline_results:
        log(f"  ViT-Tiny (from scratch):        {baseline_results['ViT-Tiny']:.1f}%")
    if "HiT-Tiny (macro)" in baseline_results:
        log(f"  HiT-Tiny macro (L4-only infer): ~{baseline_results['HiT-Tiny (macro)']:.1f}% (full), -0.5% for L4-only")
    log(f"  HiT-pretrained (denoise→cls):   {best_val_acc:.1f}%")
    log("")

    if "ViT-Tiny" in baseline_results:
        gain_vs_vit = best_val_acc - baseline_results["ViT-Tiny"]
        log(f"  Pretrained vs ViT: {gain_vs_vit:+.1f}%")
    log("")

    # ---- Plot ----
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    epochs = range(1, args.finetune_epochs + 1)

    axes[0].plot(epochs, ft_history["train_loss"], label="Train", linewidth=1.5)
    axes[0].plot(epochs, ft_history["val_loss"], label="Val", linewidth=1.5)
    axes[0].set_title("Finetune Loss (pretrained denoise → classification)")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, ft_history["train_acc"], label="Train", linewidth=1.5)
    axes[1].plot(epochs, ft_history["val_acc"], label="Val", linewidth=1.5)
    if "ViT-Tiny" in baseline_results:
        axes[1].axhline(y=baseline_results["ViT-Tiny"], color="red", linestyle="--",
                        alpha=0.5, label=f"ViT baseline ({baseline_results['ViT-Tiny']:.1f}%)")
    axes[1].set_title("Finetune Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "pretrain_denoise.png"), dpi=150)
    log(f"Saved: {fig_dir}/pretrain_denoise.png")

    log(f"\nLog saved to: {log_path}")
    log("Done.")
    log_file.close()


if __name__ == "__main__":
    main()
