"""
MAE-style pretraining with macro-prefix context on Imagenette.

Stage 1 (Pretrain): Masked Autoencoder with macro context.
  - Randomly mask 75% of L4 patches
  - Replace masked L4 with learnable mask tokens
  - L0-L3 macro tokens provide global structural context (always visible)
  - Remaining 25% L4 patches provide local context
  - Reconstruct masked L4 patch pixels

Stage 2 (Finetune): Remove L0-L3, use all L4 tokens for classification.
  - Transfer pretrained weights (patch_proj, PE, blocks, norm)
  - Add classification head, train on Imagenette

Key difference from pretrain_denoise.py:
  - Denoise: all L4 visible but noisy → reconstruct clean pixels
  - MAE: 75% L4 invisible → reconstruct from context (harder task)
  - MAE forces model to learn spatial relationships, not just denoising

Usage:
    python analysis/pretrain_mae.py --dataset imagenette
    python analysis/pretrain_mae.py --dataset imagenette --mask_ratio 0.75 --pretrain_epochs 100
    python analysis/pretrain_mae.py --dataset imagenette --finetune_epochs 200
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
from dataset.imagenette import get_imagenette


LEVELS = [1, 2, 4, 8, 14]
PATCH_SIZE = 16
NUM_FINE = 14 * 14  # 196


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


def get_data(dataset, batch_size=64, num_workers=4):
    if dataset == "cifar10":
        return get_cifar10(batch_size)
    elif dataset == "imagenette":
        train_loader, val_loader, _ = get_imagenette(
            batch_size=batch_size, num_workers=num_workers,
        )
        return train_loader, val_loader
    else:
        raise ValueError(f"Unknown dataset: {dataset}")


# ---- Stage 1: MAE Pretrain model ----

class HiTMAE(nn.Module):
    """HiT with macro-prefix + MAE-style masked reconstruction.

    During pretraining:
    - L0-L3 always visible (global structural context)
    - 75% of L4 patches are masked (replaced with learnable mask token)
    - 25% of L4 patches remain visible (local context)
    - Model reconstructs masked L4 patch pixels
    """

    def __init__(self, levels=None, mask_ratio=0.75):
        super().__init__()
        self.levels = levels or LEVELS
        self.total_patches = sum(n * n for n in self.levels)
        self.patch_size = PATCH_SIZE
        self.mask_ratio = mask_ratio
        self.num_fine = NUM_FINE
        self.num_coarse = sum(n * n for n in self.levels[:-1])  # 85

        vit = timm.create_model("vit_tiny_patch16_224", pretrained=False, num_classes=10)
        self.embed_dim = vit.embed_dim  # 192

        self.patch_proj = nn.Linear(PATCH_SIZE * PATCH_SIZE * 3, self.embed_dim)
        self.cls_token = vit.cls_token
        self.pe = ContinuousPE3D(self.embed_dim, num_freq=32)

        coords = build_pyramid_coords(self.levels)
        self.register_buffer("pyramid_coords", coords)

        self.cls_pos = nn.Parameter(torch.zeros(1, 1, self.embed_dim))

        # Learnable mask token
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        nn.init.normal_(self.mask_token, std=0.02)

        self.pos_drop = vit.pos_drop
        self.blocks = vit.blocks
        self.norm = vit.norm

        # Reconstruction head: predict pixel values of masked patches
        self.recon_head = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.GELU(),
            nn.Linear(self.embed_dim, PATCH_SIZE * PATCH_SIZE * 3),
        )

    def random_mask(self, B, device):
        """Generate random mask indices for L4 patches.

        Returns:
            mask_indices: [num_masked] indices of masked L4 patches
            visible_indices: [num_visible] indices of visible L4 patches
        """
        num_masked = int(self.num_fine * self.mask_ratio)
        num_visible = self.num_fine - num_masked

        # Random permutation of L4 patch indices
        perm = torch.randperm(self.num_fine, device=device)
        mask_indices = perm[:num_masked]
        visible_indices = perm[num_masked:]

        return mask_indices, visible_indices, num_masked, num_visible

    def forward(self, images):
        """
        Returns:
            pred_pixels: [B, num_masked, 3*16*16] predicted pixels for masked patches
            target_pixels: [B, num_masked, 3*16*16] ground truth pixels
            num_masked: int
        """
        B = images.size(0)
        device = images.device

        # Extract all pyramid patches
        all_patches = extract_pyramid_patches(images, self.levels, self.patch_size)
        # all_patches: [B, total_patches, 3*P*P]

        # Split into coarse (L0-L3) and fine (L4)
        coarse_patches = all_patches[:, :self.num_coarse, :]  # [B, 85, C*P*P]
        fine_patches = all_patches[:, self.num_coarse:, :]     # [B, 196, C*P*P]

        # Generate random mask
        mask_indices, visible_indices, num_masked, num_visible = self.random_mask(B, device)

        # Save target pixels for masked patches
        target_pixels = fine_patches[:, mask_indices, :]  # [B, num_masked, C*P*P]

        # Project coarse patches (always visible)
        coarse_emb = self.patch_proj(coarse_patches)  # [B, 85, embed_dim]

        # Project visible fine patches
        visible_fine = fine_patches[:, visible_indices, :]  # [B, num_visible, C*P*P]
        visible_emb = self.patch_proj(visible_fine)  # [B, num_visible, embed_dim]

        # Mask tokens for masked positions
        mask_tokens = self.mask_token.expand(B, num_masked, -1)  # [B, num_masked, embed_dim]

        # Reconstruct full L4 sequence: place visible and mask tokens at correct positions
        fine_emb = torch.zeros(B, self.num_fine, self.embed_dim, device=device)
        fine_emb[:, visible_indices, :] = visible_emb
        fine_emb[:, mask_indices, :] = mask_tokens

        # Concatenate: [coarse | fine (with masks)]
        x = torch.cat([coarse_emb, fine_emb], dim=1)  # [B, 281, embed_dim]

        # Add positional encoding
        pe = self.pe(self.pyramid_coords)  # [281, embed_dim]
        x = x + pe.unsqueeze(0)

        # CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1) + self.cls_pos
        x = torch.cat([cls_tokens, x], dim=1)  # [B, 282, embed_dim]

        # Transformer
        x = self.pos_drop(x)
        x = self.blocks(x)
        x = self.norm(x)

        # Extract masked L4 token features and reconstruct
        # L4 starts at position 1 + num_coarse in the full sequence
        l4_start = 1 + self.num_coarse
        l4_features = x[:, l4_start:l4_start + self.num_fine, :]  # [B, 196, embed_dim]
        masked_features = l4_features[:, mask_indices, :]  # [B, num_masked, embed_dim]

        pred_pixels = self.recon_head(masked_features)  # [B, num_masked, C*P*P]

        return pred_pixels, target_pixels, num_masked


# ---- Stage 2: Finetune model (L4 only) ----

class ViTFinetune(nn.Module):
    """ViT-Tiny using pretrained weights from HiTMAE.
    Only uses L4 tokens (196 patches) at inference.
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

def train_mae_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    total = 0
    for images, _ in loader:
        images = images.to(device)
        optimizer.zero_grad()
        pred, target, num_masked = model(images)
        loss = F.mse_loss(pred, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
        total += images.size(0)
    return total_loss / total


@torch.no_grad()
def eval_mae(model, loader, device):
    model.eval()
    total_loss = 0.0
    total = 0
    for images, _ in loader:
        images = images.to(device)
        pred, target, num_masked = model(images)
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
    """Transfer shared weights from HiTMAE to ViTFinetune."""
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
    parser.add_argument("--pretrain_epochs", type=int, default=100)
    parser.add_argument("--finetune_epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--pretrain_lr", type=float, default=1.5e-4)
    parser.add_argument("--finetune_lr", type=float, default=1e-3)
    parser.add_argument("--mask_ratio", type=float, default=0.75)
    parser.add_argument("--dataset", type=str, default="imagenette", choices=["cifar10", "imagenette"])
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--ckpt_dir", type=str, default=None)
    args = parser.parse_args()

    if args.ckpt_dir is None:
        args.ckpt_dir = f"./output/{args.dataset}_ckpt"

    device = get_device()
    print(f"Device: {device}")

    train_loader, val_loader = get_data(args.dataset, args.batch_size, args.num_workers)

    fig_dir = os.path.join(os.path.dirname(__file__), "figures")
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(args.ckpt_dir, exist_ok=True)

    ds_tag = args.dataset
    log_path = os.path.join(fig_dir, f"pretrain_mae_{ds_tag}.log")
    log_file = open(log_path, "w")

    def log(msg):
        print(msg)
        log_file.write(msg + "\n")
        log_file.flush()

    log(f"Dataset: {ds_tag}")
    log(f"Pretrain: {args.pretrain_epochs} epochs, mask_ratio={args.mask_ratio}")
    log(f"Finetune: {args.finetune_epochs} epochs")
    log(f"Pretrain LR: {args.pretrain_lr}, Finetune LR: {args.finetune_lr}")
    log("")

    num_classes = 10

    # ================================================================
    # Stage 1: MAE Pretrain
    # ================================================================
    log("=" * 70)
    log(f"Stage 1: MAE Pretraining (macro context + {args.mask_ratio:.0%} L4 mask)")
    log("=" * 70)

    pt_ckpt = os.path.join(args.ckpt_dir, f"hit_mae_pretrained_{ds_tag}.pt")

    torch.manual_seed(42)
    pretrain_model = HiTMAE(mask_ratio=args.mask_ratio).to(device)
    param_count = sum(p.numel() for p in pretrain_model.parameters())
    log(f"Parameters: {param_count:,}")
    log(f"L4 patches: {NUM_FINE}, masked: {int(NUM_FINE * args.mask_ratio)}, "
        f"visible: {NUM_FINE - int(NUM_FINE * args.mask_ratio)}")
    log(f"Macro tokens (L0-L3): {sum(n*n for n in LEVELS[:-1])} (always visible)")

    pretrain_history = []

    if os.path.exists(pt_ckpt):
        log(f"Loading pretrained from {pt_ckpt}")
        ckpt = torch.load(pt_ckpt, map_location=device, weights_only=True)
        pretrain_model.load_state_dict(ckpt["model"])
        log(f"  Loaded (epoch={ckpt['epoch']})")
        if "history" in ckpt:
            pretrain_history = ckpt["history"]
    else:
        optimizer = torch.optim.AdamW(
            pretrain_model.parameters(), lr=args.pretrain_lr,
            betas=(0.9, 0.95), weight_decay=0.05,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.pretrain_epochs,
        )

        for epoch in range(args.pretrain_epochs):
            train_loss = train_mae_epoch(pretrain_model, train_loader, optimizer, device)
            scheduler.step()
            val_loss = eval_mae(pretrain_model, val_loader, device)
            pretrain_history.append({
                "epoch": epoch + 1, "train_loss": train_loss, "val_loss": val_loss,
            })

            if (epoch + 1) % 10 == 0 or epoch == 0:
                log(f"  Epoch {epoch+1:3d}/{args.pretrain_epochs} | "
                    f"train_mse={train_loss:.6f} val_mse={val_loss:.6f} | "
                    f"lr={optimizer.param_groups[0]['lr']:.6f}")

        torch.save({
            "model": pretrain_model.state_dict(),
            "epoch": args.pretrain_epochs,
            "mask_ratio": args.mask_ratio,
            "history": pretrain_history,
        }, pt_ckpt)
        log(f"  Saved: {pt_ckpt}")

    # ================================================================
    # Stage 2: Finetune for classification (L4 only)
    # ================================================================
    log("")
    log("=" * 70)
    log("Stage 2: Finetuning (L4 only, MAE-pretrained weights)")
    log("=" * 70)

    torch.manual_seed(42)
    finetune_model = ViTFinetune(num_classes=num_classes).to(device)

    transferred = transfer_weights(pretrain_model, finetune_model)
    log(f"  Transferred {len(transferred)} weight tensors from pretrained model")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        finetune_model.parameters(), lr=args.finetune_lr, weight_decay=0.05,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.finetune_epochs,
    )

    ft_history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc = 0.0

    for epoch in range(args.finetune_epochs):
        train_loss, train_acc = train_cls_epoch(
            finetune_model, train_loader, criterion, optimizer, device,
        )
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

    ft_ckpt = os.path.join(args.ckpt_dir, f"hit_mae_finetuned_{ds_tag}.pt")
    torch.save({
        "model": finetune_model.state_dict(),
        "epoch": args.finetune_epochs,
        "best_val_acc": best_val_acc,
        "history": ft_history,
    }, ft_ckpt)
    log(f"  Saved: {ft_ckpt}")

    # ================================================================
    # Baselines (load from cache)
    # ================================================================
    log("")
    log("=" * 70)
    log("Baselines")
    log("=" * 70)

    baseline_results = {}

    for name, model_cls, ckpt_name in [
        ("ViT-Tiny", ViTTiny, f"vit_tiny_{ds_tag}.pt"),
        ("HiT-Tiny (macro)", HiTTiny, f"hit_tiny_{ds_tag}.pt"),
    ]:
        ckpt_path = os.path.join(args.ckpt_dir, ckpt_name)
        torch.manual_seed(42)
        model = model_cls(num_classes=num_classes).to(device)

        if os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
            model.load_state_dict(ckpt["model"])
            val_loss, val_acc = eval_cls(model, val_loader, criterion, device)
            log(f"  {name}: loaded from cache, val_acc={val_acc:.1f}%")
            baseline_results[name] = val_acc
        else:
            log(f"  {name}: no checkpoint found at {ckpt_path}, skipping")

    # Also check denoise pretrain result if available
    denoise_ckpt = os.path.join(args.ckpt_dir, f"hit_denoise_finetuned_{ds_tag}.pt")
    if os.path.exists(denoise_ckpt):
        torch.manual_seed(42)
        denoise_ft = ViTFinetune(num_classes=num_classes).to(device)
        ckpt = torch.load(denoise_ckpt, map_location=device, weights_only=True)
        denoise_ft.load_state_dict(ckpt["model"])
        val_loss, val_acc = eval_cls(denoise_ft, val_loader, criterion, device)
        log(f"  HiT-denoise (pretrain→finetune): loaded from cache, val_acc={val_acc:.1f}%")
        baseline_results["HiT-denoise"] = val_acc

    # ================================================================
    # Summary
    # ================================================================
    log("")
    log("=" * 70)
    log("SUMMARY (all using 196 tokens at inference)")
    log("=" * 70)

    if "ViT-Tiny" in baseline_results:
        log(f"  ViT-Tiny (from scratch):          {baseline_results['ViT-Tiny']:.1f}%")
    if "HiT-Tiny (macro)" in baseline_results:
        log(f"  HiT-Tiny macro (full 282 tokens):  {baseline_results['HiT-Tiny (macro)']:.1f}%")
    if "HiT-denoise" in baseline_results:
        log(f"  HiT-denoise (pretrain→finetune):   {baseline_results['HiT-denoise']:.1f}%")
    log(f"  HiT-MAE (pretrain→finetune):       {best_val_acc:.1f}%")
    log("")

    if "ViT-Tiny" in baseline_results:
        gain = best_val_acc - baseline_results["ViT-Tiny"]
        log(f"  MAE-pretrained vs ViT: {gain:+.1f}%")
    if "HiT-denoise" in baseline_results:
        gain = best_val_acc - baseline_results["HiT-denoise"]
        log(f"  MAE-pretrained vs Denoise-pretrained: {gain:+.1f}%")
    log("")

    # ---- Plot ----
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Top-left: pretrain loss
    if pretrain_history:
        pt_epochs = [h["epoch"] for h in pretrain_history]
        axes[0, 0].plot(pt_epochs, [h["train_loss"] for h in pretrain_history],
                        label="Train", linewidth=1.5)
        axes[0, 0].plot(pt_epochs, [h["val_loss"] for h in pretrain_history],
                        label="Val", linewidth=1.5)
    axes[0, 0].set_title(f"MAE Pretrain Loss ({args.mask_ratio:.0%} mask)")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("MSE Loss")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Top-right: finetune loss
    ft_epochs = range(1, args.finetune_epochs + 1)
    axes[0, 1].plot(ft_epochs, ft_history["train_loss"], label="Train", linewidth=1.5)
    axes[0, 1].plot(ft_epochs, ft_history["val_loss"], label="Val", linewidth=1.5)
    axes[0, 1].set_title("Finetune Loss")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("CE Loss")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Bottom-left: finetune accuracy
    axes[1, 0].plot(ft_epochs, ft_history["train_acc"], label="Train", linewidth=1.5)
    axes[1, 0].plot(ft_epochs, ft_history["val_acc"], label="Val", linewidth=1.5)
    if "ViT-Tiny" in baseline_results:
        axes[1, 0].axhline(y=baseline_results["ViT-Tiny"], color="red", linestyle="--",
                           alpha=0.5, label=f"ViT baseline ({baseline_results['ViT-Tiny']:.1f}%)")
    if "HiT-Tiny (macro)" in baseline_results:
        axes[1, 0].axhline(y=baseline_results["HiT-Tiny (macro)"], color="green", linestyle="--",
                           alpha=0.5, label=f"HiT-macro ({baseline_results['HiT-Tiny (macro)']:.1f}%)")
    axes[1, 0].set_title("Finetune Accuracy")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("Accuracy (%)")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Bottom-right: val accuracy zoom
    axes[1, 1].plot(ft_epochs, ft_history["val_acc"], label="MAE-pretrained", linewidth=2)
    if "ViT-Tiny" in baseline_results:
        axes[1, 1].axhline(y=baseline_results["ViT-Tiny"], color="red", linestyle="--",
                           alpha=0.7, label=f"ViT ({baseline_results['ViT-Tiny']:.1f}%)")
    if "HiT-Tiny (macro)" in baseline_results:
        axes[1, 1].axhline(y=baseline_results["HiT-Tiny (macro)"], color="green", linestyle="--",
                           alpha=0.7, label=f"HiT-macro ({baseline_results['HiT-Tiny (macro)']:.1f}%)")
    if "HiT-denoise" in baseline_results:
        axes[1, 1].axhline(y=baseline_results["HiT-denoise"], color="purple", linestyle="--",
                           alpha=0.7, label=f"Denoise ({baseline_results['HiT-denoise']:.1f}%)")
    axes[1, 1].set_title("Val Accuracy Comparison")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("Val Accuracy (%)")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.suptitle(f"HiT-MAE on {ds_tag} (pretrain {args.pretrain_epochs}ep → finetune {args.finetune_epochs}ep)",
                 fontsize=14)
    plt.tight_layout()
    plot_path = os.path.join(fig_dir, f"pretrain_mae_{ds_tag}.png")
    plt.savefig(plot_path, dpi=150)
    log(f"Saved: {plot_path}")

    # Save results JSON
    results = {
        "pretrain_history": pretrain_history,
        "finetune_history": ft_history,
        "best_val_acc": best_val_acc,
        "baselines": baseline_results,
        "config": {
            "dataset": ds_tag,
            "mask_ratio": args.mask_ratio,
            "pretrain_epochs": args.pretrain_epochs,
            "finetune_epochs": args.finetune_epochs,
            "pretrain_lr": args.pretrain_lr,
            "finetune_lr": args.finetune_lr,
        },
    }
    json_path = os.path.join(fig_dir, f"pretrain_mae_{ds_tag}.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    log(f"Saved: {json_path}")

    log(f"\nLog saved to: {log_path}")
    log("Done.")
    log_file.close()


if __name__ == "__main__":
    main()
