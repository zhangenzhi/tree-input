"""
DiffMAE-style pretraining: macro context + masked L4 + multi-timestep diffusion
reconstruction of (8,8)-offset patches.

Stage 1 (Pretrain):
  - Mask 75% of L4 patches
  - L0-L3 macro tokens provide global structural context (always visible)
  - For each masked L4 position (row, col), the reconstruction target is the
    (8,8)-offset patch at pixel (row*16+8, col*16+8) — cross-boundary detail
  - Multi-timestep diffusion: sample random t, add noise at level t to target,
    model predicts clean offset pixels conditioned on encoder features + timestep
  - Forces model to learn cross-boundary textures at multiple noise scales

Stage 2 (Finetune):
  - Remove L0-L3, use all L4 tokens for classification
  - Transfer pretrained encoder weights

Key innovations over pretrain_mae.py:
  - Reconstruction target is offset patches (cross-boundary), not L4 patches
  - Multi-timestep noise schedule (not single-step)
  - Timestep conditioning in reconstruction head

Usage:
    python analysis/pretrain_diffmae.py --dataset imagenette
    python analysis/pretrain_diffmae.py --dataset imagenette --pretrain_epochs 200 --finetune_epochs 120
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import math
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
OFFSET = 8  # half-patch shift


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


# ---- Diffusion noise schedule ----

def cosine_beta_schedule(timesteps, s=0.008):
    """Cosine noise schedule (Nichol & Dhariwal 2021)."""
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clamp(betas, 0.0001, 0.9999)


class DiffusionSchedule:
    """Precomputed diffusion schedule parameters."""

    def __init__(self, timesteps=1000):
        self.timesteps = timesteps
        betas = cosine_beta_schedule(timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)

    def to(self, device):
        self.sqrt_alphas_cumprod = self.sqrt_alphas_cumprod.to(device)
        self.sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod.to(device)
        return self

    def q_sample(self, x_0, t, noise=None):
        """Add noise to x_0 at timestep t.
        x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * noise
        """
        if noise is None:
            noise = torch.randn_like(x_0)

        sqrt_alpha = self.sqrt_alphas_cumprod[t]
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[t]

        # Reshape for broadcasting: [B] -> [B, 1, 1] or [B, 1]
        while sqrt_alpha.dim() < x_0.dim():
            sqrt_alpha = sqrt_alpha.unsqueeze(-1)
            sqrt_one_minus_alpha = sqrt_one_minus_alpha.unsqueeze(-1)

        return sqrt_alpha * x_0 + sqrt_one_minus_alpha * noise, noise


# ---- Timestep embedding ----

class TimestepEmbedding(nn.Module):
    """Sinusoidal timestep embedding."""

    def __init__(self, embed_dim, max_period=10000):
        super().__init__()
        self.embed_dim = embed_dim
        half = embed_dim // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(half, dtype=torch.float32) / half)
        self.register_buffer("freqs", freqs)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, t):
        """t: [B] integer timesteps -> [B, embed_dim]"""
        t_float = t.float()
        args = t_float.unsqueeze(-1) * self.freqs.unsqueeze(0)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        return self.mlp(emb)


# ---- Stage 1: DiffMAE model ----

class HiTDiffMAE(nn.Module):
    """HiT with macro-prefix + MAE masking + diffusion reconstruction of offset patches.

    Encoder: processes L0-L3 (visible) + visible L4 + mask tokens
    Reconstruction: for masked L4 positions, predict (8,8)-offset patch pixels
                    conditioned on encoder output + diffusion timestep
    """

    def __init__(self, levels=None, mask_ratio=0.75, diff_timesteps=1000):
        super().__init__()
        self.levels = levels or LEVELS
        self.patch_size = PATCH_SIZE
        self.mask_ratio = mask_ratio
        self.num_fine = NUM_FINE
        self.num_coarse = sum(n * n for n in self.levels[:-1])  # 85
        self.offset = OFFSET

        vit = timm.create_model("vit_tiny_patch16_224", pretrained=False, num_classes=10)
        self.embed_dim = vit.embed_dim  # 192

        self.patch_proj = nn.Linear(PATCH_SIZE * PATCH_SIZE * 3, self.embed_dim)
        self.cls_token = vit.cls_token
        self.pe = ContinuousPE3D(self.embed_dim, num_freq=32)

        coords = build_pyramid_coords(self.levels)
        self.register_buffer("pyramid_coords", coords)

        self.cls_pos = nn.Parameter(torch.zeros(1, 1, self.embed_dim))

        # Mask token
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        nn.init.normal_(self.mask_token, std=0.02)

        self.pos_drop = vit.pos_drop
        self.blocks = vit.blocks
        self.norm = vit.norm

        # Timestep embedding
        self.time_embed = TimestepEmbedding(self.embed_dim)

        # Reconstruction head: encoder_features + timestep -> clean offset pixels
        self.recon_head = nn.Sequential(
            nn.Linear(self.embed_dim * 2, self.embed_dim),
            nn.GELU(),
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.GELU(),
            nn.Linear(self.embed_dim, PATCH_SIZE * PATCH_SIZE * 3),
        )

        # Diffusion schedule
        self.diff_schedule = DiffusionSchedule(diff_timesteps)
        self.diff_timesteps = diff_timesteps

        # Precompute which L4 positions have valid offset patches (13x13 inner grid)
        # L4 grid: 14 rows x 14 cols. Offset at (row*16+8, col*16+8) is valid for row<13, col<13
        valid_mask = torch.zeros(14, 14, dtype=torch.bool)
        valid_mask[:13, :13] = True
        self.register_buffer("valid_offset_mask", valid_mask.flatten())  # [196]

    def _extract_offset_targets(self, images, mask_indices):
        """Extract (8,8)-offset patch pixels for masked L4 positions.

        Args:
            images: [B, 3, 224, 224]
            mask_indices: [num_masked] L4 patch indices (0-195)

        Returns:
            offset_pixels: [B, num_valid, 3*16*16] offset patch pixel values
            valid_mask: [num_masked] bool mask for which indices have valid offset
        """
        B = images.size(0)
        P = self.patch_size

        # Check which masked indices have valid offset patches
        valid = self.valid_offset_mask[mask_indices]  # [num_masked]

        offset_pixels = []
        for idx in mask_indices:
            row = idx // 14
            col = idx % 14
            if row < 13 and col < 13:
                t = row * P + self.offset
                l = col * P + self.offset
                patch = images[:, :, t:t+P, l:l+P]  # [B, 3, P, P]
                offset_pixels.append(patch.reshape(B, -1))
            else:
                # Placeholder for invalid positions (will be masked out in loss)
                offset_pixels.append(torch.zeros(B, 3 * P * P, device=images.device))

        offset_pixels = torch.stack(offset_pixels, dim=1)  # [B, num_masked, 3*P*P]
        return offset_pixels, valid

    def random_mask(self, B, device):
        num_masked = int(self.num_fine * self.mask_ratio)
        perm = torch.randperm(self.num_fine, device=device)
        mask_indices = perm[:num_masked]
        visible_indices = perm[num_masked:]
        return mask_indices, visible_indices, num_masked

    def forward(self, images):
        """
        Returns:
            pred_x0: [B, num_valid, 3*16*16] predicted clean offset pixels
            target_x0: [B, num_valid, 3*16*16] ground truth offset pixels
            num_valid: number of valid reconstruction targets
        """
        B = images.size(0)
        device = images.device

        # Extract patches
        all_patches = extract_pyramid_patches(images, self.levels, self.patch_size)
        coarse_patches = all_patches[:, :self.num_coarse, :]
        fine_patches = all_patches[:, self.num_coarse:, :]

        # Random mask
        mask_indices, visible_indices, num_masked = self.random_mask(B, device)

        # Extract offset targets for masked positions
        offset_targets, valid_mask = self._extract_offset_targets(images, mask_indices)

        # Encode
        coarse_emb = self.patch_proj(coarse_patches)
        visible_emb = self.patch_proj(fine_patches[:, visible_indices, :])
        mask_tokens = self.mask_token.expand(B, num_masked, -1)

        fine_emb = torch.zeros(B, self.num_fine, self.embed_dim, device=device)
        fine_emb[:, visible_indices, :] = visible_emb
        fine_emb[:, mask_indices, :] = mask_tokens

        x = torch.cat([coarse_emb, fine_emb], dim=1)
        pe = self.pe(self.pyramid_coords)
        x = x + pe.unsqueeze(0)

        cls_tokens = self.cls_token.expand(B, -1, -1) + self.cls_pos
        x = torch.cat([cls_tokens, x], dim=1)

        x = self.pos_drop(x)
        x = self.blocks(x)
        x = self.norm(x)

        # Extract masked position features
        l4_start = 1 + self.num_coarse
        l4_features = x[:, l4_start:l4_start + self.num_fine, :]
        masked_features = l4_features[:, mask_indices, :]  # [B, num_masked, embed_dim]

        # Sample diffusion timestep
        t = torch.randint(0, self.diff_timesteps, (B,), device=device)

        # Add noise to offset targets
        self.diff_schedule.to(device)
        noised_targets, noise = self.diff_schedule.q_sample(offset_targets, t)

        # Timestep embedding: [B, embed_dim] -> [B, 1, embed_dim] -> broadcast to [B, num_masked, embed_dim]
        t_emb = self.time_embed(t).unsqueeze(1).expand(-1, num_masked, -1)

        # Predict clean offset pixels: condition on encoder features + timestep
        recon_input = torch.cat([masked_features, t_emb], dim=-1)  # [B, num_masked, 2*embed_dim]
        pred_x0 = self.recon_head(recon_input)  # [B, num_masked, 3*P*P]

        # Filter to valid offset positions only
        valid_idx = valid_mask.nonzero(as_tuple=True)[0]
        pred_valid = pred_x0[:, valid_idx, :]
        target_valid = offset_targets[:, valid_idx, :]

        return pred_valid, target_valid, len(valid_idx)


# ---- Stage 2: Finetune (same as pretrain_mae.py) ----

class ViTFinetune(nn.Module):
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


class OffsetFinetune(nn.Module):
    """Finetune with L4 + (8,8)-offset patches. Uses DiffMAE-pretrained weights.
    Total: 1 CLS + 85 offset + 196 L4 = 282 tokens.
    """

    def __init__(self, num_classes=10, num_offset=NUM_MICRO):
        super().__init__()
        self.patch_size = PATCH_SIZE
        self.embed_dim = 192
        self.num_offset = num_offset
        self.offset = OFFSET

        vit = timm.create_model("vit_tiny_patch16_224", pretrained=False, num_classes=num_classes)

        self.patch_proj = nn.Linear(PATCH_SIZE * PATCH_SIZE * 3, self.embed_dim)
        self.cls_token = vit.cls_token
        self.pe = ContinuousPE3D(self.embed_dim, num_freq=32)

        fine_coords = build_pyramid_coords([14])
        self.register_buffer("fine_coords", fine_coords)

        # Offset grid: 13x13, select 85
        offset_positions = []
        for row in range(13):
            for col in range(13):
                t = self.offset + row * PATCH_SIZE
                l = self.offset + col * PATCH_SIZE
                offset_positions.append((t, l))
        torch.manual_seed(42)
        perm = torch.randperm(len(offset_positions))[:num_offset]
        self.offset_positions = [offset_positions[i] for i in perm]

        offset_coords = []
        for t, l in self.offset_positions:
            cx = (l + PATCH_SIZE / 2) / 224.0
            cy = (t + PATCH_SIZE / 2) / 224.0
            s = PATCH_SIZE / 224.0
            offset_coords.append([cx, cy, s])
        self.register_buffer("offset_coords", torch.tensor(offset_coords, dtype=torch.float32))

        self.cls_pos = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        self.pos_drop = vit.pos_drop
        self.blocks = vit.blocks
        self.norm = vit.norm
        self.head = nn.Linear(self.embed_dim, num_classes)

    def _extract_offset_patches(self, images):
        B = images.size(0)
        P = self.patch_size
        patches = []
        for t, l in self.offset_positions:
            patch = images[:, :, t:t+P, l:l+P]
            patches.append(patch.reshape(B, -1))
        return torch.stack(patches, dim=1)

    def forward(self, images):
        B = images.size(0)

        fine_patches = extract_pyramid_patches(images, [14], self.patch_size)
        offset_patches = self._extract_offset_patches(images)

        all_patches = torch.cat([offset_patches, fine_patches], dim=1)
        x = self.patch_proj(all_patches)

        all_coords = torch.cat([self.offset_coords, self.fine_coords], dim=0)
        pe = self.pe(all_coords)
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

def train_diffmae_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    total = 0
    for images, _ in loader:
        images = images.to(device)
        optimizer.zero_grad()
        pred, target, num_valid = model(images)
        if num_valid > 0:
            loss = F.mse_loss(pred, target)
        else:
            loss = torch.tensor(0.0, device=device)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
        total += images.size(0)
    return total_loss / total


@torch.no_grad()
def eval_diffmae(model, loader, device):
    model.eval()
    total_loss = 0.0
    total = 0
    for images, _ in loader:
        images = images.to(device)
        pred, target, num_valid = model(images)
        if num_valid > 0:
            loss = F.mse_loss(pred, target)
        else:
            loss = torch.tensor(0.0, device=device)
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
    parser.add_argument("--pretrain_epochs", type=int, default=200)
    parser.add_argument("--finetune_epochs", type=int, default=120)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--pretrain_lr", type=float, default=1.5e-4)
    parser.add_argument("--finetune_lr", type=float, default=1e-3)
    parser.add_argument("--mask_ratio", type=float, default=0.75)
    parser.add_argument("--diff_timesteps", type=int, default=1000)
    parser.add_argument("--dataset", type=str, default="imagenette", choices=["cifar10", "imagenette"])
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--ckpt_dir", type=str, default=None)
    parser.add_argument("--finetune_with_offset", action="store_true",
                        help="Finetune with L4 + offset patches (282 tokens) instead of L4 only (196)")
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
    log_path = os.path.join(fig_dir, f"pretrain_diffmae_{ds_tag}.log")
    log_file = open(log_path, "w")

    def log(msg):
        print(msg)
        log_file.write(msg + "\n")
        log_file.flush()

    log(f"Dataset: {ds_tag}")
    log(f"Pretrain: {args.pretrain_epochs} epochs, mask_ratio={args.mask_ratio}, "
        f"diff_timesteps={args.diff_timesteps}")
    log(f"Finetune: {args.finetune_epochs} epochs")
    log(f"Pretrain LR: {args.pretrain_lr}, Finetune LR: {args.finetune_lr}")
    log(f"Reconstruction target: (8,8)-offset patches (cross-boundary detail)")
    log("")

    num_classes = 10

    # ================================================================
    # Stage 1: DiffMAE Pretrain
    # ================================================================
    log("=" * 70)
    log(f"Stage 1: DiffMAE Pretraining (macro + {args.mask_ratio:.0%} mask + "
        f"diffusion offset reconstruction)")
    log("=" * 70)

    pt_ckpt = os.path.join(args.ckpt_dir, f"hit_diffmae_pretrained_{ds_tag}.pt")

    torch.manual_seed(42)
    pretrain_model = HiTDiffMAE(
        mask_ratio=args.mask_ratio, diff_timesteps=args.diff_timesteps
    ).to(device)
    param_count = sum(p.numel() for p in pretrain_model.parameters())
    log(f"Parameters: {param_count:,}")
    num_masked = int(NUM_FINE * args.mask_ratio)
    num_valid_offset = min(num_masked, 13 * 13)  # at most 169 valid offset positions
    log(f"L4 patches: {NUM_FINE}, masked: {num_masked}, "
        f"valid offset targets: up to {num_valid_offset}")
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
            train_loss = train_diffmae_epoch(pretrain_model, train_loader, optimizer, device)
            scheduler.step()
            val_loss = eval_diffmae(pretrain_model, val_loader, device)
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
    # Stage 2: Finetune
    # ================================================================
    log("")
    log("=" * 70)
    if args.finetune_with_offset:
        log("Stage 2: Finetuning (L4 + offset patches, DiffMAE-pretrained weights)")
        ft_tag = "offset"
    else:
        log("Stage 2: Finetuning (L4 only, DiffMAE-pretrained weights)")
        ft_tag = "l4only"
    log("=" * 70)

    torch.manual_seed(42)
    if args.finetune_with_offset:
        finetune_model = OffsetFinetune(num_classes=num_classes).to(device)
    else:
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

    ft_ckpt = os.path.join(args.ckpt_dir, f"hit_diffmae_finetuned_{ft_tag}_{ds_tag}.pt")
    torch.save({
        "model": finetune_model.state_dict(),
        "epoch": args.finetune_epochs,
        "best_val_acc": best_val_acc,
        "history": ft_history,
    }, ft_ckpt)
    log(f"  Saved: {ft_ckpt}")

    # ================================================================
    # Baselines
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
            log(f"  {name}: loaded, val_acc={val_acc:.1f}%")
            baseline_results[name] = val_acc
        else:
            log(f"  {name}: no checkpoint at {ckpt_path}, skipping")

    # Check other pretraining results
    for tag, label in [
        (f"hit_denoise_finetuned_{ds_tag}.pt", "HiT-denoise"),
        (f"hit_mae_finetuned_{ds_tag}.pt", "HiT-MAE"),
    ]:
        path = os.path.join(args.ckpt_dir, tag)
        if os.path.exists(path):
            torch.manual_seed(42)
            m = ViTFinetune(num_classes=num_classes).to(device)
            ckpt = torch.load(path, map_location=device, weights_only=True)
            m.load_state_dict(ckpt["model"])
            _, val_acc = eval_cls(m, val_loader, criterion, device)
            log(f"  {label}: loaded, val_acc={val_acc:.1f}%")
            baseline_results[label] = val_acc

    # ================================================================
    # Summary
    # ================================================================
    ft_tokens = "282 (L4+offset)" if args.finetune_with_offset else "196 (L4 only)"
    log("")
    log("=" * 70)
    log(f"SUMMARY (finetune: {ft_tokens})")
    log("=" * 70)

    if "ViT-Tiny" in baseline_results:
        log(f"  ViT-Tiny (from scratch, 196):       {baseline_results['ViT-Tiny']:.1f}%")
    if "HiT-Tiny (macro)" in baseline_results:
        log(f"  HiT-Tiny macro (282):               {baseline_results['HiT-Tiny (macro)']:.1f}%")
    if "HiT-denoise" in baseline_results:
        log(f"  HiT-denoise (pretrain→ft, 196):     {baseline_results['HiT-denoise']:.1f}%")
    if "HiT-MAE" in baseline_results:
        log(f"  HiT-MAE (pretrain→ft, 196):         {baseline_results['HiT-MAE']:.1f}%")
    log(f"  HiT-DiffMAE (pretrain→ft, {ft_tokens}): {best_val_acc:.1f}%")
    log("")

    if "ViT-Tiny" in baseline_results:
        log(f"  DiffMAE vs ViT: {best_val_acc - baseline_results['ViT-Tiny']:+.1f}%")
    if "HiT-MAE" in baseline_results:
        log(f"  DiffMAE vs MAE: {best_val_acc - baseline_results['HiT-MAE']:+.1f}%")

    # ---- Plot ----
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    if pretrain_history:
        pt_epochs = [h["epoch"] for h in pretrain_history]
        axes[0, 0].plot(pt_epochs, [h["train_loss"] for h in pretrain_history],
                        label="Train", linewidth=1.5)
        axes[0, 0].plot(pt_epochs, [h["val_loss"] for h in pretrain_history],
                        label="Val", linewidth=1.5)
    axes[0, 0].set_title(f"DiffMAE Pretrain Loss ({args.mask_ratio:.0%} mask, offset target)")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("MSE Loss")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    ft_epochs = range(1, args.finetune_epochs + 1)
    axes[0, 1].plot(ft_epochs, ft_history["train_loss"], label="Train", linewidth=1.5)
    axes[0, 1].plot(ft_epochs, ft_history["val_loss"], label="Val", linewidth=1.5)
    axes[0, 1].set_title("Finetune Loss")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(ft_epochs, ft_history["train_acc"], label="Train", linewidth=1.5)
    axes[1, 0].plot(ft_epochs, ft_history["val_acc"], label="Val", linewidth=1.5)
    for name, color in [("ViT-Tiny", "red"), ("HiT-Tiny (macro)", "green")]:
        if name in baseline_results:
            axes[1, 0].axhline(y=baseline_results[name], color=color, linestyle="--",
                               alpha=0.5, label=f"{name} ({baseline_results[name]:.1f}%)")
    axes[1, 0].set_title("Finetune Accuracy")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("Accuracy (%)")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(ft_epochs, ft_history["val_acc"], label="DiffMAE", linewidth=2)
    for name, color, ls in [
        ("ViT-Tiny", "red", "--"), ("HiT-Tiny (macro)", "green", "--"),
        ("HiT-denoise", "purple", ":"), ("HiT-MAE", "orange", ":"),
    ]:
        if name in baseline_results:
            axes[1, 1].axhline(y=baseline_results[name], color=color, linestyle=ls,
                               alpha=0.7, label=f"{name} ({baseline_results[name]:.1f}%)")
    axes[1, 1].set_title("Val Accuracy Comparison")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("Val Accuracy (%)")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.suptitle(f"HiT-DiffMAE on {ds_tag}", fontsize=14)
    plt.tight_layout()
    plot_path = os.path.join(fig_dir, f"pretrain_diffmae_{ds_tag}.png")
    plt.savefig(plot_path, dpi=150)
    log(f"\nSaved: {plot_path}")

    json_path = os.path.join(fig_dir, f"pretrain_diffmae_{ds_tag}.json")
    with open(json_path, "w") as f:
        json.dump({
            "pretrain_history": pretrain_history,
            "finetune_history": ft_history,
            "best_val_acc": best_val_acc,
            "baselines": baseline_results,
        }, f, indent=2)
    log(f"Saved: {json_path}")

    log(f"\nLog saved to: {log_path}")
    log("Done.")
    log_file.close()


if __name__ == "__main__":
    main()
