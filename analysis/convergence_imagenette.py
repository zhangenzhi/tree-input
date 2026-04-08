"""
Convergence comparison on Imagenette (10 classes, native high-res).

Models: ViT-Tiny, HiT-Tiny (macro), HiT-micro, HiT-random.

Usage:
    python analysis/convergence_imagenette.py --num_epochs 100
    python analysis/convergence_imagenette.py --num_epochs 100 --models hit_micro,hit_random
    python analysis/convergence_imagenette.py --num_epochs 100 --models all
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import time
import json
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import timm
from model.hit import extract_pyramid_patches, build_pyramid_coords, ContinuousPE3D
from dataset.imagenette import get_imagenette


LEVELS = [1, 2, 4, 8, 14]
NUM_CLASSES = 10
NUM_MICRO = 85
MICRO_CROP_SIZE = 8


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ---- Models ----

class ViTTiny(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        self.vit = timm.create_model(
            "vit_tiny_patch16_224", pretrained=False, num_classes=num_classes
        )

    def forward(self, x):
        return self.vit(x)


class HiTTiny(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, levels=None):
        super().__init__()
        self.levels = levels or LEVELS
        self.total_patches = sum(n * n for n in self.levels)
        self.patch_size = 16

        vit = timm.create_model(
            "vit_tiny_patch16_224", pretrained=False, num_classes=num_classes
        )
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


class HiTMicroTiny(nn.Module):
    """HiT with micro-prefix: 85 random 8x8 crops + 196 standard patches."""

    def __init__(self, num_classes=NUM_CLASSES, num_micro=NUM_MICRO, crop_size=MICRO_CROP_SIZE):
        super().__init__()
        self.num_micro = num_micro
        self.crop_size = crop_size
        self.patch_size = 16

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

    def _extract_micro(self, images):
        import torch.nn.functional as F
        B, C, H, W = images.shape
        if self.training:
            top = torch.randint(0, H - self.crop_size + 1, (self.num_micro,))
            left = torch.randint(0, W - self.crop_size + 1, (self.num_micro,))
        else:
            torch.manual_seed(0)
            top = torch.randint(0, H - self.crop_size + 1, (self.num_micro,))
            left = torch.randint(0, W - self.crop_size + 1, (self.num_micro,))
        crops, coords = [], []
        for i in range(self.num_micro):
            t, l = top[i].item(), left[i].item()
            crop = images[:, :, t:t+self.crop_size, l:l+self.crop_size]
            resized = F.interpolate(crop, size=(self.patch_size, self.patch_size),
                                    mode='bilinear', align_corners=False)
            crops.append(resized.reshape(B, -1))
            coords.append([(l + self.crop_size/2)/W, (t + self.crop_size/2)/H, self.crop_size/H])
        return torch.stack(crops, dim=1), torch.tensor(coords, dtype=torch.float32)

    def forward(self, images):
        B = images.size(0)
        micro_patches, micro_coords = self._extract_micro(images)
        micro_coords = micro_coords.to(images.device)
        fine_patches = extract_pyramid_patches(images, [14], self.patch_size)

        all_patches = torch.cat([micro_patches, fine_patches], dim=1)
        x = self.patch_proj(all_patches)

        all_coords = torch.cat([micro_coords, self.fine_coords], dim=0)
        pe = self.pe(all_coords)
        x = x + pe.unsqueeze(0)

        cls_tokens = self.cls_token.expand(B, -1, -1) + self.cls_pos
        x = torch.cat([cls_tokens, x], dim=1)
        x = self.pos_drop(x)
        x = self.blocks(x)
        x = self.norm(x)
        return self.head(x[:, 0])


class HiTRandomTiny(nn.Module):
    """HiT with random-prefix: 85 randomly duplicated L4 patches + 196 standard patches."""

    def __init__(self, num_classes=NUM_CLASSES, num_random=NUM_MICRO):
        super().__init__()
        self.num_random = num_random
        self.patch_size = 16

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
        fine_patches = extract_pyramid_patches(images, [14], self.patch_size)

        if self.training:
            indices = torch.randint(0, 196, (self.num_random,))
        else:
            torch.manual_seed(0)
            indices = torch.randint(0, 196, (self.num_random,))

        random_patches = fine_patches[:, indices, :]
        random_coords = self.fine_coords[indices]

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


class HiTRandomPETiny(nn.Module):
    """HiT with random-prefix + different PE: 85 duplicated L4 patches but with
    modified positional encoding (s offset to -1) so embeddings are NOT identical.

    Same redundant content as HiT-random, but softmax can distinguish original vs duplicate.
    Tests whether softmax identity failure or information redundancy causes overfitting.
    """

    def __init__(self, num_classes=NUM_CLASSES, num_random=NUM_MICRO):
        super().__init__()
        self.num_random = num_random
        self.patch_size = 16

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
        fine_patches = extract_pyramid_patches(images, [14], self.patch_size)

        if self.training:
            indices = torch.randint(0, 196, (self.num_random,))
        else:
            torch.manual_seed(0)
            indices = torch.randint(0, 196, (self.num_random,))

        random_patches = fine_patches[:, indices, :]

        # Same (cx, cy) as original, but s = -1 to mark as "duplicate level"
        # This makes PE different from original L4 tokens → breaks embedding identity
        random_coords = self.fine_coords[indices].clone()
        random_coords[:, 2] = -1.0  # s dimension offset

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


class HiTOffsetTiny(nn.Module):
    """HiT with offset-prefix: 85 fixed-offset 16x16 patches + 196 standard patches.

    Patches are shifted by (8, 8) pixels from the 14x14 grid, landing at the
    intersection of 4 neighboring L4 patches. They contain real image content
    that partially overlaps with L4 but is NOT identical.
    Tests whether non-redundant same-scale extra tokens provide useful information.

    Offset grid: 13x13 = 169 positions available, randomly select 85.
    """

    def __init__(self, num_classes=NUM_CLASSES, num_offset=NUM_MICRO):
        super().__init__()
        self.num_offset = num_offset
        self.patch_size = 16
        self.offset = 8  # half-patch shift

        vit = timm.create_model("vit_tiny_patch16_224", pretrained=False, num_classes=num_classes)
        self.embed_dim = vit.embed_dim

        self.patch_proj = nn.Linear(16 * 16 * 3, self.embed_dim)
        self.cls_token = vit.cls_token
        self.pe = ContinuousPE3D(self.embed_dim, num_freq=32)

        fine_coords = build_pyramid_coords([14])
        self.register_buffer("fine_coords", fine_coords)

        # Precompute the offset grid: 13x13 positions starting at (8, 8) with stride 16
        # These sit exactly at the intersection of 4 neighboring L4 patches
        offset_positions = []
        for row in range(13):
            for col in range(13):
                t = self.offset + row * self.patch_size
                l = self.offset + col * self.patch_size
                offset_positions.append((t, l))
        # Fixed random selection of 85 from 169
        torch.manual_seed(42)
        perm = torch.randperm(len(offset_positions))[:num_offset]
        self.offset_positions = [offset_positions[i] for i in perm]

        # Precompute PE coords for offset patches
        offset_coords = []
        for t, l in self.offset_positions:
            cx = (l + self.patch_size / 2) / 224.0
            cy = (t + self.patch_size / 2) / 224.0
            s = self.patch_size / 224.0
            offset_coords.append([cx, cy, s])
        self.register_buffer("offset_coords", torch.tensor(offset_coords, dtype=torch.float32))

        self.cls_pos = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        self.pos_drop = vit.pos_drop
        self.blocks = vit.blocks
        self.norm = vit.norm
        self.head = vit.head

    def _extract_offset_patches(self, images):
        """Extract 16x16 patches at fixed (8,8)-offset grid positions."""
        B = images.size(0)
        P = self.patch_size
        patches = []
        for t, l in self.offset_positions:
            patch = images[:, :, t:t+P, l:l+P]  # [B, C, P, P]
            patches.append(patch.reshape(B, -1))
        return torch.stack(patches, dim=1)  # [B, num_offset, C*P*P]

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


class HiTNoiseTiny(nn.Module):
    """HiT with noise-prefix: 85 Gaussian noise patches + 196 standard patches.

    Control for softmax failure hypothesis: noise patches have random embeddings
    that are NOT identical to any L4 patch, so softmax routing remains functional.
    Carries zero useful information — tests whether extra tokens alone cause overfitting.
    """

    def __init__(self, num_classes=NUM_CLASSES, num_noise=NUM_MICRO):
        super().__init__()
        self.num_noise = num_noise
        self.patch_size = 16

        vit = timm.create_model("vit_tiny_patch16_224", pretrained=False, num_classes=num_classes)
        self.embed_dim = vit.embed_dim

        self.patch_proj = nn.Linear(16 * 16 * 3, self.embed_dim)
        self.cls_token = vit.cls_token
        self.pe = ContinuousPE3D(self.embed_dim, num_freq=32)

        fine_coords = build_pyramid_coords([14])
        self.register_buffer("fine_coords", fine_coords)

        # Fixed random coords for noise tokens (spread uniformly)
        torch.manual_seed(99)
        noise_coords = torch.rand(num_noise, 3)  # random (cx, cy, s) in [0,1]
        self.register_buffer("noise_coords", noise_coords)

        self.cls_pos = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        self.pos_drop = vit.pos_drop
        self.blocks = vit.blocks
        self.norm = vit.norm
        self.head = vit.head

    def forward(self, images):
        B = images.size(0)
        device = images.device

        # Standard L4 patches
        fine_patches = extract_pyramid_patches(images, [14], self.patch_size)

        # Generate Gaussian noise patches (pixel-space, same dim as real patches)
        noise_patches = torch.randn(B, self.num_noise, 3 * self.patch_size * self.patch_size,
                                    device=device)

        all_patches = torch.cat([noise_patches, fine_patches], dim=1)
        x = self.patch_proj(all_patches)

        all_coords = torch.cat([self.noise_coords, self.fine_coords], dim=0)
        pe = self.pe(all_coords)
        x = x + pe.unsqueeze(0)

        cls_tokens = self.cls_token.expand(B, -1, -1) + self.cls_pos
        x = torch.cat([cls_tokens, x], dim=1)
        x = self.pos_drop(x)
        x = self.blocks(x)
        x = self.norm(x)
        return self.head(x[:, 0])


class HiTMacroOffsetTiny(nn.Module):
    """HiT with macro + offset: L0-L3 pyramid (85) + (8,8)-offset patches (85) + L4 (196).

    Combines two orthogonal sources of genuinely new information:
    - Macro (L0-L3): global multi-scale structure (top-down)
    - Offset (8,8): cross-boundary texture continuity (bottom-up)
    Total: 1 CLS + 85 macro + 85 offset + 196 L4 = 367 tokens.
    """

    def __init__(self, num_classes=NUM_CLASSES, levels=None, num_offset=NUM_MICRO):
        super().__init__()
        self.levels = levels or LEVELS
        self.total_macro = sum(n * n for n in self.levels[:-1])  # 85
        self.num_offset = num_offset
        self.patch_size = 16
        self.offset = 8

        vit = timm.create_model("vit_tiny_patch16_224", pretrained=False, num_classes=num_classes)
        self.embed_dim = vit.embed_dim

        self.patch_proj = nn.Linear(16 * 16 * 3, self.embed_dim)
        self.cls_token = vit.cls_token
        self.pe = ContinuousPE3D(self.embed_dim, num_freq=32)

        # Macro coords (L0-L3)
        macro_coords = build_pyramid_coords(self.levels[:-1])  # [85, 3]
        self.register_buffer("macro_coords", macro_coords)

        # Fine coords (L4)
        fine_coords = build_pyramid_coords([14])  # [196, 3]
        self.register_buffer("fine_coords", fine_coords)

        # Offset grid: 13x13 positions at (8,8) shift, select 85
        offset_positions = []
        for row in range(13):
            for col in range(13):
                t = self.offset + row * self.patch_size
                l = self.offset + col * self.patch_size
                offset_positions.append((t, l))
        torch.manual_seed(42)
        perm = torch.randperm(len(offset_positions))[:num_offset]
        self.offset_positions = [offset_positions[i] for i in perm]

        offset_coords = []
        for t, l in self.offset_positions:
            cx = (l + self.patch_size / 2) / 224.0
            cy = (t + self.patch_size / 2) / 224.0
            s = self.patch_size / 224.0
            offset_coords.append([cx, cy, s])
        self.register_buffer("offset_coords", torch.tensor(offset_coords, dtype=torch.float32))

        self.cls_pos = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        self.pos_drop = vit.pos_drop
        self.blocks = vit.blocks
        self.norm = vit.norm
        self.head = vit.head

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

        # Macro patches (L0-L3)
        macro_patches = extract_pyramid_patches(images, self.levels[:-1], self.patch_size)
        # Offset patches (8,8 shifted grid)
        offset_patches = self._extract_offset_patches(images)
        # Fine patches (L4)
        fine_patches = extract_pyramid_patches(images, [14], self.patch_size)

        # Concatenate: [macro | offset | fine]
        all_patches = torch.cat([macro_patches, offset_patches, fine_patches], dim=1)
        x = self.patch_proj(all_patches)

        # PE
        all_coords = torch.cat([self.macro_coords, self.offset_coords, self.fine_coords], dim=0)
        pe = self.pe(all_coords)
        x = x + pe.unsqueeze(0)

        # CLS
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


def run_experiment(num_epochs=100, batch_size=64, lr=1e-3, data_dir="./data",
                   num_workers=4, models="vit,hit"):
    device = get_device()
    print(f"Device: {device}")
    print(f"Dataset: Imagenette (10 classes, native high-res)")
    print(f"Epochs: {num_epochs}, Batch size: {batch_size}, LR: {lr}")
    print(f"Models: {models}")
    print()

    train_loader, val_loader, _ = get_imagenette(
        batch_size=batch_size, data_dir=data_dir, num_workers=num_workers,
    )
    print(f"Train: {len(train_loader.dataset)} images, Val: {len(val_loader.dataset)} images")
    print()

    criterion = nn.CrossEntropyLoss()
    results = {}

    all_models = {
        "vit": ("ViT-Tiny", ViTTiny, "vit_tiny_imagenette.pt"),
        "hit": ("HiT-Tiny", HiTTiny, "hit_tiny_imagenette.pt"),
        "hit_micro": ("HiT-micro", HiTMicroTiny, "hit_micro_tiny_imagenette.pt"),
        "hit_random": ("HiT-random", HiTRandomTiny, "hit_random_tiny_imagenette.pt"),
        "hit_noise": ("HiT-noise", HiTNoiseTiny, "hit_noise_tiny_imagenette.pt"),
        "hit_random_pe": ("HiT-random-PE", HiTRandomPETiny, "hit_random_pe_tiny_imagenette.pt"),
        "hit_offset": ("HiT-offset", HiTOffsetTiny, "hit_offset_tiny_imagenette.pt"),
        "hit_macro_offset": ("HiT-macro+offset", HiTMacroOffsetTiny, "hit_macro_offset_tiny_imagenette.pt"),
    }

    if models == "all":
        model_keys = list(all_models.keys())
    else:
        model_keys = [m.strip() for m in models.split(",")]

    models_to_run = []
    for key in model_keys:
        if key in all_models:
            models_to_run.append(all_models[key])

    ckpt_dir = os.path.join("output", "imagenette_ckpt")
    os.makedirs(ckpt_dir, exist_ok=True)

    for name, model_cls, ckpt_name in models_to_run:
        ckpt_path = os.path.join(ckpt_dir, ckpt_name)

        print(f"{'='*60}")

        torch.manual_seed(42)
        model = model_cls(num_classes=NUM_CLASSES).to(device)
        param_count = sum(p.numel() for p in model.parameters())

        # Load checkpoint if exists
        start_epoch = 0
        best_val_acc = 0.0
        if os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
            model.load_state_dict(ckpt["model"])
            ckpt_epoch = ckpt["epoch"]
            ckpt_val = ckpt.get("best_val_acc", ckpt.get("val_acc", "?"))
            print(f"Loading {name} from {ckpt_path}")
            print(f"  Loaded (epoch={ckpt_epoch}, val_acc={ckpt_val}%)")

            if ckpt_epoch >= num_epochs:
                print(f"  Skipping training, using cached results.")
                results[name] = None
                print()
                continue
            else:
                start_epoch = ckpt_epoch
                best_val_acc = ckpt.get("best_val_acc", 0.0)
                print(f"  Resuming from epoch {start_epoch} to {num_epochs}...")

        print(f"Training {name}...")
        print(f"{'='*60}")
        print(f"Parameters: {param_count:,}")

        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)
        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=num_epochs
        )
        # Advance scheduler to correct position if resuming
        for _ in range(start_epoch):
            cosine_scheduler.step()

        history = {
            "train_loss": [], "train_acc": [],
            "val_loss": [], "val_acc": [],
            "epoch_time": [],
        }

        for epoch in range(start_epoch, num_epochs):
            start = time.time()
            train_loss, train_acc = train_one_epoch(
                model, train_loader, criterion, optimizer, device
            )
            cosine_scheduler.step()
            val_loss, val_acc = validate(model, val_loader, criterion, device)
            elapsed = time.time() - start

            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)
            history["epoch_time"].append(elapsed)

            if val_acc > best_val_acc:
                best_val_acc = val_acc

            if (epoch + 1) % 10 == 0 or epoch == start_epoch:
                print(f"  Epoch {epoch+1:3d}/{num_epochs} | "
                      f"train_loss={train_loss:.4f} train_acc={train_acc:.1f}% | "
                      f"val_loss={val_loss:.4f} val_acc={val_acc:.1f}% | "
                      f"time={elapsed:.1f}s lr={optimizer.param_groups[0]['lr']:.6f}")

        results[name] = history

        torch.save({
            "epoch": num_epochs,
            "model": model.state_dict(),
            "train_acc": history["train_acc"][-1],
            "val_acc": history["val_acc"][-1],
            "best_val_acc": best_val_acc,
        }, ckpt_path)
        print(f"  Saved: {ckpt_path}")
        print()

    # ---- Save results ----
    fig_dir = os.path.join(os.path.dirname(__file__), "figures")
    os.makedirs(fig_dir, exist_ok=True)

    with open(os.path.join(fig_dir, "imagenette_convergence.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved: {fig_dir}/imagenette_convergence.json")

    # ---- Plot ----
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    epochs = range(1, num_epochs + 1)

    for name, hist in results.items():
        if hist is None:
            continue
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

    plt.suptitle(f"Imagenette Convergence ({num_epochs} epochs)", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "imagenette_convergence.png"), dpi=150)
    print(f"Saved: {fig_dir}/imagenette_convergence.png")

    # ---- Summary ----
    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for name, hist in results.items():
        if hist is None:
            continue
        best_val = max(hist["val_acc"])
        best_epoch = hist["val_acc"].index(best_val) + 1
        avg_time = np.mean(hist["epoch_time"])
        final_train = hist["train_acc"][-1]
        final_val = hist["val_acc"][-1]
        gap = final_train - final_val
        print(f"  {name:10s} | best_val={best_val:.1f}% (ep{best_epoch}) | "
              f"final train={final_train:.1f}% val={final_val:.1f}% gap={gap:.1f}% | "
              f"avg_time={avg_time:.1f}s/ep")


def run_ablation(batch_size=64, data_dir="./data", num_workers=4):
    """Level ablation test on trained HiT-macro checkpoint."""
    device = get_device()
    print(f"Device: {device}")
    print()

    _, val_loader, _ = get_imagenette(
        batch_size=batch_size, data_dir=data_dir, num_workers=num_workers,
    )
    print(f"Val: {len(val_loader.dataset)} images")
    print()

    # Load HiT-macro checkpoint
    ckpt_dir = os.path.join("output", "imagenette_ckpt")
    ckpt_path = os.path.join(ckpt_dir, "hit_tiny_imagenette.pt")

    torch.manual_seed(42)
    model = HiTTiny(num_classes=NUM_CLASSES).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model"])
    print(f"Loaded HiT-macro from {ckpt_path}")
    print(f"  epoch={ckpt['epoch']}, best_val_acc={ckpt.get('best_val_acc', '?')}%")
    print()

    # Level offsets
    level_offsets = {}
    offset = 1
    for lvl_idx, n in enumerate(LEVELS):
        level_offsets[lvl_idx] = (offset, offset + n * n)
        offset += n * n

    ablation_configs = [
        ("Full (L0-L4)", None),
        ("L4 only (fine)", [4]),
        ("L0-L3 only (coarse)", [0, 1, 2, 3]),
        ("L3-L4", [3, 4]),
        ("L0+L4", [0, 4]),
        ("L2-L4", [2, 3, 4]),
        ("L1-L4", [1, 2, 3, 4]),
    ]

    print("=" * 60)
    print("Level Ablation Test (Imagenette)")
    print("=" * 60)

    model.eval()
    for config_name, keep_levels in ablation_configs:
        correct, total_count = 0, 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                B = images.size(0)

                patches = extract_pyramid_patches(images, model.levels, model.patch_size)
                x = model.patch_proj(patches)
                pe = model.pe(model.pyramid_coords)
                x = x + pe.unsqueeze(0)
                cls_tokens = model.cls_token.expand(B, -1, -1) + model.cls_pos
                x = torch.cat([cls_tokens, x], dim=1)

                if keep_levels is not None:
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
        n_tokens = len(keep_indices) if keep_levels is not None else 1 + sum(n * n for n in LEVELS)
        print(f"  {config_name:25s} | tokens={n_tokens:4d} | val_acc={acc:.1f}%")

    print("=" * 60)


def run_offset_ablation(batch_size=64, data_dir="./data", num_workers=4):
    """Offset ablation: load HiT-offset checkpoint, evaluate with/without offset tokens."""
    device = get_device()
    print(f"Device: {device}")
    print()

    _, val_loader, _ = get_imagenette(
        batch_size=batch_size, data_dir=data_dir, num_workers=num_workers,
    )
    print(f"Val: {len(val_loader.dataset)} images")
    print()

    ckpt_dir = os.path.join("output", "imagenette_ckpt")

    # Test HiT-offset
    ckpt_path = os.path.join(ckpt_dir, "hit_offset_tiny_imagenette.pt")
    if os.path.exists(ckpt_path):
        torch.manual_seed(42)
        model = HiTOffsetTiny(num_classes=NUM_CLASSES).to(device)
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model"])
        print(f"Loaded HiT-offset from {ckpt_path}")
        print(f"  epoch={ckpt['epoch']}, best_val_acc={ckpt.get('best_val_acc', '?')}%")
    else:
        print(f"No checkpoint at {ckpt_path}")
        return

    print()
    print("=" * 60)
    print("Offset Ablation Test (Imagenette)")
    print("=" * 60)

    model.eval()
    # Config: (name, use_offset)
    configs = [
        ("Full (offset + L4)", True),
        ("L4 only (drop offset)", False),
    ]

    for config_name, use_offset in configs:
        correct, total_count = 0, 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                B = images.size(0)

                fine_patches = extract_pyramid_patches(images, [14], model.patch_size)

                if use_offset:
                    offset_patches = model._extract_offset_patches(images)
                    all_patches = torch.cat([offset_patches, fine_patches], dim=1)
                    x = model.patch_proj(all_patches)
                    all_coords = torch.cat([model.offset_coords, model.fine_coords], dim=0)
                    pe = model.pe(all_coords)
                    x = x + pe.unsqueeze(0)
                else:
                    # L4 only: skip offset tokens
                    x = model.patch_proj(fine_patches)
                    pe = model.pe(model.fine_coords)
                    x = x + pe.unsqueeze(0)

                cls_tokens = model.cls_token.expand(B, -1, -1) + model.cls_pos
                x = torch.cat([cls_tokens, x], dim=1)
                x = model.pos_drop(x)
                x = model.blocks(x)
                x = model.norm(x)
                logits = model.head(x[:, 0])

                _, predicted = logits.max(1)
                correct += predicted.eq(labels).sum().item()
                total_count += labels.size(0)

        acc = 100.0 * correct / total_count
        n_tokens = 282 if use_offset else 197
        print(f"  {config_name:30s} | tokens={n_tokens:4d} | val_acc={acc:.1f}%")

    # Also test HiT-macro+offset if available
    ckpt_path2 = os.path.join(ckpt_dir, "hit_macro_offset_tiny_imagenette.pt")
    if os.path.exists(ckpt_path2):
        torch.manual_seed(42)
        model2 = HiTMacroOffsetTiny(num_classes=NUM_CLASSES).to(device)
        ckpt2 = torch.load(ckpt_path2, map_location=device, weights_only=True)
        model2.load_state_dict(ckpt2["model"])
        print()
        print(f"Loaded HiT-macro+offset from {ckpt_path2}")
        print(f"  epoch={ckpt2['epoch']}, best_val_acc={ckpt2.get('best_val_acc', '?')}%")
        print()

        # Configs for macro+offset: full, L4+offset, L4+macro, L4 only
        mo_configs = [
            ("Full (macro+offset+L4)", True, True),
            ("Offset+L4 (drop macro)", False, True),
            ("Macro+L4 (drop offset)", True, False),
            ("L4 only", False, False),
        ]

        model2.eval()
        for config_name, use_macro, use_offset in mo_configs:
            correct, total_count = 0, 0

            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(device), labels.to(device)
                    B = images.size(0)

                    fine_patches = extract_pyramid_patches(images, [14], model2.patch_size)
                    x_fine = model2.patch_proj(fine_patches)

                    parts = []
                    coord_parts = []

                    if use_macro:
                        macro_patches = extract_pyramid_patches(
                            images, model2.levels[:-1], model2.patch_size)
                        x_macro = model2.patch_proj(macro_patches)
                        parts.append(x_macro)
                        coord_parts.append(model2.macro_coords)

                    if use_offset:
                        offset_patches = model2._extract_offset_patches(images)
                        x_offset = model2.patch_proj(offset_patches)
                        parts.append(x_offset)
                        coord_parts.append(model2.offset_coords)

                    parts.append(x_fine)
                    coord_parts.append(model2.fine_coords)

                    x = torch.cat(parts, dim=1)
                    all_coords = torch.cat(coord_parts, dim=0)
                    pe = model2.pe(all_coords)
                    x = x + pe.unsqueeze(0)

                    cls_tokens = model2.cls_token.expand(B, -1, -1) + model2.cls_pos
                    x = torch.cat([cls_tokens, x], dim=1)
                    x = model2.pos_drop(x)
                    x = model2.blocks(x)
                    x = model2.norm(x)
                    logits = model2.head(x[:, 0])

                    _, predicted = logits.max(1)
                    correct += predicted.eq(labels).sum().item()
                    total_count += labels.size(0)

            acc = 100.0 * correct / total_count
            n_tokens = 1 + (85 if use_macro else 0) + (85 if use_offset else 0) + 196
            print(f"  {config_name:30s} | tokens={n_tokens:4d} | val_acc={acc:.1f}%")

    print("=" * 60)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--models", type=str, default="vit,hit",
                        help="Comma-separated: vit,hit,hit_micro,hit_random or 'all'")
    parser.add_argument("--ablation", action="store_true",
                        help="Run level ablation on trained HiT-macro checkpoint")
    parser.add_argument("--offset_ablation", action="store_true",
                        help="Run offset ablation: test offset/macro internalization")
    args = parser.parse_args()

    if args.ablation:
        run_ablation(
            batch_size=args.batch_size,
            data_dir=args.data_dir,
            num_workers=args.num_workers,
        )
    elif args.offset_ablation:
        run_offset_ablation(
            batch_size=args.batch_size,
            data_dir=args.data_dir,
            num_workers=args.num_workers,
        )
    else:
        run_experiment(
            num_epochs=args.num_epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            data_dir=args.data_dir,
            num_workers=args.num_workers,
            models=args.models,
        )
