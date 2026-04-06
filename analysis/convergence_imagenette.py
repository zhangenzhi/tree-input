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
        if os.path.exists(ckpt_path):
            print(f"Loading {name} from {ckpt_path}")
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
            model.load_state_dict(ckpt["model"])
            print(f"  Loaded (epoch={ckpt['epoch']}, val_acc={ckpt.get('best_val_acc', ckpt.get('val_acc', '?'))}%)")
            print(f"  Skipping training, using cached results.")
            results[name] = None  # no history, just checkpoint
            print()
            continue

        print(f"Training {name}...")
        print(f"{'='*60}")
        print(f"Parameters: {param_count:,}")

        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)
        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=num_epochs
        )

        history = {
            "train_loss": [], "train_acc": [],
            "val_loss": [], "val_acc": [],
            "epoch_time": [],
        }

        best_val_acc = 0.0

        for epoch in range(num_epochs):
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

            if (epoch + 1) % 10 == 0 or epoch == 0:
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
    args = parser.parse_args()

    if args.ablation:
        run_ablation(
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
