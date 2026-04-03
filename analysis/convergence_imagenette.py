"""
Convergence comparison: HiT vs ViT on Imagenette (10 classes, native high-res).

Uses ViT-Tiny scale (embed_dim=192, depth=12, heads=3) for speed.
Imagenette images are natively high-resolution, so sub-patch detail is real.

Usage:
    python analysis/convergence_imagenette.py --num_epochs 100
    python analysis/convergence_imagenette.py --num_epochs 100 --skip_vit
    python analysis/convergence_imagenette.py --num_epochs 100 --skip_hit
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
                   num_workers=4, skip_vit=False, skip_hit=False):
    device = get_device()
    print(f"Device: {device}")
    print(f"Dataset: Imagenette (10 classes, native high-res)")
    print(f"Epochs: {num_epochs}, Batch size: {batch_size}, LR: {lr}")
    if skip_vit:
        print("Skipping ViT-Tiny (--skip_vit)")
    if skip_hit:
        print("Skipping HiT-Tiny (--skip_hit)")
    print()

    train_loader, val_loader, _ = get_imagenette(
        batch_size=batch_size, data_dir=data_dir, num_workers=num_workers,
    )
    print(f"Train: {len(train_loader.dataset)} images, Val: {len(val_loader.dataset)} images")
    print()

    criterion = nn.CrossEntropyLoss()
    results = {}

    models_to_run = []
    if not skip_vit:
        models_to_run.append(("ViT-Tiny", ViTTiny))
    if not skip_hit:
        models_to_run.append(("HiT-Tiny", HiTTiny))

    ckpt_dir = os.path.join("output", "imagenette_ckpt")
    os.makedirs(ckpt_dir, exist_ok=True)

    for name, model_fn in models_to_run:
        print(f"{'='*60}")
        print(f"Training {name}...")
        print(f"{'='*60}")

        torch.manual_seed(42)
        model = model_fn(num_classes=NUM_CLASSES).to(device)
        param_count = sum(p.numel() for p in model.parameters())
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

        # Save checkpoint
        tag = "vit" if "ViT" in name else "hit"
        ckpt_path = os.path.join(ckpt_dir, f"{tag}_tiny_imagenette.pt")
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

    for name in ["ViT-Tiny", "HiT-Tiny"]:
        if name not in results:
            continue
        hist = results[name]
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

    plt.suptitle(f"ViT-Tiny vs HiT-Tiny on Imagenette ({num_epochs} epochs)", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "imagenette_convergence.png"), dpi=150)
    print(f"Saved: {fig_dir}/imagenette_convergence.png")

    # ---- Summary ----
    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for name in ["ViT-Tiny", "HiT-Tiny"]:
        if name not in results:
            continue
        hist = results[name]
        best_val = max(hist["val_acc"])
        best_epoch = hist["val_acc"].index(best_val) + 1
        avg_time = np.mean(hist["epoch_time"])
        final_train = hist["train_acc"][-1]
        final_val = hist["val_acc"][-1]
        gap = final_train - final_val
        print(f"  {name:10s} | best_val={best_val:.1f}% (ep{best_epoch}) | "
              f"final train={final_train:.1f}% val={final_val:.1f}% gap={gap:.1f}% | "
              f"avg_time={avg_time:.1f}s/ep")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--skip_vit", action="store_true")
    parser.add_argument("--skip_hit", action="store_true")
    args = parser.parse_args()
    run_experiment(
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        data_dir=args.data_dir,
        num_workers=args.num_workers,
        skip_vit=args.skip_vit,
        skip_hit=args.skip_hit,
    )
