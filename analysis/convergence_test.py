"""
Quick convergence comparison: HiT vs ViT on CIFAR-10 (resized to 224).

Uses ViT-Tiny scale (embed_dim=192, depth=6, heads=3) for speed.
Trains both models on CIFAR-10 for 20 epochs on MPS/CPU.
Plots training loss and validation accuracy curves side by side.

Purpose: verify whether multi-scale input leads to faster early convergence.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import timm
from model.hit import extract_pyramid_patches, build_pyramid_coords, ContinuousPE3D


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_cifar10(batch_size=64):
    transform_train = transforms.Compose([
        transforms.Resize(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    transform_val = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    train_set = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform_train)
    val_set = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform_val)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=2)
    return train_loader, val_loader


class ViTTiny(nn.Module):
    """ViT-Tiny from timm for fast experiments."""
    def __init__(self, num_classes=10):
        super().__init__()
        self.vit = timm.create_model(
            "vit_tiny_patch16_224",
            pretrained=False,
            num_classes=num_classes,
        )

    def forward(self, x):
        return self.vit(x)


class HiTTiny(nn.Module):
    """HiT-Tiny: multi-scale pyramid input with ViT-Tiny backbone."""
    def __init__(self, num_classes=10, levels=None):
        super().__init__()
        self.levels = levels or [1, 2, 4, 8, 14]
        self.total_patches = sum(n * n for n in self.levels)
        self.patch_size = 16

        vit = timm.create_model("vit_tiny_patch16_224", pretrained=False, num_classes=num_classes)
        self.embed_dim = vit.embed_dim  # 192

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


def run_experiment(num_epochs=20, batch_size=64, lr=1e-3):
    device = get_device()
    print(f"Device: {device}")
    print(f"Epochs: {num_epochs}, Batch size: {batch_size}, LR: {lr}")
    print()

    train_loader, val_loader = get_cifar10(batch_size)
    criterion = nn.CrossEntropyLoss()

    results = {}

    for name, model_fn in [("ViT-Tiny", ViTTiny), ("HiT-Tiny", HiTTiny)]:
        print(f"{'='*50}")
        print(f"Training {name}...")
        print(f"{'='*50}")

        torch.manual_seed(42)
        model = model_fn(num_classes=10).to(device)
        param_count = sum(p.numel() for p in model.parameters())
        print(f"Parameters: {param_count:,}")

        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)

        history = {
            "train_loss": [], "train_acc": [],
            "val_loss": [], "val_acc": [],
            "epoch_time": [],
        }

        for epoch in range(num_epochs):
            start = time.time()
            train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
            val_loss, val_acc = validate(model, val_loader, criterion, device)
            elapsed = time.time() - start

            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)
            history["epoch_time"].append(elapsed)

            print(f"  Epoch {epoch+1:2d}/{num_epochs} | "
                  f"train_loss={train_loss:.4f} train_acc={train_acc:.1f}% | "
                  f"val_loss={val_loss:.4f} val_acc={val_acc:.1f}% | "
                  f"time={elapsed:.1f}s")

        results[name] = history
        print()

    # ---- Plot ----
    fig_dir = os.path.join(os.path.dirname(__file__), "figures")
    os.makedirs(fig_dir, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    epochs = range(1, num_epochs + 1)

    for name, hist in results.items():
        axes[0, 0].plot(epochs, hist["train_loss"], label=name, marker="o", markersize=3)
        axes[0, 1].plot(epochs, hist["val_loss"], label=name, marker="o", markersize=3)
        axes[1, 0].plot(epochs, hist["train_acc"], label=name, marker="o", markersize=3)
        axes[1, 1].plot(epochs, hist["val_acc"], label=name, marker="o", markersize=3)

    axes[0, 0].set_title("Train Loss")
    axes[0, 1].set_title("Val Loss")
    axes[1, 0].set_title("Train Acc (%)")
    axes[1, 1].set_title("Val Acc (%)")

    for ax in axes.flat:
        ax.set_xlabel("Epoch")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.suptitle("ViT-Tiny vs HiT-Tiny on CIFAR-10 (224x224)", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "convergence_comparison.png"), dpi=150)
    print(f"Saved: {fig_dir}/convergence_comparison.png")

    # Print summary
    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for name, hist in results.items():
        best_val = max(hist["val_acc"])
        best_epoch = hist["val_acc"].index(best_val) + 1
        avg_time = np.mean(hist["epoch_time"])
        print(f"  {name:10s} | best_val_acc={best_val:.1f}% (epoch {best_epoch}) | avg_time={avg_time:.1f}s/epoch")
    print()

    # Early convergence comparison (first 5 epochs)
    print("Early convergence (first 5 epochs):")
    for name, hist in results.items():
        print(f"  {name:10s} | val_acc@5 = {hist['val_acc'][4]:.1f}%  train_loss@5 = {hist['train_loss'][4]:.4f}")


if __name__ == "__main__":
    run_experiment(num_epochs=20, batch_size=64, lr=1e-3)
