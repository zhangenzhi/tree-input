import os
import time
import logging
import argparse

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import GradScaler, autocast

from model.hit import create_hit_base
from dataset.imagenet import imagenet_dataloader


def setup_logging(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    logging.basicConfig(
        filename=os.path.join(output_dir, "hit_train.log"),
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logging.getLogger().addHandler(console)


def setup_distributed():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank


def train_one_epoch(model, loader, criterion, optimizer, scaler, epoch, local_rank):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for step, (images, labels) in enumerate(loader):
        images = images.cuda(local_rank, non_blocking=True)
        labels = labels.cuda(local_rank, non_blocking=True)

        optimizer.zero_grad()
        with autocast():
            outputs = model(images)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        if step % 200 == 0 and local_rank == 0:
            logging.info(
                f"  [Epoch {epoch}][Step {step}/{len(loader)}] "
                f"loss={running_loss / (step + 1):.4f} acc={100.0 * correct / total:.2f}%"
            )

    return running_loss / len(loader), 100.0 * correct / total


@torch.no_grad()
def validate(model, loader, criterion, local_rank):
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.cuda(local_rank, non_blocking=True)
        labels = labels.cuda(local_rank, non_blocking=True)

        with autocast():
            outputs = model(images)
            loss = criterion(outputs, labels)

        val_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    return val_loss / len(loader), 100.0 * correct / total


def main():
    parser = argparse.ArgumentParser(description="HiT-B ImageNet Training (DDP)")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output", type=str, default="./output/hit_b")
    parser.add_argument("--batch_size", type=int, default=256, help="per-GPU batch size")
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--num_epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.3)
    parser.add_argument("--warmup_epochs", type=int, default=10)
    parser.add_argument("--label_smoothing", type=float, default=0.1)
    parser.add_argument("--levels", type=str, default="1,2,4,8,14", help="pyramid levels, comma-separated")
    args = parser.parse_args()

    local_rank = setup_distributed()
    setup_logging(args.output)

    levels = [int(x) for x in args.levels.split(",")]

    if local_rank == 0:
        logging.info(f"Args: {args}")
        logging.info(f"Pyramid levels: {levels}, total patches: {sum(n*n for n in levels)}")

    # Data
    dataloaders, samplers = imagenet_dataloader(args)

    # Model
    model = create_hit_base(num_classes=1000, pretrained=False, levels=levels)
    model = model.cuda(local_rank)
    model = DDP(model, device_ids=[local_rank])

    # Loss / Optimizer / Scheduler
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    total_steps = args.num_epochs * len(dataloaders["train"])
    warmup_steps = args.warmup_epochs * len(dataloaders["train"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps)
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1e-4, total_iters=warmup_steps)
    lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup_scheduler, scheduler], milestones=[warmup_steps]
    )

    scaler = GradScaler()
    best_val_acc = 0.0

    for epoch in range(args.num_epochs):
        samplers["train"].set_epoch(epoch)
        start = time.time()

        train_loss, train_acc = train_one_epoch(
            model, dataloaders["train"], criterion, optimizer, scaler, epoch, local_rank
        )

        lr_scheduler.step()

        val_loss, val_acc = validate(model, dataloaders["val"], criterion, local_rank)

        elapsed = time.time() - start

        if local_rank == 0:
            logging.info(
                f"Epoch {epoch}/{args.num_epochs} | "
                f"train_loss={train_loss:.4f} train_acc={train_acc:.2f}% | "
                f"val_loss={val_loss:.4f} val_acc={val_acc:.2f}% | "
                f"time={elapsed:.1f}s lr={optimizer.param_groups[0]['lr']:.6f}"
            )

            ckpt = {
                "epoch": epoch,
                "model": model.module.state_dict(),
                "optimizer": optimizer.state_dict(),
                "val_acc": val_acc,
            }
            torch.save(ckpt, os.path.join(args.output, "latest.pt"))

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(ckpt, os.path.join(args.output, "best.pt"))
                logging.info(f"  New best val_acc: {best_val_acc:.2f}%")

    if local_rank == 0:
        logging.info(f"Training complete. Best val_acc: {best_val_acc:.2f}%")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
