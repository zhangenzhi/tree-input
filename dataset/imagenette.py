import os
import tarfile
import urllib.request

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


IMAGENETTE_URL = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2.tgz"


def download_imagenette(data_dir="./data"):
    """Download and extract Imagenette2 (full-size) if not present."""
    dest = os.path.join(data_dir, "imagenette2")
    if os.path.isdir(dest) and os.path.isdir(os.path.join(dest, "train")):
        return dest

    os.makedirs(data_dir, exist_ok=True)
    tgz_path = os.path.join(data_dir, "imagenette2.tgz")

    if not os.path.exists(tgz_path):
        print(f"Downloading Imagenette2 to {tgz_path} ...")
        urllib.request.urlretrieve(IMAGENETTE_URL, tgz_path)
        print("Download complete.")

    print(f"Extracting to {data_dir} ...")
    with tarfile.open(tgz_path, "r:gz") as tar:
        tar.extractall(path=data_dir)
    print("Extraction complete.")

    return dest


def get_imagenette(batch_size=64, data_dir="./data", num_workers=4):
    """Get Imagenette train/val loaders for single-GPU training.

    Returns:
        train_loader, val_loader, num_classes (10)
    """
    root = download_imagenette(data_dir)

    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    transform_val = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    train_set = datasets.ImageFolder(os.path.join(root, "train"), transform=transform_train)
    val_set = datasets.ImageFolder(os.path.join(root, "val"), transform=transform_val)

    use_pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=use_pin_memory, drop_last=True,
    )
    val_loader = DataLoader(
        val_set, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=use_pin_memory,
    )

    return train_loader, val_loader, 10


def imagenette_dataloader(args):
    """Distributed dataloader for Imagenette (same interface as imagenet_dataloader)."""
    root = download_imagenette(args.data_dir)

    data_transforms = {
        "train": transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.AutoAugment(transforms.AutoAugmentPolicy.IMAGENET),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=0.25),
        ]),
        "val": transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]),
    }

    image_datasets = {
        x: datasets.ImageFolder(os.path.join(root, x), transform=data_transforms[x])
        for x in ["train", "val"]
    }

    samplers = {
        x: torch.utils.data.distributed.DistributedSampler(image_datasets[x])
        for x in ["train", "val"]
    }

    dataloaders = {
        x: DataLoader(
            image_datasets[x],
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=True,
            sampler=samplers[x],
            drop_last=(x == "train"),
        )
        for x in ["train", "val"]
    }
    return dataloaders, samplers
