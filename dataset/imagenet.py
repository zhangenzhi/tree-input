import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def imagenet_dataloader(args):
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.AutoAugment(transforms.AutoAugmentPolicy.IMAGENET),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=0.25),
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]),
    }

    image_datasets = {
        x: datasets.ImageNet(root=args.data_dir, split=x, transform=data_transforms[x])
        for x in ['train', 'val']
    }

    samplers = {
        x: torch.utils.data.distributed.DistributedSampler(image_datasets[x])
        for x in ['train', 'val']
    }

    dataloaders = {
        x: DataLoader(
            image_datasets[x],
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=True,
            sampler=samplers[x],
            drop_last=(x == 'train'),
        )
        for x in ['train', 'val']
    }
    return dataloaders, samplers
