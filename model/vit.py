import timm
from torch import nn


def create_vit_base(num_classes=1000, pretrained=False):
    """Create standard ViT-Base/16 from timm."""
    model = timm.create_model(
        "vit_base_patch16_224",
        pretrained=pretrained,
        num_classes=num_classes,
    )
    return model
