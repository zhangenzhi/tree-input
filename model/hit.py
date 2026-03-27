import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


def build_pyramid_coords(levels, image_size=224, patch_size=16):
    """Precompute (cx, cy, s) for each patch at each pyramid level.

    Args:
        levels: list of int, number of splits per side at each level.
            e.g. [1, 2, 4, 8, 14] means 1x1, 2x2, 4x4, 8x8, 14x14 grids.
        image_size: original image size.
        patch_size: target patch size after resize.

    Returns:
        coords: Tensor of shape [total_patches, 3] with (cx, cy, s) in [0, 1].
    """
    coords = []
    for n in levels:
        s = 1.0 / n
        for row in range(n):
            for col in range(n):
                cx = (col + 0.5) / n
                cy = (row + 0.5) / n
                coords.append([cx, cy, s])
    return torch.tensor(coords, dtype=torch.float32)


class ContinuousPE3D(nn.Module):
    """Learned 3D positional encoding from continuous (cx, cy, s) coordinates."""

    def __init__(self, embed_dim, num_freq=64):
        super().__init__()
        self.num_freq = num_freq
        # input: 3 coords * num_freq * 2 (sin + cos)
        input_dim = 3 * num_freq * 2
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )
        # frequency bands (log-spaced)
        freqs = torch.exp(torch.linspace(0, math.log(1000.0), num_freq))
        self.register_buffer("freqs", freqs)

    def forward(self, coords):
        """
        Args:
            coords: [N, 3] tensor of (cx, cy, s).
        Returns:
            pe: [N, embed_dim] positional encoding.
        """
        # coords: [N, 3] -> [N, 3, 1] * [1, 1, num_freq] -> [N, 3, num_freq]
        x = coords.unsqueeze(-1) * self.freqs.unsqueeze(0).unsqueeze(0)
        # sin/cos -> [N, 3, num_freq*2] -> [N, 3*num_freq*2]
        x = torch.cat([x.sin(), x.cos()], dim=-1)
        x = x.reshape(x.size(0), -1)
        return self.mlp(x)


def extract_pyramid_patches(images, levels, patch_size=16):
    """Extract multi-scale pyramid patches from a batch of images.

    Args:
        images: [B, 3, H, W] tensor.
        levels: list of int, grid splits per side. e.g. [1, 2, 4, 8, 14].
        patch_size: size to resize each sub-region to.

    Returns:
        patches: [B, total_patches, 3*patch_size*patch_size] tensor.
    """
    B, C, H, W = images.shape
    all_patches = []

    for n in levels:
        if n == H // patch_size:
            # finest level: no resize needed, just unfold
            p = images.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
            # p: [B, C, n, n, patch_size, patch_size]
            p = p.contiguous().view(B, C, n * n, patch_size, patch_size)
            p = p.permute(0, 2, 1, 3, 4)  # [B, n*n, C, patch_size, patch_size]
        else:
            # split image into n x n grid, resize each to patch_size
            cell_h = H // n
            cell_w = W // n
            patches_level = []
            for row in range(n):
                for col in range(n):
                    crop = images[:, :, row*cell_h:(row+1)*cell_h, col*cell_w:(col+1)*cell_w]
                    resized = F.interpolate(crop, size=(patch_size, patch_size), mode='bilinear', align_corners=False)
                    patches_level.append(resized)
            p = torch.stack(patches_level, dim=1)  # [B, n*n, C, patch_size, patch_size]

        p = p.reshape(B, -1, C * patch_size * patch_size)  # [B, n*n, C*P*P]
        all_patches.append(p)

    return torch.cat(all_patches, dim=1)  # [B, total_patches, C*P*P]


class HiTBase(nn.Module):
    """Hierarchical Input Transformer - Base (HiT-B).

    Uses timm's ViT-Base backbone with replaced positional encoding
    and multi-scale pyramid input.
    """

    def __init__(
        self,
        image_size=224,
        patch_size=16,
        num_classes=1000,
        levels=None,
        pretrained=False,
        pe_num_freq=64,
    ):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.levels = levels or [1, 2, 4, 8, 14]
        self.total_patches = sum(n * n for n in self.levels)

        # Load timm ViT-Base and extract components
        vit = timm.create_model(
            "vit_base_patch16_224",
            pretrained=pretrained,
            num_classes=num_classes,
        )
        self.embed_dim = vit.embed_dim  # 768

        # Reuse timm's patch projection (16x16x3 -> 768)
        self.patch_proj = nn.Linear(patch_size * patch_size * 3, self.embed_dim)

        # CLS token from timm
        self.cls_token = vit.cls_token  # [1, 1, 768]

        # Replace positional encoding with 3D continuous PE
        self.pe = ContinuousPE3D(self.embed_dim, num_freq=pe_num_freq)

        # Precompute pyramid coordinates
        coords = build_pyramid_coords(self.levels, image_size, patch_size)
        self.register_buffer("pyramid_coords", coords)  # [total_patches, 3]

        # CLS token PE: learnable
        self.cls_pos = nn.Parameter(torch.zeros(1, 1, self.embed_dim))

        # Transformer blocks and head from timm
        self.pos_drop = vit.pos_drop
        self.blocks = vit.blocks
        self.norm = vit.norm
        self.head = vit.head

    def forward(self, images):
        """
        Args:
            images: [B, 3, 224, 224] tensor.
        Returns:
            logits: [B, num_classes].
        """
        B = images.size(0)

        # 1. Extract pyramid patches -> [B, total_patches, 3*16*16]
        patches = extract_pyramid_patches(images, self.levels, self.patch_size)

        # 2. Linear projection -> [B, total_patches, 768]
        x = self.patch_proj(patches)

        # 3. Add 3D positional encoding
        pe = self.pe(self.pyramid_coords)  # [total_patches, 768]
        x = x + pe.unsqueeze(0)

        # 4. Prepend CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1) + self.cls_pos
        x = torch.cat([cls_tokens, x], dim=1)  # [B, 1+total_patches, 768]

        # 5. Transformer blocks
        x = self.pos_drop(x)
        x = self.blocks(x)
        x = self.norm(x)

        # 6. Classification head (CLS token)
        return self.head(x[:, 0])


def create_hit_base(num_classes=1000, pretrained=False, levels=None):
    """Create HiT-Base model."""
    return HiTBase(
        num_classes=num_classes,
        pretrained=pretrained,
        levels=levels,
    )
