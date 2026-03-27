# Tree-Input ViT: Multi-Scale Image Pyramid as ViT Input

## 1. Experiment Idea

Standard ViT treats an image as a flat sequence of non-overlapping patches, lacking explicit multi-scale structural information. The model must rely on stacking self-attention layers to gradually build up global context and cross-scale feature associations.

**Core idea**: Construct an image pyramid at the input level. For a given image, generate multi-scale representations by recursively subdividing the image and resizing each sub-region to a uniform patch size (e.g., 16x16). Concatenate all scales into a single token sequence and feed it into a standard ViT. This introduces hierarchical, multi-scale structural information purely through data organization, without modifying the model architecture.

### Input Construction

Given an image of size H x W and patch size P x P (e.g., 16x16):

| Level | Description | Number of Patches | Each Patch Covers |
|-------|------------|-------------------|-------------------|
| 0 | Whole image resized to P x P | 1 | 100% of image |
| 1 | 2x2 split, each resized to P x P | 4 | 25% of image |
| 2 | 4x4 split, each resized to P x P | 16 | 6.25% of image |
| ... | ... | ... | ... |
| L | 2^L x 2^L split, each resized to P x P | 4^L | original resolution patches |

Total token count = 1 + 4 + 16 + ... + 4^L = (4^(L+1) - 1) / 3

### Position Encoding

Keep PE design as simple as possible to avoid introducing confounding factors. Use a 3D continuous coordinate system based on each patch's absolute position in the original image:

- (cx, cy): normalized center coordinates of the patch in the original image, range [0, 1]
- s: normalized scale, the fraction of the image edge that the patch covers

Encode (cx, cy, s) using standard sinusoidal PE or learned PE, linearly extended from 2D to 3D. No special structural encoding (e.g., tree-aware or containment-aware PE) is introduced — the model learns spatial and hierarchical relationships from data.

## 2. Experiment Hypotheses

We compare **Hiera** (tree-input ViT) against **Baseline** (standard ViT) on image classification.

### Hypothesis A: Hiera > Baseline

Multi-scale input is effective. The model benefits from explicit hierarchical structure at the data level. This enables:
- Faster global context establishment (macro-level patches provide a global view from layer 1)
- Learned cross-scale feature associations via self-attention
- Improved robustness to scale variation in objects

Follow-up: investigate *how* the multi-scale input helps (attention patterns, convergence speed, per-class analysis).

### Hypothesis B: Hiera ≈ Baseline

Two possible explanations:
- **B1**: Standard ViT already captures sufficient hierarchical information through depth — explicit multi-scale input provides no additional benefit at this task/data scale.
- **B2**: The benefit of hierarchical input and the cost of information redundancy cancel out.

Follow-up: use controlled ablations (e.g., scale dropout, partial pyramid) to distinguish B1 from B2.

### Hypothesis C: Hiera < Baseline

The introduced redundancy or increased learning complexity harms model performance.

Possible causes:
- Attention dilution from redundant tokens
- Model over-relying on low-frequency coarse patches
- Insufficient training (more complex input may require longer training)

Follow-up: before concluding, verify both models are fully converged. Then investigate the specific failure mode.

## 3. Experiment Setup

### Task
Image classification on ImageNet-1K.

### Dataset
ImageNet-1K (1.28M training images, 50K validation images, 1000 classes).
Input resolution: 224x224.

### Models

Both models use [timm](https://github.com/huggingface/pytorch-image-models) standard ViT-Base as backbone.

| Model | Input | Token Count | Backbone | PE |
|-------|-------|-------------|----------|----|
| ViT-B (Baseline) | Standard 14x14 patches | 196 | timm `vit_base_patch16_224` | Standard 2D learned PE |
| HiT-B | Multi-scale image pyramid | 1+4+16+64+196 = 281 | timm `vit_base_patch16_224` (modified PE) | 3D (cx, cy, s) PE |

#### ViT-B Baseline
- Directly use timm's `vit_base_patch16_224` with default config
- ViT-Base: embed_dim=768, depth=12, num_heads=12, patch_size=16

#### HiT-B (Hierarchical Input Transformer - Base)
- Same backbone as ViT-B, only modifications:
  1. Input pipeline: construct image pyramid (levels 0-4), resize each sub-region to 16x16, project all patches through the same patch embedding layer
  2. Replace 2D learned PE with 3D PE based on (cx, cy, s) continuous coordinates
- Everything else (embed_dim, depth, heads, classifier head) stays identical to timm ViT-B

#### HiT-B Token Construction (224x224 input, patch_size=16)

| Level | Split | Patches | Each Patch Covers | Resize From |
|-------|-------|---------|-------------------|-------------|
| 0 | 1x1 | 1 | 224x224 (whole image) | 224x224 -> 16x16 |
| 1 | 2x2 | 4 | 112x112 | 112x112 -> 16x16 |
| 2 | 4x4 | 16 | 56x56 | 56x56 -> 16x16 |
| 3 | 8x8 | 64 | 28x28 | 28x28 -> 16x16 |
| 4 | 14x14 | 196 | 16x16 | 16x16 (no resize) |

Total: 281 tokens (vs. baseline 196 tokens, +43% overhead)

#### 3D Position Encoding for HiT-B
For each patch, compute its continuous coordinates in the original image:
- cx = (col_index + 0.5) / num_cols_at_level, normalized to [0, 1]
- cy = (row_index + 0.5) / num_rows_at_level, normalized to [0, 1]
- s = 1.0 / num_cols_at_level (fraction of image edge covered)

Encoding method: learned PE or sinusoidal PE over (cx, cy, s), projected to embed_dim=768. Start with learned PE to match baseline's PE type.

### Training Config
Follow timm's standard ViT-B ImageNet recipe for both models:
- Optimizer: AdamW
- Base learning rate: 1e-3 (with warmup + cosine decay)
- Weight decay: 0.3
- Batch size: 1024 (effective, with gradient accumulation if needed)
- Epochs: 300
- Augmentation: RandAugment, Mixup, CutMix, Random Erasing
- Label smoothing: 0.1
- Drop path rate: 0.1

### Control Experiment
- **HiT-B-Random**: same 281 tokens, but extra 85 tokens are duplicated random fine-level patches instead of multi-scale pyramid. Same 3D PE. Isolates whether gains come from multi-scale structure or additional tokens.

### Evaluation
- Top-1 / Top-5 accuracy on ImageNet validation set
- Training loss curves
- Throughput (images/sec) comparison
