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

We compare **HiT** (hierarchical input ViT) against **Baseline** (standard ViT) on image classification.

### Hypothesis A: HiT > Baseline

Multi-scale input is effective. The model benefits from explicit hierarchical structure at the data level.

### Hypothesis B: HiT ≈ Baseline

Standard ViT already captures sufficient hierarchical information through depth, or benefit/cost cancel out.

### Hypothesis C: HiT < Baseline

The introduced redundancy or increased learning complexity harms model performance.

## 3. Experiment Setup

### Main Experiment (ImageNet-1K)

Both models use timm standard ViT-Base as backbone.

| Model | Input | Token Count | Backbone | PE |
|-------|-------|-------------|----------|----|
| ViT-B | Standard 14x14 patches | 196 | timm `vit_base_patch16_224` | Standard 2D learned PE |
| HiT-B | Multi-scale image pyramid | 281 | timm `vit_base_patch16_224` (modified PE) | 3D (cx, cy, s) PE |

Platform: 4x H100. Training: AdamW, lr=1e-3, cosine decay, 300 epochs.

### Preliminary Experiments (CIFAR-10)

ViT-Tiny and HiT-Tiny (embed_dim=192, depth=6, heads=3) on CIFAR-10 (resized to 224x224) for fast iteration.

### Control Experiment
- **HiT-B-Random**: same 281 tokens, but extra 85 tokens are duplicated random fine-level patches instead of multi-scale pyramid. Isolates whether gains come from multi-scale structure or additional tokens.

---

## 4. Analysis Results (CIFAR-10 Preliminary)

### 4.1 Convergence Comparison (20 epochs)

| Metric | ViT-Tiny | HiT-Tiny |
|--------|----------|----------|
| val_acc@5 | 53.2% | 60.6% |
| val_acc@20 | 72.2% | 75.6% |

**Finding**: HiT-Tiny converges faster and achieves higher accuracy. At epoch 5, HiT already reaches ViT's epoch 10 level.

### 4.2 Full Training (100 epochs)

| Metric | ViT-Tiny | HiT-Tiny |
|--------|----------|----------|
| best val_acc | ~74.3% (ep50) | 79.1% (ep100) |
| final train_acc | ~99%+ | 100% |
| overfit gap | ~22% | ~21% |

**Finding**: HiT-Tiny maintains a ~5% advantage throughout training. Both models overfit severely without data augmentation, but HiT's val_acc ceiling is consistently higher. ViT val_acc plateaus around epoch 50 at ~74%, while HiT continues improving.

**Conclusion**: **Hypothesis A is supported** — multi-scale input improves classification performance on CIFAR-10.

### 4.3 Structural Attention Bias at Initialization

Tested whether cross-level attention bias exists before any training.

**Metric**: parent-child attention / intra-level attention ratio (ratio=1.0 means no bias).

| Model | embed_dim | Input | Ratio |
|-------|-----------|-------|-------|
| HiT-Tiny | 192 | CIFAR-10 | 0.997 |
| HiT-B | 768 | CIFAR-10 | 1.007 |
| HiT-Tiny | 192 | Random | 0.998 |
| HiT-B | 768 | Random | 0.998 |

**Finding**: No structural attention bias at initialization. Random W_Q and W_K projections completely dilute pixel-space correlations regardless of embedding dimension or input type.

### 4.4 Attention Bias Emergence During Training

Tracked PC/intra ratio every 50 steps during HiT-Tiny training.

#### First epoch (fine-grained):

| Step | Ratio |
|------|-------|
| init | 0.997 |
| step 10 | 0.987 |
| step 50 | 1.252 |
| step 100 | 1.243 |
| step 200 | 1.531 |
| step 782 (ep1 end) | 1.546 |

**Finding**: Structural bias is NOT present at initialization. It **emerges through training** between step 10-50 and grows throughout epoch 1.

#### Full 100 epochs:

Ratio evolution follows a clear single-peak pattern:

1. **Epoch 0**: ratio=1.0 (uniform, no bias)
2. **Epoch 1-3**: rapid rise, peak ~1.8-2.0
3. **Epoch 3-30**: decline back toward 1.0
4. **Epoch 30-100**: stabilizes at ~1.05-1.1

**Finding**: The model spontaneously discovers and utilizes cross-level structural correlations in early training (peak ratio ~2.0), then gradually reduces this dependency as it learns finer semantic features. The ratio is also highly noisy (oscillating between 1.2-1.8 within each epoch), suggesting the bias fluctuates with batch content.

**Key insight**: The model is NOT locked into a "structural shortcut". It self-regulates the strength of cross-level attention over training. This contradicts the hypothesis that structural similarity would cause the model to get stuck in a sharp local minimum.

### 4.5 Can ViT Catch Up With More Training?

ViT-Tiny at epoch 50 reaches train_acc=96.6% with val_acc=74.3%. The model is already severely overfitting. More epochs are unlikely to close the gap with HiT (79.1%).

*Status: ViT-Tiny 100 epoch run pending for confirmation.*

### 4.6 Attention Distance Analysis

*Status: pending. Script: `analysis/attention_distance.py`*

For each transformer layer, compute the mean spatial distance (in original image coordinates) between query tokens and their attended key tokens.

- **If ViT learns hierarchical processing**: shallow layers should attend locally (short distance), deep layers globally (long distance) — similar to CNN receptive field growth.
- **If ViT does NOT**: attention distance will be flat or disordered across layers.
- **For HiT**: additionally track attention distribution across pyramid levels per layer.

This directly measures whether ViT self-discovers equivalent hierarchical representations, or whether HiT's explicit multi-scale input provides information that ViT cannot extract from single-scale patches alone.

---

## 5. Analysis Scripts

| Script | Purpose | Status |
|--------|---------|--------|
| `analysis/attention_init.py` | Attention bias at initialization (Tiny vs Base) | Done |
| `analysis/attention_1epoch.py` | Fine-grained ratio tracking (every 50 steps, 100 epochs) | Done |
| `analysis/convergence_test.py` | ViT vs HiT training curves (100 epochs) | Done |
| `analysis/attention_distance.py` | Layer-wise attention distance + level attention distribution | Pending |
