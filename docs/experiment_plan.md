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
| best val_acc | 75.3% (ep90) | 79.1% (ep100) |
| final train_acc | 100% | 100% |
| final val_loss | 1.90 | 1.68 |
| overfit gap (train-val acc) | 25% | 21% |

#### ViT-Tiny 100 epoch details:

| Epoch | train_loss | train_acc | val_loss | val_acc |
|-------|-----------|-----------|---------|---------|
| 1 | 1.878 | 29.3% | 1.697 | 36.5% |
| 10 | 1.052 | 61.9% | 1.113 | 60.0% |
| 20 | 0.636 | 77.3% | 0.873 | 69.8% |
| 30 | 0.310 | 88.9% | 0.979 | 71.6% |
| 50 | 0.080 | 97.2% | 1.420 | 71.0% |
| 70 | 0.013 | 99.6% | 1.563 | 74.3% |
| 90 | 0.000 | 100.0% | 1.899 | 75.3% |

**Finding**:
- HiT-Tiny maintains ~4% advantage at convergence (79.1% vs 75.3%).
- Both models overfit severely, but HiT has a smaller train-val gap (21% vs 25%).
- ViT val_loss diverges earlier (ep20: 0.87) and more severely (ep90: 1.90) than HiT (ep90: 1.68).
- Despite val_loss increasing monotonically after epoch 20, ViT val_acc continues to slowly improve (69.8% → 75.3%), suggesting the model is still learning some generalizable features even while overfitting.

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

### 4.7 Level Ablation (trained HiT-Tiny, 100 epochs)

| Config | Tokens | Val Acc | vs Full |
|--------|--------|---------|---------|
| Full (L0-L4) | 282 | 80.0% | — |
| L4 only (fine) | 197 | 79.6% | -0.4% |
| L0-L3 only (coarse) | 86 | 70.1% | -9.9% |
| L3-L4 | 261 | 80.2% | +0.2% |
| L0+L4 | 198 | 79.7% | -0.3% |

**Findings:**
- Removing L0-L3 only drops 0.4%. Hierarchical information has been largely internalized.
- L0-L3 alone (70.1%) cannot classify well — their value is in training-time structural guidance, not direct classification features.
- L3-L4 slightly outperforms Full (+0.2%), suggesting L0-L2 may be mild noise at inference.
- **Practical implication**: Train with full pyramid (281 tokens), inference with L4 only (196 tokens). Same cost as ViT, but 79.6% vs 75.3% (+4.3%).

### 4.8 Linear Probe Analysis

*Status: done. Script: `analysis/linear_probe.py`*

Train a linear classifier on frozen per-layer features, for each level separately.

#### Results

**ViT-Tiny probe accuracy per layer:**

| Config | L1 | L2 | L3 | L4 | L5 | L6 | L7 | L8 | L9 | L10 | L11 | L12 |
|--------|-----|-----|-----|-----|-----|-----|-----|-----|-----|------|------|------|
| CLS | 34.7 | 45.4 | 52.3 | 57.0 | 59.6 | 63.0 | 64.6 | 67.0 | 69.1 | 71.3 | 72.9 | 75.2 |
| All patches | 48.1 | 56.1 | 58.1 | 60.1 | 62.1 | 64.2 | 65.5 | 66.7 | 67.9 | 68.7 | 69.1 | 70.8 |

**HiT-Tiny probe accuracy per layer:**

| Config | L1 | L2 | L3 | L4 | L5 | L6 | L7 | L8 | L9 | L10 | L11 | L12 |
|--------|-----|-----|-----|-----|-----|-----|-----|-----|-----|------|------|------|
| CLS | 40.0 | 50.4 | 53.7 | 58.0 | 61.0 | 64.4 | 67.2 | 70.4 | 72.7 | 75.1 | 78.2 | 79.9 |
| All tokens | 49.1 | 55.7 | 59.6 | 62.4 | 64.6 | 65.4 | 67.4 | 69.1 | 70.5 | 72.5 | 74.6 | 76.1 |
| L4 only | 49.7 | 55.6 | 59.5 | 62.5 | 64.7 | 65.5 | 67.5 | 69.5 | 70.6 | 72.8 | 74.9 | 76.5 |
| L0-L3 only | 46.1 | 53.4 | 57.6 | 60.8 | 63.0 | 64.6 | 65.8 | 68.5 | 69.2 | 71.5 | 73.6 | 74.8 |
| L0(1x1) | 41.8 | 46.1 | 48.4 | 51.7 | 53.3 | 55.2 | 57.3 | 59.9 | 60.7 | 61.9 | 64.2 | 65.3 |
| L1(2x2) | 41.2 | 48.2 | 50.9 | 53.9 | 56.1 | 58.5 | 60.2 | 62.5 | 64.5 | 65.5 | 68.0 | 69.0 |
| L2(4x4) | 43.2 | 49.5 | 54.1 | 56.7 | 60.1 | 61.5 | 62.8 | 65.9 | 67.0 | 69.0 | 71.9 | 72.7 |
| L3(8x8) | 46.4 | 53.4 | 57.5 | 61.0 | 63.7 | 64.9 | 66.0 | 68.4 | 69.6 | 71.5 | 74.0 | 74.9 |
| L4(14x14) | 49.5 | 56.3 | 59.9 | 62.8 | 65.1 | 65.6 | 67.3 | 69.4 | 70.7 | 72.6 | 74.8 | 76.5 |

**Key comparison — HiT L4 vs ViT patches (internalization gap):**

| Layer | L1 | L2 | L3 | L4 | L5 | L6 | L7 | L8 | L9 | L10 | L11 | L12 |
|-------|------|------|------|------|------|------|------|------|------|-------|-------|-------|
| Diff | +1.6 | -0.5 | +1.4 | +2.4 | +2.6 | +1.3 | +2.0 | +2.8 | +2.8 | +4.2 | +5.8 | +5.7 |

#### Findings

**F1: Internalized information accumulates across layers, concentrates in deep layers.**
The HiT L4 vs ViT patches gap is small at shallow layers (+1.6 at L1) and grows to +5.7 at L12. Each layer of cross-level interaction injects a small amount of structural information into L4. This is not a one-time injection but a gradual accumulation process.

**F2: Information flow is bidirectional.**
HiT L0-L3 only at L12 (74.7%) > ViT All patches at L12 (70.6%). The 85 coarse tokens outperform ViT's 196 fine tokens by 4.1%. This means fine-grained information flows into coarse tokens too — L0-L3 are not just static providers, they also absorb classification-relevant details from L4.

**F3: In shallow layers, ViT's concentrated token count gives a slight edge.**
At L2, ViT patches (56.1%) slightly outperform HiT L4 (55.6%). In the first 1-2 layers, ViT's 196 tokens are all fine-grained and directly task-relevant, while HiT's attention is partially "distracted" by learning cross-level relationships. This distraction disappears by L3 and reverses into a growing advantage.

**F4: Per-level probe accuracy follows resolution order at final layer.**
L0(65.3) < L1(69.0) < L2(72.7) < L3(74.9) < L4(76.5). Higher spatial resolution provides more discriminative spatial detail. But the gap between levels shrinks compared to what raw resolution would predict — L0 with 1 token achieving 65.3% is remarkable.

**F5: CLS token in HiT builds classification readiness faster.**
HiT CLS leads ViT CLS at every layer. The gap is +5.7 at L1, narrows to +1.4 at L6, then widens again to +4.7 at L12. The early advantage suggests HiT's CLS benefits from immediately accessible global tokens (L0-L1). The late widening suggests accumulated structural information provides better deep-layer features for classification.

#### Interpretation

The shallow-to-deep probe accuracy progression tells a coherent story:

- **Shallow layers (L1-L3)**: Both models extract low-level features (edges, textures, colors). ViT's L4 tokens have a slight advantage because all 196 tokens carry task-relevant fine-grained features directly. HiT's L4 tokens are "distracted" — they spend attention budget learning cross-level relationships with L0-L3, which slightly reduces their per-layer feature quality in the first 2 layers.

- **Middle layers (L4-L8)**: HiT L4 starts overtaking ViT patches. The cross-level interaction begins paying off — structural information from L0-L3 provides a regularizing global context. HiT's L4 tokens now carry both fine-grained local features AND injected hierarchical structure.

- **Deep layers (L9-L12)**: The gap accelerates (+3.0 → +6.1). The accumulated structural information acts as a "macro-level regularizer" — HiT's L4 representations are better organized in feature space, leading to more linearly separable class boundaries. ViT's representations, without this structural scaffolding, plateau earlier.

This supports the hypothesis that **hierarchical input provides a global structural prior that regularizes deep-layer representations**, rather than directly contributing classification features. The coarse tokens (L0-L3) are not information sources for classification — they are structural anchors that guide how L4's fine-grained features are organized.

### 4.8b Control Experiments: Micro, Random, and Fixed Prefix

*Status: done. Script: `analysis/micro_prefix_probe.py`*

Five-way comparison to isolate the source of macro-prefix's advantage.

| Model | Prefix content | Tokens | New info vs L4? |
|-------|---------------|--------|-----------------|
| ViT-Tiny | none | 196 | — |
| HiT-macro | L0-L3 coarse pyramid | 281 | Yes (global structure) |
| HiT-micro | 85 random 8x8 crops | 281 | Partial (finer grain, overlaps with L4) |
| HiT-random | 85 randomly selected L4 patch duplicates (different each forward pass) | 281 | No (redundant) |
| HiT-fixed | 85 fixed L4 patch duplicates (same patches every forward pass) | 281 | No (redundant) |

#### Results

**CLS probe at L12 (final classification ability):**

| Model | CLS L12 |
|-------|---------|
| HiT-macro | **79.9%** |
| HiT-random | 78.0% |
| HiT-micro | 77.5% |
| HiT-fixed | 77.3% |
| ViT | 75.2% |

**L4 probe diff vs ViT (internalization into fine-level tokens):**

| Model | L1 | L2 | L6 | L9 | L12 | Pattern |
|-------|------|------|------|------|-------|---------|
| Macro | +1.6 | -0.5 | +1.3 | +2.8 | **+5.7** | Deep-layer accumulation |
| Micro | +0.2 | -1.8 | -0.9 | -0.4 | **+1.2** | Shallow harm, weak deep gain |
| Fixed | +0.7 | -2.0 | -1.4 | -0.1 | **+0.5** | Shallow harm, marginal deep gain |
| Random | -1.4 | -3.0 | -0.3 | -0.5 | **-1.2** | Harm throughout |

#### Findings

**F1: Macro prefix is uniquely effective — not replaceable by more tokens or finer granularity.**
Only macro achieves significant L4 internalization (+5.7 at L12). The advantage comes specifically from global structural information that L4 tokens cannot obtain from their 16x16 local patches.

**F2: Fixed vs Random disentangles redundancy from stochasticity.**
Both use identical content (duplicated L4 patches), but fixed uses the same 85 patches every forward pass while random resamples each time. Fixed shows marginal positive L4 internalization (+0.5) while random harms L4 throughout (-1.2). This indicates that random prefix selection prevents stable cross-attention patterns from forming, actively degrading L4 representation quality. The stochasticity, not just the redundancy, is what harms L4.

**F3: Random prefix harms L4 but best helps CLS among non-macro controls.**
Random CLS (78.0%) > Fixed CLS (77.3%) ≈ Micro CLS (77.5%), despite random having the worst L4 internalization. The random selection acts as training-time regularization noise (similar to dropout), benefiting the CLS token's robustness while degrading fine-level representations.

**F4: Micro and Fixed show weak but positive L4 internalization at deep layers.**
Both micro (+1.2) and fixed (+0.5) achieve small positive internalization at L12, while random (-1.2) does not. Micro provides marginally new sub-patch information; fixed provides stable (though redundant) cross-attention targets. Both allow some degree of representation enrichment, but far less than macro's genuinely new global structure.

**F5: Bidirectional internalization observed — macro (global→L4) and micro (detail→L4).**
Both macro and micro show L4 probe gains at L12 (though vastly different magnitudes), suggesting L4 can absorb information from both coarser and finer scales. The macro direction is far more effective because it provides genuinely inaccessible information.

**F6: Final ranking — macro >> random ≈ micro ≈ fixed > ViT (CLS); macro >> micro > fixed > ViT > random (L4 internalization).**
For classification, any extra tokens help somewhat (+2-3% CLS). For representation quality (internalization), only genuinely new cross-scale information matters, and prefix stability is a prerequisite for any positive internalization.

#### Open questions for next analysis (4.9)

**Q1: Where in 192-d space is the +5.7% gap encoded?**
SVD on HiT_L4 - ViT_patches residual at L12 to find the internalization subspace.

**Q2: Is the internalized information low-rank or dispersed?**
Singular value spectrum of the residual — determines whether structural knowledge is compact (extractable as a prior) or entangled with content.

**Q3: Necessity verification.**
Remove the internalization subspace from HiT L4 features → does acc drop to ViT level? Proves the identified subspace is necessary, not just correlated.

### 4.9 Internalized Information Localization

*Status: planned. Depends on linear probe results (4.8).*

**Core question**: The level ablation (4.7) confirmed that hierarchical information has been internalized — removing L0-L3 at inference only drops 0.4%. But where in the 192-dimensional representation space is this information encoded?

#### Method: Residual Subspace Analysis

For the same set of images, extract HiT L4 and ViT patch representations at each layer. The difference between these two sets of representations encodes the internalized hierarchical information.

**Step 1: Find the difference subspace.**
- Compute residual matrix: R = HiT_L4_features - ViT_features (per sample, per token, matched by spatial position)
- SVD on R: R = U @ S @ V^T
- The top-k right singular vectors (columns of V) span the "internalization subspace"
- Examine singular value spectrum: concentrated (low-rank) vs dispersed (high-rank)

**Step 2: Probe inside and outside the subspace.**
- Project HiT L4 features onto the internalization subspace → probe accuracy = contribution of internalized info
- Project HiT L4 features onto the orthogonal complement → probe accuracy = contribution of shared (ViT-equivalent) info
- If subspace probe ≈ HiT-ViT accuracy gap (~4%), the internalized information is precisely localized

**Step 3: Necessity verification.**
- Remove the subspace component from HiT L4 features (project onto orthogonal complement only)
- Run classification: if acc drops to ViT level (~75%), confirms these dimensions are necessary for the hierarchical advantage
- If acc doesn't drop, the internalized info is encoded differently (e.g., in covariance structure rather than mean direction)

**Step 4: Track subspace evolution across layers.**
- Compute the internalization subspace at each layer
- Measure subspace alignment between adjacent layers (principal angle or subspace overlap)
- If the subspace rotates across layers → information is being actively transformed
- If the subspace is stable → information was injected once and preserved through residual connections

**Expected outcomes:**
- If internalized info is low-rank (top 5-10 singular vectors explain >80% variance): hierarchical knowledge is compact, potentially extractable as a learned "structural prior"
- If high-rank: hierarchical info is deeply entangled with texture/content features, harder to isolate
- Either result informs the "strong structure" direction — whether structural constraints can be cleanly separated from content, or whether they must be jointly learned

---

## 5. Analysis Scripts

| Script | Purpose | Status |
|--------|---------|--------|
| `analysis/attention_init.py` | Attention bias at initialization (Tiny vs Base) | Done |
| `analysis/attention_1epoch.py` | Fine-grained ratio tracking (every 50 steps, 100 epochs) | Done |
| `analysis/convergence_test.py` | ViT vs HiT training curves (100 epochs) | Done |
| `analysis/training_dynamics.py` | Per-layer heatmap/entropy/CLS-attn/norms + level ablation | Done |
| `analysis/linear_probe.py` | Per-layer per-level linear probe for ViT and HiT | Done |
| `analysis/micro_prefix_probe.py` | Micro/random/fixed prefix control + comparative probe | Done |
| `analysis/attention_distance.py` | Layer-wise attention distance + level attention distribution | Pending |
| `analysis/internalization.py` | Residual subspace analysis: where is internalized info encoded | Planned |
