# 4-Channel MRI Input and Resolution Feasibility

## Decision

Adopt the native 4-channel MRI route as the primary plan:

- Do not compress the four MRI modalities into three pseudo-RGB channels.
- Keep all four modalities as independent input channels.
- Adapt the pretrained input stem from 3 input channels to 4 input channels.
- Freeze most pretrained layers initially, and only fine-tune the changed input path plus later pretrained layers and the segmentation decoder.

There are now two viable spatial-resolution routes:

1. Conservative baseline: resize `240x240` slices to `224x224`, preserve the pretrained `14x14 + CLS` Transformer geometry exactly, and map predictions back to `240x240` before volume computation.
2. Native-resolution route: keep `240x240`, interpolate the pretrained positional embedding from `14x14` to `15x15`, and decode directly back to `240x240`.

The updated preference is:

- use `224` if the immediate goal is the lowest-risk pretrained-model MVP;
- use `240 + positional interpolation` if the goal is the cleaner formal segmentation and volumetry pipeline.

Because this project ultimately computes tumour volume in the original voxel grid, the `240` native-resolution route is conceptually stronger once positional-embedding interpolation is implemented.

The key architectural conclusion remains:

`3x224x224 -> 4x240x240` cannot be handled cleanly by the first convolution alone while keeping the original downstream Transformer token format unchanged.

Channel adaptation can be done in the first convolution. Spatial resizing should be done before the model, or by an explicit pre-stem resize module. It should not be hidden inside the pretrained first convolution.

## Inspected Model

Local checkpoint:

`models/pretrained/timm/vit_base_r50_s16_224.orig_in21k/model.safetensors`

Model:

`timm/vit_base_r50_s16_224.orig_in21k`

This is a timm `VisionTransformer` using a hybrid `ResNetV2` patch embedding path.

Important modules:

- model class: `VisionTransformer`
- patch embedding: `HybridEmbed`
- CNN backbone: `patch_embed.backbone`, a `ResNetV2`
- first input conv: `patch_embed.backbone.stem.conv`
- projection into ViT embedding space: `patch_embed.proj`
- transformer blocks: `12`
- embedding dimension: `768`
- cls token shape: `(1, 1, 768)`
- positional embedding shape: `(1, 197, 768)`

The local safetensors checkpoint loads into the original 3-channel architecture with:

- missing keys: `0`
- unexpected keys: `0`

## First Convolution Structure

The original first convolution is:

```text
StdConv2dSame(3, 64, kernel_size=(7, 7), stride=(2, 2), bias=False)
```

Weight shape:

```text
(64, 3, 7, 7)
```

So the pretrained model expects `3` input channels at the first layer and produces `64` stem channels.

If the model is instantiated with `in_chans=4`, timm changes the first convolution to:

```text
StdConv2dSame(4, 64, kernel_size=(7, 7), stride=(2, 2), bias=False)
```

Weight shape:

```text
(64, 4, 7, 7)
```

This confirms that the channel-side adaptation is local: only the first conv weight changes from `(64, 3, 7, 7)` to `(64, 4, 7, 7)`. The output channel count remains `64`, so the rest of the ResNet-ViT path can keep the same tensor interface.

Recommended initialization for the 4-channel first conv:

```python
w3 = pretrained_first_conv.weight              # (64, 3, 7, 7)
w_mean = w3.mean(dim=1, keepdim=True)          # (64, 1, 7, 7)
w4 = w_mean.repeat(1, 4, 1, 1) * (3.0 / 4.0)   # (64, 4, 7, 7)
```

The `3/4` scale keeps the approximate activation magnitude closer to the original 3-channel model when four modalities are summed by the convolution.

## Shape Audit: 224 Input

For input:

```text
(B, 3, 224, 224)
```

Observed shapes:

```text
stem.conv      -> (B, 64, 112, 112)
stem.pool      -> (B, 64, 56, 56)
ResNet stage0  -> (B, 256, 56, 56)
ResNet stage1  -> (B, 512, 28, 28)
ResNet stage2  -> (B, 1024, 14, 14)
Hybrid proj    -> (B, 768, 14, 14)
patch_embed    -> (B, 196, 768)
with CLS token  -> (B, 197, 768)
```

This matches the pretrained positional embedding:

```text
pos_embed -> (1, 197, 768)
```

Therefore the pretrained Transformer receives exactly the expected token format:

```text
196 image tokens + 1 CLS token
```

For segmentation, we can discard the CLS token and reshape:

```text
(B, 196, 768) -> (B, 768, 14, 14)
```

Then a light decoder can upsample:

```text
14 -> 28 -> 56 -> 112 -> 224
```

## Shape Audit: 240 Input

For input:

```text
(B, 3, 240, 240)
```

The CNN path still runs, but produces a larger token grid:

```text
stem.conv      -> (B, 64, 120, 120)
stem.pool      -> (B, 64, 60, 60)
ResNet stage0  -> (B, 256, 60, 60)
ResNet stage1  -> (B, 512, 30, 30)
ResNet stage2  -> (B, 1024, 15, 15)
Hybrid proj    -> (B, 768, 15, 15)
patch_embed    -> (B, 225, 768)
with CLS token  -> (B, 226, 768)
```

This does not match the pretrained positional embedding:

```text
tokens    -> (B, 226, 768)
pos_embed -> (1, 197, 768)
```

The actual failure is:

```text
RuntimeError: The size of tensor a (226) must match the size of tensor b (197) at non-singleton dimension 1
```

So raw `240x240` input is not format-compatible with the pretrained `224x224` positional embedding as-is. It becomes compatible if we interpolate the image-token positional embedding from a `14x14` grid to a `15x15` grid and use the resulting `(1, 226, 768)` positional embedding.

## Can the First Conv Also Do 240-to-224?

No, not in the sense we need.

The first conv can solve the channel mismatch:

```text
3 channels -> 4 channels
```

while preserving the same downstream feature interface:

```text
stem output channels remain 64
```

But the first conv cannot cleanly solve the spatial mismatch:

```text
240x240 -> 224x224
```

because it is a local convolution, not a global resize operation. With the existing pretrained geometry, the model effectively downsamples spatial size by a factor of 16 before the Transformer:

```text
224 / 16 = 14
240 / 16 = 15
```

That means:

```text
224x224 -> 14x14 image tokens -> 196 tokens
240x240 -> 15x15 image tokens -> 225 tokens
```

The Transformer checkpoint is tied to the `14x14 + CLS` token format through its positional embedding. Forcing the first conv stride, padding, or kernel to make `240` eventually become `14` would alter the pretrained ResNet geometry and receptive fields. That would discard much of the value of the pretrained stem and make debugging harder.

Therefore:

- Channel adaptation belongs in the first conv.
- Spatial resizing belongs before the first conv, or in an explicit pre-stem resize adapter.
- The first conv should not be asked to do both jobs.

## Route A: 224 Baseline

For the first baseline, resize each 2D axial slice from `240x240` to `224x224` during dataset loading.

For image channels:

```python
image = torch.nn.functional.interpolate(
    image[None],
    size=(224, 224),
    mode="bilinear",
    align_corners=False,
)[0]
```

Expected image tensor:

```text
(4, 224, 224)
```

For the segmentation label:

```python
label = torch.nn.functional.interpolate(
    label[None, None].float(),
    size=(224, 224),
    mode="nearest",
)[0, 0].long()
```

Nearest-neighbor resizing is required for labels so class ids are not interpolated into invalid fractional values.

MRI intensity normalization should remain medical-image specific:

- normalize per modality,
- use nonzero brain voxels when estimating statistics,
- do not use ImageNet RGB mean/std.

For the decoder:

```text
14 -> 28 -> 56 -> 112 -> 224
```

The output prediction is naturally:

```text
(B, 1, 224, 224)
```

Before patient-level volumetry, the predicted probability map or mask must be mapped back to the original `240x240` grid.

Preferred evaluation path:

```text
P_224 -> bilinear resize -> P_240 -> threshold -> mask_240
```

For an already-thresholded binary mask:

```text
mask_224 -> nearest resize -> mask_240
```

Then stack the `155` original axial slices and compute volume in the original physical voxel space:

```text
V_pred = number_of_predicted_tumour_voxels_in_240x240x155 * 1 mm^3
```

This separation is important:

- model space: `224x224`;
- physical/medical evaluation space: original `240x240x155`.

## Route B: 240 Native Resolution

Keeping `240x240` is technically possible, but it changes the Transformer input format:

```text
197 tokens -> 226 tokens
```

To do this properly, we would need:

- instantiate or adapt the model for a `15x15` token grid,
- interpolate positional embeddings from `14x14` to `15x15`,
- adjust the decoder to reshape `(B, 225, 768)` into `(B, 768, 15, 15)`,
- upsample from `15x15` to `240x240`.

The pretrained positional embedding can be adapted as:

```text
pos_embed                 -> (1, 197, 768)
CLS positional token       -> (1, 1, 768)
image positional tokens    -> (1, 196, 768)
reshape image positions    -> (1, 14, 14, 768)
2D interpolate             -> (1, 15, 15, 768)
flatten                    -> (1, 225, 768)
concat CLS                 -> (1, 226, 768)
```

This does not require changing the structure of most pretrained weights:

- ResNetV2 stem after the first conv: unchanged;
- ResNet stages: unchanged;
- `HybridEmbed.proj`: unchanged;
- ViT attention and MLP weights: unchanged;
- LayerNorm weights: unchanged;
- number of Transformer blocks: unchanged.

Transformer blocks naturally accept variable token length:

```text
X -> (B, N, 768)
```

The attention matrix changes size, but the learned projections do not:

```text
N = 197 for 224 input
N = 226 for 240 input
```

For segmentation and volume tracking, Route B has a strong geometry advantage:

```text
input grid = label grid = prediction grid = physical voxel grid
```

The decoder is also clean because:

```text
240 / 16 = 15
15 -> 30 -> 60 -> 120 -> 240
```

So prediction can be directly:

```text
(B, 1, 240, 240)
```

No post-prediction resize is needed before stacking slices and computing patient-level volume.

## Fine-Tuning Scope

The planned fine-tuning policy is compatible with native 4-channel input.

Initial trainable modules should be:

- changed 4-channel input conv, and likely its immediately adjacent stem normalization if needed,
- segmentation decoder,
- late pretrained layers, for example ResNet stage2 and ViT blocks 9-12.

Initially frozen modules should be:

- early and middle ResNet layers,
- early ViT blocks, for example blocks 0-8,
- most pretrained parameters not needed for first-stage domain adaptation.

This keeps the comparison controlled:

```text
same frozen patient split
same 4 MRI modalities
same patient-level volume targets
changed input stem + selected later layers + decoder
```

## Final Recommendation

Use native 4-channel input for all routes.

For the lowest-risk MVP, Route A is acceptable:

```text
raw MRI slice:        (4, 240, 240)
dataset resize:       (4, 224, 224)
4-channel stem conv:  (4 -> 64)
ResNetV2 hybrid path: 14x14 feature grid
HybridEmbed proj:     1024 -> 768
ViT input:            196 image tokens + CLS = 197 tokens
discard CLS:          (B, 196, 768)
reshape:              (B, 768, 14, 14)
decoder output:       (B, 1, 224, 224)
evaluation resize:    (B, 1, 240, 240)
```

This preserves the pretrained Transformer interface exactly while retaining all four MRI modalities. It must map predictions back to `240x240` before volumetry.

For the formal segmentation and volume-tracking model, prefer Route B if implementation time allows:

```text
raw MRI slice:        (4, 240, 240)
no image resize:      (4, 240, 240)
4-channel stem conv:  (4 -> 64)
ResNetV2 hybrid path: 15x15 feature grid
HybridEmbed proj:     1024 -> 768
pos_embed:            interpolate 14x14 -> 15x15
ViT input:            225 image tokens + CLS = 226 tokens
discard CLS:          (B, 225, 768)
reshape:              (B, 768, 15, 15)
decoder output:       (B, 1, 240, 240)
volumetry grid:       original 240x240x155 voxel grid
```

This route changes slightly more model geometry, but it keeps the entire segmentation and volumetry pipeline on the original image grid.

## Positional Interpolation Mathematics

The pretrained absolute positional embedding is:

```text
E_pos in R^{1 x 197 x 768}
```

It consists of one prefix/CLS positional vector and 196 image positional vectors:

```text
E_cls = E_pos[:, :1, :]     in R^{1 x 1 x 768}
E_img = E_pos[:, 1:, :]     in R^{1 x 196 x 768}
```

Since:

```text
196 = 14 x 14
```

the image positional vectors can be reshaped into a 2D spatial grid:

```text
E_img -> R^{1 x 14 x 14 x 768}
```

For PyTorch interpolation, move the channel dimension forward:

```text
R^{1 x 14 x 14 x 768} -> R^{1 x 768 x 14 x 14}
```

Then apply bicubic 2D interpolation:

```text
Interp(E_img, size=(15, 15)) -> R^{1 x 768 x 15 x 15}
```

Move channels back and flatten:

```text
R^{1 x 768 x 15 x 15}
-> R^{1 x 15 x 15 x 768}
-> R^{1 x 225 x 768}
```

Finally concatenate the unchanged CLS positional vector:

```text
E_pos' = concat(E_cls, E_img') in R^{1 x 226 x 768}
```

Only the positional embedding tensor changes shape:

```text
(1, 197, 768) -> (1, 226, 768)
```

The Transformer block weights do not change shape. Self-attention accepts:

```text
X in R^{B x N x 768}
```

so changing `N` from `197` to `226` changes the runtime attention matrix size, but not the learned projection matrices:

```text
W_Q, W_K, W_V, W_O remain 768 x 768
MLP weights remain unchanged
LayerNorm weights remain unchanged
```

Therefore Route B does not cause a large structural disturbance to the pretrained model. It only adapts:

- first conv weight: `(64, 3, 7, 7) -> (64, 4, 7, 7)`;
- positional embedding: `(1, 197, 768) -> (1, 226, 768)`;
- decoder grid: `14x14 -> 15x15`.

All ResNet stages, HybridEmbed projection, and Transformer block parameter shapes remain compatible.

## Implemented Route B Interface

Implemented file:

`src/pretrained_hybrid_vit.py`

Main class:

```python
PretrainedHybridViTSegmenter
```

Main helper functions:

```python
interpolate_pos_embed_2d
adapt_first_conv_weight
adapt_hybrid_vit_state_dict
count_parameters
```

Confirmed smoke-test interface:

```text
input:        (B, 4, 240, 240)
feature grid: (B, 768, 15, 15)
logits:       (B, 1, 240, 240)
```

Confirmed adapted pretrained tensors:

```text
first conv: (64, 4, 7, 7)
pos_embed:  (1, 226, 768)
```

The checkpoint adaptation loads with no missing or unexpected encoder keys after converting the first conv and positional embedding.

## Code-Level Audit

Audit script:

`scripts/03_route_b_smoke_audit.py`

Output:

`outputs/model_audit/route_b_smoke_audit.json`

The audit verifies:

- first conv in the loaded model exactly equals the RGB-mean, `3/4`-scaled adapted checkpoint tensor;
- positional embedding in the loaded model exactly equals bicubic `14x14 -> 15x15` interpolation of the checkpoint image-position tokens;
- CLS positional token is unchanged;
- positional embedding is not mutated or repeatedly interpolated during forward passes;
- input `(1, 4, 240, 240)` gives feature grid `(1, 768, 15, 15)`;
- model output is raw logits with shape `(1, 1, 240, 240)`;
- forward output is finite for zero and random inputs;
- `eval()` forward is deterministic for repeated identical input;
- decoder contains no `Sigmoid`, so it is compatible with `BCEWithLogitsLoss`;
- there are no BatchNorm modules with running statistics in this Route B model;
- optimizer coverage check sees each trainable parameter object exactly once.

Latest audit result:

```text
first_conv_matches_expected_rgb_mean_scaled_3_over_4: 0.0
pos_embed_matches_expected_bicubic_14_to_15:          0.0
cls_pos_embed_unchanged:                              0.0
pos_embed_changed_during_two_forwards:                false
feature_grid_shape_ok:                                true
logits_shape_ok:                                      true
zero_forward_finite:                                  true
random_forward_finite:                                true
eval_forward_deterministic:                           true
decoder_has_sigmoid:                                  false
has_batchnorm_modules:                                false
optimizer covers all trainable params once:           true
```

Runtime shape chain from hooks:

```text
stem.conv      -> (1, 64, 120, 120)
stem.pool      -> (1, 64, 60, 60)
resnet.stage2  -> (1, 1024, 15, 15)
hybrid.proj    -> (1, 768, 15, 15)
vit.block8     -> (1, 226, 768)
vit.block11    -> (1, 226, 768)
feature grid   -> (1, 768, 15, 15)
logits         -> (1, 1, 240, 240)
```

Current default trainable scope:

```text
trainable params: 42,515,793
frozen params:    58,926,336
total params:     101,442,129
```

Trainable modules are:

- `encoder.pos_embed`;
- `encoder.patch_embed.backbone.stem`;
- `encoder.patch_embed.backbone.stages.2`;
- last 4 ViT blocks;
- final encoder norm;
- decoder.

This is a moderate fine-tuning setup, not a minimal input-conv-only setup.

One ad hoc normalized-MRI activation check was also run on train case `BRATS_001`, tumour-positive slice `66`, using per-modality nonzero-brain z-score normalization. This is not a substitute for the final dataset-normalization audit, but it checks the immediate numerical path:

```text
input          -> mean -0.0000, std 0.5589, finite true
stem.conv      -> mean  0.0014, std 11.2769, finite true
resnet.stage2  -> mean  0.0343, std  0.0854, finite true
hybrid.proj    -> mean -0.0002, std  0.1285, finite true
vit.block11    -> mean  0.0030, std  0.5736, finite true
logits         -> mean -0.2524, std  0.3152, finite true
```

The first conv has large raw extrema, but the downstream normalized ResNetV2 path returns to stable finite activations. During real training, activation stats, loss, logits, and gradient norms should still be logged for early NaN/Inf detection.
