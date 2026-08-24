# Pretrained ViT-R50 Hybrid Notes

## Local snapshot

The pretrained model was downloaded with `huggingface_hub.snapshot_download`, not `git clone`.
There is no `.git` directory in the model folder. The local folder is a direct Hugging Face snapshot:

`models/pretrained/timm/vit_base_r50_s16_224.orig_in21k`

Pinned source:

- repo: `timm/vit_base_r50_s16_224.orig_in21k`
- revision: `f8a432f48978c1a676f62d61c1f5a4f9eef52d08`
- weights: `model.safetensors`

The snapshot files are set read-only to reduce accidental edits.

## Model behavior

The model is a timm `VisionTransformer` with a hybrid ResNetV2 patch embed:

- input expected by pretraining config: `(B, 3, 224, 224)`
- first conv: `patch_embed.backbone.stem.conv`
- patch embedding module: `HybridEmbed`
- CNN backbone: `ResNetV2`
- projection: `Conv2d(1024, 768, kernel_size=1)`
- transformer blocks: `12`
- embedding dim: `768`
- cls token shape: `(1, 1, 768)`
- positional embedding shape: `(1, 197, 768)`
- parameter count: `97,890,112`

Verified local load:

- `model.load_state_dict(..., strict=False)` has `0` missing keys and `0` unexpected keys.
- `model.forward_features(torch.randn(1, 3, 224, 224))` returns `(1, 197, 768)`.
- `model(torch.randn(1, 3, 224, 224))` with `num_classes=0` returns `(1, 768)`.

## Training data usage

The frozen split is patient-level and volume-stratified:

`outputs/splits/frozen_patient_split_seed42.csv`

Training must use only rows with `split == "train"`:

- train patients: `339`
- validation patients: `72`
- test patients: `73`

Each patient has `155` axial slices:

- train all axial slices: `52,545`
- train tumour-positive slices: `23,597`
- train tumour-negative slices: `28,948`
- validation all axial slices: `11,160`
- test all axial slices: `11,315`

Validation and test patients must not be used for optimizer updates.

## Segmentation adaptation

The pretrained model is a feature backbone, not a segmentation model. For brain tumour segmentation:

1. Adapt MRI slice input from 4 channels to the pretrained 3-channel input path.
   Candidate strategies:
   - learned 1x1 projection from 4 to 3 channels,
   - modality selection,
   - PCA/linear projection to 3 channels.
2. Resize or crop slices from `240x240` to model-compatible `224x224`, or interpolate position embeddings for `240x240`.
3. Use `forward_features` tokens as the encoder output.
4. Remove image-level classification use.
5. Attach a dense decoder that predicts `(B, 4, H, W)` segmentation logits.
6. Train with pixel-wise segmentation losses, such as cross entropy plus Dice loss.

