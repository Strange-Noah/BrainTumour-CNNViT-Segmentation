---
title: Brain Tumour Binary Segmentation MVP
emoji: 🧠
colorFrom: blue
colorTo: indigo
sdk: gradio
app_file: app.py
python_version: 3.10
---

# Brain Tumour Binary Segmentation MVP

Minimal Hugging Face Space for the first deployment milestone.

## Scope

This MVP only does the core task:

1. Accept one `.nii` or `.nii.gz` file.
2. Reject the upload unless the loaded NIfTI shape is exactly `(240, 240, 155, 4)`.
3. Run the frozen Route B binary CNN-ViT segmentation model on GPU.
4. Report predicted tumour volume and tumour voxel fraction of the input grid.
5. Show the largest predicted-positive slice as a 2D overlay.
6. Export `prediction.nii.gz`.

No resizing, resampling, registration, 4-label segmentation, uncertainty, 3D rendering, reference-label evaluation, or arbitrary clinical scan compatibility is included in this first step.

## Checkpoint

Provide the binary checkpoint in either of these ways:

- place it at `checkpoints/binary_route_b_best_checkpoint.pt`; or
- set the environment variable `BINARY_CHECKPOINT_PATH`.

The checkpoint must contain:

```python
checkpoint["model_state_dict"]
```

from the frozen binary Route B model.

## Expected Input

The uploaded file must load via nibabel with:

```text
shape == (240, 240, 155, 4)
```

The app does not infer modality order. The user must upload a NIfTI whose channel order matches the training pipeline.
