# Data and Weight Setup

This repository excludes raw imaging data and large model weights.

## Dataset

Use MSD Task01 BrainTumour. Prepare files as:

```text
data/processed/Task01_BrainTumour/
├── imagesTr/
│   ├── BRATS_001.nii.gz
│   └── ...
└── labelsTr/
    ├── BRATS_001.nii.gz
    └── ...
```

Kaggle may expose decompressed `.nii` files instead of `.nii.gz`; the scripts support both.

## Pretrained Initialization

Download the pretrained hybrid R50-ViT checkpoint from:

```text
https://huggingface.co/timm/vit_base_r50_s16_224.orig_in21k
```

Expected local path:

```text
models/pretrained/timm/vit_base_r50_s16_224.orig_in21k/model.safetensors
```

## Fine-tuned Checkpoints

Large fine-tuned checkpoints should be stored outside GitHub.

Suggested paths:

```text
checkpoints/binary_route_b_best_checkpoint.pt
checkpoints/route_b_4label_best_checkpoint.pt
```

The deployment app defaults to these names. The 4-label checkpoint can also be loaded from:

```text
TLOMMP/route-b-4label-full-train/best_checkpoint.pt
```

## Frozen Split

The frozen patient split is included:

```text
outputs/splits/frozen_patient_split_seed42.csv
```

All training, validation, and test scripts should use this manifest.
