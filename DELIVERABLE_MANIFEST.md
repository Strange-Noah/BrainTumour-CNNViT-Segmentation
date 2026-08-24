# Deliverable Manifest

This directory is intended as the lightweight GitHub-ready project deliverable.

## Included

- Source code:
  - `src/`
  - `scripts/`
  - `configs/`
  - `deployment/binary_seg_mvp/`
- Kaggle launcher notebooks:
  - `kaggle/vit-train.ipynb`
  - `kaggle/vit-infer.ipynb`
  - `kaggle/vit-train-4label.ipynb`
  - `kaggle/vit-infer-4label.ipynb`
- Frozen patient split:
  - `outputs/splits/frozen_patient_split_seed42.csv`
  - `outputs/splits/frozen_patient_split_seed42_audit.json`
- Lightweight result summaries:
  - `outputs/binary_training/`
  - `outputs/binary_inference/`
  - `outputs/4label_training/`
  - `outputs/4label_inference/`
- Reports and figures:
  - `docs/`

## Excluded

- Raw MSD NIfTI data
- Decompressed training volumes
- `.tar` archives
- Pretrained ImageNet-21k weights
- Fine-tuned `.pt` checkpoints
- Local Hugging Face clone metadata
- Runtime cache directories

## Size Intent

The deliverable should remain small enough for normal GitHub use. Large files belong in Hugging Face, Kaggle Datasets, Google Drive, or another artifact store.
