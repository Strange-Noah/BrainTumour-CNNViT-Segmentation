# Swin UNETR on UCSD-BMETS-Longitudinal

Run these commands from the repository root in the `bmets-swin` Conda environment. The scripts use the frozen Protocol v1 patient split, the canonical R2 manifest, the shared loader, and the shared evaluator. Keep R2 credentials in a local AWS profile or environment; never commit credentials.

## Install and verify

```powershell
python -m pip install -r .\research\requirements.txt
python -m pip install monai
python -c "import torch, monai; print('torch', torch.__version__); print('cuda', torch.cuda.is_available()); print('gpu', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'not detected'); print('monai', monai.__version__)"
```

The Swin implementation uses MONAI's 3D Swin UNETR. The starting configuration is feature size 24, 96-cubed patches, batch size 1, CUDA mixed precision, and gradient checkpointing. If the 8 GB GPU runs out of memory, use 64-cubed patches. Both ROI sizes are multiples of 32. Each epoch selects a checkpoint using 12 visits from distinct validation patients by default; the later full validation run computes the shared metrics for every validation visit.

## Download the verified manifest

Set the same R2 profile, bucket, and endpoint variables used for `research.real_smoke`, then run:

```powershell
$project = "C:\Users\chany\Desktop\BlendEd\Novo Nordisk Project\bmets\bmets-project"
$cache = "E:\bmets-cache"
New-Item -ItemType Directory -Force $cache | Out-Null
Set-Location $project
python -m experiments.download_manifest --output "$cache\visits.jsonl"
```

The helper checks the manifest SHA-256 against the fixed Protocol v1 value. The manifest and downloaded NIfTI cache stay outside Git.

## GPU sanity run

This deliberately uses four training visits and two validation visits only to check the training path. It is not a reportable baseline result.

```powershell
python -m experiments.train_swin_unetr `
  --manifest "$cache\visits.jsonl" `
  --cache "$cache" `
  --output "$project\outputs\swin_unetr_bmets\sanity" `
  --epochs 1 `
  --max-train-visits 4 `
  --max-val-visits 2
```

If CUDA runs out of memory, choose a new output folder and repeat with `--roi 64`. Do not reuse an output folder; the trainer refuses to overwrite runs.

## Full training

The completed run used every visit belonging to the 210 frozen training patients for 21 epochs. Checkpoint selection used the same patient-balanced sample of 12 visits from the 45 frozen validation patients each epoch; the best subset score was at epoch 21. To reproduce this configuration, use:

```powershell
python -m experiments.train_swin_unetr `
  --manifest "$cache\visits.jsonl" `
  --cache "$cache" `
  --output "$project\outputs\swin_unetr_bmets\run_f24_roi96" `
  --epochs 21 `
  --roi 96 `
  --feature-size 24
```

Use a distinct output name and `--roi 64` if reducing the crop size. The trainer writes `best_checkpoint.pt`, `run_metadata.json`, and `history.json`; checkpoint selection uses mean visit Dice within each patient, then averages across validation patients. In the completed run, training loss fell from 1.1823 at epoch 1 to 0.5960 at epoch 21. The selected checkpoint was from epoch 21. The model was randomly initialized; no Swin pretrained checkpoint was used.

## Full validation evaluation

Use the complete validation partition to generate original-T1c-grid predictions and the shared metrics. This run directory must not already exist.

```powershell
$checkpoint = "$project\outputs\swin_unetr_bmets\run_f24_roi96\best_checkpoint.pt"
$metadata = "$project\outputs\swin_unetr_bmets\run_f24_roi96\run_metadata.json"
$codeSha = (git rev-parse HEAD).Trim()

python -m experiments.run_swin_partition `
  --manifest "$cache\visits.jsonl" `
  --cache "$cache" `
  --checkpoint "$checkpoint" `
  --metadata "$metadata" `
  --partition validation `
  --model-version "swinunetr-f24-roi96-run1" `
  --git-sha $codeSha `
  --output "$project\outputs\swin_unetr_bmets\validation_run1"
```

### Completed full-validation result

The local full-validation run evaluated all 159 visits from the 45 frozen validation patients. Results below are patient-macro means as specified by Protocol v1; visit metrics are averaged within each patient before averaging across patients.

| Metric | Result |
| --- | ---: |
| Dice | 0.2552 |
| HD95 | 74.03 mm |
| Lesion-level sensitivity | 0.4808 |
| False-positive components | 56.10 per patient (visit-averaged) |
| Absolute volume error | 2,503.68 mm³ |

All 45 patients had defined patient-level aggregates for these metrics. At visit level, 15 visits had no enhancing-tissue reference lesions; lesion sensitivity and HD95 are undefined for those visits under the shared evaluator convention. Across all 159 visits, 464 of 1,238 reference lesion components were matched, while 8,349 predicted components were unmatched. The high false-positive count, low Dice, and high HD95 indicate that this first model is not yet a reliable segmentation source for lesion tracking. Review the saved overlays and per-visit metrics to check model errors and confirm alignment.

The detailed report is `outputs/swin_unetr_bmets/validation_run1/metrics.json`; one original-T1c-grid prediction and candidate sidecar per visit are under `outputs/swin_unetr_bmets/validation_run1/visits/`. Training history and run metadata are under `outputs/swin_unetr_bmets/run_f24_roi96/`.

The test partition has **not** been evaluated. Keep it untouched until the team reviews these validation predictions and decides whether to freeze this preliminary baseline or run another validation-only iteration.

## Final test evaluation

Run this once after choosing the checkpoint and configuration from validation. Do not use test results to select or tune the model.

```powershell
python -m experiments.run_swin_partition `
  --manifest "$cache\visits.jsonl" `
  --cache "$cache" `
  --checkpoint "$checkpoint" `
  --metadata "$metadata" `
  --partition test `
  --model-version "swinunetr-f24-roi96-run1" `
  --git-sha $codeSha `
  --output "$project\outputs\swin_unetr_bmets\test_run1"
```

The per-visit prediction folders contain `seg.nii.gz` and `prediction.json` with visit-local lesion candidates and physical volumes/centroids. The report contains patient-macro Dice, HD95, lesion sensitivity, false positives, and absolute volume error plus defined-patient counts.
