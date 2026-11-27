# Swin UNETR on UCSD-BMETS-Longitudinal

Run these commands from the repository root in the `bmets-swin` Conda environment. The scripts use the frozen Protocol v1 patient split, the canonical R2 manifest, the shared loader, and the shared evaluator. Keep R2 credentials in a local AWS profile or environment; never commit credentials.

## Install and verify

```powershell
python -m pip install -r .\research\requirements.txt
python -m pip install monai
python -c "import torch, monai; print('torch', torch.__version__); print('cuda', torch.cuda.is_available()); print('gpu', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'not detected'); print('monai', monai.__version__)"
```

The Swin implementation uses MONAI's 3D Swin UNETR. The starting configuration is feature size 24, 96-cubed patches, batch size 1, CUDA mixed precision, and gradient checkpointing. If the 8 GB GPU runs out of memory, use 64-cubed patches. Both ROI sizes are multiples of 32. The original run selected checkpoints from 12 visits, which resolved to the first 12 sorted validation patients and stayed fixed across epochs. The updated trainer defaults to one fixed visit from each of the 45 validation patients for future runs. Full partition evaluation still uses all 159 validation visits.

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
  --feature-size 24 `
  --max-val-visits 12
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

## Controlled lesion-aware Swin UNETR experiments

The current code supports explicit ablations while preserving the original settings as the defaults for sampling, loss, and support masking. Each experimental run uses a unique output folder and retains the frozen patient split, four input channels, raw label-3 target, MONAI model, and common evaluator.

New trainer options:

| Option | Purpose |
| --- | --- |
| `--sampling voxel` | Original 70% foreground-voxel sampler, for historical reproduction. |
| `--sampling lesion_balanced` | Keeps the epoch update count fixed, gives each patient two or three updates, rotates across visits, selects connected lesions with a clipped inverse-square-root volume preference, and avoids repeating a lesion in the same visit until all components have been used. It reserves 25% of updates for negative crops. |
| `--local-crop-probability 0.5 --local-crop-scale 0.75` | For half of selected-lesion crops, extracts a smaller 3D source field of view and resizes it to the same 96³ model input. The selected lesion is retained with a margin; all labels in the crop remain supervised. |
| `--support-mask input` | Uses the largest 26-connected component of the union of nonzero MRI voxels to zero isolated support outside that component after normalization. |
| `--support-mask loss` | Ignores loss outside that MRI-derived support. Training stops if the support excludes any labelled enhancing voxels. |
| `--support-mask output` | Gates predictions outside the MRI-derived support at inference; partition evaluation also checks that no reference label lies outside the support. |
| `--loss lesion_weighted_ce` | Keeps a soft Dice overlap term and weights CE positive voxels by component volume, clipped to 0.5–2.0. A train-only pre-pass computes a global positive-voxel normalizer, cached under `$cache\derived`, so total positive weight remains one on average across the training partition rather than cancelling out within a single-lesion patch. |
| `--loss tversky --tversky-alpha 0.6 --tversky-beta 0.4` | Uses a Tversky overlap term plus CE. Alpha weights false positives; beta weights false negatives. |

The support mask is an image-derived **acquisition-support proxy**, not a brain-extraction mask. It removes disconnected nonzero islands but may still include skull/scalp and must be inspected before any result is promoted. Do not describe it as skull stripping. Local crops are fully 3D and resampled to the fixed network ROI; they are not 512 × 512 2D images.

### Debug the new code path

Use this only to detect configuration or shape errors. It is not a baseline result.

```powershell
python -m experiments.train_swin_unetr `
  --manifest "$cache\visits.jsonl" `
  --cache "$cache" `
  --output "$project\outputs\swin_unetr_bmets\debug_lesion_sampler" `
  --epochs 1 --roi 96 --feature-size 24 `
  --max-train-visits 4 --max-val-visits 2 `
  --sampling lesion_balanced
```

### Train the first controlled comparison

This run changes patient/lesion sampling only. Leave local magnification, masking, and loss weighting off so its effect is interpretable.

```powershell
python -m experiments.train_swin_unetr `
  --manifest "$cache\visits.jsonl" `
  --cache "$cache" `
  --output "$project\outputs\swin_unetr_bmets\lesion_balanced_v1" `
  --epochs 21 --roi 96 --feature-size 24 `
  --max-val-visits 45 `
  --sampling lesion_balanced --loss baseline --support-mask none
```

After this completes, run full validation with:

```powershell
$run = "$project\outputs\swin_unetr_bmets\lesion_balanced_v1"
$checkpoint = "$run\best_checkpoint.pt"
$metadata = "$run\run_metadata.json"
$codeSha = (git rev-parse HEAD).Trim()
python -m experiments.run_swin_partition `
  --manifest "$cache\visits.jsonl" --cache "$cache" `
  --checkpoint "$checkpoint" --metadata "$metadata" `
  --partition validation --model-version "swinunetr-lesion-balanced-v1" `
  --git-sha $codeSha --output "$run\validation_full"
```

This evaluates all 159 validation visits. The run metadata also records whether the Git working tree was dirty and a SHA-256 of the experiment code files; commit the reviewed feature branch before a reportable run so its Git SHA identifies the exact source revision.

### Run one-change-at-a-time ablations

Define this helper once in PowerShell. It fixes the manifest, patient split, 21 epochs, ROI, feature size, validation coverage, and seed; each invocation writes to a unique output folder.

```powershell
function Start-SwinAblation {
  param(
    [Parameter(Mandatory=$true)][string]$Name,
    [string]$Loss = "baseline",
    [string]$SupportMask = "none",
    [double]$LocalCropProbability = 0.0,
    [double]$LocalCropScale = 0.75,
    [double]$TverskyAlpha = 0.6,
    [double]$TverskyBeta = 0.4
  )
  $arguments = @(
    "-m", "experiments.train_swin_unetr",
    "--manifest", "$cache\visits.jsonl",
    "--cache", $cache,
    "--output", "$project\outputs\swin_unetr_bmets\$Name",
    "--epochs", "21", "--roi", "96", "--feature-size", "24",
    "--max-val-visits", "45", "--seed", "42",
    "--sampling", "lesion_balanced",
    "--loss", $Loss,
    "--support-mask", $SupportMask,
    "--local-crop-scale", "$LocalCropScale",
    "--local-crop-probability", "$LocalCropProbability",
    "--tversky-alpha", "$TverskyAlpha",
    "--tversky-beta", "$TverskyBeta"
  )
  python @arguments
}
```

Run the sampling-only comparison first, followed by the single-change ablations:

```powershell
Start-SwinAblation -Name "lesion_balanced_v1"
Start-SwinAblation -Name "lesion_multiscale_v1" -LocalCropProbability 0.5
Start-SwinAblation -Name "lesion_support_loss_v1" -SupportMask "loss"
Start-SwinAblation -Name "lesion_support_input_v1" -SupportMask "input"
Start-SwinAblation -Name "lesion_weighted_ce_v1" -Loss "lesion_weighted_ce"
Start-SwinAblation -Name "lesion_tversky_fp_v1" -Loss "tversky" -TverskyAlpha 0.6 -TverskyBeta 0.4
```

Keep the same seed and number of updates across the comparison. The saved `last_checkpoint.pt` and `last_run_metadata.json` correspond to the final epoch; `best_checkpoint.pt` and `run_metadata.json` correspond to the best 45-patient selection score. Record each run's Git SHA, checkpoint, metadata and full-validation `metrics.json`.

### Analyze lesion size without replacing shared metrics

First lock the volume-bin edges using the training-patient lesion-volume distribution. Then run the supplementary lesion-size report on validation predictions. The example below uses the BraTS small-instance boundary of 27 mm³ as a first supplementary split; it is not a claim that BraTS and BMETS are directly comparable.

```powershell
$valOutput = "$project\outputs\swin_unetr_bmets\lesion_balanced_v1\validation_full"
python -m experiments.analyze_swin_lesion_sizes `
  --manifest "$cache\visits.jsonl" --cache "$cache" `
  --predictions $valOutput --partition validation `
  --bins-mm3 27 `
  --output "$valOutput\lesion_size_metrics.json"
```

This supplementary report uses the shared 26-connected, one-to-one overlap matching rule and reports lesion sensitivity by reference-volume bin plus false positives per visit. Do not modify the shared `metrics.json` evaluator results. Inspect overlays for small lesions and mask-gated cases before deciding whether a variant is useful.

## Resume training after a computer interruption

The trainer writes `last_training_state.pt` atomically after each completed epoch. It contains model weights, AdamW state, AMP scaler, epoch and history, and Python/NumPy/PyTorch/sampler random states. Resume with the same settings and the same output folder; `--epochs` is the total target epoch count, so it must be greater than the saved epoch.

```powershell
$run = "$project\outputs\swin_unetr_bmets\lesion_balanced_v1"
python -m experiments.train_swin_unetr `
  --manifest "$cache\visits.jsonl" --cache "$cache" `
  --output $run --epochs 21 --roi 96 --feature-size 24 `
  --max-val-visits 45 --seed 42 `
  --sampling lesion_balanced --loss baseline --support-mask none `
  --resume "$run\last_training_state.pt"
```

Resume is supported at completed-epoch boundaries. If the computer fails during an epoch, the unfinished epoch is repeated from the preceding saved epoch. The older `best_checkpoint.pt` and `last_checkpoint.pt` files contain model weights only; they cannot restore optimizer state. To initialize a fresh run from one of those older weights-only files, use `--init-checkpoint` and a new, unused `--output` folder. That warm-start resets the optimizer and epoch count; it is not an exact resume.
