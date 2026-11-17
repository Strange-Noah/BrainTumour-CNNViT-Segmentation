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

After the sanity run completes, train with every visit belonging to the 210 frozen training patients. Checkpoint selection uses a patient-balanced sample of 12 visits from the 45 validation patients each epoch; the later validation partition run evaluates every validation visit:

```powershell
python -m experiments.train_swin_unetr `
  --manifest "$cache\visits.jsonl" `
  --cache "$cache" `
  --output "$project\outputs\swin_unetr_bmets\run_f24_roi96" `
  --epochs 30 `
  --roi 96 `
  --feature-size 24
```

Use a distinct output name and `--roi 64` if reducing the crop size. The trainer writes `best_checkpoint.pt`, `run_metadata.json`, and `history.json`; checkpoint selection uses mean visit Dice within each patient, then averages across validation patients.

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

Review `validation_run1\metrics.json` and several saved masks in `validation_run1\visits`. Confirm prediction overlays align with T1c before freezing the model configuration.

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
