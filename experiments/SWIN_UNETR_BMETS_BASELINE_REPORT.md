# Swin UNETR on UCSD-BMETS-Longitudinal: Initial Baseline Report

## Executive summary

This report documents the initial 3D Swin UNETR segmentation experiment for the UCSD-BMETS-Longitudinal project. The goal was to adapt a MONAI Swin UNETR to the team's frozen Protocol v1 workflow and establish an initial baseline for enhancing tissue segmentation.

The experiment ran for 21 training epochs on an NVIDIA GeForce RTX 5060 Laptop GPU. Checkpoint selection identified epoch 21 as the best epoch using a 12-visit validation subset. The selected checkpoint was then evaluated on the full validation partition: 45 patients and 159 visits.

| Patient-macro validation metric | Mean | Median |
| --- | ---: | ---: |
| Dice | 0.2552 | 0.2390 |
| HD95 (mm) | 74.03 | 71.13 |
| Lesion-level sensitivity | 0.4808 | 0.4444 |
| False-positive components per patient (visit-averaged) | 56.10 | 58.50 |
| Absolute volume error (mm³) | 2,503.68 | 1,397.10 |

The training loss decreased from 1.1823 at epoch 1 to 0.5960 at epoch 21. However, the full validation results indicate substantial segmentation errors and many false-positive components. This is a useful first pipeline and model baseline, but it is not yet suitable as a reliable lesion-tracking input. The test partition has not been evaluated.

## Project objective and benchmark protocol

The Swin UNETR task is to produce a comparable segmentation baseline that can later supply visit-level predictions to the longitudinal tracking work. Each examination is processed independently; this model does not use information from prior or later visits.

The experiment follows the shared Protocol v1 decisions:

- Patient-level split: 210 train, 45 validation, and 45 test patients, stratified by visit count. All visits for each patient remain in one partition.
- Input channels, in order: T1, T1c, T2, FLAIR.
- Primary target: binary enhancing tissue, derived from original segmentation label 3. Original source labels are preserved.
- Validation/test outputs: binary masks in original T1c reference geometry.
- Shared evaluation: Dice, HD95 in mm, lesion-level sensitivity, false positives, and volume error, with visit metrics aggregated within patient before calculating patient-macro results.

Protocol artifacts are in `protocol/v1/`; the split hash recorded in the run is `57f5af55141e16ecb0b85fdda63cc355175d9b35271126784eb22766698b36f0`.

The split contains 746 training visits, 159 validation visits, and 160 test visits. These visit totals differ from patient totals because participants have two to four examinations each.

## Data and shared pipeline

The model uses the shared canonical manifest and `research.dataset.Dataset` loader rather than creating a second patient split or a separate scan-matching workflow. The loader reads the canonical manifest from R2, verifies its pinned checksum, selects visits using the frozen patient split, and fetches scans into a local cache. MRI modalities are resampled to the T1c reference grid when needed; the segmentation geometry is checked by the shared pipeline.

The loader returns a four-channel image array with shape `4 × X × Y × Z`, the raw source labels, the derived task target, the T1c NIfTI reference, and visit metadata. One successful real-data smoke check loaded visit `BMETS0001_01`, with image shape `4 × 512 × 512 × 184` and observed source labels `[0, 2, 3]`. That smoke check used the reference mask as its prediction and verified data/export plumbing only; it was not a learned model result.

## Implementation

### Code added or adapted

| File | Role |
| --- | --- |
| `experiments/download_manifest.py` | Downloads the canonical visit manifest from R2 and verifies its SHA-256. It does not download all MRI scans. |
| `experiments/swin_unetr_common.py` | Defines shared per-modality normalization and the four-input/two-output MONAI model constructor. |
| `experiments/train_swin_unetr.py` | Loads frozen train/validation visits, samples training patches, optimizes the model, selects a checkpoint, and writes history/metadata. |
| `experiments/swin_unetr.py` | Loads the selected checkpoint, applies the same normalization, runs sliding-window inference, and returns a binary mask. |
| `experiments/run_swin_partition.py` | Runs inference over validation or test visits, invokes shared evaluation/export functions, and writes per-visit predictions and aggregate metrics. |
| `research/dataset.py` | Shared R2-backed data loader and source geometry/label checks. |
| `research/evaluate.py` | Shared lesion component, metric, candidate export, and prediction provenance implementation. |

### Model and preprocessing

- Architecture: MONAI 3D Swin UNETR.
- Input/output channels: 4 input MRI modalities, 2 output classes (background and enhancing tissue).
- Feature size: 24.
- Initialization: random initialization; no Swin pretrained checkpoint was used.
- Spatial training crop: `96 × 96 × 96` voxels.
- Training batch size: 1 visit patch per optimization step.
- Per-modality normalization: z-score over nonzero voxels within each volume, preserving zero-valued background.
- Patch sampling: one patch per training visit per epoch; 70% probability of centering the crop on a foreground voxel and otherwise selecting a random center.
- Loss: MONAI `DiceCELoss`, with foreground Dice and softmax cross-entropy.
- Optimizer: AdamW, learning rate `1e-4`, weight decay `1e-5`.
- Mixed precision: CUDA float16 autocast with gradient scaling.
- Memory: gradient checkpointing enabled during training.
- Validation inference: full-volume sliding windows using the same 96-voxel ROI and 0.25 overlap; output accumulation is on CPU.
- Output decision: `argmax` over background/enhancing logits; no validation-tuned probability threshold or component-size filtering was applied.

At validation checkpoint selection each epoch, the script uses up to 12 visits from distinct validation patients. It selects a deterministic patient-balanced subset by round-robin visit order. The selected checkpoint was epoch 21, whose subset patient-macro Dice was 0.1742. The later full validation run measured all 159 visits and is the more comprehensive validation result reported here.

## Training run

The trainer saved a checkpoint after validation when the subset score improved. It saved model weights and run metadata, but not optimizer/scaler state for resuming a run.

| Run detail | Value |
| --- | --- |
| Training epochs | 21 |
| Best checkpoint epoch | 21 |
| GPU | NVIDIA GeForce RTX 5060 Laptop GPU |
| PyTorch | 2.15.0+cu132 |
| MONAI | 1.6.1 |
| Training loss, epoch 1 | 1.1823 |
| Training loss, epoch 21 | 0.5960 |
| Checkpoint | `outputs/swin_unetr_bmets/run_f24_roi96/best_checkpoint.pt` |
| Training history | `outputs/swin_unetr_bmets/run_f24_roi96/history.json` |
| Run metadata | `outputs/swin_unetr_bmets/run_f24_roi96/run_metadata.json` |

The decrease in training loss shows the optimization loop was learning on its training patches. It does not by itself establish generalization; the full validation metrics below are the relevant evidence for that.

## Full validation evaluation

The full evaluation used all 45 validation patients and 159 visits, with no test visits. The recorded evaluation code revision was `6c6c01943ca5402f1d2bb5148c7228e4a02cfbaf`. The report is `outputs/swin_unetr_bmets/validation_run1/metrics.json`; 159 exported masks and their prediction sidecars are in `outputs/swin_unetr_bmets/validation_run1/visits/`.

The table reports patient-macro results: each visit's defined metric values are averaged per patient, then patient values are averaged across the 45 patients. False-positive counts and volume errors therefore represent visit-averaged patient metrics, not pooled voxel-level statistics.

| Metric | Mean | Median | Interpretation |
| --- | ---: | ---: | --- |
| Dice | 0.2552 | 0.2400 | Low overlap between predictions and reference enhancing tissue. |
| HD95 (mm) | 74.03 | 71.13 | Large boundary distances on average. |
| Lesion-level sensitivity | 0.4808 | 0.4400 | About half of reference lesion components are detected under the shared overlap matching rule. |
| False-positive components | 56.10 | 58.50 | Many unmatched predicted components per patient after averaging over that patient's visits. |
| Absolute volume error (mm³) | 2,503.68 | 1,397.10 | Considerable patient-level volume disagreement, with a skewed mean suggesting some large errors. |

Across all visits, the evaluator counted 1,238 reference lesion components, matched 464, and counted 8,349 unmatched predicted components. Fifteen visits had no reference enhancing-tissue lesion; lesion sensitivity and HD95 for those visits are undefined under the evaluator convention. All 45 patients had defined patient-level aggregates for each metric in the summary table. Forty-five visits had Dice equal to zero.

### Assessment

The pipeline and evaluation export completed over the full validation partition, and the predictions are available in original T1c geometry for downstream inspection. The segmentation quality is weak: Dice is low, HD95 is high, and the large number of false-positive components is a major problem for lesion tracking. The model should be described as an initial baseline, not as a reliable tracking input or a competitive result.

The low validation performance may reflect limited training (21 epochs), random initialization, the simple one-patch-per-visit sampling scheme, lack of augmentation, or model/prediction behavior that fragments foreground into many components. The metric summary alone cannot distinguish these causes. Before changing the test set or making claims about data quality, inspect prediction overlays and representative false-positive cases in a NIfTI viewer. Confirm that each exported mask aligns with its T1c scan; then use validation only for any further training or threshold/model changes.

## Limitations and reproducibility notes

1. **Run restartability.** The checkpoint is usable, but the run cannot be resumed faithfully because optimizer and AMP scaler states were not saved.
2. **Random initialization.** The baseline did not use the available self-supervised Swin weights. This should be stated in comparisons.
3. **One patch per visit per epoch.** With hundreds of training visits, this is computationally expensive and provides limited patch diversity per epoch.
4. **Checkpoint selection subset.** The 12 validation visits are used repeatedly for epoch selection. The full 159-visit validation pass is reported separately, but the checkpoint selection score is not independent of model selection.
5. **No augmentation beyond stochastic patch location.** This is a simple first implementation, not an exhaustive training pipeline.
6. **High false-positive burden.** The shared evaluator intentionally performs no component-size filtering, so every unmatched 26-connected component is counted. This follows the agreed protocol and should not be changed only to improve the reported score.
7. **No test evaluation.** The frozen test partition remains untouched and must only be evaluated after the team has finished validation-based decisions.
8. **Runtime.** The 21-epoch run took multiple days on the laptop GPU with the current repeated volume-loading workflow. Future experiments should profile cached data access and patch sampling before committing to longer schedules.

## Recommended follow-up

1. Visually review representative validation cases across high/low Dice, high false-positive counts, empty-reference visits, and differing lesion burdens. Verify mask-to-T1c alignment.
2. Preserve this run and its outputs as the first benchmark checkpoint; do not overwrite its records.
3. If the overlays confirm model errors, make a new validation-only run with a documented change. Suitable controlled experiments include using the official Swin self-supervised initialization if checkpoint compatibility is confirmed, increasing patch diversity per visit, and adding modest 3D spatial/intensity augmentation. Change one major factor at a time where feasible.
4. Improve restartability and runtime instrumentation in the trainer before another long run: save optimizer/scaler/epoch state, checkpoint periodically, and avoid re-hashing already verified cached volumes on every sample load while retaining checksum verification at cache creation.
5. Once settings and checkpoint are selected using validation, run the frozen test partition once with the shared evaluator and report the same metrics with patient/visit defined counts.

## Reproduction commands

Run from the repository root in the `bmets-swin` Conda environment. Configure the project R2 profile/bucket/endpoint locally first; never put credentials in this repository.

```powershell
$project = "C:\Users\chany\Desktop\BlendEd\Novo Nordisk Project\bmets\bmets-project"
$cache = "C:\Users\chany\Desktop\BlendEd\Novo Nordisk Project\bmets\local_cache"
Set-Location $project

# Confirm the canonical manifest used by the shared loader.
python -m experiments.download_manifest --output "$cache\visits.jsonl"

# Example new run capped at the 21 completed epochs from the first experiment.
# Use a new output folder; the training script refuses to overwrite one.
python -m experiments.train_swin_unetr `
  --manifest "$cache\visits.jsonl" `
  --cache "$cache" `
  --output "$project\outputs\swin_unetr_bmets\run_f24_roi96_reproduction" `
  --epochs 21 `
  --roi 96 `
  --feature-size 24
```

The completed validation command and output directory are documented in `experiments/SWINUNETR_BMETS.md`. The frozen test evaluation should be run only after validation review.
