# Swin UNETR BMETS Experiment Code: Implementation Notes

This document describes the Swin UNETR experiment work on branch `swinunetr-lesion-aware`: what each changed or added file does, how the training options work, how to run the controlled experiments, and how to read the saved results. It is a technical record of the implementation; it does not claim that any proposed option improves validation performance until that option has been run and evaluated.

## 1. Purpose in the project

The Swin UNETR task supplies an independent 3D segmentation model for the UCSD-BMETS longitudinal project. The output is a binary prediction of enhancing tissue, using original UCSD label 3 as foreground. This model is trained one examination at a time; it does not use earlier visits as model input. Its exported per-visit masks and lesion candidates can later be consumed by longitudinal tracking.

The new work keeps the frozen Protocol v1 patient split, the same four MRI inputs (T1, T1c, T2, FLAIR), the same enhancing-tissue target, and the shared evaluation/export contract. The first recommended experiment changes only training-patch sampling. Cropping, support masking, and alternative losses are separate ablations, so their effects can be assessed one at a time.

## 2. Files and responsibilities

| File | Responsibility |
|---|---|
| `experiments/train_swin_unetr.py` | Training entry point; loads the frozen manifest and split, creates the MONAI Swin UNETR, samples patches, computes losses, validates, and writes checkpoints and history. |
| `experiments/swin_sampling.py` | Utilities for connected enhancing-lesion components, patient-balanced visit draws, lesion-aware crop centers, optional local crops, MRI support masks, and support-mask caching. |
| `experiments/swin_unetr_common.py` | Shared model construction and per-modality normalization used by training and prediction. |
| `experiments/swin_unetr.py` | Prediction adapter used by the shared partition runner; loads the configured checkpoint and predicts on original-grid MRI volumes with sliding-window inference. |
| `experiments/run_swin_partition.py` | Runs inference and shared metrics over a partition, checks the frozen split/manifest provenance, and exports predictions in reference geometry. |
| `experiments/analyze_swin_lesion_sizes.py` | Supplementary lesion-size sensitivity analysis. It reads exported predictions and does not alter the shared evaluator or `metrics.json`. |
| `research/dataset.py` | R2-backed data retrieval and local cache verification. Successful cache verification is memoized per `Dataset` instance, and SHA-256 is now calculated in bounded-size chunks. |
| `experiments/SWINUNETR_BMETS.md` | Operational guide with commands for training, validation, ablations, and resume. |

## 3. Input preparation and baseline model behavior

For each selected visit, `research.dataset.Dataset.load` obtains T1, T1c, T2, FLAIR, and the segmentation. The images are aligned to the T1c reference geometry when required. The training target is binary: label 3 becomes positive and all other labels become background. The other original mask labels are not redefined as enhancing tissue.

Each modality is normalized separately using the existing `normalize_modalities` routine. It computes a per-volume z-score over nonzero voxels and leaves zero background at zero. The model receives four channels in the fixed order T1, T1c, T2, FLAIR. Training uses a batch size of one, a 96³ input ROI by default, feature size 24, CUDA mixed precision, gradient checkpointing, and AdamW with learning rate `1e-4` and weight decay `1e-5`.

For every optimizer update, the selected 3D patch and target are transferred to CUDA. Model forward, loss, backward, and optimizer update use the GPU. File retrieval, NIfTI decompression, resampling, normalization, connected-component labeling, and crop construction are CPU-side operations.

## 4. Patient-balanced, lesion-aware sampling

### 4.1 Balance updates across patients

The frozen training partition contains 210 patients and 746 visits. The lesion-balanced sampler distributes 746 crop draws approximately evenly across patients: each patient receives three draws, and the remaining 116 draws are distributed one each to 116 patients. Thus, **94 patients receive three draws and 116 receive four** in an epoch. This is a draw count, not a claim that every visit is seen each epoch.

For a patient with multiple visits, draws cycle through that patient's visits. The starting visit rotates by epoch. This limits the influence of patients who happen to have more examinations while allowing their visits to contribute over successive epochs. There are still 746 optimizer updates per full training epoch.

The implementation is in `patient_balanced_draw_counts` in `experiments/swin_sampling.py`. Some older prose in experiment notes says “two or three” draws per patient; the current arithmetic and code give three or four for 746 visits and 210 patients.

### 4.2 Choose a lesion rather than a random foreground voxel

The binary enhancing mask is split into 26-connected components. Each component is treated as one lesion candidate for sampling in that visit. Its physical volume is computed from voxel count multiplied by the voxel volume derived from the T1c affine, in mm³.

On a lesion draw, the sampler chooses a component with a volume-aware probability. Within that visit, the weight is proportional to the square root of median remaining lesion volume divided by that component's volume, clipped to `[0.5, 2.0]`. This increases exposure for smaller components, but caps the preference so an extremely small annotation cannot monopolize sampling. Components selected in a visit are not selected again in that same visit during the epoch until all its components have been used; then the set can be reused.

About 25% of draws are deliberately negative crops when the visit has one or more lesions. If the visit contains no enhancing component, the draw is negative. This retains explicit background examples.

The selected lesion determines the crop center. The crop is adjusted as needed to retain that component plus a small margin. Other annotated enhancing lesions that fall inside the resulting crop remain labelled and contribute to training loss; only the crop center is chosen using one lesion.

### 4.3 Compare with the original sampler

The `voxel` mode reproduces the prior style of choosing a foreground voxel with probability 0.7 (when foreground exists), otherwise a random center. The first controlled comparison uses `lesion_balanced`, changing sampling while keeping the model, target, loss, ROI, and support-mask behavior fixed. Use a distinct output directory for every run.

## 5. Optional experiment switches

These are available in `train_swin_unetr.py`, but they are not all enabled in the recommended sampling-only run.

### 5.1 Local magnification and jitter

`--local-crop-probability` controls how often a selected lesion gets a smaller source crop. Its default is `0.0`, so local magnification is off. For example, `--local-crop-probability 0.5 --local-crop-scale 0.75` starts with a crop about 75% of the normal voxel extent on each axis for half of lesion-centered draws, then resizes it to the same 96³ network input. For a 96³ ROI this is initially 72³ voxels, though the crop grows if needed to retain the selected lesion plus margin. This makes a small lesion occupy more input voxels but removes surrounding context. The center starts at a randomly selected voxel in the lesion and is adjusted as needed to keep the component in view. The actual physical field of view also depends on image voxel spacing. A local crop is a separate ablation because it changes field of view and interpolation.

### 5.2 MRI support mask

`acquisition_support` estimates the union of nonzero voxels across the four MRI modalities and retains its largest 26-connected component. This estimates acquired MRI support; it is not a brain segmentation or skull-stripping algorithm. The derived mask is cached under the cache directory so it need not be recomputed every epoch.

`--support-mask` choices:

- `none` (default): no support mask.
- `input`: zeros image voxels outside support during training and validation.
- `loss`: excludes voxels outside support from the training loss. It errors if the MRI-derived support excludes any positive reference label.
- `output`: leaves model input unchanged and zeros prediction voxels outside support during validation and prediction export. The trainer also rejects a support mask that excludes positive validation labels.

The support mask must be audited visually before a result using it is considered useful. Input/loss/output modes change the experiment and should not be silently mixed with the unmasked baseline.

### 5.3 Lesion-volume-weighted cross-entropy

`--loss lesion_weighted_ce` computes training-only lesion volumes and uses them to weight positive cross-entropy voxels. The raw positive weight is proportional to the square root of training-median lesion volume divided by the component volume, clipped to `[0.5, 2.0]`, then normalized so the mean weight over training positive voxels is one. Background CE weight remains one. The overlap term remains a Dice-style term. Statistics are cached by manifest and split hashes under `cache/derived`.

This is a separate loss ablation from volume-aware sampling: sampling changes which crop is selected; weighted CE changes the gradient contributions inside the crop.

### 5.4 Tversky loss

`--loss tversky` replaces the Dice overlap term with Tversky overlap, while retaining the cross-entropy term. The Tversky denominator is `TP + alpha * FP + beta * FN`; increasing alpha penalizes false positives more, while increasing beta penalizes false negatives more. Defaults are alpha `0.6` and beta `0.4`. These coefficients are hypotheses to evaluate, not validated optima.

## 6. Validation, checkpoint selection, and final evaluation

By default, checkpoint selection evaluates one fixed visit from each of the 45 validation patients (`--max-val-visits 45`). `validation_indices` selects visits round-robin across patients; the selected list remains fixed between epochs. Dice is first averaged within patient and then averaged over patients, so each patient contributes equally.

This 45-visit score chooses the checkpoint; it is not the final partition report. After choosing the checkpoint, run `experiments.run_swin_partition` on the full validation partition (159 visits) with the shared evaluation/export code. The shared output reports scan- and lesion-level metrics and saves each binary prediction in original T1c reference geometry for subsequent tracking. If comparing against an older checkpoint selected on a different validation subset, compare both models on the same full validation partition and evaluator, and document the difference in checkpoint-selection policy.

`experiments.analyze_swin_lesion_sizes.py` is a supplementary analysis. It bins reference lesions by physical volume, performs one-to-one lesion matching on any positive voxel overlap, then reports pooled and patient-macro sensitivity by bin and false positives per visit. Bin edges must be selected and locked using training-patient information before validation analysis. The example edge of 27 mm³ is a proposed starting point, not a BMETS-validated threshold. The script writes a separate JSON report and leaves the shared metrics unchanged.

## 7. Recommended controlled run

Run this first to isolate the effect of patient/lesion-aware sampling:

```powershell
python -m experiments.train_swin_unetr `
  --manifest "$cache\visits.jsonl" `
  --cache "$cache" `
  --output "$project\outputs\swin_unetr_bmets\lesion_balanced_v1" `
  --epochs 21 --roi 96 --feature-size 24 `
  --max-val-visits 45 --seed 42 `
  --sampling lesion_balanced --local-crop-probability 0 `
  --support-mask none --loss baseline
```

The equivalent defaults are local-crop probability `0`, support mask `none`, and loss `baseline`; spelling them out makes the experiment configuration explicit. Run with the project repository as the working directory so `protocol/v1/patients.csv` and Python modules resolve correctly. The output folder must not already exist for a new run.

Then run the full validation partition, using the best checkpoint and its matching metadata, as shown in `experiments/SWINUNETR_BMETS.md`. Do not use the test partition for repeated experiment selection.

## 8. Saved artifacts

Each training output folder contains:

- `history.json`: epoch-by-epoch mean training loss, 45-patient selection Dice, selection patient count, and experiment settings.
- `best_checkpoint.pt` and `run_metadata.json`: model weights selected by the best validation selection score and their configuration/provenance.
- `last_checkpoint.pt` and `last_run_metadata.json`: weights and metadata from the latest completed epoch.
- `last_training_state.pt`: full training state written atomically after every completed epoch, including model/optimizer/scaler state, epoch, history, and RNG states for recovery.

Use `--resume "$run\last_training_state.pt"` with the same output folder and matching experiment settings to continue at the next epoch. `--epochs` is the total target epoch number, not the number of additional epochs. Resume is at completed-epoch boundaries; a failure inside an epoch means that epoch is repeated. The older model-only checkpoints can be loaded with `--init-checkpoint` for a warm start, but that resets optimizer and epoch state and is not an exact resume. Do not treat the currently running process as having new resume support unless it was started from the updated script: a Python process does not load source edits made after startup.

## 9. Cache and runtime behavior

`research.dataset.Dataset.fetch` now remembers successful raw-file verification for the lifetime of a dataset object. Subsequent requests for the same file check its size and modification time rather than issuing another R2 `HEAD` request and rereading the entire local file to calculate its SHA-256. A missing or changed file falls back to the checksum and download path. Hashing reads in 8 MiB chunks to bound memory use.

This removes repeated remote metadata requests and checksum scans, but it does not cache decompressed or normalized NIfTI volumes. `Dataset.load` still reads and preprocesses each selected visit when it is sampled. Data loading and preprocessing therefore remain CPU-side work and can leave GPU utilization intermittent. The one-time training device print (`Training devices: model=cuda:0, image=cuda:0`) confirms device placement for the first training batch.

## 10. Interpretation and limitations

- The changes are methods to test; the code alone does not establish that lesion-balanced sampling, masking, local crops, or either alternative loss is better.
- The sampling-only run is the primary interpretable comparison. Change one setting at a time afterward and use unique output directories.
- Patient/lesion sampling affects training exposure, but training remains one independent visit per update. There is no temporal input or lesion tracking in this trainer.
- A fixed 45-patient validation selection makes checkpoint selection more patient-balanced than the former 12-visit subset, but full validation remains necessary for reporting.
- Predictions are not lesion tracks. Tracking should consume exported candidates across visits after segmentation predictions are reviewed.
- Record the code revision, manifest/split hashes, command, checkpoint, metadata, history, and full validation metrics for each run.
