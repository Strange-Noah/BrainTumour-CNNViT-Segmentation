# Route B Training Sanity

## Purpose

The first training phase should not be a full experiment. It should verify that the Route B transfer-learning pipeline can learn without immediate numerical or sampling failure.

Main checks:

- train loss decreases;
- validation loss is tracked separately;
- BCE, Dice loss, and total loss are logged separately;
- predicted foreground fraction does not collapse silently to all background;
- slice Dice is logged;
- gradient norm is finite;
- train/validation curves are saved for mentor review.

## Training Script

Script:

`scripts/04_train_route_b.py`

Primary outputs:

```text
outputs/training/<run_name>/training_metadata.json
outputs/training/<run_name>/loss_history.csv
outputs/training/<run_name>/loss_curves.png
outputs/training/<run_name>/best_checkpoint.pt
```

The loss is:

```text
L = 0.5 * BCEWithLogitsLoss + 0.5 * DiceLoss
```

The model returns raw logits:

```text
(B, 1, 240, 240)
```

No sigmoid is applied inside the model.

## Patient Split

The script reads the frozen patient split:

`outputs/splits/frozen_patient_split_seed42.csv`

Training samples are expanded only from patients with:

```text
split == "train"
```

Validation samples are expanded only from patients with:

```text
split == "val"
```

The DataLoader may shuffle train slices, but it does not create a new random patient split.

## Differential Learning Rates

Current optimizer groups:

```text
decoder:       3e-4
stem:          3e-4
pos_embed:     3e-5
ResNet stage2: 3e-5
ViT last4:     3e-5
encoder norm:  3e-5
```

This matches the current moderate fine-tuning policy:

```text
trainable params: 42,515,793
frozen params:    58,926,336
```

Fallback if overfitting or collapse appears:

```text
stem + decoder + ViT last2
```

## CPU Micro-Run

The local machine currently has no CUDA/MPS device, so only a tiny CPU smoke training was run:

```bash
python3 PBL_BrainTumour/scripts/04_train_route_b.py \
  --epochs 2 \
  --batch-size 1 \
  --max-train-steps-per-epoch 2 \
  --max-val-steps-per-epoch 1 \
  --max-train-patients 4 \
  --max-val-patients 2 \
  --max-slices-per-val-patient 2 \
  --output-dir PBL_BrainTumour/outputs/training/route_b_cpu_sanity_2epoch
```

Outputs:

```text
outputs/training/route_b_cpu_sanity_2epoch/loss_history.csv
outputs/training/route_b_cpu_sanity_2epoch/loss_curves.png
outputs/training/route_b_cpu_sanity_2epoch/best_checkpoint.pt
```

Observed:

```text
epoch 1 train_loss 0.7837 val_loss 0.6737 train_dice 0.0351 val_dice 0.0000 train_fg 0.3598 val_fg 0.0065
epoch 2 train_loss 0.6688 val_loss 0.6551 train_dice 0.0000 val_dice 0.0000 train_fg 0.0039 val_fg 0.0001
```

Interpretation:

- the forward/backward/update/logging/checkpoint/plot path works;
- loss decreased in this tiny run;
- predicted foreground fraction collapsed sharply, but the run has only four optimizer steps and is not representative;
- foreground fraction should be watched carefully in the real 200-500 step sanity run.

## Recommended GPU Sanity Run

On a GPU machine, run:

```bash
python3 PBL_BrainTumour/scripts/04_train_route_b.py \
  --epochs 5 \
  --batch-size 4 \
  --max-train-steps-per-epoch 100 \
  --max-val-steps-per-epoch 25 \
  --max-val-patients 8 \
  --max-slices-per-val-patient 16 \
  --output-dir PBL_BrainTumour/outputs/training/route_b_sanity_gpu
```

This gives 500 optimizer steps total. Use the resulting loss diagram for mentor review before starting full epoch training.

## Dual-GPU Benchmark

DataParallel support is implemented as a small wrapper around the existing Route B model. It does not change the model, loss, optimizer groups, validation aggregation, or checkpoint metric.

Notebook parameters:

```text
USE_DATA_PARALLEL = True
GPU_IDS = "0,1"
BATCH_SIZE = 8
FULL_TRAIN = False
BUILD_ONLY = False
MAX_TRAIN_STEPS_PER_EPOCH = 200
MAX_VAL_STEPS_PER_EPOCH = 25
OUTPUT_DIR = "run_outputs/route_b_dualgpu_benchmark"
```

Expected metadata:

```text
parallelism.mode = "DataParallel"
parallelism.gpu_ids = [0, 1]
parallelism.gpu_count_used = 2
effective_batch.batch_size = 8
effective_batch.per_gpu_batch_size = 4
```

Checkpoint saving unwraps `torch.nn.DataParallel`, so checkpoint keys remain stable and do not gain a `module.` prefix. This keeps single-GPU and dual-GPU checkpoints load-compatible.

AMP is intentionally not enabled yet. First benchmark DataParallel with batch 8; if both T4s are used and throughput improves, test batch 16 or AMP as a separate change.

The trainer records epoch-level timing in `loss_history.csv`:

```text
train_seconds
val_seconds
epoch_seconds
train_samples_per_sec
```

This is the intended comparison signal for single T4 vs dual T4:

```text
epoch=1 ... timing: train=1834.2s val=391.5s total=2225.7s throughput: train=28.65 slices/s
```

Use the train throughput and the separate train/validation timings to decide whether DataParallel is worth keeping.

## Full-Train Validation Guard

`scripts/train.py --full-train` now explicitly disables validation subsetting and passes:

```text
--max-train-steps-per-epoch none
--max-val-steps-per-epoch none
--max-val-patients none
--max-slices-per-val-patient none
--require-full-val
```

The lower-level trainer asserts:

```text
val patients = 72
val samples  = 72 x 155 = 11,160
```

Verified wrapper build-only command:

```bash
python3 scripts/train.py \
  --config configs/pretrained_route_b.yaml \
  --full-train \
  --build-only \
  --max-train-patients 2 \
  --num-workers 0 \
  --output-dir run_outputs/route_b_wrapper_full_val_build_check
```

Observed:

```text
val_summary = {
  samples: 11160,
  unique_patients: 72,
  positive_slices: 4945,
  negative_slices: 6215,
  positive_fraction: 0.4431
}
```

This prevents the dangerous failure mode where full training still validates on the sanity subset.

It also prevents capped full epochs. A true full epoch with batch size 4 is expected to process approximately:

```text
train: ceil(52,545 / 4) = 13,137 batches
val:   11,160 / 4       = 2,790 batches
```

`--full-train` now uses the CLI/notebook `--epochs` value directly rather than silently falling back to the YAML default. Non-build-only full training rejects `--max-train-patients`, so a true full run cannot accidentally train on a patient subset.

## Current Limitations

Training still uses 2D slice-wise BCEWithLogits plus soft Dice loss. This is intentional. The patient-level 3D metrics are validation metrics, not training losses.

When validation contains complete patient volumes, the trainer computes:

```text
patient-level 3D Dice
RVE = |V_pred - V_GT| / V_GT
```

These are computed after collecting all 155 slices per validation patient and stacking them into:

```text
240 x 240 x 155
```

Implementation note: the trainer does not store all binary masks in RAM. It computes mathematically equivalent online sufficient statistics per patient:

```text
intersection = sum(P & G)
pred_sum     = sum(P)
gt_sum       = sum(G)
slice_ids    = observed slice indices
```

Then:

```text
Dice3D = (2 * intersection + eps) / (pred_sum + gt_sum + eps)
RVE    = |pred_sum - gt_sum| / gt_sum
```

If validation is a sanity subset such as `8 patients x 16 slices`, patient-level 3D metrics are skipped:

```text
patient_3d_metrics_status = skipped_incomplete_patient_volumes
```

In that case checkpoint selection falls back to:

```text
val_loss
```

When complete validation is available, checkpoint selection uses:

```text
patient_dice3d_mean
```

The loss history CSV now records:

```text
patient_3d_metrics_status
patient_dice3d_mean
patient_dice3d_median
patient_rve_mean
patient_rve_median
patient_3d_count
```

Full validation also checks that the observed validation patient set exactly matches the expected frozen validation IDs, not just that some patients have complete slices.
