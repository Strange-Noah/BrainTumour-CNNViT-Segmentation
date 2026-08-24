# Route B 4-label Segmentation Plan

## Decision

The next CNN-ViT experiment should be a 4-label segmentation task, not another attempt to fine-tune the binary model.

The binary CNN-ViT is already finalized:

```text
Held-out test patients: 73
Mean 3D Dice:           0.8481
Median 3D Dice:         0.8725
Mean RVE:               13.94%
Median RVE:              8.75%
```

Further tuning after seeing held-out test results would weaken the frozen-test evaluation story. The next scientifically clean extension is therefore to keep the same Route B backbone and split, but change the segmentation target from binary whole tumour to multi-class tumour subregion segmentation.

## Label Encoding

The processed MSD/BraTS labels use:

```text
0 = background
1 = label 1
2 = label 2
3 = label 3
```

A quick local audit confirmed that labels contain `{0, 1, 2, 3}`. Therefore the model should output:

```text
logits shape: (B, 4, 240, 240)
target shape: (B, 240, 240), dtype long
```

This is a mutually exclusive multi-class segmentation problem, so the activation/decision rule is:

```text
probabilities = softmax(logits, dim=1)
prediction = argmax(probabilities, dim=1)
```

It should not use four independent sigmoid heads unless we intentionally redefine the labels as overlapping regions. The current labels are exclusive, so softmax is the correct first implementation.

Do not assign biological names such as edema, necrotic core, or enhancing tumour until the official `dataset.json` / Task01 label mapping is re-checked. Code and reports should initially refer to `label 1`, `label 2`, and `label 3`.

## What Stays the Same

The following parts should remain unchanged from the finalized Route B CNN-ViT:

- Same frozen patient split: 339 train / 72 validation / 73 test
- Same native 240 x 240 Route B geometry
- Same 4-channel MRI input
- Same positional embedding interpolation from 14 x 14 to 15 x 15
- Same pretrained hybrid R50-ViT backbone
- Same decoder upsampling geometry: 15 -> 30 -> 60 -> 120 -> 240
- Same training hardware pattern: dual T4 DataParallel + AMP
- Same patient-local sampling and NIfTI proxy caching
- Same train/validation/test separation

This isolates the new experiment to the target formulation and final segmentation head, rather than changing the architecture in many places at once.

## Required Model Change

The binary model used:

```text
decoder head: Conv2d(..., out_channels=1)
```

The 4-label model should use:

```text
decoder head: Conv2d(..., out_channels=4)
```

The pretrained encoder can still be loaded from the same ImageNet-21k model. The binary checkpoint cannot be directly reused with strict loading because the decoder head shape differs:

```text
binary head:  (1, C, 1, 1)
4-label head: (4, C, 1, 1)
```

For a clean experiment, initialize from the same ImageNet-21k pretrained backbone rather than from the binary tumour checkpoint. If later desired, binary-to-multiclass warm-starting can be a separate ablation.

## Dataset Change

Current binary training code does:

```python
label = (label > 0).astype(np.float32)
label = torch.from_numpy(label).unsqueeze(0)
```

The 4-label dataset should instead preserve the raw integer class:

```python
label = np.asarray(label_nii.dataobj[:, :, z], dtype=np.int64)
label = torch.from_numpy(label.copy())
```

Expected batch:

```text
image: (B, 4, 240, 240), float32
label: (B, 240, 240), int64
```

Assertions should check:

```python
assert image.shape[1:] == (4, 240, 240)
assert label.shape[1:] == (240, 240)
assert set(unique(label)) subset {0,1,2,3}
```

## Loss Function

Use a multi-class version of the existing BCE + Dice idea:

```text
L = lambda * CrossEntropyLoss + (1 - lambda) * SoftDiceLoss
```

A good first setting is:

```text
lambda = 0.5
```

Implementation:

```python
ce = F.cross_entropy(logits, target.long())
probs = torch.softmax(logits, dim=1)
target_1h = one_hot(target, num_classes=4)
dice_loss = 1 - (Dice_1 + Dice_2 + Dice_3) / 3
loss = 0.5 * ce + 0.5 * dice_loss
```

Important detail: background is not included in the Dice loss:

```text
L = 0.5 * CE(z, y) + 0.5 * (1 - (D1 + D2 + D3) / 3)
```

Cross-entropy still uses all four classes, including background. Dice loss focuses only on foreground labels 1, 2, and 3 so that a high background Dice cannot dilute tumour-subclass errors.

The first version should log both:

- Dice including background
- Dice excluding background

For model selection, use tumour-class metrics excluding background.

## Validation Metrics

Validation should compute both 2D slice-level and patient-level 3D metrics.

Per patient, reconstruct:

```text
pred_volume: (240, 240, 155), values in {0,1,2,3}
gt_volume:   (240, 240, 155), values in {0,1,2,3}
```

Then compute:

```text
Dice_class_1
Dice_class_2
Dice_class_3
Mean tumour-subclass Dice = mean(Dice_1, Dice_2, Dice_3)
Whole-tumour Dice = Dice(pred > 0, gt > 0)
```

Also compute class-specific volumes:

```text
V_class_c = number_of_voxels(pred == c) * voxel_volume_mm3
RVE_class_c = |V_pred_c - V_gt_c| / V_gt_c
```

Whole-tumour volume should still be reported:

```text
V_whole = number_of_voxels(label > 0) * voxel_volume_mm3
```

This preserves compatibility with the binary CNN-ViT volumetry endpoint.

Absent-class convention:

- If a patient has no GT voxels for class `c`, class-specific RVE is `NaN`.
- Class-specific RVE aggregates should ignore `NaN` and summarize only GT-positive patients for that class.
- For class-specific patient Dice used in checkpointing, average each class only over patients where that class is present in GT.
- If GT is absent but prediction is present, that false positive is still captured by predicted fraction / volume diagnostics, but it should not create an artificial Dice=1 for absent classes.

This avoids inflating macro Dice when many patients lack a small subclass.

## Checkpoint Criterion

For the 4-label experiment, the primary validation criterion should be:

```text
mean patient-level Dice across tumour classes 1, 2, 3
```

That is:

```text
score = mean_patients(mean(Dice_1, Dice_2, Dice_3))
```

Whole-tumour Dice should be reported as a secondary metric, not used as the main checkpoint criterion, because whole-tumour Dice collapses the task back toward the binary endpoint.

## Inference Output

The 4-label inference script should save:

```text
pred_label.nii.gz
```

with integer values:

```text
0, 1, 2, 3
```

It should preserve the reference MRI affine and spacing, as in the binary inference script.

Optional probability maps:

```text
prob_class0.nii.gz
prob_class1.nii.gz
prob_class2.nii.gz
prob_class3.nii.gz
```

For routine test evaluation, saving the label map is enough.

## Recommended File Boundary

Do not overwrite the finalized binary scripts. Add separate files:

```text
configs/pretrained_route_b_4label.yaml
scripts/06_train_route_b_4label.py
scripts/07_infer_route_b_4label.py
vit-train-4label.ipynb
vit-infer-4label.ipynb
```

The binary model should remain frozen and reproducible.

## First Implementation Order

1. Add 4-label dataset path while preserving integer labels.
2. Add model construction with `num_classes=4`.
3. Add multi-class CE + Dice loss returning a metric dictionary.
4. Add slice-level metrics:
   - Dice class 1
   - Dice class 2
   - Dice class 3
   - whole-tumour Dice
   - mean foreground-class Dice
5. Add patient-level 3D metrics using online sufficient statistics per class.
6. Run build-only gate.
7. Run 100-200 step sanity training.
8. If stable, train for full epochs and select checkpoint by validation mean foreground-class Dice.
9. Run frozen 73-patient test once.

## Main Risk

The tumour subclasses may be strongly imbalanced, and some classes may be absent or occupy very small volumes in individual patients. Class frequencies should therefore be audited before introducing any class weighting.

The first training run should therefore log per-class foreground fractions and per-class Dice from the beginning.

Do not judge this experiment only by whole-tumour Dice. A model can preserve the binary tumour mask while failing to separate tumour subregions.
