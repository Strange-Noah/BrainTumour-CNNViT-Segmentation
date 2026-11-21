# Frozen v2 Gate1 128 zoom — bounded CPU audit

**Conclusion:** no crop, interpolation, raw-label, mask, affine or gradient-sign corruption was found in the exact failed example. The 128 zoom is implemented correctly for this case. There is a confirmed diagnostic limitation: its metrics combine **three** lesions, and the selected small lesion contributes only **352/3,312 target pixels**. Component weighting helps that lesion relatively but does not provide absolute volume preference across single-component examples. Empty-target Dice has a strong resolution-dependent calibration design limitation. Neither design issue is established as the cause of this failure.

Frozen scientific source: `3ddc324e114d0a0cd740234b5ba26ec750238d2a`; unchanged science `1d6fa16a689916e4225d561b7f2217e61f0ec423cddb9894b5ac3f8636ee8110`. Exact private input v3 receipts and canonical plan digest `359433d8d4e5c8bdfa1008b2db26dd85945ff9683e8638fc4d290e3430d9e2a3` verified. No validation/test images, GPU, new training, inference, checkpoint reconstruction or R2 write. Original image reads were limited to this training examination's headers and one checksum-verified T2 volume to resolve an observed zero-channel question.

The reported failed-run Dice/probability/400-updates/55-seconds are **owner-provided existing GPU evidence**, not a new measurement by this CPU audit. The diagnostic checkpoint was not accessible through the current Kaggle output interface (empty listing). Therefore this report cannot compare actual failed-checkpoint train/eval forwards or recover its probability maps.

## 1. Exact sample and geometry

Replayed [fixed_rows at the frozen source](https://github.com/Strange-Noah/BrainTumour-CNNViT-Segmentation/blob/3ddc324e114d0a0cd740234b5ba26ec750238d2a/experiments/unet25d/cohort/v2/train.py#L41-L54) against the full frozen plan and train-only component inventory. Gate0 contains **one** real 128-scale example, and it is exactly this Gate1 row; there were not 128 separately audited real examples.

Examination/component identifiers, spacing, native/canonical affines, 3D volume/bbox/centroid, slice, exact crop origin/jitter, physical FOV and source-coordinate mapping are preserved only in private `exact_geometry_private.json` and `source_headers_private.json`, outside Git. No identifiers, private coordinates or images are published here.

Independent checks passed:

- Recomputed the exact crop origin from component plane/seed/jitter policy. The selected component is <100 mm³; a local 26-connected reference from original GT agrees with stored component identity, voxel count and physical volume.
- Original four-modality and annotation headers use mm and match the T1c reference affine. Original native GT reorientation/padding equals the prepared raw-label plane exactly. All 22 selected voxels are inside every original modality FOV.
- Modality-major channel order is T1n/T1c/T2/FLAIR, each z−1/z/z+1; independently checked against decoded shard arrays. Output `[12,512,512]`, all finite.
- The 128 crop does not cross the padded image canvas; all 16,384 crop pixels are inside geometric FOV. Anatomy/loss mask retains 15,089 native pixels, becoming 241,424. Zero target pixels excluded.
- Independent index-based crop and repeated-neighbor label/ID/mask references match production exactly. Bilinear pixel-center reference `(i+0.5)/4−0.5`, clamped to crop edges, matches within **4.77e−7**. MRI bilinear and label nearest-exact sampling describe the same grid; no extra row/column swap or half-pixel displacement was found.

| Crop component | Native pixels | Resized pixels | Share of positive BCE weight |
|---|---:|---:|---:|
| Selected small lesion | 22 | 352 | 39.1724% |
| Incidental lesion A | 84 | 1,344 | 30.4138% |
| Incidental lesion B | 101 | 1,616 | 30.4138% |
| **All enhancing GT** | **207** | **3,312** | **100%** |

Every component's count is multiplied by exactly **16**; selected foreground is fully retained. The target intentionally includes all label-3 tissue, rather than erasing incidental lesions. [Frozen production loading](https://github.com/Strange-Noah/BrainTumour-CNNViT-Segmentation/blob/3ddc324e114d0a0cd740234b5ba26ec750238d2a/experiments/unet25d/cohort/v2/loading.py#L42-L73), [geometry](https://github.com/Strange-Noah/BrainTumour-CNNViT-Segmentation/blob/3ddc324e114d0a0cd740234b5ba26ec750238d2a/experiments/unet25d/cohort/v2/geometry.py#L36-L61).

## 2. Intensity and scale comparison

All 12 channels have finite mean/std, seven quantiles, zero fractions and valid occupancy recorded at prepared-plane, native-crop, resized-before-final-mask and final-input stages in [the numeric receipt](ZOOM128_CPU_AUDIT.json). Visit normalization is reused; no crop normalization was introduced. “Valid occupancy” is the geometry/anatomy supervision mask, not a claim that every valid voxel contains brain signal.

Central T1c crop std changes **0.65472 → 0.61686** after bilinear enlargement (−5.78%); mean remains −0.32349. Zero fraction changes **74.70% → 73.51%**, so interpolation does not create extra zero borders. Selected-lesion T1c mean changes −0.23325 → −0.31507; contrast against the five-native-pixel background ring changes **0.79624 → 0.70309**. This is measurable expected smoothing, not disappearance of the enhancing structure. Negative normalized intensities do not mean absent MRI signal.

**Important source-data limitation:** all three T2 input planes are zero before cropping. The checksum-verified original T2 NIfTI also has those three zero planes, while the volume contains 4,380,085 nonzero voxels, exactly the preparation metadata count. All four original affines match; the lesion is inside T2 FOV. Thus this is an original zero-plane/input-information limitation, not zoom-created zeros, a wholly missing T2 volume or an out-of-FOV resampling artifact. No imputation or data policy change was made.

Same-lesion controls retain the exact failed crop center and selected component. Their private physical extents are recorded using actual affine basis vectors, including obliquity.

| Representation | Selected pixels | All GT pixels | Valid pixels | Zero-logit loss | Central T1c std |
|---|---:|---:|---:|---:|---:|
| Native 512 context | 22 | 207 | 21,472 | 0.837071 | 0.18145 |
| Native 256 → 512 | 88 | 828 | 85,888 | 0.837104 | 0.33734 |
| Native 128 → 512 | 352 | 3,312 | 241,424 | 0.833217 | 0.61686 |
| Same 128 embedded unzoomed in 512 | 22 | 207 | 15,089 | 0.833159 | — |

Zoom changes feature scale, occupied area and input statistics materially. That is **SCALE_DISTRIBUTION_SHIFT**, not evidence of an incorrect resize. No model accuracy comparison was executed.

## 3. Exact loss and gradients

[Unchanged frozen balanced_loss](https://github.com/Strange-Noah/BrainTumour-CNNViT-Segmentation/blob/3ddc324e114d0a0cd740234b5ba26ec750238d2a/experiments/unet25d/cohort/v2/loss.py#L5-L25) was run on the real reconstructed target/mask/weights with **controlled zero logits**. These are CPU objective checks, not the unavailable failed model's actual logits.

- Positive/negative counts **3,312 / 238,112**; invalid **20,720**.
- Positive BCE contribution to total loss **0.1732868**; negative **0.1732868**. Soft Dice = 1−3313/124025 = **0.9732876**, contributes 0.4866438; total **0.8332174**.
- Selected-positive per-pixel weight **3.68577**; incidental weights **0.74948 / 0.62333**. Selected-positive gradient mean **−1.410955e−4**; incidental-positive mean **−2.767611e−5**; nearby and far valid-background gradient **+5.518854e−7**; all invalid gradients **exactly zero**.
- Selected/other-positive gradient L1 totals **0.04967 / 0.08192**. Independent float64 central differences at representative logits agree with production autograd to about **1.1e−11 absolute error**. Gradient direction is correct; selected foreground has not lost its BCE signal.
- A second *hypothetical uniform* p=0.0269 objective check increases positive loss and retains strong upward foreground gradients. It is not reconstructed model output: the reported mean alone cannot recover per-pixel probabilities.

**A — confirmed loss-design limitation.** Let component j have n_j pixels and preference a_j. Before final normalization w_i=a_j/n_j. Weighted positive BCE is Σ_j a_j·mean(BCE_j)/Σ_j a_j. Common normalization cancels; with one component, a_j cancels completely. Numerical common-weight ×3 difference is zero; synthetic one-component volume preferences yield equal loss within 1.2e−7 rounding. Relative weighting *within this three-component crop works*: the selected 10.63% of pixels receives 39.17% of positive BCE. Absolute small-lesion preference *across single-component crops* is absent. Volume-aware sampling preference still exists; it was not cancelled.

**B — confirmed loss-design limitation, not a sign bug.** With empty GT and N valid pixels, Dice is 1−1/(Σp+1), gradient p(1−p)/(Σp+1)². At N=241,424, Dice is 0.9999917 at p=.5, 0.9998460 at .0269, 0.9958750 at .001, and 0.7071090 at 1e−5. It is nearly a constant penalty over much of the range and then encourages extremely low background probabilities; its scaling depends on N. A 50% loss coefficient does not imply 50% gradient influence. These checks do **not** establish that it caused this positive example's failure. Disabling/reweighting it would be a scientific objective change and was not implemented.

## 4. DDP, BatchNorm and interpretation

Enumerated the frozen four-example rotation for all 400 updates: **each example appears 100 times on each rank, 200 times globally**. There is no unequal exposure of 128 versus 512. Rank 0 cycles [512-positive,128-positive,negative-enhancing,negative-empty]; rank 1 has a one-position offset. Example pairings include both-positive and both-negative updates. This is a deterministic order/statistics difference, not unequal frequency.

[Gate1 and measure](https://github.com/Strange-Noah/BrainTumour-CNNViT-Segmentation/blob/3ddc324e114d0a0cd740234b5ba26ec750238d2a/experiments/unet25d/cohort/v2/train.py#L56-L87): training uses local BN (momentum .01), not SyncBN; DDP broadcasts buffers before forward. Each rank then updates its own statistics on its local example. Interim eval uses those local buffers and requires both ranks to pass, but prints rank-0 metrics. Final evaluation first broadcasts rank-0 buffers. Train/eval transitions are correct. Mixed-scale/empty-plane input distributions can plausibly impair running-statistics evaluation, but **BN as the root cause remains unsupported without the existing checkpoint's controlled train/eval comparison**. Finite BN statistics alone would not settle that question.

The reported 128 Dice/probability are **all-three-lesion metrics**, not selected-lesion metrics. However, the overall mean .0269 still implies the selected-lesion mean cannot exceed approximately .0269×3312/352 = **0.2531**, even if all positive probability mass were concentrated there. Thus this diagnostic limitation does not convert the failed strict gate into a pass. The 512 and 128 GPU examples also use different lesions, so their accuracy gap is not a controlled scale experiment.

| Finding | Classification | What is established |
|---|---|---|
| Wrong crop/axes/resize or missing selected target | NOT_SUPPORTED_BY_EVIDENCE | Independent geometry/label/reference checks pass for exact sample. |
| Absolute small-volume weight disappears in one-component positive BCE | CONFIRMED_LOSS_DESIGN_ISSUE | Algebra and numerical invariance; relative multi-component weighting works. |
| Empty Dice calibration/scaling | CONFIRMED_LOSS_DESIGN_ISSUE | Formula/gradients verified; causal role unresolved. |
| “128 small-lesion metric” pools incidental lesions | GATE1_DIAGNOSTIC_LIMITATION | Three components, selected only 10.63% of GT pixels. |
| Zoom statistics and original zero T2 planes | SCALE_DISTRIBUTION_SHIFT | Measured; source zeros verified, not introduced by zoom. |
| Unequal 128 training exposure | NOT_SUPPORTED_BY_EVIDENCE | Equal per-rank/global counts. |
| BN caused failed 128 prediction | NOT_SUPPORTED_BY_EVIDENCE | Plausible mechanism, checkpoint comparison unavailable. |

No numerical **CONFIRMED_IMPLEMENTATION_BUG** was found in the 128 preprocessing/loss path. The confirmed low-risk correction is diagnostic scope/count reporting, described below.

## 5. Minimal correction and next decision

Added only `target_scope`, selected/incidental foreground counts and selected positive-BCE weight share to sample identity/Gate1 reporting. It does not claim selected-lesion probability/Dice without a selected-component mask. Model tensors, GT, mask, weight values, loss, sampler, normalization, thresholds and training order are unchanged. Config/science hash and 220,384 mandatory / 73,462 replay plan remain unchanged. **No new source was registered or deployed; no GPU job was started or restarted.**

[CPU reconstruction](../../cohort/v2/audit_zoom128.py), [six regression tests](../../cohort/v2/test_zoom128_audit.py), [reporting correction](../../cohort/v2/loading.py), [Gate1 reporting](../../cohort/v2/train.py). Six targeted tests and 18 existing v2 tests passed. The read-only audit requires the verified private v3 metadata/GET lease in a mode-700 directory outside the repository:

```bash
python -m experiments.unet25d.cohort.v2.audit_zoom128 --private-dir /PRIVATE/VERIFIED_V3_AUDIT_DIRECTORY
```

Do not rerun unchanged Gate1 just for more time or lower thresholds. First preserve/retrieve the already generated diagnostic checkpoint, compute selected-versus-incidental metrics and a controlled CPU train/eval BN comparison on model copies. If a future GPU experiment is then authorized, use the **same lesion** at paired native 512 / zoomed 128 contexts with explicit per-lesion metrics, comparable negatives and unchanged strict criteria; predefine the limited BN/objective question. Changing absolute volume weighting, empty Dice or BN policy requires a new scientific config/source identity and established review. None is authorized for GPU execution by this audit. Stop here after bounded CPU forensics.
