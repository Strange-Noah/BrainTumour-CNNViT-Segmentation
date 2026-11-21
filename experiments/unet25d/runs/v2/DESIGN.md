# v2: foreground-aware, lesion-centric, multi-scale Phase 1

This replaces the earlier v2 proposal with the executable implementation in [cohort/v2](../../cohort/v2/). Historical v1 files/configurations/tags are unchanged. No Tversky, architecture change, precision experiment, or v1 state migration.

Frozen: SMP 0.5 U-Net/ResNet34, the pinned ImageNet revision/SHA, four modalities × three adjacent slices, `[B,12,512,512]`, raw label 3, full FP32, dual T4, local batch 1/global 2, AdamW and original initial learning rates, frozen 210/45/45 patient split. All BN layers use ordinary training statistics with momentum 0.01; evaluation uses rank-0 running buffers. This is fresh initialization, never v1's special normalization migration. Gate 1 tests actual eval-mode positive predictions, not loss alone.

## Spatial support and physical geometry

FOV validity maps canonical voxel centers through `inverse(native_affine) @ canonical_affine` into the native acquired lattice; intensity zeros never determine FOV. Existing closest-RAS orientation, padding and full oblique native affine are retained. All 746 training label volumes are checked against that geometry and existing catalogue positives.

Anatomical support uses all 12 input channels, hole filling, 3 mm closing and a conservative outward 20 mm margin. The minimum singular value of the physical in-plane basis produces an enclosing, conservative index-space margin for oblique/sheared grids. Tiny/uncertain support or pathological morphological scale falls back to FOV. It is not a strict parenchyma mask. For training only, any excluded label-3 voxel forces a recorded whole-plane FOV fallback; a GT voxel outside FOV fails outright. This checked fallback applies to every component occurrence before suppression/loss. The bounded CPU real audit verifies examples; runtime coverage guards apply to every training draw. Inference support is MRI-only and never uses GT to choose/fix its mask. Final native predictions are not clipped by anatomical masks.

Only the original prepared catalogue/shards are read. Auxiliary component IDs and metadata go into a new private science-addressed v2 namespace; no original prepared examination is rewritten or re-ingested.

## Foreground-aware loss

`L = 0.5 BCE + 0.5 SoftDice`. This preserves the family, **not** v1's internal weighting.

For component j, `a_j = clip((100 mm³ / V_j)^0.25, 1, 4)`. Each positive crop pixel receives `a_j / n_j`, where n_j is that component's pixel count in the actual resized crop. Positive weights are normalized to mean one; the positive BCE is their weighted mean. Thus each component's aggregate contribution depends on a bounded physical-volume preference, not its voxel count. A one-voxel component cannot acquire an unbounded weight.

Positive samples use `BCE = 0.5 mean_weighted_positive + 0.5 mean_valid_negative`. Entirely negative samples use valid-background BCE; positive-only ROIs use positive BCE. Soft Dice is `(2 Σ p y m + 1)/(Σ p m + Σ y m + 1)` with explicit empty-case smoothing. Invalid pixels contribute zero gradient. Valid normal tissue and false positives remain supervised. Full-volume validation retains the original unweighted BCE+Dice reporting reduction for comparison; its inference, native geometry and denominators are unchanged.

## Lesion replay and exhaustive interleaving

Training-only 26-connected 3D components retain visit-local IDs, original physical volumes, centroids, bboxes, slice membership and slice-level bounds. Patient/visit/component inventories remain private. Replay balances patients/visits, samples lesions with the bounded small-volume rule, and caps selected replay at 1,536/patient, 768/visit and 128/component. Incidental lesions in full-field patches are measured separately; mandatory coverage is never capped or removed.

Every one of 220,384 mandatory tile-plane sites occurs exactly once. Add 73,462 lesion-centered replays: 293,846 samples / 146,923 optimizer updates. Given the existing 37,170 positive mandatory sites, planned foreground exposure is 110,632/293,846 = 37.65%; the actual frozen catalogue/plan must confirm it. Desired replay scales: 512/256/128 at 50/30/20%. 192 is supported for compatible escalation, not assigned an initial quota. Oversized lesions escalate; >512 in-plane extent is explicitly marked partial rather than falsely claiming whole-lesion coverage. Bounded jitter and aligned padding apply before bilinear MRI and nearest-neighbor label/mask/ID resizing. Physical MRI resolution does not increase.

A deterministic update-level class sequence pairs positives with negatives and distributes positive-containing updates across both ranks. Bounded four-patient/32-update blocks retain locality; sorting occurs only inside visit queues, never over the final global plan. All mandatory negative-enhancing and negative-empty sites remain. Complete-plan checks enforce ≥35% positive samples, maximum eight negative-only updates and positive exposure in every sliding 32-update rank window. Cursor advances only after successful optimizer updates. Plan hash and prefix counts are checked on resume; prefetch/replay never count as unique coverage.

Training-only mining records peak and top-256 negative probabilities inside valid tissue. Pure negative versus positive-containing patches have explicit different eligibility; positive patches cannot silently become empty hard negatives. Records are bounded to 32/visit/category. Phase 2 allocation is not predetermined.

## Integrated run and stopping

The full-scale capacity example has at least 16 actual component pixels on its center plane; the local 128 example is a true <100 mm³ lesion. These training-only criteria avoid testing full-scale capacity solely on a pathological one-voxel target, while still testing tiny-lesion learning. The same fixed examples are reused for early observation.

The first GPU session performs up to 400 diagnostic updates / 600 seconds on four fixed training examples (512/128 positives and both negative types). Passing requires actual eval-mode tumor probability ≥0.60, foreground Dice ≥0.10 and nonempty predictions on each positive; negative probability ≤0.40 and predicted fraction ≤0.10 on each negative; finite model/optimizer/gradients and healthy BN. A decreasing total loss alone fails. Formal model, optimizer and RNG are then independently initialized from the pinned encoder; no diagnostic update carries over.

After ~5% mandatory coverage, evaluate the same fixed training examples and the immutable first two sorted validation exams using full-FOV/native inference. This detects fatal positive-learning/numerical issues, not scientific model selection. Healthy positive learning continues automatically. Ten-minute observation uses actual formal computation timestamps.

At complete mandatory/replay coverage, run all 45 validation patients / 159 exams with full-FOV overlapping original-scale tiles, positive Gaussian logit merging and native restoration before thresholding. Preserve 0.5 and 0.3/0.7 metrics, actual per-threshold lesion strata, pooled counts, empty prediction frequency, predicted volumes, positive/negative probability histograms and private predictions. No test access. Then set `AWAIT_PHASE2_DESIGN` and disable campaign continuation.

## Resources and provenance

All output state, checkpoint pointers, collections and auxiliary data are isolated under `novo-nordisk/experiments/unet25d/v2/<science_sha>/`. Private input ID and kernel identity are new v2 resources on the same existing Kaggle account. Source execution remains pinned by main's existing trusted registry/actor/branch checks. No source bypass or permission expansion.

At the historical 5.19 samples/s, Phase 1 training alone is about 15.73 wall-hours; native validation adds roughly 2.22 hours at the same per-plane rate, plus initialization/data I/O. These are estimates, not measurements or a scientific time cap. Actual available free quota establishes the cumulative bound with existing 1.25 charge-rate safety and 900-second reserve; each provider session is ≤12 hours with 600-second finalization margin. No future-week quota renewal is authorized. If insufficient, persist and pause. Existing scheduled CPU recovery independently collects and GET/SHA-verifies checkpoints/metrics and resumes only healthy unfinished Phase 1. No speculative GPU retries or Phase 2 launches.
