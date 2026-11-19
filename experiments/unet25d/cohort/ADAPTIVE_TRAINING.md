# Adaptive two-phase formal training

Owner request: exhaustive coverage followed by training-only targeted sampling; existing free Kaggle resources only. This supersedes the old 30-pass stopping proposal and 13,440-draw training passes. The latter remains only as a backwards-compatible accepted CPU input-read helper. No MRI preparation, data-policy, architecture, precision, augmentation or loss change.

## Frozen science

SMP 0.5.0 U-Net / pinned ImageNet ResNet34; 12 channels; enhancing raw label 3; full FP32; dual T4 DDP, local/global batch 1/2. AdamW encoder/decoder LR 1e-4/3e-4, weight decay 1e-4. Full 512×512 BCEWithLogits + SoftDice, equal weights, including background. No brain mask. Test cohort remains sealed. [Configuration](config.json).

**Phase 1:** seeded shuffled patient and visit blocks, then tile and consecutive center slices. Exact immutable 220,384-position plan / 110,192 optimizer updates, strided rank partition, no padding/drop. Checkpoint cursor advances only after actual finite synchronized optimizer updates; prefetch is never consumed coverage. Plan SHA, unique count and update count persist. Full validation follows exhaustion, including when interrupted across a provider boundary.

**Phase 2:** frozen 67,200-example plans, initially baseline/hard-positive/hard-negative/ordinary-negative 40/25/20/15%. Baseline floor 30%; patient/visit caps 1%/0.5% of each cycle. Fixed multiset is sorted for visit/shard locality. All categories share these caps. A hard pool initially unavailable falls back to its GT stratum and records that fallback rather than claiming a mined example.

Normal training logits provide FN, low confidence, soft overlap, boundary error, and background peak/top-256 probabilities. Small-lesion weighting uses 3D 26-connected training-GT components with actual affine determinant, not 2D cross-section area. One existing native GT per encountered positive visit is read lazily on CPU, retained only as tiny per-position physical-volume properties. No new MRI cache or additional GPU inference campaign. Difficulty EMA .8, half-life 50,000 updates, maximum age 200,000; at most 32 records per visit/category, synchronized every 200 updates and before checkpoints. Active plan, records, RNG and control history persist.

## Complete validation and control

All 45 validation patients / 159 visits, original T1c geometry; Gaussian continuous-logit merging and restoration before threshold. Fixed primary threshold .5. Secondary .3/.7 reuse logits, never select per-patient thresholds. Full-distribution BCE+Dice averages canonical planes with all pixels/background; threshold-invariant. Existing HD95 and 26-connectivity / maximum-cardinality one-to-one positive-overlap matching retained.

Control Dice averages **GT-positive visits within each patient**, then defined patients; mutually empty visits never contribute Dice=1 to that score. Sensitivity averages GT-positive visit sensitivities per patient, then defined patients. FP/scan averages all scans per patient, then patients; empty/positive strata also reported. Small-lesion bins are predeclared <100, 100–<1,000, ≥1,000 mm³ (internal physical-volume groups, not clinical acceptance criteria): matched/total lesions within each patient/bin, then defined patients. Denominators reported. Partial validation preserves per-visit progress but cannot adapt or stop training.

Only complete validation updates sampling. At most one direction ≤10 percentage points; unchanged baseline anchor, bounded concentration, reassessment before repeated bias, revert recent failed adjustments when loss improves but quality regresses or both sensitivity and FP worsen. Improving metrics keep current ratios. FP calibration grid is diagnostic; primary threshold remains .5.

TARGET_REACHED requires positive Dice ≥.70, sensitivity ≥.80, FP/scan ≤2 for two consecutive complete validations, never before exhaustive coverage. PLATEAU_BELOW_TARGET requires ≥3 Phase 2 validation cycles and 3 consecutive intervals without meaningful nonregressive gains (Dice .01, sensitivity .02, FP reduction .5). Sampling changes do not reset patience. Targets are research operating goals, not clinical validation.

## Execution and recovery

Reuse accepted actual CPU input read from [44556878634](https://github.com/Strange-Noah/BrainTumour-CNNViT-Segmentation/actions/runs/44556878634), bound to unchanged catalogue/data policy and encoder. Refresh only the existing private small signed-input package; exact-version receipt binding remains. No MRI republication. Updated scientific identity rejects legacy diagnostic or old-policy training checkpoints.

Latest checkpoints every ten minutes and provider boundary include phase, cursor, unique coverage, frozen plan, bounded mining, adaptation/validation history, rank RNG and optimizer state. Preserve best positive-Dice and best sensitivity≥.8 / FP≤2 checkpoint separately; no acceptable trade-off checkpoint is claimed before one exists. Existing independent CPU collector SHA-verifies private artifacts, restores latest and retains best pointers. Numerical/data failures halt without GPU retry. Provider boundary resumes the same phase/cursor; it is not convergence.

Owner authorizes currently available free allowance, not unlimited renewals. Query quota before each submission. Safety conversion starts conservatively at 1.25 reported quota-hours/session-hour: prior accepted dual-T4 campaign charged .45h for at least 1,305.61 measured compute seconds (upper observed rate <1.25). Reserve 900 reported quota seconds; maintain a cumulative reserved-session guard, ≤12h provider sessions, tighten rate from actual charge observations. Renewal requires fresh allowance; exhausted allowance reports PAUSED_RESOURCE_LIMIT, never scientific success. Actual device-hours are reported separately. No paid resources.

After integrated source and CPU checks pass, launch through owner-authorized existing workflow. Observe ten minutes of actual optimizer updates, report both GPU utilization, throughput, steps/s, RAM/loading, finite updates, coverage/checkpoint status and conservative ETAs; then end active supervision. Existing half-hourly CPU collector continues independently within recorded allowance.

## Evidence and current status

[Minimal CPU acceptance](ADAPTIVE_ACCEPTANCE.json). Source, CI and actual execution links are added with publication/run evidence. The CPU checks allocated no GPU. Formal training subsequently started from reviewed main; [actual first-ten-minute results and independent handoff](FORMAL_EXECUTION.md). GPU_TRAINING_STARTED=true. [Project handoff](../../../docs/longitudinal/handoff.md).
