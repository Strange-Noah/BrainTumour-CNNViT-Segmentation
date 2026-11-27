# LesionLocator zero-shot tracking on UCSD-BMETS

## Frozen inputs

- [Official source](https://github.com/MIC-DKFZ/LesionLocator/tree/ef073baaa0dcba8dd036ec5a29f130db18c9d580), GPL-3.0.
- [Official weights record v1](https://zenodo.org/records/15174217), CC BY-NC-SA 4.0. The record publishes ZIP MD5 `1f9b8a64703346df9bcc532a0a32a07b`. Runtime acceptance requires identifying and hashing the tracking dataset, plans, and fold-0 checkpoint. Selective extraction must record member-level verification separately from full ZIP verification.
- [Canonical split](../../protocol/v1/split.json) and manifest SHA-256 pinned in the CPU entry point. Verified cohort inventory is 45 validation patients, 159 visits, and 114 adjacent pairs. Sequence lengths are 8×2, 5×3, and 32×4 visits. All 159 records reference T1c and labels. Relative day fields are absent for these rows. Follow-up order comes from visit ordinal.
- Native T1c, raw label 3, all foreground voxels, 26-connectivity. Historical reference IDs are visit-local. Expert longitudinal identities require their own evidence. Geometry-derived identities are reference proxies.

## Execution and endpoints

The CPU entry point `experiments.lesionlocator_benchmark.baseline --first 0 --count 1` checks G0 and scores L0 using the shared evaluator. Patient index is a fixed sorted order. Outputs are private and use create-once writes with hash readback. Test patients and other MRI modalities are excluded from imaging fetches. Per-visit metadata and scores stay in R2.

L1 uses the previous GT instance, while L2 carries forward the immediately preceding predicted mask after initial GT prompting. Official tracking stays intact in prev_mask mode, fold 0, mirroring disabled, and native checkpoint preprocessing. The official CUDA path activates FP16 autocast. Pin actual checkpoint plans, registration shape, model patch, normalization and export interpolation before accepting results. Tap preprocessor, z crop, 175-grid prompt, learned deformation, warped prompt, ROI and exporter as observations. Keep stable identity keyed to the input instance and track empty or skipped states independently of the official file enumeration.

G1–G5 reuse the pinned official operations to measure axis/crop, spacing, z-window, 175-grid and export loss. Any target GT used by a diagnostic must stay outside formal inference. Score L0, L1-Warp and L1-Final on common native grids with identical denominators. Record same-lesion reference coverage separately from spatial matching and whole-brain union performance. Report patient and lesion denominators, absent or ambiguous labels and resource limits. Treat stage deltas as paired observations. A loss of overlap can reflect real lesion evolution.

## Current execution evidence

The inventory above is verified from existing immutable metadata. The CPU baseline and its synthetic tests are prepared for review. The credential-bearing workstream registry currently has a single U-Net route. LesionLocator requires its own exact-source registration before a credentialed execution. Add a narrow route through protected review, then run CPU G0/L0 independently and prepare cloud-only assets. Use short-lived input/output grants in the upstream inference process. Keep permanent R2 and Kaggle credentials in the trusted orchestration process.

The tracking checkpoint, native-image pilot, G1–G5, inference and stage-wise outcome tables remain pending execution.
