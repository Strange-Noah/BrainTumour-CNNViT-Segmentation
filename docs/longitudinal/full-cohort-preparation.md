# One-time full-cohort preparation — execution record

**Status: running; not yet ready for formal training.** No GPU has been allocated.

The owner-approved PR #27 revision `af451052d0ae5d0140fbea9a9ca6ddd4b5436357` passed CI and merged as [`b35e75dabc8dde693e6cc1991168e909c5a60f58`](https://github.com/Strange-Noah/BrainTumour-CNNViT-Segmentation/commit/b35e75dabc8dde693e6cc1991168e909c5a60f58). Every CPU job pins that integrated source. No infrastructure or credential changes.

Confirmed completed visits so far: **94/905** (train 67/746; validation 27/159), per-object SHA-256 readback and immutable visit receipts. These are completed visits, not a complete-cohort readiness claim. Four existing free standard CPU jobs run concurrently. Batches reaching their normal 30-minute preparation envelope resume with completed visits skipped. No test pixels are prepared.

The one-time queue includes all remaining patient groups. After the queue succeeds, the existing CPU workflow assembles the catalogue and the existing private publisher runs in **prepare** mode only. A correctness failure stops further dispatch for diagnosis. No user notebook operation is needed. Formal training remains disabled (`formal_training_authorized=false`, budget zero).

## Confirmed CPU runs

- [Run 44492202268](https://github.com/Strange-Noah/BrainTumour-CNNViT-Segmentation/actions/runs/44492202268): train group start 0; 22 visits persisted; bounded partial batch, resumed.
- [Run 44492323799](https://github.com/Strange-Noah/BrainTumour-CNNViT-Segmentation/actions/runs/44492323799): train group start 10; 21 visits persisted; bounded partial batch, resumed.
- [Run 44492356865](https://github.com/Strange-Noah/BrainTumour-CNNViT-Segmentation/actions/runs/44492356865): validation group start 0; 27 visits persisted; bounded partial batch, resumed.
- [Run 44493010312](https://github.com/Strange-Noah/BrainTumour-CNNViT-Segmentation/actions/runs/44493010312): train group start 20; 24 visits persisted; bounded partial batch, resumed.

## Private saved-output inspection

[CPU run 44492301532](https://github.com/Strange-Noah/BrainTumour-CNNViT-Segmentation/actions/runs/44492301532) succeeded: prediction is nonempty and its native grid matches original T1c. No model inference or GPU. A three-plane T1c/prediction/ground-truth overlay and quantitative diagnostic report remain private in R2. Receipt SHA-256 `b5147ed80c91f12e038b3936166e431c1439960670172935e722f3aa12dea1a7` (714 bytes). Native header agreement alone does not establish anatomical registration. The previous Dice ~0.0216 remains a diagnostic, not satisfactory segmentation evidence. This record does not claim a human visual review of the private overlay.

[Frozen configuration, geometry, coverage, storage/loading and budget handoff](../../experiments/unet25d/cohort/README.md). Actual compressed cohort size and final catalogue/publication run IDs will be recorded after those jobs finish; they are not yet known. No full training authorization is assumed.
