# Active experiments

Experiment Protocol v1 is frozen in protocol/v1. The 300-patient mapping is 210/45/45, stratified by two, three and four visits. Split SHA-256 `57f5af55141e16ecb0b85fdda63cc355175d9b35271126784eb22766698b36f0`. Canonical CSV and split specification were uploaded to private R2 under `novo-nordisk/datasets/ucsd-bmets/v1/manifests/splits/<split-sha>/` and passed readback.

The shared evaluator, original-T1c prediction export, per-visit candidate centroids/volumes and patient aggregation are implemented in research/evaluate.py. The R2 loader is research/dataset.py. It retrieves only selected visits into a checksum-keyed cache. It preserves official raw labels and supplies independent visit targets for enhancing tissue (3) and core (1,3). Native preprocessing remains a model responsibility. Model owners can start from experiments/README.md and run their own checkpoint adapters via research.run_visit.

The shared metric checks passed for perfect overlap, empty-mask conventions and physical component volumes. Actual learned inference still requires the owners' checkpoint adapters. The reference-mask adapter is an explicitly labeled data/evaluation smoke, not a segmentation model.

Governance was simplified per owner instruction. Main has no merge-blocking rules, and CI remains diagnostic. Both TCIA environments retain their secrets and main-only policies with no required human approvals. Summer archive remains frozen. Documentation PRs #3 and #4 were merged.

The resumed pilot failed because the GitHub runner timed out on the collection page. A private Kaggle CPU job retrieved the official Faspex public link for the existing environment secret, avoiding dependence on that page during transfer. The retry passed. All 43 pilot files are stored and independently readback-verified in R2. As of 2026-11-16, 40 patients have completion receipts. R2 contains 1,417 raw objects, 23,086,652,241 bytes across 133 examination prefixes. cloud/tcia_batch.py supports bounded 1–10-patient jobs and completion-receipt recovery, to be run after the pilot. A completed patient is readback-verified before skipping. Partial jobs may re-download the current patient but conditional R2 writes never overwrite different bytes.

Full training for the model owners awaits coordination. Hosted deployment remains deferred.

## Confirmed real-data evidence

The real loader returned a 4×512×512×184 first visit from R2, with raw mask values 0,2,3. Both task mappings passed reference-mask evaluation and prediction/lesion candidate export on the real geometry. These identity smoke metrics (Dice 1, HD95 0, lesion sensitivity 1, zero false positives and zero volume error) validate wiring only. Learned model inference has not run.

- [43-file upload and independent verification](https://github.com/Strange-Noah/BrainTumour-CNNViT-Segmentation/actions/runs/43713004163)
- [Real loader, evaluator and export run](https://github.com/Strange-Noah/BrainTumour-CNNViT-Segmentation/actions/runs/43713470661)
- [First progressive patient batch](https://github.com/Strange-Noah/BrainTumour-CNNViT-Segmentation/actions/runs/43713512344)

Use `python -m research.evaluate_files CASES.csv OUTPUT.json --task enhancing` to evaluate actual model predictions. The input CSV joins patient and exam IDs to original-grid prediction and raw truth filenames. It produces visit metrics, patient means, macro patient means and defined-metric counts.

## Ingestion recovery on 2026-11-16

Patient ranges 0–9, 10–19, 20–29 and 40–49 completed. The start=30 and start=50 jobs failed at root-level Aspera browse, before their patient transfers. The recovery implementation derives its 1,065 exam folders from the checksum-verified canonical manifest, avoiding repeated whole-release source listing. Recognized transient Aspera errors receive at most three attempts with backoff. Failed commands report sanitized indicators only.

The interrupted ranges are retrying on the existing free runners, with already-completed patient receipts reused. Subsequent patient ranges continue in bounded batches. [Retry 30–39](https://github.com/Strange-Noah/BrainTumour-CNNViT-Segmentation/actions/runs/43735419540) · [Retry 50–59](https://github.com/Strange-Noah/BrainTumour-CNNViT-Segmentation/actions/runs/43735423231) · [Machine-readable availability snapshot](../../protocol/v1/ingestion-status.json)

## Owner integration confirmation and progressive availability

The owner confirmed Zihan/Kaile (CNN–ViT), Yilong (Swin UNETR) and Qiu (nnU-Net v2) integrate independently. The first run uses enhancing tissue (raw 3). Owners select checkpoint, framework and compute when ready and record those with model configuration and Git SHA through the run metadata file. Core (1+3) remains optional secondary work. No checkpoint collection is required before owners start integration.

Latest live availability snapshot has 70 complete patient receipts, 2,601 raw objects and 38,716,196,788 bytes. Object totals may include partially uploaded subsequent patients. Four disjoint ten-patient jobs are active; the coordinator replenishes completed ranges and halts new scheduling on an unexpected failure. Canonical R2 completion receipts and GitHub run history support recovery without a local imaging copy. [Current snapshot](../../protocol/v1/ingestion-status.json)

The full 300-patient collection is still in progress. Learned architecture results are pending each owner's first checkpoint run. The reference-mask identity metrics remain data/evaluation/export verification.

## Full ingestion reconciliation

All 300 patient receipts and 1,065 canonical examinations now reconcile to 11,405 R2 raw objects totaling 175,705,024,365 bytes (about 163.64 GiB). No required modality or segmentation is missing. The stored object keys and sizes agree with the completion receipts; no unexpected objects or invalid receipts were found. Original scans and masks were preserved.

[Machine-readable final ingestion reconciliation](../../protocol/v1/final-ingestion.json). The complete MRI/header/label audit runs separately; ingestion completion does not imply that every audit is finished.
