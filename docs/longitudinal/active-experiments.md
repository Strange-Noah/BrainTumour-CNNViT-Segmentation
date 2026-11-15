# Active experiments

Experiment Protocol v1 is frozen in protocol/v1. The 300-patient mapping is 210/45/45, stratified by two, three and four visits. Split SHA-256 `57f5af55141e16ecb0b85fdda63cc355175d9b35271126784eb22766698b36f0`. Canonical CSV and split specification were uploaded to private R2 under `novo-nordisk/datasets/ucsd-bmets/v1/manifests/splits/<split-sha>/` and passed readback.

The shared evaluator, original-T1c prediction export, per-visit candidate centroids/volumes and patient aggregation are implemented in research/evaluate.py. The R2 loader is research/dataset.py. It retrieves only selected visits into a checksum-keyed cache. It preserves official raw labels and supplies independent visit targets for enhancing tissue (3) and core (1,3). Native preprocessing remains a model responsibility. Model owners can start from experiments/README.md and run their own checkpoint adapters via research.run_visit.

The shared metric checks passed for perfect overlap, empty-mask conventions and physical component volumes. Actual learned inference still requires the owners' checkpoint adapters. The reference-mask adapter is an explicitly labeled data/evaluation smoke, not a segmentation model.

Governance was simplified per owner instruction. Main has no merge-blocking rules, and CI remains diagnostic. Both TCIA environments retain their secrets and main-only policies with no required human approvals. Summer archive remains frozen. Documentation PRs #3 and #4 were merged.

The resumed pilot failed because the GitHub runner timed out on the collection page. A private Kaggle CPU job retrieved the official Faspex public link for the existing environment secret, avoiding dependence on that page during transfer. Persistent raw imaging and full ingestion remain unverified until the retry passes. cloud/tcia_batch.py supports bounded 1–10-patient jobs and completion-receipt recovery, to be run after the pilot. A completed patient is readback-verified before skipping. Partial jobs may re-download the current patient but conditional R2 writes never overwrite different bytes.

Full training for the model owners awaits coordination. Hosted deployment remains deferred.
