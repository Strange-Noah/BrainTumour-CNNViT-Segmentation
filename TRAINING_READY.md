# TRAINING_READY = true

Finalized 2026-11-19. **No GPU was allocated. Formal training authorization remains false and approved budget remains zero. The only remaining action is the owner's total GPU-budget decision.** No new preparation project or discretionary validation is needed after that decision.

## Data and actual acceptance

| Partition | Patients | Prepared visits |
|---|---:|---:|
| Training | 210/210 | 746/746 |
| Validation | 45/45 | 159/159 |
| Total | 255/255 | 905/905 |

Seventy bounded CPU preparation runs completed without a failed preparation run, with at most four standard free CPU jobs concurrent. Completed immutable receipts were reused, including nine compatible original normalized visits; the last unstarted/partial ranges were split by patient within that same limit. Test imaging was never prepared. Original TCIA ingestion/audit were not repeated.

The single final catalogue assembly passed: missing/duplicated visits, incomplete slice/tile ranges and conflicting object identities all zero; all four oversized and all 51 oblique examinations present. All four modalities and raw labels are represented. 640×640 uses the exact four approved windows; Gaussian-weighted continuous logits are merged before native-T1c restoration and thresholding. Native T1c oblique affine/spacing is retained; differing image modalities use documented physical-coordinate linear interpolation. No resolution or field-of-view reduction.

Private R2 catalogue identifier:

`<private catalogue locator retained in owner handoff>`

SHA-256: `04b59724c5cc6ac654f85d51ad19e8a8a0766944d8211ecfd03a3495a418de1e`; 16,609,873 bytes. Frozen patients.csv SHA-256: `57f5af55141e16ecb0b85fdda63cc355175d9b35271126784eb22766698b36f0`.

**Actual distinct referenced compressed data: 66,057,921,137 bytes = 66.058 GB decimal / 61.521 GiB**, excluding catalogue/receipt metadata and unrelated previous outputs. There are **32,308 logical axial shard ranges, 25,384 distinct compressed shard objects and 905 native-label objects** (26,289 distinct objects). Identical compressed content can serve multiple recorded ranges. Summing logical references gives 67,385,116,201 bytes; that is not the distinct-object storage footprint. Integrity uses SHA-verified immutable preparation receipts and object identities; no routine full-data rehash.

Private Kaggle input: **version 3**, verified; the account/reference is retained in the private owner handoff. The publisher binds the exact version by a private publication receipt, not an eventually consistent search version number. Catalogue, configuration and lease bytes matched the published private package. The actual **TrainingInput** reader resolved that package and read one real, finite CPU batch: input **[1, 12, 512, 512]**, binary target **[1, 1, 512, 512]**. No inference/model update/GPU. Pinned encoder SHA-256 `6549ccd46a2d20a0a74fca866cfc5c6f9b3672d468b0a92aa94a58545a61103e` was verified and included with its configuration.

Private CPU-readiness receipt: `<private readiness locator retained in owner handoff>`; SHA-256 `cc203c0490b92ca8c1b58038012d989744a6651ae16e6fb960627bfaccfb3e30`, 606 bytes. Dataset objects, pixels, signed URLs, model outputs and credentials remain private. A later launch automatically refreshes expiring private GET-only leases and resume dependencies; it reuses the accepted catalogue and this input check.

## Reviewed execution ready

Integrated training source: [`c4b70a933ce5f36d41798f9051b3db197185b162`](https://github.com/Strange-Noah/BrainTumour-CNNViT-Segmentation/commit/c4b70a933ce5f36d41798f9051b3db197185b162). Scientific configuration: full FP32; SMP U-Net/ResNet34; pinned ImageNet initialization, never the diagnostic checkpoint; dual T4 DDP; local/global batch 1/2; four modalities × three slices; label 3; AdamW encoder/decoder LR 1e-4/3e-4; equal BCE/soft Dice. No precision, augmentation or architecture experiment. Patient-aware coverage and deterministic sampler/model/optimizer/RNG continuation remain frozen.

[Kaggle documents a 12-hour CPU/GPU session maximum](https://www.kaggle.com/docs/notebooks#technical-specifications). Formal sessions request the smaller of that maximum and remaining cumulative owner budget, with ten minutes reserved internally for checkpoint/finalization and initialization counted in-session. No artificial hourly restart. At most 30 coverage passes; every pass has 13,440 examples / 6,720 updates and guarantees patient/visit/tile coverage, not exhaustive axial-slice training.

After authorization, existing Actions launches the private Kaggle session and observes approximately ten minutes of actual optimizer updates: finite loss/gradients, global throughput, loading wait, both GPUs' utilization/memory and host RAM. It reports the measured health, kernel/source/input identity and approximate end time, then the agent ends active supervision. Training continues independently.

Checkpoints every ten minutes and session end; complete 45-patient/159-visit native-space validation every five passes; best checkpoint uses complete patient-mean Dice. Partial scheduled validation resumes from the same model. No test evaluation. The existing workflow's active half-hourly CPU collector (`17,47 * * * *`) recovers terminal Kaggle outputs to private R2 with SHA readback, advances the resume pointer only after persistence, and continues a successful campaign only within its original cumulative budget. No active authorization record means no GPU submission. Correctness errors stop continuation; two collection errors block blind retries. Whole requested session windows are reserved conservatively, so interruptions do not reset the budget.

[GitHub-hosted jobs have a six-hour cap](https://docs.github.com/en/actions/reference/limits). Our initial Actions job keeps its 100-minute cap because multi-hour GPU work no longer depends on that job staying alive. The narrow scheduled collector operates independently of the chat/launch job. The workflow is deployed and active; CPU control/data tests pass. **No new GPU or 12-hour empirical GPU test was performed.** Scheduled delivery can be delayed by GitHub, so recovery timing is not an exact-time guarantee.

Exact authorized launch entrypoint is the existing **kaggle-benchmark.yml, mode=formal**; an example for a future explicitly approved eight-hour cumulative budget:

```sh
gh workflow run kaggle-benchmark.yml --repo Strange-Noah/BrainTumour-CNNViT-Segmentation --ref main \
  -f mode=formal -f dataset=<private-input-reference> \
  -f kernel=<private-kernel-reference> \
  -f formal_training_authorized=true -f approved_gpu_budget_seconds=28800
```

Do not execute that command before budget approval. [Operational handoff](https://github.com/Strange-Noah/BrainTumour-CNNViT-Segmentation/blob/c4b70a933ce5f36d41798f9051b3db197185b162/experiments/unet25d/cohort/EXECUTION.md). The user-approved unattended policy is also saved in global AGENTS.md. No buckets, credentials, provider, environment/main protections or archived summer history were changed.

## Budget: measured reference versus projections

The measured accepted bounded-cache throughput is **15.403 examples/s**. Full-cohort throughput has not yet been measured and no benchmark was launched for this estimate.

| Coverage passes | Examples / updates | Compute-only lower reference | Full validation rounds | Projected total dual-T4 session hours | Physical two-device hours |
|---|---|---:|---:|---:|---:|
| 10 | 134,400 / 67,200 | 2.42 h | 2 | 6–14 h | 12–28 |
| 20 | 268,800 / 134,400 | 4.85 h | 4 | 12–27 h | 24–54 |
| 30 | 403,200 / 201,600 | 7.27 h | 6 | 18–40 h | 36–80 |

Metadata-only replay of the frozen first-pass plan and actual cache bounds predicts **27,239,432,882 compressed cold-read bytes, 8,820 GETs and 410,732,855,296 decoded bytes** across both ranks. One full validation cold-reads about **11,650,348,444 bytes**. No imaging or GPU was used for those calculations; subsequent sampled passes vary.

The final receipt-assembly step took 365 seconds for about 905 receipt reads plus assembly. This shows that request latency must be budgeted; it does not measure Kaggle transfer speed. A planning scenario of 0.2–0.4 s/request, 50–100 MiB/s aggregate transfer and 300–600 MiB/s aggregate decoded throughput gives about 30–60 minutes of cold input service per pass. Prefetch overlaps some input work with GPU work; do not simply add every service estimate as independent elapsed time. Full validation allows 30–90 minutes per round (1–3 / 2–6 / 3–9 hours for the three plans). Allow roughly 5–15 minutes startup/transition per provider session and 1–3% checkpoint overhead. The ten-minute finalization margin is counted inside each reserved session. Artifact recovery is CPU-side, roughly 5–30 minutes depending on output volume, and consumes no GPU session hours.

These conservative ranges revise the initial 12–24-hour planning envelope using real cohort size/request counts. They remain projections, not guaranteed completion times. **Recommend an initial cap of 8 dual-T4 session hours (16 physical T4 device-hours)**, targeting the first complete five-pass validation; ten passes are not guaranteed. Recommend an initial cumulative ceiling no higher than **24 session hours / 48 physical device-hours** before reviewing real research results; this cap does not promise all 30 passes and is not approved by this report. The first formal run supplies the actual streamed-data rate.

Current provider quota was read privately and is sufficient for the recommended first eight-hour session under a conservative two-device interpretation. Provider quota units are separate from physical device-hours. Larger cumulative budgets require available quota; no paid extension or subscription is authorized.

## Direct evidence

- [Accepted full preparation implementation PR #27](https://github.com/Strange-Noah/BrainTumour-CNNViT-Segmentation/pull/27); [CPU execution summary](docs/longitudinal/cpu-preparation-summary.json).
- [Final catalogue/completeness run 44547458632](https://github.com/Strange-Noah/BrainTumour-CNNViT-Segmentation/actions/runs/44547458632): exact cohort, all four oversized / 51 oblique, zero missing/duplicates/conflicts.
- [Successful private publication and real CPU batch read 44556878634](https://github.com/Strange-Noah/BrainTumour-CNNViT-Segmentation/actions/runs/44556878634): version 3, finite actual batch and persisted receipt.
- [Reviewed long-session PR #28](https://github.com/Strange-Noah/BrainTumour-CNNViT-Segmentation/pull/28), [visibility fix PR #29](https://github.com/Strange-Noah/BrainTumour-CNNViT-Segmentation/pull/29), [exact-version binding PR #30](https://github.com/Strange-Noah/BrainTumour-CNNViT-Segmentation/pull/30), [26-test source CI](https://github.com/Strange-Noah/BrainTumour-CNNViT-Segmentation/actions/runs/44556158115).
- [Machine-readable readiness](docs/longitudinal/training-ready-evidence.json), [storage/I/O projections](docs/longitudinal/training-budget.json).

The two earlier CPU publication attempts stopped before imaging-batch execution (temporary status visibility, then wrong-version control-file mismatch). Both were diagnosed and fixed without re-preparation or GPU usage. The final real batch check above succeeded once. No scientific quality claim is made from the old single-visit Dice ~0.0216; its diagnostic overlay remains private.
