# Benchmark acceptance record — 2026-11-17

**Campaign pending; no GPU result is claimed.** Free Kaggle quota before any launch was 0.00 h used / 30.00 h remaining. Allocated session time and GPU-device time for this task are both zero. Quota accounting semantics (session time versus summed device-hours) have not been empirically established; these units are not interchangeable.

| Configuration | Local batch | Workers/rank | Samples/s | Speedup | Peak VRAM/GPU | Peak RAM | Late throughput | Data wait % |
|---|---:|---:|---|---|---|---|---|---|
| A single T4 | 1 | 1 | Not run | — | — | — | — | — |
| B DDP T4×2 | 1 | 1 | Not run | — | — | — | — | — |
| C best sustained / worker tuning | 1 | 2 | Not run | — | — | — | — | — |
| D local batch feasibility | 2 | 1 | Not run | — | — | — | — | — |

No utilization/RSS/throughput plots are fabricated. `report.py` generates figures from actual timestamped rank/system records once runs exist. Sustained reliability, DDP speedup, best useful throughput, RAM plateau, bottleneck attribution, recommended worker/cache/batch choice, and the R2 checkpoint/resume acceptance all remain open.

Independent [Kaggle CPU correctness checks](evidence/kaggle-cpu-checks.json) also passed; that kernel needs neither R2 Secrets nor imaging. This confirms CPU portability, not CUDA/DDP execution.

## CPU preparation and persistence update

The owner confirmed successful authenticated interactive CPU preparation. Do not repeat it. API inspection still exposes only private version 7 with preflight script source and zero saved outputs, so mountability is not verified. [Preservation steps, evidence and mount checker](CACHE_PERSISTENCE.md). No GPU was requested.

## Confirmed access blocker

Private CPU notebook `mathewzhou/novo-tcia-and-r2-verification` saved versions 5 and 6 returned `ConnectionError` for all four R2 Secret names. [v5 evidence](evidence/saved-preflight-v5.json) · [v6 evidence](evidence/saved-preflight-v6.json). Version 6 was the one bounded retry after the owner confirmed all four are available in the **interactive CPU** session. Record that owner confirmation separately from the agent-observed saved-run failure; do not infer either result applies to the other execution mode. No credential values were printed or copied to source/artifacts. Dataset permissions were unchanged.

Saved-run credential acquisition is therefore not verified. Desktop browser control also timed out, so the agent could not operate the owner's authenticated interactive session. There is no basis to consume GPU quota on a saved run that cannot prepare its data or persist checkpoints. Repeating identical submissions is not a remedy. A usable authenticated interactive runtime or a platform-supported saved-run secret path is required before end-to-end execution can proceed. Do not put credential values in notebook code, Kaggle outputs, GitHub or chat as a workaround.

## Confirmed geometry findings

All 1,065 existing T1c audit headers were read without imaging transfers. Native 512×512: 660; 256×256: 368; 640×640: 6; other rectangular: 31. Orientations RAS 1,058, LPS 4, LAS 3; oblique 53. Padding and explicit physical resampling are implemented, but the six 640 cases (and any oblique bounding grid >512) require a full-cohort policy decision. They cannot silently disappear from the frozen cohort.

## Continuation checklist

1. Run the [pinned CPU entry script](kaggle_cpu_entry.py) in the confirmed authenticated CPU session (the agent browser-control connection is currently unavailable), or run the prepared CPU launcher in the confirmed authenticated CPU session; retain private cache, config and receipt outputs. Resolve reported cache/geometry/preflight failures before allocating GPUs.
2. Verify mounted preparation receipts and actual two-T4 hardware, then execute the bounded campaign under a platform session cap of 60 minutes. Use no more free quota than remains.
3. Review single/DDP sustained telemetry (>=10 steady-state minutes), cache turnovers, process-tree memory, finite gradients and early/late rates. Validate native-space metrics and fresh-process checkpoint continuation/readback.
4. Publish measured comparison and figures, GPU allocation/session/device-time records and any failures. Recommend a configuration for owner review; do not start full-cohort training.

Implementation is a feature-branch candidate with CPU evidence, not a completed or approved GPU benchmark. [Specification and code map](README.md).

## Selected architecture implementation — 2026-11-18

The owner selected Actions → private Kaggle inputs → GPU → Actions. The new manual `kaggle-benchmark.yml` separates prepare, CPU validate, benchmark and collect modes. `KAGGLE_API_TOKEN` stays solely in the Actions orchestration step. GPU code uses artifact mode and contains no R2 credentials or Secrets lookup. Existing R2 independently verified caches are restored first; otherwise an immutable claim permits one bounded regeneration, persisted to R2 before private Kaggle publication. The private CPU validation must pass before benchmark submission. The API hard runtime is 3,600 seconds; GPU code has a shorter internal deadline. Results and checkpoints are collected into R2 and read back through a new client. Mode collect can retry retrieval without rerunning GPU training.

This implementation still requires actual cloud validation; it is not benchmark acceptance. The token setup has been confirmed by the owner. Integrated-source policy remains in force. PR18 and the final PR19 revision require explicit revision approvals demanded by automatic review; no notebook cells or credential values are requested. Do not investigate the old interactive session or its Secrets further. No full training is authorized.

## First cloud preparation run — upload failure

PR18/19 merged at their approved revisions. Run [44150569383](https://github.com/Strange-Noah/BrainTumour-CNNViT-Segmentation/actions/runs/44150569383), source `744d1b3cca21d19d7624df4284bce008c3242f51`, completed bounded CPU cache preparation but failed in the first chunk readback: `StreamingBody.__enter__` returned a raw HTTP response without `iter_chunks`. No complete cache manifest/independent verification was published and no GPU was allocated. The completed cache was on an ephemeral runner, not a saved public artifact. We cannot claim it is recoverable after job completion.

The fix wraps the streaming body with `contextlib.closing`; regression tests now use actual botocore `StreamingBody`, rather than a fake that masked its context-manager behavior. Per-visit normalized cache files are also persisted to output R2 before advancing to the next visit, with whole-file SHA readback and a completion receipt, scoped by the frozen configuration hash. Future recovery restores those completed visits and skips their preprocessing. This changes persistence only, not preprocessing or model configuration.

The first regeneration authorization has been consumed. The fixed recovery job is not started without explicit permission for one additional bounded recovery build. The new immutable recovery claim uses a separate v2 key rather than deleting or weakening the original claim. [Failure record](evidence/cache-upload-failure.json). GPU benchmark and performance acceptance remain open.

## Durable cache recovery — 2026-11-18

[Recovery run 44153000932](https://github.com/Strange-Noah/BrainTumour-CNNViT-Segmentation/actions/runs/44153000932) completed R2 persistence: 45 files / 11,737,028,514 bytes, independently read back in a fresh process with block and whole-file SHA-256 verification. The prepared subset is 8 train visits / 1 validation visit. Configuration SHA is `ab91d206fefff7d65ac2437f9402e9b161a5772d7341ec2fb2b6d279afec5bf4`; transfer manifest SHA is `37176945ceff56d7d8411e74a40e676a66a168f39780aea58d7659793034547e`. No more preprocessing is necessary for publication retries. [Aggregate evidence](evidence/cache-recovery-44153000932.json).

The job's publication receipt alone did not prove Kaggle dataset readiness. The initial post-publication queries returned HTTP 403, but later official API metadata confirmed the dataset exists, is private, version 1, and ready. No actual structured creation error was observed in this run. Independent CPU workflow [44164635211](https://github.com/Strange-Noah/BrainTumour-CNNViT-Segmentation/actions/runs/44164635211) failed querying the intended dataset; GPU was not submitted. A separate [cloud private-artifact retrieval 44165085593](https://github.com/Strange-Noah/BrainTumour-CNNViT-Segmentation/actions/runs/44165085593) succeeded, confirming the cloud token can read the account's existing private CPU artifacts. Do not diagnose this as missing Kaggle authentication or restart preprocessing.

Earlier proposed publication response checks were not needed to proceed: the unchanged integrated path passed once the dataset became ready. Those unrelated changes are removed from the GPU stability fix. One attempted read-only CLI error-text check was not executed because automatic approval review returned `Bad Request` (review-service failure, not an unsafe-action finding); it was not bypassed.

Independent CPU validation then passed in run 44166879820: all 38 prepared files, 8 training visits, one validation visit, exact configuration/receipt hashes and offline weights verified. The initial 403 resolved without token changes or data retransmission; its exact backend cause is not established. GPU campaign run 44167758270 was submitted only after that CPU acceptance. PR22 response checks are defensive improvements, not a prerequisite for the now-ready input.

## Actual GPU attempt and bounded continuation

[First real measurements, plots and failure analysis](FIRST_GPU_ATTEMPT.md) · [Current handoff](../../docs/longitudinal/unet-benchmark-handoff.md). Single short 5.537 samples/s, DDP short 9.443 samples/s (1.706× short speedup); the longer single run stopped after ~168 steady seconds at a combined nonfinite safeguard. No sustained acceptance is claimed. Existing cache and credentials are ready. The tested continuation uses bounded standard loss-scale recovery and a 3,000-second session cap within the remaining original campaign budget.
