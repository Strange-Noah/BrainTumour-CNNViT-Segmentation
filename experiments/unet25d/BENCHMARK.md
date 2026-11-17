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
