# Integrated robust FP32 campaign

The owner revised execution policy: establish sustained full-FP32 training before considering precision optimization. This supersedes PR #24's selective-FP32 candidate. No mixed precision, GradScaler, TF32, architecture, learning-rate, target, split or preprocessing changes are used. The frozen preparation configuration file and existing private cache bytes stay unchanged; checkpoints record a separate **effective FP32 training configuration**, loader policy and the original preparation-config SHA.

## Direct bounded loader changes

Original patient-aware draws are generated unchanged. Each rank's fixed 64-sample window is reordered by visit (first-appearance order), then by slice position. Every original patient/visit/slice draw and its multiplicity is retained in that rank/window; this changes temporal correlation, not window sampling weights or the positive/negative mixture. Full windows are generated before truncation so checkpoint continuation has the exact same next-draw prefix. Do not silently substitute another sampler.

One persistent worker per rank, two prefetched batches, at most two mapped visits and **3 GiB of mapped files per worker**. Byte bounds additionally limit the visit count; a single visit exceeding the bound fails explicitly. Mappings are closed on eviction. No decoded whole-volume copies are held, and the redundant full-sample float32 copy was removed. Available-RAM pressure (<15%) and the absolute session deadline stop execution; RSS trends are measured without treating normal cache warmup as failure. Summed RSS can double-count shared mapped pages.

Synthetic CPU profiling (128×128, eight visits, 1,024 draws) reduced visit-map opens from **889 to 127 (85.7%)**; numeric timings are in [the receipt](evidence/fp32-loader-profile.json). These local synthetic rates are **not** MRI/Kaggle throughput. Eight tests cover FP32 updates/guards, model state and dtype, exact per-rank window multisets, resume prefixes, optimizer moment restoration, pixel/context correctness, map byte bounds, persistent worker reuse and resumable storage. Geometry/gradient/BN contracts also passed. [CPU evidence](evidence/fp32-cpu-checks.json).

## One automatic campaign

Reuse `mathewzhou/novo-unet25d-benchmark-cache`, version 1. No cache rebuild or new infrastructure/authentication steps.

1. Full-FP32 single T4: first ten real updates provide fail-fast warmup, then the same process continues to 660 seconds.
2. Full-FP32 dual-T4 DDP: the same settings and 660-second duration; no preliminary short-job campaign or parameter sweep.
3. Save a healthy final checkpoint; launch a new DDP process and perform four additional successful updates. Check strict model restoration, optimizer step, per-rank RNG and next 32 global draw positions; reject failed or incompatible checkpoints.
4. Run complete held-out-visit native-space validation from the resumed checkpoint under RAM/deadline monitoring. Fresh CPU checkpoint restoration and Actions → private R2 new-client full SHA recovery follow.

Nonfinite loss/gradients/checkpoint tensors cause a genuine stop; no speculative automatic retry. Healthy checkpoints are saved atomically during sustained execution and at the end. Failure diagnostics use a different filename so they cannot overwrite the last healthy periodic checkpoint. Actual late throughput, cache hits/misses, loader wait, available RAM, summed RSS, GPU utilization and memory remain telemetry, not predictions.

The next job has a **2,100-second platform cap**. Its absolute in-process deadline starts at the notebook entry point, including dependency/setup time, with 120 seconds reserved for finalization. Observed preceding account GPU usage was 0.33 hours; adding 35 minutes remains below the original 60-minute campaign envelope. No full-cohort training is authorized. Six oversized examinations remain a separate full-cohort policy issue and do not block this existing bounded cache.

Status: implementation and CPU checks complete; main integration and the single integrated GPU execution are the next operations. Actual GPU results must be recorded after the run, never inferred from these CPU checks.
