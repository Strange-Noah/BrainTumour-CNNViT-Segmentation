# Integrated FP32 benchmark handoff

The owner replaced selective precision debugging with a full-FP32 robust baseline and direct loader improvements. [Implementation, CPU evidence and exact campaign plan](../../experiments/unet25d/INTEGRATED_FP32.md). PR #24 is being revised to this strategy; its previous selective-FP32 head must not be used.

Next execution: integrate tested source, then dispatch the existing `kaggle-benchmark.yml`, mode `benchmark`, `gpu_session_seconds=2100`, from main. Run one continuous campaign: single T4 660 seconds, DDP 660 seconds, new-process four-update continuation, native-space validation, CPU state restoration and independent private R2 artifact recovery. First actual steps are fail-fast warmup inside the sustained process. Healthy stages continue automatically. Genuine numerical/resource failure stops the job without speculative relaunch. No full training.

Reuse the verified private cache version 1 (45 persisted files / 11.74 GB); do not rebuild preprocessing, republish inputs or revisit old interactive Secrets. Full-FP32 effective training config and loader policy are checkpointed separately from the unchanged preparation receipt. Sampling preserves each rank's 64-draw-window multiset; cache bounds are two mappings and 3 GiB per worker, one worker/rank, prefetch two, persistent workers.

Prior GPU account use was 0.33 hours / 29.67 remaining; the next platform cap is 35 minutes inside the original 60-minute campaign budget. Do not reset the budget. Oversized full-cohort MRI handling and hosted deployment do not block this bounded cache. Raw dataset credentials remain read-only; output credentials remain in Actions, never the GPU notebook.

Historical [attempt 3 results and readback receipts](../../experiments/unet25d/ATTEMPT_3.md): run 44255895457 stopped at 835 updates, single short 5.995 samples/s, DDP short 10.072; none establishes sustained performance. All 38 diagnostic files passed new-client R2 SHA readback. CPU diagnostic version 5 located residual overflow and its 3 outputs were recovered in Actions run 44261391821. No previous failed checkpoint is valid for formal resume acceptance.

## Export contract correction found during the run

Run 44292134002 is executing the approved immutable full-FP32 source `d8ac3240b58a193cce79636695153396f9bdddd2`. Review found that final export omits the required split SHA and passes an existing destination. Preserve the useful training/continuation/inference outputs; do not modify or restart the running GPU source. A [CPU-only export recovery](../../experiments/unet25d/CPU_EXPORT_RECOVERY.md) is prepared with three synthetic contract tests. It must be separately reviewed/integrated before using existing Actions credentials. It can reuse verified saved logits only when sustained training, resume and inference are complete; it allocates no GPU and performs no new inference/training.
