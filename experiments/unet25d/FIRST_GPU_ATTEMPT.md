# First real T4×2 attempt — partial measurements, not acceptance

[Actions run 44167758270](https://github.com/Strange-Noah/BrainTumour-CNNViT-Segmentation/actions/runs/44167758270) used integrated source `1cb04c851e94e2dd5a302c5174beb5bcc6d33fc9`, the verified private cache version 1 and real train data. Full-network forward, backward and AdamW updates ran at 12×512×512, with FP16 and the fixed BatchNorm policy. Both devices were Tesla T4s according to nvidia-smi telemetry. No synthetic timing substitutes or test-label evaluation were used.

| Configuration | Batch/GPU | Workers/rank | Samples/s | Steady measurement | Peak GPU memory used (MiB) | Peak summed RSS (GiB) | Data wait |
|---|---:|---:|---:|---:|---|---:|---:|
| A short, single T4 | 1 | 1 | 5.537 | 40 updates / 7.22 s after warm-up | 731 / idle second device | 4.36 | 65.5% |
| B short, DDP T4×2 | 1 | 1 | 9.443 | 40 updates per rank / 8.47 s | 917 / 917 | 8.43 | 35.7% |
| A attempted sustained | 1 | 1 | 5.956 | 167.90 s, stopped | 731 / idle second device | 4.94 | 64.6% |

Short-to-short speedup is **1.706×**. This is not a measured sustained DDP speedup. The single-card longer segment's late throughput was 5.869 samples/s and late/middle RSS ratio 0.929; no progressive RSS growth was seen in that limited window, but it is far shorter than the required >=600 seconds. Two-GPU sustained operation, worker tuning, batch 2, full-visit validation and training resume have not passed.

The original combined nonfinite-loss/gradient safeguard stopped A at the next update after step 1010. Its log did not distinguish bad loss from FP16 gradient overflow, so the precise numeric failure is not proven from that message alone. No OOM or memory guard stop occurred. We must not claim model/RAM acceptance or fix this by silently changing architecture, resolution or learning rates.

The bounded remedy distinguishes nonfinite loss (fatal) from nonfinite scaled gradients with finite loss. In the latter case all ranks skip the optimizer update, halve the loss scale and clear gradients, at most three times. Every skipped update is logged and excluded from useful samples/sec. Sampler draw position and overflow count are checkpointed separately from optimizer steps. Real CPU GradScaler tests verify no weight update on overflow, successful next update after scale reduction, and fatal/cap behavior. GPU stability of this remedy remains unverified.

[Aggregate and compressed telemetry](evidence/gpu-attempt-1/) · [GPU/RAM/data-wait/cache figures](../../docs/longitudinal/figures/unet25d/). Data wait dominates measured iteration time; isolated NCCL overhead is not measurable from backward-including-sync timers. RSS sums can double-count shared mappings; PSS/available-memory traces are retained. GPU memory values above use sampled nvidia-smi; CUDA allocator peaks are separately recorded in performance.json.

The controller collected 38 private artifacts into output R2 and verified every SHA using a new client. Its collection receipt is `novo-nordisk/experiments/unet25d/bridge/44167758270/collection.json`, SHA `58c18ca17f25d04b3f7eaef1c61f80b04f45e66b37e06522cc70576100505271`. Full checkpoint deserialization/continued training was not reached, so hash recovery must not be called training-resume acceptance.

Kaggle quota changed from 0.00 to **0.14 hours** (29.86 remaining). Quota accounting is not assumed to equal summed GPU-device hours. Instrumented device timestamps cover 344.14 seconds (5.74 minutes), giving a two-device time lower bound of 0.1912 GPU-hours; container startup and output finalization are outside that telemetry. The next session will be capped at **3,000 seconds (50 minutes)**, keeping the first quota charge plus the new hard session cap below the authorized 60-minute campaign budget with margin. No full training is authorized; no cache regeneration or credential change is needed.
