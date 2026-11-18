# U-Net GPU benchmark continuation

Current verified state: autonomous Actions/R2 execution passed; prepared cache independently persisted (45 required files / 11.74 GB); private Kaggle dataset `mathewzhou/novo-unet25d-benchmark-cache`, version 1, is ready; independent CPU mount validation 44166879820 passed. Reuse it. No preprocessing, publication, credential or old-session investigation is needed.

First real GPU attempt: Actions run 44167758270, integrated source `1cb04c851e94e2dd5a302c5174beb5bcc6d33fc9`. Single and DDP short forward/backward/optimizer tests passed. The longer single-card test stopped at the nonfinite loss/gradient safeguard after step 1010; neither ten-minute sustained acceptance nor full checkpoint continuation was reached. [Measurements and limits](../../experiments/unet25d/FIRST_GPU_ATTEMPT.md).

The continuation records expected FP16 overflow separately from fatal nonfinite loss, skips at most three bad-gradient optimizer updates while reducing scale, and preserves draw position in checkpoints. CPU tests use the actual GradScaler. Failed diagnostic checkpoints do not masquerade as valid recovery inputs. Architecture, learning rates, normalization, 512 resolution, split and cache bytes are unchanged.

After the tested source revision is integrated, dispatch `kaggle-benchmark.yml`, mode `benchmark`, **gpu_session_seconds=3000**. The first charge was 0.14 quota hours; the shorter second hard cap keeps the campaign within the original 60-minute allocation ceiling. This is not a fresh 60-minute budget. Do not run mode prepare. Collect/resume must preserve source and input receipts. No full-cohort training is authorized.

Accept only actual >=600-second single and DDP steady-state records, bounded process-tree memory across cache turnovers, successful complete-visit native-space evaluation and fresh-process state continuation plus R2 readback. Short data already shows substantial loader wait; do not infer that the eight-visit OS-hot cache predicts full-cohort cold-I/O throughput. Formal configuration selection stays with the owner after measured results.
