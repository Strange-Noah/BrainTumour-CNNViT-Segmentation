# Formal training execution and recovery

The scientific configuration and data recipe in [config.json](config.json) remain frozen. `formal_training_authorized=false` and `approved_budget_seconds=0`; no GPU may be submitted until the owner authorizes a cumulative budget. Operational authorization is supplied by the owner-controlled manual workflow inputs, so that later budget approval requires no new preparation or code revision.

## Provider limits and project limits

[Kaggle's current specifications](https://www.kaggle.com/docs/notebooks#technical-specifications) document 12 hours per CPU/GPU session. The pinned Kaggle 2.2.4 CLI accepts a requested timeout in seconds; [official CLI documentation](https://github.com/Kaggle/kaggle-cli/blob/main/docs/kernels.md) describes it. Our earlier 3,600-second source guards, 4,500-second polling deadline and 2,100-second benchmark default were project bounds, not provider limits.

Formal sessions now request `min(43,200 seconds, remaining cumulative authorized budget)`. Training reserves ten minutes inside that session for checkpoint/finalization, while initialization also counts against the window. The scientific maximum remains 30 coverage passes and six complete validation rounds. No forced hourly restart. A final budget remainder under 20 minutes is left unused rather than spending it on setup with little useful training.

[GitHub-hosted jobs have a six-hour maximum](https://docs.github.com/en/actions/reference/limits). The existing bridge keeps its 100-minute CPU job timeout because it no longer waits for a multi-hour Kaggle session: it publishes/launches and observes the first approximately ten minutes of actual CUDA optimizer updates, then exits. A narrow `17,47 * * * *` schedule in the same workflow checks the one private active-run record and collects terminal outputs. Scheduled delivery may be delayed by GitHub; it is not an exact-time completion guarantee. This is not an independently deployed scheduler.

## Launch after an explicit budget decision

Only the owner account may authorize a formal launch. For a hypothetical **approved** eight-hour cumulative budget (not authorized by this document):

```sh
gh workflow run kaggle-benchmark.yml \
  --repo Strange-Noah/BrainTumour-CNNViT-Segmentation --ref main \
  -f mode=formal \
  -f dataset=mathewzhou/novo-unet25d-full-cohort-input \
  -f kernel=mathewzhou/novo-unet25d-full-cohort-training \
  -f formal_training_authorized=true \
  -f approved_gpu_budget_seconds=28800
```

The formal path ignores the legacy benchmark session default and derives its session from the documented provider cap and cumulative budget. The run records integrated source SHA, catalogue SHA, pinned private input version, campaign/session identity, requested timeout and owner budget. Unique private kernel names prevent a later session overwriting uncollected outputs. Source and scientific configuration remain pinned across resumed sessions.

## Initial observation and independent continuation

A successful real optimizer update emits a sanitized `training_update` record; fresh training emits step 1 immediately and later updates every 100 steps. It includes finite loss/gradient status, sustained global samples/s, loading wait, host RAM availability, CUDA allocated/peak memory, and both GPUs' utilization/memory from `nvidia-smi` when available. No patient identifiers, images, private object URLs or secrets are printed. The existing API's live kernel log reads supply the initial observation.

After approximately ten minutes from first observed actual computation, Actions reports the latest measured health, expected session end and direct private kernel link. The conversational agent reports this and ends active supervision. Closing the chat or ending Actions' initial job does not stop the Kaggle subprocess. The session itself enforces finite loss/gradients/checkpoint state, actual memory pressure, ten-minute atomic checkpoints, deterministic sampler/model/optimizer/RNG state, budget/session deadlines and validation every five complete coverage passes. It never resumes the diagnostic low-Dice checkpoint.

## Persistent completion recovery

The existing private output bucket holds `novo-nordisk/experiments/unet25d/full-training-active.json`. It contains only this project's explicitly authorized active campaign, reserved cumulative timeout and collection history. No active record, a completed budget, a completed model or a blocked correctness failure makes scheduled execution a no-op. The workflow uses serial concurrency to prevent duplicate collection/submission; no pull-request or push event receives these secrets.

On terminal Kaggle status, the existing collector transfers checkpoints, validation predictions/metrics, telemetry and provenance to private R2 and independently SHA-verifies its writes. It updates the existing resume pointer only after verified output persistence. Successful sessions may continue from that checkpoint, using a refreshed private GET-only data lease and input version, only while reserved session time remains in the original cumulative owner budget and the maximum passes/final validation are unfinished. Whole requested session windows are reserved conservatively, including initialization and margins; this prevents an interrupted process from silently resetting the budget. Increasing a completed campaign's budget requires another explicit **cumulative** owner authorization, not a fresh zeroed counter.

An ambiguous interrupted GPU submission is not blindly resubmitted. A numerical/provider failure is collected and stops GPU continuation. Two collection failures block blind retries. Manual recovery uses the same workflow with `mode=recover`; it does not create a new GPU run unless an already authorized successful campaign genuinely needs its next provider-bounded session. No new provider, bucket, environment, permission or long-lived GPU secret is introduced.

## Final preparation, once

The existing four-way CPU preparation continues using PR #27's immutable source and the exact same data-policy SHA. Only after all batches finish does `cohort.index` read the SHA-verified visit receipts, check exact 746/159 unique visits, complete axial/tile coverage, four modalities/raw labels and all four oversized / 51 oblique examinations, and persist the catalogue and actual storage/shard totals. This uses immutable preparation evidence, not another full image audit or data rehash.

The existing CPU publisher's `mode=prepare` publishes that catalogue, configuration, pinned encoder dependencies and private expiring data leases. It then downloads the versioned private input's three small control files and constructs **one actual finite CPU training batch through `TrainingInput`**. The private batch checksum and sanitized shape/readiness receipt are saved in R2. Later authorized launches refresh only expiring leases/resume dependencies; they reuse that catalogue and successful input check. Medical pixels and private URL leases stay off GitHub.

The final evidence/actual totals are recorded in [PR #28](https://github.com/Strange-Noah/BrainTumour-CNNViT-Segmentation/pull/28). Do not claim `TRAINING_READY=true` until those real finalization steps succeed and this reviewed execution source is integrated.
