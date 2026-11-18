# Full-FP32 execution handoff — campaign complete

[Measured results, plots, exact configuration and receipts](../../experiments/unet25d/INTEGRATED_RESULTS.md).

One GPU job, run 44292134002 at source `d8ac3240b58a193cce79636695153396f9bdddd2`, achieved 8.439 samples/s single T4 and 15.403 dual T4 (1.825×) over >600 steady seconds each, with finite training and stable RAM. Loader wait was 0.343%/0.799%. A new DDP process resumed 5,076 → 5,080 with model/optimizer/RNG/sampler checks. Native inference completed; final export failed on a missed shared metadata contract. Do not describe the original Actions run as successful.

Approved CPU-only PR #25 fixed the export contract. Run 44333112852 at source `8a0276a96e8f313c28a72897b53b4d9708d93a17` successfully restored three checkpoints, exported the existing saved logits to native NIfTI and independently read back private outputs. No new GPU, training or inference occurred. GPU collection: 53 private files; CPU report SHA `b4bf5e24d76622ac84fa214a3177f6acefe6fd9391a816897f97fdfbb92d87d1`.

Recommend the tested full-FP32 dual-T4 configuration: local batch 1/global 2, one persistent worker/rank, prefetch two, at most two mappings/3 GiB per worker, 64-draw exact-multiset locality windows, unchanged input/split/loss/LRs. Do not introduce more precision tuning or rebuild cache version 1. One held-out visit is an inference/export check, not model-quality acceptance.

GPU quota after all attempts: 0.78 hours used / 29.22 remaining. Stop GPU work; full-cohort training is not authorized. Full-run authorization and resource schedule, full-cohort prepared inputs, and oversized/native-grid policy remain separate decisions. Do not repeat this benchmark automatically. Credentials remain in Actions, dataset access read-only, outputs private in the permanent output bucket. Summer archive and repository governance are unchanged.
