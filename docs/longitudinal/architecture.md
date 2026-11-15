# Cloud architecture and access

TCIA → approved Linux staging → private Cloudflare R2 → private Kaggle sessions → R2 and GitHub. GitHub deploys the viewer package to the project owner's Hugging Face Space. The Space operates on synthetic examples or user-provided volumes. It requires no R2 credentials.

| System | Access and responsibility |
|---|---|
| GitHub | Source, review, manifests' format and hashes, environment pins and aggregate reports. Each contributor branches from the reviewed baseline. |
| Cloudflare R2 | Set `R2_ENDPOINT_URL`, `R2_BUCKET`, `R2_ACCESS_KEY_ID`, and `R2_SECRET_ACCESS_KEY` in the runtime secret store. Keep canonical data and manifests in private `novo-nordisk-pbl`, with read-only access from Kaggle. Write all experiment outputs to private `novo-nordisk-pbl-outputs` using separate bucket-scoped credentials. The owner will provision the output bucket and credentials. |
| Kaggle | Each contributor uses their own private notebook, Internet enabled and a free GPU when Gate D is reached. Attach the R2 secrets to that notebook via Kaggle Secrets. |
| Hugging Face | A dedicated Space on the project owner's login, using CPU Basic and the Gradio SDK once the owner account has the required plan. Set repository variable `HF_SPACE_ID` and the `hugging-face` GitHub environment secret `HF_TOKEN` with access scoped to that Space. |

Use these prefixes under `novo-nordisk/`. The datasets prefix belongs in `novo-nordisk-pbl`. Experiments, checkpoints, predictions, reports, and new output verification artifacts belong in `novo-nordisk-pbl-outputs`.

- `datasets/ucsd-bmets/v1/{raw,metadata,processed,manifests}/`. Raw and metadata preserve source bytes. Processed data lives under a recipe digest. Reviewed manifests are immutable and addressed by digest.
- `experiments/<method>/<run-id>/`. Use UTC timestamp, owner handle and a random suffix as the unique run ID. Keep config, code SHA, input manifest SHA, partition SHA, runtime versions and metrics together.
- `checkpoints/<run-id>/`, `predictions/<run-id>/`, `reports/<run-id>/`. Record output sizes and SHA-256. Upload artifacts before publishing the completion report. On a retry, accept an existing object only after SHA-256 readback agrees. For outputs produced nondeterministically, use a new run ID after a conflict.
- `verification/<run-id>/`. Synthetic probes and receipts.

Conditional PUT protects cooperating tools. Review bucket locks, access controls and disaster recovery before declaring the raw prefix immutable against all clients. The script uses independent GET and SHA-256 because S3 ETags can vary by upload mode.

At the published Standard rate, 163 GB costs roughly USD 2.45/month for raw bytes before any available free tier, or USD 2.30 if all 10 GB-month remain. Requests and derived data are additional. Direct R2 egress is free. The account's free-tier consumption and staged dataset size remain unverified. [R2 pricing](https://developers.cloudflare.com/r2/pricing/)

Kaggle should stream or cache cohorts of complete patients with local disk limits, using SHA-256 as the cache key. Use R2 as the canonical store. For repeated full-cohort work, benchmark immutable patient shards or versioned private Kaggle datasets partitioned into working subsets, subject to the platform's observed limits and TCIA terms. Avoid duplicating all 163 GB into each session. Benchmark listing, first-byte latency, request count, sequential throughput, GPU memory and free disk before choosing shard size. Private Kaggle caches are replaceable and must reference a manifest digest.
