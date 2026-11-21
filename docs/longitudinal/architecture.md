# Cloud architecture and access

TCIA → approved Linux staging → private Cloudflare R2 → private Kaggle sessions → R2 and GitHub. GitHub deploys the viewer package to the project owner's Hugging Face Space. The Space operates on synthetic examples or user-provided volumes. It requires no R2 credentials.

| System | Access and responsibility |
|---|---|
| GitHub | Source, review, manifests' format and hashes, environment pins and aggregate reports. Each contributor continues their existing logical workstream; shared changes are selectively reviewed into main. |
| Cloudflare R2 | Set `R2_ENDPOINT_URL`, `R2_BUCKET`, `R2_ACCESS_KEY_ID`, and `R2_SECRET_ACCESS_KEY` in the runtime secret store. Keep canonical data and manifests in private `novo-nordisk-pbl`, with read-only access from Kaggle. Write all experiment outputs to private `novo-nordisk-pbl-outputs` using separate bucket-scoped credentials. Both private buckets and the authorized runtime credentials are already provisioned. |
| Kaggle | The owner workstream uses accepted GitHub Actions → private Kaggle inputs → GPU → GitHub Actions recovery, without manual notebook operations. Teammate runtimes remain independently authorized; model-specific configuration stays on their own workstreams. |
| Hugging Face | A dedicated Space on the project owner's login, using CPU Basic and the Gradio SDK once the owner account has the required plan. Set repository variable `HF_SPACE_ID` and the `hugging-face` GitHub environment secret `HF_TOKEN` with access scoped to that Space. |

Use these prefixes under `novo-nordisk/`. The datasets prefix belongs in `novo-nordisk-pbl`. Experiments, checkpoints, predictions, reports, and new output verification artifacts belong in `novo-nordisk-pbl-outputs`.

- `datasets/ucsd-bmets/v1/{raw,metadata,processed,manifests}/`. Raw and metadata preserve source bytes. Processed data lives under a recipe digest. Reviewed manifests are immutable and addressed by digest.
- `experiments/<method>/<run-id>/`. Use UTC timestamp, owner handle and a random suffix as the unique run ID. Keep config, code SHA, input manifest SHA, partition SHA, runtime versions and metrics together.
- `checkpoints/<run-id>/`, `predictions/<run-id>/`, `reports/<run-id>/`. Record output sizes and SHA-256. Upload artifacts before publishing the completion report. On a retry, accept an existing object only after SHA-256 readback agrees. For outputs produced nondeterministically, use a new run ID after a conflict.
- `verification/<run-id>/`. Synthetic probes and receipts.

Conditional PUT protects cooperating tools. Review bucket locks, access controls and disaster recovery before declaring the raw prefix immutable against all clients. The script uses independent GET and SHA-256 because S3 ETags can vary by upload mode.

Canonical raw storage is 175,705,024,365 bytes, as recorded in the [final ingestion receipt](../../protocol/v1/final-ingestion.json). Request charges, derived storage and account quota are separate; this reorganization does not provision resources or repeat a cost audit.

Kaggle should stream or cache cohorts of complete patients with local disk limits, using SHA-256 as the cache key. Use R2 as the canonical store. Use the accepted workstream's immutable patient shards/private inputs and bounded cache rather than duplicating the entire cohort in each session. Reorganization does not authorize additional preparation or benchmark campaigns. Private Kaggle caches are replaceable and must reference a manifest digest.
