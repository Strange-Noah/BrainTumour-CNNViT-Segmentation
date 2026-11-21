# Shared dataset availability

Canonical UCSD-BMETS v1 ingestion and integrity audit are complete, not pending: **300 patients / 1,065 examinations / 11,405 NIfTI objects / 175,705,024,365 bytes**. [Final immutable receipt](../../protocol/v1/final-ingestion.json) · [full audit](full-dataset-audit.md) · [aggregate audit](../../protocol/v1/full-audit-summary.json). No missing expected objects, missing required modalities or size conflicts were found.

Private canonical bucket: `novo-nordisk-pbl`; outputs/checkpoints/reports: `novo-nordisk-pbl-outputs`, isolated by unique run prefixes. Teammate dataset access remains read-only. R2 credentials and patient-level artifacts are never stored on GitHub. The four official metadata TSVs and manifest identity `040ec6c523d5beae43a30b64fede2bee226e1a08e8c14b2cca8043b01b9c39f4` remain canonical; an older manifest is historical, not a new ingestion target.

Use the frozen **210 train / 45 validation / 45 sealed test** patient split and [protocol](../../protocol/v1/README.md). Model development must not evaluate the test cohort. Preserve raw tissue labels; primary enhancing target is label 3, optional core is labels 1+3. Tissue values are not individual lesion IDs. Preserve native physical geometry and follow [spatial-unit policy](spatial-unit-policy.md).

[Portable CPU-only S3 reader](../../cloud/read_smoke.py) accepts a configurable HTTPS endpoint, bucket, key, digest and standard AWS credential chain. It performs no writes. Use authorized read-only references and existing receipts, not public data copies.

Model-specific prepared caches and scientific results live on their workstreams. They are not prerequisites for every teammate or shared accepted model results. [Onboarding](../../TEAM_ONBOARDING.md) · [model contracts](model-contracts.md) · [execution handoff](handoff.md).
