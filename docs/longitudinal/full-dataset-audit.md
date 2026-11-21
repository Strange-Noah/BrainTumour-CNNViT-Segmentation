# Full-cohort acquisition and NIfTI audit

Completed 2026-11-16 with existing free GitHub CPU runners. All 300 acquisition receipts reconcile to 1,065 exams and 11,405 original NIfTI files totaling 175,705,024,365 bytes. No expected object, required modality or mask is missing; no size conflicts or unexpected objects were found. [Acquisition reconciliation](../../protocol/v1/final-ingestion.json).

The separate all-file audit downloaded each object directly from private R2 to cloud scratch, checked SHA-256 against its acquisition metadata, inspected shape/dtype, all-voxel finiteness, spatial/temporal units, spacing, orientation, qform/sform codes and matrices, effective affine and alignment to T1c. It also recorded mask label histograms and empty masks. Raw files were never rewritten.

All 300 patients, 1,065 unique exams and 11,405 unique files are accounted for. There are **11,333 PASS files and 72 WARNING files; zero failures**. Exam totals are 993 PASS and 72 WARNING. The only warnings are 70 empty masks and the two previously reproduced unknown-unit masks. No exam requires quarantine under this audit. Empty masks are legitimate negative-target data; use the shared empty-case evaluation conventions.

BMETS0016_01 and BMETS0095_01 retain their original unknown-unit headers. Their masks match the corresponding millimeter T1c shape, spacing and effective affine. The explicit loader policy interprets these two aligned masks in millimeters without altering source bytes. [Unit policy](spatial-unit-policy.md).

[Aggregate audit evidence](../../protocol/v1/full-audit-summary.json) includes report hashes and observed mask values. Original label values remain tissue classes, never lesion identities. Primary segmentation remains enhancing tissue (3), with optional core (1+3); connected components yield visit-local lesion candidates.

Detailed headers, matrices, voxel histograms, per-exam statuses and the report index remain in private `novo-nordisk-pbl` under `novo-nordisk/datasets/ucsd-bmets/v1/reports/audit/full-v1/`. Final `summary.json`, `index.json` and `examination-status.csv` were written create-once and independently GET-checked against SHA-256. This final report supersedes partial audit snapshots.

The audit proves integrity and the stated geometry checks. It does not establish annotation accuracy or model quality. No paid compute or full training was initiated.
