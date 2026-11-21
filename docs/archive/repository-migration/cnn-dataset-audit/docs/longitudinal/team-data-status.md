# Team data availability and verification

Latest ingestion reconciliation: 2026-11-16. All 300 patients and 1,065 examinations are stored in private R2: 11,405 original NIfTI objects, 175,705,024,365 bytes. No missing required modalities, masks, unexpected objects, size conflicts or invalid patient receipts were found. [Final reconciliation](../../protocol/v1/final-ingestion.json).

The four official metadata TSVs and minimized canonical manifest remain private. Use manifest revision `040ec6c5`; the older `4f3c7caa` manifest is superseded. Acquisition receipts provide original-object SHA-256 values. Raw labels and source headers remain unchanged. Team accounts retain dataset read-only access.

| Object under `novo-nordisk/datasets/ucsd-bmets/v1/` | SHA-256 |
|---|---|
| `manifests/040ec6c523d5beae43a30b64fede2bee226e1a08e8c14b2cca8043b01b9c39f4/visits.jsonl` | `040ec6c523d5beae43a30b64fede2bee226e1a08e8c14b2cca8043b01b9c39f4` |
| `manifests/4f3c7caa8fb4a19b8706bae0a1d1b1a82a1b0c8862f84194d21cae35907cbcf7/visits.jsonl` | `4f3c7caa8fb4a19b8706bae0a1d1b1a82a1b0c8862f84194d21cae35907cbcf7` |
| `metadata/UCSD-BMETS-Longitudinal_MRImetadata_Dictionary.tsv` | `b2647b0f8936676208bb3e7bc986eb8888f76376ea83d29412da82074ec51797` |
| `metadata/UCSD-BMETS-Longitudinal_TCIA_Clinical_Data.tsv` | `d701179cba86124d833de30bed496337c19f054c49e48d6ebc25137542daf0a2` |
| `metadata/UCSD-BMETS-Longitudinal_TCIA_Clinical_Data_dictionary.tsv` | `77fd461369f8b0e1f9bc762e79d6346d0a0f733cc5e4923ddbe8de17b65cb3a1` |
| `metadata/UCSD-BMETS-Longitudinal_TCIA_MRImetadata.tsv` | `53ac850fcd31094b2fc131cad08b23497b9539bbf75cf2b3679b7e7108fa41c7` |

The portable CPU smoke passed with `verified=true`, 480 bytes and shape `[4,4,4]` for the Gate B synthetic NIfTI. It uses the standard AWS credential chain and a configurable S3 endpoint. It performs no write operations. Live validation used R2; other S3 implementations still need their own endpoint and reader credential check.

The 43-file pilot passed independent read-only verification. The complete header/geometry/label audit reconciles all files: zero failures, 70 empty-mask warnings and two aligned unknown-unit masks. [Full audit and evidence](full-dataset-audit.md). See [active experiments](active-experiments.md), [acquisition records](tcia.md) and [runtime handoff](handoff.md). No hosted deployment or complete model training is required to use the data.
