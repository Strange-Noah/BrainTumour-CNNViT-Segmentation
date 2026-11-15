# Team data availability and verification

Verified by live S3 listing and SHA-256 readback on 2026-11-15, Asia/Shanghai.

Private bucket `novo-nordisk-pbl` has four original official metadata TSVs (136,665 bytes combined), two immutable manifest revisions (1,619,865 bytes combined), and two synthetic verification objects (777 bytes combined). The raw imaging prefix contains zero objects. The 43-file real patient was acquired and audited in Kaggle scratch, and still needs persistent ingestion. The remaining release is pending transfer.

The canonical pending-availability manifest is the revision beginning `040ec6c5`. The earlier `4f3c7caa` manifest uses the unverified T2 filename convention and is superseded. Both are immutable. The canonical manifest records expected paths and pending integrity for images. Its presence does not imply those image objects are available.

| Object under `novo-nordisk/datasets/ucsd-bmets/v1/` | SHA-256 |
|---|---|
| `manifests/040ec6c523d5beae43a30b64fede2bee226e1a08e8c14b2cca8043b01b9c39f4/visits.jsonl` | `040ec6c523d5beae43a30b64fede2bee226e1a08e8c14b2cca8043b01b9c39f4` |
| `manifests/4f3c7caa8fb4a19b8706bae0a1d1b1a82a1b0c8862f84194d21cae35907cbcf7/visits.jsonl` | `4f3c7caa8fb4a19b8706bae0a1d1b1a82a1b0c8862f84194d21cae35907cbcf7` |
| `metadata/UCSD-BMETS-Longitudinal_MRImetadata_Dictionary.tsv` | `b2647b0f8936676208bb3e7bc986eb8888f76376ea83d29412da82074ec51797` |
| `metadata/UCSD-BMETS-Longitudinal_TCIA_Clinical_Data.tsv` | `d701179cba86124d833de30bed496337c19f054c49e48d6ebc25137542daf0a2` |
| `metadata/UCSD-BMETS-Longitudinal_TCIA_Clinical_Data_dictionary.tsv` | `77fd461369f8b0e1f9bc762e79d6346d0a0f733cc5e4923ddbe8de17b65cb3a1` |
| `metadata/UCSD-BMETS-Longitudinal_TCIA_MRImetadata.tsv` | `53ac850fcd31094b2fc131cad08b23497b9539bbf75cf2b3679b7e7108fa41c7` |

The portable CPU smoke passed with `verified=true`, 480 bytes and shape `[4,4,4]` for the Gate B synthetic NIfTI. It uses the standard AWS credential chain and a configurable S3 endpoint. It performs no write operations. Live validation used R2; other S3 implementations still need their own endpoint and reader credential check.

The TCIA pilot was manually dispatched on promoted main and is waiting at the existing credential approval gate. The account used by this agent is the required environment reviewer and the dispatch actor, so the current self-review prevention blocks this account from approving that run. A trusted collaborator can dispatch a new main run for owner approval under the existing policy. No environment policy was changed for onboarding. [Pilot run](https://github.com/Strange-Noah/BrainTumour-CNNViT-Segmentation/actions/runs/43710790508)

The complete per-file inventory enumeration previously ended in cancellation. Full ingestion remains gated on persistent pilot verification, inventory completeness and staging validation. [Acquisition audit](tcia.md) · [Runtime handoff](handoff.md)
