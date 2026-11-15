# Shared splits and prediction handoff

## Patient split

All model families consume the same owner-reviewed, immutable split artifact. No canonical train/validation/test assignment has been approved yet. Synthetic development and loader work may proceed. A model must not publish held-out comparison metrics against a self-selected split.

Use UTF-8 CSV with columns `patient_id,partition`. Each canonical patient appears exactly once. Partition is exactly `train`, `validation` or `test`. Join visits through the canonical manifest's patient ID, and keep every visit of a patient in that partition. Check unique IDs, complete coverage of the selected cohort, no partition overlap and nonempty required partitions.

Publish the CSV in private R2 `datasets/ucsd-bmets/v1/manifests/splits/<sha256>/patients.csv`, with a JSON specification containing dataset version, canonical manifest SHA-256, CSV SHA-256, cohort inclusion rules, seed, assignment method, partition counts and approving Git commit. Publish only hashes and aggregate counts in public GitHub. Freeze the artifact before training. New assignments require a new version and a reviewed rationale. Fit preprocessing only on train.

## Prediction records

Every model exports one integer 3D NIfTI mask per visit in the source MRI grid. Preserve the reference affine, shape and declared spatial units. Include a separate versioned resampling recipe if processing uses another grid, and restore the source grid before handoff. An absent prediction is an explicit missing record, never an empty mask.

Store the NIfTI under `novo-nordisk/predictions/<run-id>/<exam-id>/seg.nii.gz` in the private output bucket. Pair it with JSON carrying these required fields.

| Field | Contract |
|---|---|
| `schema_version`, `run_id`, `patient_id`, `exam_id` | Version 1 and canonical IDs. Run IDs are unique across contributors. |
| `git_sha`, `config_sha256`, `environment` | Immutable model code, configuration digest and dependency/runtime versions. |
| `dataset_version`, `manifest_sha256`, `split_sha256` | The same canonical inputs for every compared model. |
| `reference` | MRI object key and SHA-256, shape, affine and units. |
| `prediction` | Private object key, SHA-256, size, integer dtype, observed labels and explicit value-to-class map. |
| `processing` | Normalization, channel order, threshold/argmax, resampling, postprocessing and original-grid restoration. |
| `status` | `complete`, `missing` or `failed`. Only complete records carry validated prediction objects. |

TCIA defines tissue labels 1 non-enhancing core, 2 surrounding non-enhancing FLAIR hyperintensity, 3 enhancing tissue and 4 resection cavity. Background is 0. A model declaring a subset must name excluded classes and its task. Binary conversions require a reviewed mapping. These labels do not identify individual lesions. Keep instance masks and longitudinal lesion IDs in separately versioned artifacts.

Upload mask bytes first and validate SHA-256, geometry and label values. Publish the completion record last without overwriting an existing run. Future tracking code joins by patient/exam and compares visits using the manifest's timing precision. Tracking identity, matching and evaluation rules remain future reviewed feature work.
