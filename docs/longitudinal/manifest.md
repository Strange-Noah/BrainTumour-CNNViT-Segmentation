# Patient and visit manifest

Use one UTF-8 JSON object per line, sorted by patient ID and numeric visit order. The primary key is dataset version, patient ID and examination ID. Keep one reviewed canonical revision and save its SHA-256 alongside every run. A local generator produces pending records from MRI and clinical TSVs. Its column mappings and expected relative paths must be checked against the official files.

| Field | Contract |
|---|---|
| `dataset_version`, `patient_id`, `exam_id`, `visit_order` | Version 1, source patient and examination IDs, and positive within-patient index. |
| `relative_days`, `timing_precision` | Integer day offset only when the source defines the origin and units. Otherwise null and `ordinal`; if source timing is coarser, record the actual precision and source separately. |
| `modalities`, `segmentation` | Modality-to-path map and segmentation path relative to `raw/`. Add optional sequences only where present. Record missing artifacts explicitly in a verified manifest. |
| `treatment` | Source treatment fields with documented units and origin. Unavailable values are null. |
| `spatial_reference` | Published first-visit T1c reference. Record verified shape, affine, axes, units, voxel volume and any geometry exceptions in the reviewed manifest. |
| `availability`, `integrity`, `provenance` | Expected, available or missing. Pending, verified or failed. Preserve metadata hashes, verified object digests and acquisition receipt references. |

Keep all visits from a patient in the same split. Freeze a separate patient→partition mapping with seed, selection criteria and hashes. Validate no overlap across partitions before preprocessing, training or evaluation. Keep held-out data out of any normalization fit and parameter selection. Report complete and missing visits separately.

For future segmentation evaluation, record matching and resampling decisions, class semantics, whole-lesion overlap and absolute volume error, units, empty-mask handling and patient-level aggregates before comparing models. Keep lesion matching and longitudinal correspondence undefined until their evaluation protocol is reviewed. Segmentation outputs are integer NIfTI masks with background 0, a declared label map and geometry traceable to the source MRI. TCIA defines non-enhancing core 1, surrounding non-enhancing FLAIR hyperintensity 2, enhancing tissue 3 and resection cavity 4. The four-visit pilot observed labels 0, 2 and 3. See [shared split and prediction contracts](model-contracts.md).
