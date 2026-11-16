# Spatial-unit findings and source-preserving interpretation

The original masks for BMETS0016_01 and BMETS0095_01 independently report unknown spatial units. Their T1c references declare millimetres. Both masks share the corresponding reference shape, effective affine and header voxel spacing.

| Examination | Mask spacing | Result |
|---|---|---|
| BMETS0016_01 | 0.5 × 0.5 × 0.5 | WARNING, interpretable as mm |
| BMETS0095_01 | 0.5 × 0.5 × 0.800000011920929 | WARNING, interpretable as mm |

The first patient audit covered 2 visits and 22 files, with 21 file PASS and one WARNING. The second covered 4 visits and 44 files, with 43 file PASS and one WARNING. Neither audit had a FAIL.

Original NIfTI bytes remain unchanged. research.geometry.mask_geometry permits this explicit interpretation only for the two known examinations and only when the matching T1c establishes valid mm geometry and all grid/spacing checks pass. It returns a named warning. Other undefined or inconsistent spatial units require investigation. Prediction export requires an explicitly millimetre T1c reference. Evaluation cases for a missing-unit truth mask must provide the separate reference CSV field, rather than treating the truth header as its own unit reference.

Private machine-readable files.jsonl, exams.jsonl and summary.json retain the full header matrices, codes, label counts, warnings and failures under canonical R2 prefix novo-nordisk/datasets/ucsd-bmets/v1/reports/audit/full-v1/015-016/ and 094-095/. Each audit output is checksum-verified on upload. The complete-dataset audit is still running.

[BMETS0016 audit run](https://github.com/Strange-Noah/BrainTumour-CNNViT-Segmentation/actions/runs/43808343105) · [BMETS0095 audit run](https://github.com/Strange-Noah/BrainTumour-CNNViT-Segmentation/actions/runs/43808347515)
