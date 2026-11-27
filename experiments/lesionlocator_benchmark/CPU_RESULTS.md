# Validation G0/L0 results

All nine trusted CPU batches completed and reported private SHA-256 readback.
Execution source: `3eeb0c747927a8931ada80643fe3e83dd80245ba`.
The input manifest, source SHA and configuration digest identify the private
output namespace. Per-patient label maps are reproducible from the hashed
native label and T1c sources, original affines, 26-connected label-3
instance encoding, and nearest-neighbor native-grid mapping.

| Measure | Result |
|---|---:|
| Validation patients / G0 exams / L0 pairs | 45 / 159 / 114 |
| G0 foreground identity checks passed | 159 / 159 |
| L0 Dice mean over follow-up pairs | 0.331737 |
| Spatial positive-overlap matches | 382 / 597 |
| False-positive union components | 646 |
| Pooled FP per pair | 5.666667 |

The paired Dice includes the shared evaluator's empty-empty convention. The
match count uses a maximum-cardinality positive-overlap assignment on the
propagated union. Per-prompt records preserve stable IDs and explicit absent
outputs. Same-grid unique mutual overlap between two GT visits is labeled
a reference proxy. The aggregate does not estimate clinical identity accuracy.
The canonical manifest provides visit order. The MRI dictionary identifies
scanner and sequence fields, and the clinical dictionary describes age and
treatment windows. Use follow-up step for temporal strata.

## Evidence

[46031609960](https://github.com/Strange-Noah/BrainTumour-CNNViT-Segmentation/actions/runs/46031609960) · [46031922045](https://github.com/Strange-Noah/BrainTumour-CNNViT-Segmentation/actions/runs/46031922045) · [46031927829](https://github.com/Strange-Noah/BrainTumour-CNNViT-Segmentation/actions/runs/46031927829) · [46031934306](https://github.com/Strange-Noah/BrainTumour-CNNViT-Segmentation/actions/runs/46031934306) · [46031940537](https://github.com/Strange-Noah/BrainTumour-CNNViT-Segmentation/actions/runs/46031940537) · [46031947670](https://github.com/Strange-Noah/BrainTumour-CNNViT-Segmentation/actions/runs/46031947670) · [46031953712](https://github.com/Strange-Noah/BrainTumour-CNNViT-Segmentation/actions/runs/46031953712) · [46031961423](https://github.com/Strange-Noah/BrainTumour-CNNViT-Segmentation/actions/runs/46031961423) · [46031967768](https://github.com/Strange-Noah/BrainTumour-CNNViT-Segmentation/actions/runs/46031967768)

The first execution exposed NumPy float32 spacing serialization. The fix
converts NumPy scalars to JSON-native values and preserves the numeric result.
All five synthetic tests passed. The nine results above use the corrected
source and its isolated private namespace.

[Official asset retrieval](https://github.com/Strange-Noah/BrainTumour-CNNViT-Segmentation/actions/runs/46031603433)
selectively extracts the published tracking fold and verifies ZIP member CRC
and R2 SHA-256. The source includes separate geometry and tracker adapters
for review. The GPU campaign requires runtime qualification, private input
and output grants, and an exact registered execution source.
