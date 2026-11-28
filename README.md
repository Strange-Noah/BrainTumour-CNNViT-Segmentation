# Novo Nordisk Longitudinal MRI Research

Infrastructure for patient-level, multi-visit brain metastases research using UCSD-BMETS-Longitudinal v1. The viewer accepts a 3D MRI and an optional aligned mask from any segmentation pipeline. A synthetic visit is available in the UI.

## Swin UNETR BMETS baseline

The first local Swin UNETR run used the frozen Protocol v1 split and the enhancing-tissue target (raw label 3). The model was trained for 21 epochs on an NVIDIA GeForce RTX 5060 Laptop GPU. The best checkpoint was selected at epoch 21 using the training script's 12-patient validation subset. Full validation was then run on all 45 validation patients (159 visits); the test set has not been evaluated.

| Full-validation metric (patient-macro) | Result |
| --- | ---: |
| Dice | 0.2552 |
| HD95 | 74.03 mm |
| Lesion sensitivity | 0.4808 |
| False-positive components | 56.10 per patient, averaged over visits |
| Absolute volume error | 2,503.68 mm³ |

Interpretation: this is a completed first baseline run, but segmentation quality is weak for reliable lesion tracking. Training loss decreased from 1.1823 at epoch 1 to 0.5960 at epoch 21, while full-validation results show low overlap, substantial boundary error, and many false-positive components. Treat these results as preliminary. The full-validation report and per-visit predictions are in `outputs/swin_unetr_bmets/validation_run1/`; training history, configuration, and checkpoint are in `outputs/swin_unetr_bmets/run_f24_roi96/`. See [the Swin experiment notes](experiments/SWINUNETR_BMETS.md) and [detailed baseline report](experiments/SWIN_UNETR_BMETS_BASELINE_REPORT.md) for implementation, results, and limitations.

## Swin UNETR lesion-aware experiment

The follow-up `lesion_balanced_v1` experiment retained the four input channels (T1, T1c, T2, FLAIR), enhancing-tissue target (raw label 3), MONAI Swin UNETR architecture (feature size 24; 96³ ROI), baseline Dice plus cross-entropy loss, seed 42, and 21-epoch training budget. The model was randomly initialized. Each epoch contained 746 optimizer updates. Checkpoint selection used one fixed visit from each of the 45 validation patients; the best selection score was at epoch 16, and that checkpoint was evaluated across all 159 validation visits.

### Code added or extended

- `experiments/train_swin_unetr.py` adds experiment options and training support for patient-balanced draws, lesion-level sampling, optional lesion-centered local crops, MRI support masking, lesion-volume-weighted cross-entropy, Tversky loss, fixed patient-balanced checkpoint selection, and resumable training state.
- `experiments/swin_sampling.py` identifies 26-connected enhancing components, measures their volumes in mm³ using the T1c geometry, balances training draws across patients and visits, and samples lesion-centered or negative crops.
- `experiments/analyze_swin_lesion_sizes.py` measures lesion sensitivity in reference-lesion volume bins, with one-to-one matching based on any positive voxel overlap.
- `experiments/run_swin_partition.py` and shared Swin inference utilities export predictions in the original T1c geometry and compute the team's shared metrics.

For `lesion_balanced_v1`, patient-balanced visit selection and lesion-aware crop-center sampling were enabled. Lesion volume affected the probability of choosing each crop center, with smaller lesions receiving higher sampling weight. Local magnification (`local_crop_probability=0`), support masking (`none`), lesion-weighted CE, and Tversky loss were available in the code but were **not enabled** for this run. This isolates the initial experiment to the sampling strategy and avoids attributing results to optional changes that were not tested.

### Full-validation comparison

Both runs were evaluated on the same 159 validation visits from 45 patients using the same frozen split, manifest, and shared evaluator. Values below are patient-macro means; lower is better for HD95, false-positive candidates, and absolute volume error.

| Metric | Earlier baseline | Lesion-aware v1 | Change | Direction |
| --- | ---: | ---: | ---: | --- |
| Dice | 0.2552 | 0.2853 | +0.0301 | Improved |
| HD95 | 74.03 mm | 103.37 mm | +29.34 mm | Worse |
| Lesion sensitivity | 0.4808 | 0.4928 | +0.0121 | Small increase |
| False-positive candidates per scan | 56.10 | 144.61 | +88.50 | Worse; 2.58× baseline |
| Absolute volume error | 2,503.68 mm³ | 1,700.76 mm³ | −802.92 mm³ | Improved |

The paired patient bootstrap supported an increase in Dice (mean change +0.030; exploratory 95% interval +0.005 to +0.056) and lower absolute volume error (−803 mm³; −1,195 to −413 mm³). The lesion-sensitivity interval crossed zero (mean +0.012; −0.016 to +0.045). HD95 increased by 29.3 mm (+22.3 to +36.3), and false positives increased by 88.5 per scan (+64.5 to +112.4). These are descriptive validation intervals, not test-set results, and do not remove checkpoint-selection optimism.

**Evaluation:** the lesion-aware checkpoint improved average overlap and volume error, but it produced much poorer boundary distances and substantially more candidate lesions. This is a mixed result, not a clear overall improvement. The baseline's checkpoint was selected using 12 validation patients, whereas lesion-aware v1 used 45. Therefore, the full-validation comparison is informative as a preliminary model comparison, but it does not isolate the effect of lesion-aware sampling. Do not use either result alone to initialize lesion tracking without reviewing false-positive locations and candidate quality.

### Sensitivity by reference-lesion volume

The supplementary analysis used 26-connected reference components, T1c-based mm³ volumes, and one-to-one any-overlap matching on the lesion-aware v1 predictions.

| Reference lesion volume | Lesions matched | Pooled sensitivity | Patient-macro sensitivity | Patients with lesions |
| --- | ---: | ---: | ---: | ---: |
| <27 mm³ | 114 / 736 | 15.5% | 17.1% (95% patient-bootstrap interval 10.4–24.5%) | 36 / 45 |
| ≥27 mm³ | 349 / 502 | 69.5% | 69.7% (95% patient-bootstrap interval 61.6–77.6%) | 44 / 45 |

Sensitivity is about 53 percentage points lower for the smaller-lesion bin on a patient-macro basis. Any-overlap matching is permissive, so the result points to a meaningful small-lesion detection/localization gap. Two validation patients contribute 414 of the 736 lesions below 27 mm³ (56.3%); inspect patient-level results and those high-burden cases alongside the aggregate. This subgroup describes lesion-aware v1 only; without the same analysis on a matched voxel-sampling run, it cannot show that lesion-aware sampling improved small-lesion detection.

### Current follow-up and result locations

A matched voxel-sampling control is underway in `outputs/swin_unetr_bmets/voxel_selector45_control/`. The recorded checkpoint/history currently show **epoch 1 of 21** (746 updates) with the same seed, model capacity, loss, and 45-patient validation selector. Its epoch-1 score is an in-training checkpoint-selection value and is not a fair final comparison. Complete the planned run, evaluate its selected checkpoint on the same 159 validation visits, then run the same 27 mm³ subgroup analysis before comparing sampling strategies.

The lesion-aware training history and checkpoint are in `outputs/swin_unetr_bmets/lesion_balanced_v1/`; full-validation metrics and predictions are in `outputs/swin_unetr_bmets/lesion_balanced_v1/validation_full/`, with the volume-bin results in `lesion_size_metrics_corrected.json`. The test partition has not been evaluated. See [the Swin experiment notes](experiments/SWINUNETR_BMETS.md), the [lesion-aware implementation guide](docs/longitudinal/swinunetr-new-code-implementation.md), and the [updated progress report](docs/longitudinal/Swin_UNETR_LesionAware_Results_and_Next_Steps.docx) for further details.

- [Active experiments and protocol](docs/longitudinal/active-experiments.md)
- [Team onboarding](TEAM_ONBOARDING.md)
- [Promotion plan and acceptance review](docs/longitudinal/promotion.md)
- [Architecture and access](docs/longitudinal/architecture.md)
- [TCIA metadata audit and acquisition](docs/longitudinal/tcia.md)
- [Manifest and evaluation conventions](docs/longitudinal/manifest.md)
- [Verification and migration report](docs/longitudinal/verification.md)
- [Recovery and team handoff](docs/longitudinal/handoff.md)

## Local viewer

Use Python 3.12 in a virtual environment.

```sh
python -m pip install gradio==6.1.0 -r deployment/longitudinal_viewer/requirements.txt
python deployment/longitudinal_viewer/app.py
```

Review one visit at a time with the orthogonal slices and mask volumetry. Use approved de-identified uploads. This release implements mask review and synthetic visualization. The complete summer project is preserved on [summer-archive](https://github.com/Strange-Noah/BrainTumour-CNNViT-Segmentation/tree/summer-archive).

Work on independent branches, request review through pull requests, and keep main stable. See [AGENTS.md](AGENTS.md).
