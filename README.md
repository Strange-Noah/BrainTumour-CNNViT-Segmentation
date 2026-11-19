# Novo Nordisk Longitudinal MRI Research

Infrastructure for patient-level, multi-visit brain metastases research using UCSD-BMETS-Longitudinal v1. The viewer accepts a 3D MRI and an optional aligned mask from any segmentation pipeline. A synthetic visit is available in the UI.

## Swin UNETR BMETS baseline

The first local Swin UNETR run used the frozen Protocol v1 split and the enhancing-tissue target (raw label 3). The model was trained for 21 epochs on an NVIDIA GeForce RTX 5060 Laptop GPU. The best checkpoint was selected at epoch 21 using the training script's 12-visit validation subset. Full validation was then run on all 45 validation patients (159 visits); the test set has not been evaluated.

| Full-validation metric (patient-macro) | Result |
| --- | ---: |
| Dice | 0.2552 |
| HD95 | 74.03 mm |
| Lesion sensitivity | 0.4808 |
| False-positive components | 56.10 per patient, averaged over visits |
| Absolute volume error | 2,503.68 mm³ |

Interpretation: this is a completed first baseline run, but segmentation quality is weak for reliable lesion tracking. Training loss decreased from 1.1823 at epoch 1 to 0.5960 at epoch 21, while full-validation results show low overlap, substantial boundary error, and many false-positive components. Treat these results as preliminary. The full-validation report and per-visit predictions are in `outputs/swin_unetr_bmets/validation_run1/`; training history, configuration, and checkpoint are in `outputs/swin_unetr_bmets/run_f24_roi96/`. See [the Swin experiment notes](experiments/SWINUNETR_BMETS.md) and [detailed baseline report](experiments/SWIN_UNETR_BMETS_BASELINE_REPORT.md) for implementation, results, and limitations.

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
