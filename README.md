# Novo Nordisk Longitudinal MRI Research

Infrastructure for patient-level, multi-visit brain metastases research using UCSD-BMETS-Longitudinal v1. The viewer accepts a 3D MRI and an optional aligned mask from any segmentation pipeline. A synthetic visit is available in the UI.

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

Review one visit at a time with the orthogonal slices and mask volumetry. Use approved de-identified uploads. This release implements mask review and synthetic visualization. The summer CNN–ViT app is isolated in [deployment/binary_seg_mvp](deployment/binary_seg_mvp), and the exact summer deliverable is preserved in [summer-archive](https://github.com/Strange-Noah/BrainTumour-CNNViT-Segmentation/tree/summer-archive).

Work on independent branches, request review through pull requests, and keep main stable. See [AGENTS.md](AGENTS.md).
