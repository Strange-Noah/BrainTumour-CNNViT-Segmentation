# Novo Nordisk Longitudinal MRI Research

A collaborative research project developing segmentation and
longitudinal analysis tools for brain metastases.

This branch contains shared infrastructure, dataset and evaluation
contracts, common deployment components and accepted project-level
results. Individual model development is maintained separately.

## Shared dataset

UCSD-BMETS-Longitudinal v1:

- 300 patients
- 1,065 examinations
- Frozen patient-level split: 210 train, 45 validation, 45 test
- Canonical imaging data stored in private Cloudflare R2

See [dataset protocol](protocol/v1/README.md),
[architecture](docs/longitudinal/architecture.md) and
[model contracts](docs/longitudinal/model-contracts.md).

## Research workstreams

- [U-Net 2.5D / Kaggle](https://github.com/Strange-Noah/BrainTumour-CNNViT-Segmentation/tree/model/unet25d-kaggle)
- [Swin UNETR](https://github.com/Strange-Noah/BrainTumour-CNNViT-Segmentation/tree/swinunetr-baseline)

Additional workstreams are registered as contributors establish them.

No unfinished model experiment is automatically considered a
shared longitudinal segmentation baseline.

## Project demonstrations

### Summer brain tumour segmentation

[Historical Hugging Face demonstration](https://huggingface.co/spaces/TLOMMP/brain-tumour-segmentation)

[Archived source and experimental results](https://github.com/Strange-Noah/BrainTumour-CNNViT-Segmentation/tree/summer-archive)

The summer model was trained for its original brain tumour task;
it is not the current UCSD brain metastasis baseline.

### Shared longitudinal MRI viewer

The model-independent viewer accepts a 3D MRI and an aligned mask.
It also provides a synthetic demonstration.

```sh
python -m pip install gradio==6.1.0 -r deployment/longitudinal_viewer/requirements.txt
python deployment/longitudinal_viewer/app.py
```

## Contributing

- [Team onboarding](TEAM_ONBOARDING.md)
- [Active research lines](docs/longitudinal/active-experiments.md)
- [Repository governance](AGENTS.md)
- [Shared execution and recovery](docs/longitudinal/handoff.md)
- [Dataset verification](docs/longitudinal/verification.md)

This is a research project, not a validated clinical tool.
