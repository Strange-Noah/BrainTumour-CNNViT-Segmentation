# Team onboarding

## Repository structure

Main contains the shared baseline. Each contributor develops on
their own existing logical research workstream.

Clone the repository:

```sh
git clone https://github.com/Strange-Noah/BrainTumour-CNNViT-Segmentation.git
cd BrainTumour-CNNViT-Segmentation
git fetch origin
```

Enter your already assigned workstream branch:

```sh
git switch YOUR_EXISTING_WORKSTREAM
```

Do not create a new branch for each task. A new permanent workstream
requires explicit agreement.

## Shared data

Use [Protocol v1](protocol/v1/README.md) and the frozen patient split.

GitHub access and private R2 access are provisioned separately.
Obtain bucket-scoped dataset **read-only** credentials through the authorized secret store. Output writers are separately scoped to the output bucket; teammates must not receive dataset write permission.

Never commit credentials, signed URLs, medical images, checkpoints
or patient-level results.

Install shared CPU dependencies:

```sh
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install -r research/requirements.txt
python -m pip install -r cloud/requirements.txt
```

See [cloud architecture](docs/longitudinal/architecture.md) for the
existing R2 interface and [model contracts](docs/longitudinal/model-contracts.md)
for model integration requirements.

## Model development

Choose model-specific dependencies and training strategies on your
own workstream. Record the source SHA, configuration, data identity,
checkpoint provenance and evaluation protocol.

Use only authorized compute.

## Sharing results

Promote shared improvements and accepted milestones selectively
through pull requests to main. Do not merge an entire evolving
personal workstream into main.

Temporary review branches must be cleaned up after integration.

See [AGENTS.md](AGENTS.md).
