# Team onboarding

Start from reviewed `main`. GitHub access and dataset access are provisioned separately by the owner. Development can start with the synthetic MRI today. [Available data](docs/longitudinal/team-data-status.md) · [Shared model contracts](docs/longitudinal/model-contracts.md)

## GitHub

Ask the owner for repository access or fork the public repository. Clone your repository, fetch upstream main and create `feat/<name>-<topic>` from it. Open a PR against upstream main with the code SHA, validation and any aggregate results. Main requires CI and an owner merge. Summer history is on frozen `summer-archive`.

```sh
git clone https://github.com/Strange-Noah/BrainTumour-CNNViT-Segmentation.git
cd BrainTumour-CNNViT-Segmentation
git switch -c feat/alice-baseline origin/main
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install -r cloud/requirements.txt
```

## CPU data access

The owner provides individual bucket-scoped read-only access to `novo-nordisk-pbl`. Use a private AWS profile, runtime role, or the standard `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY` and optional `AWS_SESSION_TOKEN` environment variables. Obtain values through your secret store. Pass the project S3 endpoint as configuration. The read smoke performs a GET and validates a NIfTI checksum on CPU.

```sh
python cloud/read_smoke.py --endpoint "$AWS_ENDPOINT_URL_S3" \
  --profile your-reader-profile --region auto --bucket novo-nordisk-pbl \
  --key novo-nordisk/verification/r2-gate-b-20261114-124417-c454f3dc/synthetic.nii \
  --sha256 7a9b9082712814448816bacb582c7faabd0dc3a8037910413e0d7c841d571562
```

Omit `--profile` for environment credentials or runtime roles. For AWS S3, omit `--endpoint` and use the bucket's region. Other HTTPS S3-compatible endpoints are supported. Never place credentials in commands, Git files, notebook cells or outputs.

## Personal compute

Use your own private Kaggle notebook with Internet enabled. Install the cloud requirements, clone GitHub and check out the chosen immutable commit. Attach your own reader credentials through Kaggle Secrets and load them into the standard AWS environment variables in memory. Run the same read smoke on CPU first. A notebook with Internet disabled cannot retrieve GitHub or R2. Saved Kaggle versions previously failed to resolve attached Secrets, while interactive sessions passed.

For personal local GPU development, create a separate Python environment and install the GPU framework and CUDA build documented by your chosen model's official instructions. Record the versions, GPU model, memory and commit in each experiment. The project's cloud and viewer smoke requirements are CPU-only. Start with synthetic arrays, manifest parsing and inference adapters. Full training remains pending owner authorization.

## Model responsibilities and handoff

Each model contributor owns an isolated branch and experiment configuration, loader/normalization recipe, model adapter and compatible prediction export. Use the shared patient split contract and label definitions before comparing results. Coordinate changes to those shared contracts in a PR. Preserve raw labels. Lesion tracking consumes prediction handoffs and defines correspondence across visits separately from tissue classification.

Canonical datasets stay read-only. Experiment outputs belong in `novo-nordisk-pbl-outputs` under unique run IDs using separately authorized output credentials. Until that access is ready, save development outputs in your private environment and publish code and aggregate checks in PRs. [Runtime handoff](docs/longitudinal/handoff.md)
