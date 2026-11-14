# Team and cross-account recovery

Clone [GitHub](https://github.com/Strange-Noah/BrainTumour-CNNViT-Segmentation) and check out the migration PR head in detached mode for validation. For edits, create your own branch and open a PR. The archived branch is the exact summer code and report tree. Data access is separate from GitHub membership. Read [verification](verification.md) and resume at the first open gate.

## Gate B

The owner selected `novo-nordisk-pbl`. Use the cloud runtime or credential manager where the owner configured its S3 endpoint and bucket-scoped credentials. Set `R2_BUCKET=novo-nordisk-pbl` and load the other three values from that runtime’s secret store. For local verification with an AWS CLI profile, set `R2_AWS_PROFILE=novo-nordisk-r2`. Supply `R2_ENDPOINT_URL` unless the profile resolves the Cloudflare R2 S3 endpoint. The script checks that the destination is an account-level Cloudflare R2 URL. [Cloudflare S3 endpoint setup](https://developers.cloudflare.com/r2/get-started/s3/)

Use Python 3.12 and `python -m pip install -r cloud/requirements.txt`. Run `python cloud/smoke.py write UNIQUE_RUN_ID` and save its JSON receipt. In a fresh cloud runtime with read-only access, run `python cloud/smoke.py read UNIQUE_RUN_ID --sha256 RECEIPT_SHA`. Publish the digest, byte size and both outcomes. Do not store credentials in the repo.

## Gate C

Follow [TCIA acquisition](tcia.md). Record source links and checksums, official count reconciliation, one complete patient transfer with verified visits and observed labels, then stage the full dataset.

## Gate D

Use your own private Kaggle notebook. Enable Internet and select an available free GPU. Store the four R2 values in Kaggle Secrets. In a Python cell, import the secrets and launch the version-pinned smoke. Substitute the validated GitHub SHA, a new run ID, a verified MRI object key and its SHA-256.

```python
import os
import subprocess
from kaggle_secrets import UserSecretsClient

secrets = UserSecretsClient()
for name in ("R2_ENDPOINT_URL", "R2_BUCKET", "R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY"):
    os.environ[name] = secrets.get_secret(name)

checkout = "/kaggle/working/novo-source"
subprocess.run(["git", "clone", "https://github.com/Strange-Noah/BrainTumour-CNNViT-Segmentation.git", checkout], check=True)
subprocess.run(["git", "-C", checkout, "checkout", "--detach", "<commit>"], check=True)
subprocess.run(["python", "-m", "pip", "install", "-r", "cloud/requirements.txt"], cwd=checkout, check=True)
subprocess.run(["python", "cloud/kaggle_smoke.py", "<run-id>", "<object-key>", "<sha256>"], cwd=checkout, check=True)
```

Kaggle's CLI can submit staged notebook code with `kaggle kernels push -p PATH`, then inspect with `kaggle kernels status USER/SLUG`. Use a private kernel and confirm secret attachments in the UI before submitting. In a fresh session, verify the output and report keys with `cloud/storage.py` and recorded hashes. Publish the redacted report through a GitHub PR, with timing, hardware, free disk, runtime limits and storage readback. [Kaggle kernel metadata](https://github.com/Kaggle/kaggle-cli/blob/main/docs/kernels_metadata.md)

## Gate E

Confirm `hf auth whoami` reports `Mathew-ZHZ`. Creation of the private Gradio Space `Mathew-ZHZ/longitudinal-mri-viewer` returned HTTP 402. CPU Basic has no hourly compute cost, and Hugging Face requires PRO to create this Gradio Space. Resume creation if the owner enables the required plan. A free Static Space would require a browser-only port of the viewer and a new hosted smoke test. Scope the write token to that Space. Configure the GitHub variable and environment secret named in [architecture](architecture.md). For the pilot, set repository variable `HF_ENABLE_MIGRATION_PILOT=true` and select the migration branch plus main in the environment deployment branch policies. Require environment approval where available. After a reviewed viewer or workflow commit triggers Deploy MRI viewer, approve that run. Once rollout is complete, clear the pilot variable and narrow environment deployments to main. Verify the run, the Space's source commit message with GitHub SHA, startup, and the synthetic three-plane preview and volumetry. Only deploy paths under `deployment/longitudinal_viewer`. [Hugging Face GitHub Actions guide](https://huggingface.co/docs/hub/spaces-github-actions)

## Gate F

With a separate GitHub/Kaggle account, follow the same checkout and secret provisioning. Reopen the mirrored Space and reproduce the R2 readback from receipts alone. Download the canonical manifest by its digest, verify it, and recover a completed experiment report and output from R2. Record the checkout SHA and readback evidence in a PR. Runtime storage and Codex chat history are disposable.
