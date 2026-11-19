## Active instruction — adaptive formal training (2026-11-20)

The owner now authorizes actual formal training using currently available free Kaggle quota. Follow [two-phase science and execution](../../experiments/unet25d/cohort/ADAPTIVE_TRAINING.md) and [focused CPU evidence](../../experiments/unet25d/cohort/ADAPTIVE_ACCEPTANCE.json). Supersedes older preparation-only/30-pass/budget-awaiting notes below. Exhaust all 220,384 training positions before performance stopping; then adaptive 67,200-example cycles and complete validation. Keep model/FP32/BCE+Dice/data policy and existing infrastructure. No paid compute, new preparation, separate GPU benchmark, or test-cohort evaluation. Integrated source is required before credential-bearing launch. GPU_TRAINING_STARTED=true: [formal execution and completed ten-minute handoff](../../experiments/unet25d/cohort/FORMAL_EXECUTION.md). Initial observation ended successfully; the original private dual-T4 job continues under the existing independent collector.

# Team and cross-account recovery

Clone [GitHub](https://github.com/Strange-Noah/BrainTumour-CNNViT-Segmentation) and create a feature branch from reviewed main. Pin an immutable commit for experiments and open PRs against main. Start with [team onboarding](../../TEAM_ONBOARDING.md). The archived branch is the exact summer code and report tree. Data access is separate from GitHub membership. Read [verification](verification.md) and resume at the first open gate.

## Training transition — 2026-11-19

Infrastructure is frozen. The owner accepted full FP32, dual-T4 DDP, local batch 1/global batch 2; no new GPU benchmark, manual notebook operation or ingestion audit is needed. [Frozen training inputs and execution handoff](../../experiments/unet25d/cohort/README.md) supersedes historical open-gate instructions below for the 2.5D U-Net. PR #27 is merged at `b35e75dabc8dde693e6cc1991168e909c5a60f58`; preparation resumes immutable receipts without redoing ingestion/audit. Read [PR #28](https://github.com/Strange-Noah/BrainTumour-CNNViT-Segmentation/pull/28) for current final readiness evidence and [long-session execution/recovery](../../experiments/unet25d/cohort/EXECUTION.md). Formal training needs explicit total GPU budget authorization; once readiness is confirmed, that decision enables the existing launch without another preparation task. There are 746 training visits / 159 validation visits; test pixels remain excluded. Existing saved-output inspection will be private CPU only. Do not treat the old single-visit Dice as scientific acceptance.

## Current execution status — 2026-11-16

Full ingestion is complete: 300 patients, 1,065 visits and 11,405 original objects. [Final reconciliation](../../protocol/v1/final-ingestion.json) supersedes earlier progressive availability snapshots. All bounded full-audit runs and consolidated audit reconciliation completed: zero failures, 70 empty-mask warnings and two documented unknown-unit mask warnings. [Full audit](full-dataset-audit.md). The frozen split and primary enhancing-tissue label 3 contract are unchanged.

The owner configured separate output-bucket secrets in `tcia-verification`. The manual CNN CPU workflow already wires these names to `novo-nordisk-pbl-outputs`; dataset credentials remain read-only. Run 43808895351 preceded secret availability and retained only ephemeral outputs. Replacement run 43810710239 passed persistent checkpoint, prediction and report upload with SHA-256 GET checks. [Output verification](cnn-output-verification.md). PR #16 merged at its approved revision. Independent fresh-session recovery passed in run 43811539630, including the 405,913,387-byte checkpoint SHA-256 and both prediction artifacts. This GitHub output cycle does not by itself close the Kaggle-originated Gate D test.

## Gate B

The owner selected `novo-nordisk-pbl`. Use the cloud runtime or credential manager where the owner configured its S3 endpoint and bucket-scoped credentials. Set `R2_BUCKET=novo-nordisk-pbl` and load the other three values from that runtime’s secret store. For local verification with an AWS CLI profile, set `R2_AWS_PROFILE=novo-nordisk-r2`. Supply `R2_ENDPOINT_URL` unless the profile resolves the Cloudflare R2 S3 endpoint. The script checks that the destination is an account-level Cloudflare R2 URL. [Cloudflare S3 endpoint setup](https://developers.cloudflare.com/r2/get-started/s3/)

Use Python 3.12 and `python -m pip install -r cloud/requirements.txt`. Run `python cloud/smoke.py write UNIQUE_RUN_ID` and save its JSON receipt. In a fresh cloud runtime with read-only access, run `python cloud/smoke.py read UNIQUE_RUN_ID --sha256 RECEIPT_SHA`. Publish the digest, byte size and both outcomes. Do not store credentials in the repo.

Gate B passed in an interactive Kaggle CPU session according to the project owner. To reproduce it, open the private notebook, enable the two read-only R2 key attachments in Add-ons → Secrets, and run the readback cell interactively. The CLI-saved run returned a Kaggle Secrets ConnectionError. The endpoint URL and bucket name are configuration values. [Private verification notebook](https://www.kaggle.com/code/mathewzhou/novo-tcia-and-r2-verification)

```sh
python cloud/smoke.py read r2-gate-b-20261114-124417-c454f3dc \
  --sha256 7a9b9082712814448816bacb582c7faabd0dc3a8037910413e0d7c841d571562
```

## Gate C

Follow [TCIA acquisition](tcia.md). The four official TSVs and minimized manifest are stored in R2. The 43-file patient transfer passed in private Kaggle scratch.

For persistence, run `python cloud/tcia_pilot.py ingest --stage /path/to/private-staging` on existing cloud Linux with IBM Aspera CLI v4.28.0 and its Transfer SDK. Provide the project dataset writer through the runtime's secret store. The script validates the inventory and known pilot digest before publishing each raw object with conditional PUT and a byte-for-byte SHA-256 readback. It writes a completion receipt only after all 43 readbacks pass. Running again with the same staged files reuses matching existing R2 objects and rejects conflicts.

The manual main workflow has passed the 43-file pilot and independent readback. TCIA environments keep their existing credentials and main-only policies, with human approvals removed by owner instruction. For subsequent batches use mode=batch, start (zero-based sorted patient index), count (1–10). Completion receipts make reruns recoverable. The first batch was started with start=0 and count=10.

Use [active experiment records](active-experiments.md) and the frozen [protocol v1](../../protocol/v1/README.md). The shared loader and reference-mask evaluator/export passed on a real R2 visit. Keep raw labels unchanged and use the exact shared split. Feature owners can run their adapters independently while ingestion progresses.

## Gate D

Use the private notebook in an interactive CPU session with Internet enabled. Keep the current dataset credentials read-only. Once the owner provisions `novo-nordisk-pbl-outputs`, attach bucket-scoped write credentials as `R2_OUTPUT_ACCESS_KEY_ID` and `R2_OUTPUT_SECRET_ACCESS_KEY`. Set `R2_OUTPUT_BUCKET=novo-nordisk-pbl-outputs` and use a unique run ID. The same R2 endpoint serves both buckets. For the first run use the Gate B synthetic key and SHA-256 from [the verification receipt](verification.md#live-r2-receipt).
```python
import os
import subprocess
from kaggle_secrets import UserSecretsClient

secrets = UserSecretsClient()
os.environ["R2_ENDPOINT_URL"] = "<project R2 account endpoint>"
os.environ["R2_BUCKET"] = "novo-nordisk-pbl"
os.environ["R2_OUTPUT_BUCKET"] = "novo-nordisk-pbl-outputs"
for name in ("R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY",
             "R2_OUTPUT_ACCESS_KEY_ID", "R2_OUTPUT_SECRET_ACCESS_KEY"):
    os.environ[name] = secrets.get_secret(name)

checkout = "/kaggle/working/novo-source"
subprocess.run(["git", "clone", "https://github.com/Strange-Noah/BrainTumour-CNNViT-Segmentation.git", checkout], check=True)
subprocess.run(["git", "-C", checkout, "checkout", "--detach", "<commit>"], check=True)
subprocess.run(["python", "-m", "pip", "install", "-r", "cloud/requirements.txt"], cwd=checkout, check=True)
subprocess.run(["python", "cloud/kaggle_smoke.py", "<run-id>", "<object-key>", "<sha256>", "--device", "cpu"], cwd=checkout, check=True)
```

Save the reported keys and SHA-256 digests for both the normalized NIfTI and the report. In a fresh CPU session, configure `R2_OUTPUT_*` with output-bucket read-only credentials. From the checked-out repository, put `cloud` on `sys.path`, then call `client, bucket = storage.connect(output=True)` and `storage.verify(client, bucket, key, sha256)` for each receipt. Record both digests and the pinned Git SHA in the verification report.

The interim recovery fixture was generated and written by the local R2 writer. It uses the existing dataset bucket's `verification/` prefix, so it can be checked interactively using the existing read-only secrets and `storage.connect()`. The owner confirmed the fixture checksum in an interactive Kaggle CPU session. The automated write uses the separate output bucket. [Recovery fixture receipt](verification.md#local-output-recovery-fixture)

## Gate E

Confirm `hf auth whoami` reports `Mathew-ZHZ`. Creation of the private Gradio Space `Mathew-ZHZ/longitudinal-mri-viewer` returned HTTP 402. CPU Basic has no hourly compute cost, and Hugging Face requires PRO to create this Gradio Space. Resume creation if the owner enables the required plan. A free Static Space would require a browser-only port of the viewer and a new hosted smoke test. Scope the write token to that Space. Configure the GitHub variable and environment secret named in [architecture](architecture.md). Restrict the `hugging-face` environment to `main` and require approval by a designated reviewer. Set `HF_DEPLOY_ENABLED=true` once the Space and its credentials are ready. Launch Deploy MRI viewer from reviewed `main`. Verify the run, the Space's source commit message with GitHub SHA, startup, and the synthetic three-plane preview and volumetry. Only deploy paths under `deployment/longitudinal_viewer`. [Hugging Face GitHub Actions guide](https://huggingface.co/docs/hub/spaces-github-actions)

## Gate F

With a separate GitHub/Kaggle account, follow the same checkout and secret provisioning. Reopen the mirrored Space and reproduce the R2 readback from receipts alone. Download the canonical manifest by its digest, verify it, and recover a completed experiment report and output from R2. Record the checkout SHA and readback evidence in a PR. Runtime storage and Codex chat history are disposable.

## 2.5D U-Net acquisition audit — 2026-11-17

SMP 0.5.0 ResNet34 ImageNet weights were acquired and hashed. Twelve-channel initialization and synthetic CPU inference passed locally and in the private Kaggle CPU runtime. Two real train visits were inspected, plus existing training-partition audit metadata. [Architecture, preprocessing alternatives and acceptance limits](../../experiments/smp_audit/README.md) · [Kaggle evidence](../../experiments/smp_audit/evidence/kaggle-model.json). Exact training positive/negative slice counts remain unavailable from existing histograms; no full-cohort pixel pass was launched. No training or final preprocessing configuration was implemented. Await the separate preprocessing/training decision.

## Full-resolution U-Net benchmark candidate — 2026-11-17

[Specification, implementation and CPU evidence](../../experiments/unet25d/README.md) · [Benchmark status and runtime blockers](../../experiments/unet25d/BENCHMARK.md). Fixed 512×512 / 12-channel full-network fine-tuning code, bounded file cache, DDP entry point, monitoring and checkpoint/recovery stages are prepared. GPU execution and end-to-end recovery remain unverified. Saved Kaggle CPU versions 5/6 cannot retrieve Secrets even though the owner confirmed interactive CPU availability. No GPUs were allocated. Cohort header audit found six native 640×640 cases and 53 oblique acquisitions; preserving full FOV/native resolution within fixed 512 is an explicit unresolved full-cohort conflict. Do not silently resize or exclude these cases. Continue with authenticated CPU cache preparation, then the authorized <=60-minute dual-T4 campaign; do not launch full training.

CPU preparation update: the owner reports successful interactive completion. API-visible private version 7 still contains the preflight script and zero saved files. Preserve the live CPU session and use Quick Save with output files enabled; do not repeat preparation or request GPUs until a separate CPU mount/integrity check passes. [Persistence record and minimal steps](../../experiments/unet25d/CACHE_PERSISTENCE.md).

2026-11-18: Quick Save output persistence is blocked. A chunk-resumable, output-bucket-only R2 fallback is prepared in `experiments/unet25d/persist_cache.py`. Preserve the existing CPU cache; do not preprocess again. See the R2 fallback section in the cache persistence record. GPU allocation remains forbidden until fresh-process remote verification and CPU cache reconstruction/mount checks pass.

2026-11-18 autonomous execution: PR #20 approved/merged. CPU job 44145453007 read private R2, computed counts, wrote output and passed fresh-process readback; independent job 44145990561 recovered the same output. No manual notebook execution. Use main project-execute.yml for authorized main-integrated entrypoints. [Acceptance, diagnosis and GPU continuation](autonomous-execution.md). GPU/cache recovery gates remain open.
