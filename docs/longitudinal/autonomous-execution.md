# Agent-operated cloud execution — acceptance 2026-11-18

**CPU execution acceptance passed. GPU benchmark acceptance remains open.** PR #20 was explicitly approved at `0873e421bd8131452740be68e22a7582cc3aa733`, passed CI and merged as `c55546eac8b663dd02dda76f3c381c32f9980144`. No manual notebook cells were needed for the demonstration or recovery.

## Actual execution and integrity evidence

The agent dispatched [CPU run 44145453007](https://github.com/Strange-Noah/BrainTumour-CNNViT-Segmentation/actions/runs/44145453007) through `gh workflow run project-execute.yml`. It completed successfully on the existing standard GitHub Actions `ubuntu-24.04` CPU runner. Source was exactly `0873e421bd8131452740be68e22a7582cc3aa733`; the workflow itself came from merge `c55546eac8b663dd02dda76f3c381c32f9980144`.

The job read the private canonical R2 manifest (810,465 bytes), independently checked SHA-256 `040ec6c523d5beae43a30b64fede2bee226e1a08e8c14b2cca8043b01b9c39f4`, parsed it and checked 300 unique patients / 1,065 visits. It wrote a 488-byte aggregate result into private output bucket `novo-nordisk-pbl-outputs`, key `novo-nordisk/execution/acceptance/44145453007/result.json`, using conditional create-once PUT and full GET/SHA verification. A new Python process then created a new client and verified that persisted object again.

Output SHA-256: **`e9b94972bd24b6fa9723afd66f24aaadfca7dd1172a5d3791030b8d958a2ecd9`**. The agent next launched an entirely separate [recovery job 44145990561](https://github.com/Strange-Noah/BrainTumour-CNNViT-Segmentation/actions/runs/44145990561); it also completed successfully, retrieving and checking that same output without the first runner's filesystem. [Original receipt](../../protocol/v1/execution/acceptance-44145453007.json) · [Independent-job receipt](../../protocol/v1/execution/recovery-44145990561.json).

The first job ran 02:13:20–02:13:35 Asia/Shanghai on November 18 (the underlying Actions timestamps are UTC November 17). These are cloud execution results, not local tests or submission acknowledgments. No GPU or new paid resource was allocated. Dataset credentials were used only for GET; existing dataset-reader and output-writer scopes were unchanged. No credentials, medical images or per-patient clinical values appear in logs or these public reports.

## Repeat, monitor, fix and recover

[Workflow implementation](https://github.com/Strange-Noah/BrainTumour-CNNViT-Segmentation/blob/0873e421bd8131452740be68e22a7582cc3aa733/.github/workflows/project-execute.yml) · [Acceptance computation](https://github.com/Strange-Noah/BrainTumour-CNNViT-Segmentation/blob/0873e421bd8131452740be68e22a7582cc3aa733/cloud/execution_demo.py).

```sh
gh workflow run project-execute.yml --repo Strange-Noah/BrainTumour-CNNViT-Segmentation \
  --ref main -f source_sha=0873e421bd8131452740be68e22a7582cc3aa733 \
  -f entrypoint=cloud.execution_demo
```

`gh run list`, `gh run view RUN --json status,conclusion,jobs`, and `gh run view RUN --log` provide agent-controlled state, failures and logs. For another authorized project module, supply its `cloud.*`, `research.*` or `experiments.*` entrypoint, JSON string-list arguments and project-relative dependency file. Arguments must not contain credentials. Failed jobs may be dispatched again with the same pinned source; output run IDs isolate acceptance results. To recover an existing output, use the same entrypoint with JSON arguments `["--verify","KEY","SHA256"]`, as demonstrated by the second job.

The workflow has read-only GitHub token permissions, no persisted checkout credential, a 45-minute job limit and 40-minute entrypoint limit, and uses existing `tcia-verification` environment secrets. Source must be an exact commit already integrated into main, checked **before** injecting R2 credentials. Fixes follow ordinary feature PR integration, then dispatch at the new integrated SHA. No new human approval gate or environment policy was added. The guard preserves the existing boundary against executing unintegrated branch code with private-data credentials. CPU preparation jobs requiring more resources/time need separately measured bounds; this acceptance does not establish full preprocessing throughput.

## Kaggle diagnosis: capabilities tested separately

The agent launched [private interface probe](https://www.kaggle.com/code/mathewzhou/novo-execution-interface-probe) by supported API submission, without modifying the original interactive notebook. It executed to completion and produced [sanitized evidence](../../protocol/v1/execution/kaggle-interface-probe.json). Public HTTPS returned **200**, a runtime Secrets token was present, and the nested cause of `ConnectionError` was **HTTP 400** from Secrets lookup. The same job programmatically mounted another private notebook's output using `kernel_sources`, read its synthetic CPU-check artifact and validated its content.

This distinguishes a service rejection from missing internet or total API authentication failure. The new probe had no manually attached secrets; it does not prove the exact server-side binding state of historical versions 5/6/7. Those original jobs logged only the outer error type. We therefore do not claim every historical `ConnectionError` had the same status. What is now proven is that notebook-submission authentication, private input access and runtime secret authorization are distinct capabilities.

The installed official CLI's `ApiSaveKernelRequest` populates source, visibility, internet, accelerator, private dataset sources and private kernel sources; its submission path has no secret-attachment field. The official [runtime Secrets client](https://github.com/Kaggle/docker-python/blob/main/patches/kaggle_secrets.py) delegates to [web client code](https://github.com/Kaggle/docker-python/blob/main/patches/kaggle_web_client.py). That implementation catches `URLError` before its `HTTPError` handler; because `HTTPError` subclasses `URLError`, HTTP failures can be relabeled as `ConnectionError`. Our actual error-chain probe confirms this behavior in the runtime. Do not diagnose the failure as generic network loss or reuse the API token as a runtime Secrets token.

Supported API submission/status/output retrieval and private notebook-input mounting work. No supported public API path for arbitrary execution inside the owner's existing live interactive session was established. The supported desktop computer-use connection was tested again and timed out; we did not extract cookies, impersonate runtime tokens, install tunnels or bypass access controls. The original notebook still exposed zero saved output files when rechecked. Its live cache is not proven lost, but currently cannot be recovered by the verified API path. Do not ask the owner to run operational cells.

## Selected execution architecture and GPU continuation

| Path | Autonomous execution | Private R2 | GPU / current status | Decision |
|---|---|---|---|---|
| Existing GitHub Actions standard CPU | Demonstrated, including independent recovery | Existing scoped environment secrets; demonstrated | CPU only | Selected for CPU and orchestration |
| Kaggle API jobs with private inputs | Submission, completion, output retrieval and private mounting demonstrated | Runtime Secrets binding is not verified | Free T4×2 remains candidate; no GPU allocated | Use private prepared inputs and collect outputs through the API, keeping R2 credentials in Actions |
| Existing interactive Kaggle session | Owner execution works; agent browser connection unavailable | Owner confirmed interactive Secrets | Not autonomously accessible here | Do not depend on manual cells or lost session state |
| New paid cloud GPU | Not provisioned or tested | Would require scoped integration | No paid authorization | Excluded |

The smallest secure GPU bridge is Actions CPU preparation/recovery → **private** Kaggle cache input → API-submitted bounded GPU job without R2 secrets → private Kaggle checkpoint output → Actions R2 persistence and independent readback. This uses supported dataset/kernel input fields; it does not put R2 keys into a private dataset. A minimal cloud-side Kaggle API credential is still needed for publishing private inputs and collecting results without routing medical data through the Mac. The existing local Kaggle API login alone does not make a credential available to GitHub runners. A supported [Kaggle API token](https://github.com/Kaggle/kaggle-cli/blob/main/skills/references/auth.md) may be stored once as `KAGGLE_API_TOKEN` in the project environment and exposed only to the orchestration step, not GPU notebooks; that one-time cross-service credential authorization has **not** been performed or assumed. It grants Kaggle account API capabilities, not merely one notebook's Secrets access.

PR #19's benchmark/preparation code also needs its normal reviewed main integration before this guarded CPU runner may execute it. PR #20 approval did not authorize substituting a different PR revision. Once code and the narrowly exposed Kaggle integration credential are approved, first retry supported API recovery of the original outputs. If still inaccessible, the owner has authorized exactly one autonomous regeneration of the bounded cache in the now-verified CPU environment. Preserve source/config/manifest/split hashes, stream existing R2 cache chunks when available, and verify the private prepared input in a CPU job **before** requesting T4×2. Respect the existing <=60-minute GPU campaign cap; do not start full training.

CPU autonomy is accepted; cache recovery, the cloud-to-Kaggle private dataset publisher, full CPU cache throughput, GPU execution, sustained memory stability and the training checkpoint/resume loop remain explicitly unverified. These are research execution follow-ups, not reasons to describe the demonstrated CPU bridge as merely a proposal.
