# Migration and acceptance report

2026-11-14 · Asia/Shanghai

| Gate | Status | Evidence and remaining work |
|---|---|---|
| A · Repository | Verified | GitHub initially reported only main at `6d953c65badd06ff2f81109b068ef016bf723a93`. Created summer-archive and read back the same full SHA via GitHub API. Created a separate migration branch. Summer history includes the binary and 4-label configs, notebooks, source, reports and results. The old deployment used fixed 4-channel shape, model checkpoints, orthogonal slices, a 3D rendering, masks and volumetry. The new viewer adapts the orthogonal overlay and volumetry for aligned 3D NIfTI masks, while the original 3D viewer and inference code remain in the historical deployment. |
| B · R2 | Local passed, cloud pending | Live upload, HEAD, same-key retry and fresh local process recovery passed. Private Kaggle CPU version 4 stopped at `UserSecretsClient` with `ConnectionError` before an R2 request. Confirm the attachments on the exact notebook and repeat interactively. |
| C · TCIA | Metadata audited, imaging pending | Direct official MRI, clinical and MRI dictionary TSVs were validated via private Kaggle. The clinical dictionary response failed TSV validation. Generated a pending manifest for 1,065 examinations and 300 matched patients. Automatic approval review blocked clinical metadata and manifest upload to R2. TCIA describes binary masks. Actual voxel values, package contents and one complete patient transfer remain unverified. See [acquisition audit](tcia.md). |
| D · Kaggle | CPU runtime verified, R2 cycle pending | Private CPU notebook successfully reached TCIA and produced original metadata files. CPU preprocessing is available. Output write and fresh-session recovery require the Gate B secret readback and a narrowly scoped output writer. GPU work remains pending. |
| E · Space | Blocked by Hugging Face plan | The authenticated CLI confirmed the owner account `Mathew-ZHZ`. Creation of private Gradio Space `Mathew-ZHZ/longitudinal-mri-viewer` on CPU Basic failed with HTTP 402 because Hugging Face requires PRO for this configuration. The historical TLOMMP Space was untouched. No subscription was purchased. [Hugging Face Spaces requirements](https://huggingface.co/docs/hub/spaces-overview) On local Python 3.14, Gradio 6.1.0 started and its HTTP review endpoint returned all three synthetic slice images and a volume of 0.502 mL. Synthetic and uploaded fixture rendering matched. Fractional labels were rejected. Space Python is pinned to 3.12 and still needs hosted validation. Inference, new Space startup and remote deployment remain unverified. |
| F · Recovery | Prepared | Version-controlled steps, dependency pins and storage conventions are available. A second runtime still needs to perform R2 recovery, reconstruct the dataset manifest and recover a run's outputs. |

## Live R2 receipt

Uploaded at 2026-11-14 12:44:17 UTC using the project AWS profile. The synthetic object remains in the private bucket for cloud readback.

- Object key `novo-nordisk/verification/r2-gate-b-20261114-124417-c454f3dc/synthetic.nii`
- SHA-256 `7a9b9082712814448816bacb582c7faabd0dc3a8037910413e0d7c841d571562`
- Size in bytes `480`

## Collaboration audit

GitHub API reported ADMIN access, public visibility, enabled Actions with all actions allowed, no active rules on main and branch protection API response `404 Branch not protected`. Before integration a repository admin should enable a main ruleset requiring PR review and passing applicable validation checks, and restrict branch deletion and force-pushes. Protect summer-archive from deletion and updates. The admin also configures the deployment environment, reviewer access, branch restrictions and scoped Hugging Face credentials. No repository-wide settings were changed. [GitHub branch protection](https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/managing-protected-branches/about-protected-branches)

## Cost and next execution

No paid workload or subscription was initiated. See [architecture](architecture.md) for estimated raw Standard R2 storage and [handoff](handoff.md) for access setup and gate commands.
