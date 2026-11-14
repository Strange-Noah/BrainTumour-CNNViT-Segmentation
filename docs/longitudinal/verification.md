# Migration and acceptance report

2026-11-14 · Asia/Shanghai

| Gate | Status | Evidence and remaining work |
|---|---|---|
| A · Repository | Verified | GitHub initially reported only main at `6d953c65badd06ff2f81109b068ef016bf723a93`. Created summer-archive and read back the same full SHA via GitHub API. Created a separate migration branch. Summer history includes the binary and 4-label configs, notebooks, source, reports and results. The old deployment used fixed 4-channel shape, model checkpoints, orthogonal slices, a 3D rendering, masks and volumetry. The new viewer adapts the orthogonal overlay and volumetry for aligned 3D NIfTI masks, while the original 3D viewer and inference code remain in the historical deployment. |
| B · R2 | Awaiting endpoint | The `novo-nordisk-r2` AWS profile resolves local credentials. Its config contains region and output settings. The SDK resolves AWS S3 by default, so the smoke command rejected the destination before sending a request. Set the account-level R2 S3 endpoint for `novo-nordisk-pbl` to run live upload and readback. Local S3 stub checks verified conditional writes and conflicting-content rejection. |
| C · TCIA | Blocked | Official collection description and DataCite record reviewed. Repeated direct collection requests timed out. Original MRI and clinical metadata, actual labels, complete patient sequence and a supported headless Aspera transfer remain unverified. Fixture manifest validated ordinal sorting, day offsets and pending integrity. |
| D · Kaggle | Pending B/C | Authenticated list succeeded and returned a notebook under `mathewzhou`, which the owner confirmed. No notebook was submitted. GPU type, quota, runtime limits, scratch disk, transfer rates, R2 upload/readback and report publication remain unverified. A small version-pinned GPU preprocessing script and recovery steps are available. |
| E · Space | Blocked by Hugging Face plan | The authenticated CLI confirmed the owner account `Mathew-ZHZ`. Creation of private Gradio Space `Mathew-ZHZ/longitudinal-mri-viewer` on CPU Basic failed with HTTP 402 because Hugging Face requires PRO for this configuration. The historical TLOMMP Space was untouched. No subscription was purchased. [Hugging Face Spaces requirements](https://huggingface.co/docs/hub/spaces-overview) On local Python 3.14, Gradio 6.1.0 started and its HTTP review endpoint returned all three synthetic slice images and a volume of 0.502 mL. Synthetic and uploaded fixture rendering matched. Fractional labels were rejected. Space Python is pinned to 3.12 and still needs hosted validation. Inference, new Space startup and remote deployment remain unverified. |
| F · Recovery | Prepared | Version-controlled steps, dependency pins and storage conventions are available. A second runtime still needs to perform R2 recovery, reconstruct the dataset manifest and recover a run's outputs. |

## Collaboration audit

GitHub API reported ADMIN access, public visibility, enabled Actions with all actions allowed, no active rules on main and branch protection API response `404 Branch not protected`. Before integration a repository admin should enable a main ruleset requiring PR review and passing applicable validation checks, and restrict branch deletion and force-pushes. Protect summer-archive from deletion and updates. The admin also configures the deployment environment, reviewer access, branch restrictions and scoped Hugging Face credentials. No repository-wide settings were changed. [GitHub branch protection](https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/managing-protected-branches/about-protected-branches)

## Cost and next execution

No paid workload or subscription was initiated. See [architecture](architecture.md) for estimated raw Standard R2 storage and [handoff](handoff.md) for access setup and gate commands.
