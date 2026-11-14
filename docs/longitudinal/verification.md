# Migration and acceptance report

Updated 2026-11-15 · Asia/Shanghai

| Gate | Status | Evidence and remaining work |
|---|---|---|
| A · Repository | Verified | GitHub initially reported only main at `6d953c65badd06ff2f81109b068ef016bf723a93`. Created summer-archive and read back the same full SHA via GitHub API. Created a separate migration branch. Summer history includes the binary and 4-label configs, notebooks, source, reports and results. The old deployment used fixed 4-channel shape, model checkpoints, orthogonal slices, a 3D rendering, masks and volumetry. The new viewer adapts the orthogonal overlay and volumetry for aligned 3D NIfTI masks, while the original 3D viewer and inference code remain in the historical deployment. |
| B · R2 | Passed | Local upload, HEAD, same-key retry and fresh local process recovery passed. The CLI-saved Kaggle CPU run failed at Kaggle Secrets. The project owner then ran the readback interactively in the private Kaggle CPU notebook with bucket-scoped read-only credentials and reported `{"verified": true}`. The readback cell reports success after SHA-256, voxel and affine checks pass. |
| C · TCIA | Metadata stored, imaging pending | The four official TSVs and minimized 1,065-visit manifest were written to private project R2 and each passed SHA-256 readback. The Faspex public package landing page responded. A complete four-visit patient transferred to private Kaggle scratch. All 43 files matched official sizes and were hashed. All four masks matched their T1c grids and contained values 0, 2 and 3. TCIA identifies labels 2 and 3 as surrounding non-enhancing FLAIR hyperintensity and enhancing tissue. The pilot ingestion and independent readback commands are ready. GitHub detected the draft-branch workflow and skipped execution while disabled. The two branch-restricted environments are configured. Both credential names are present in each environment. The pilot now accepts manual dispatch only. Both environments require owner approval with self-review and administrator bypass disabled. GitHub requires the workflow on the default branch before manual dispatch. The pilot remains pending. [Infrastructure audit](infrastructure-audit.md) · [Workflow run](https://github.com/Strange-Noah/BrainTumour-CNNViT-Segmentation/actions/runs/43618376777) See [acquisition audit](tcia.md). |
| D · Kaggle | CPU runtime verified, R2 write pending | The owner selected permanent output bucket `novo-nordisk-pbl-outputs` with separate scoped credentials. The existing profile received AccessDenied on creation of an output bucket. A locally written synthetic output passed R2 readback and the owner reported successful checksum recovery in a fresh Kaggle CPU session. A Kaggle-originated write remains pending. |
| E · Space | Blocked by Hugging Face plan | The authenticated CLI confirmed the owner account `Mathew-ZHZ`. Creation of private Gradio Space `Mathew-ZHZ/longitudinal-mri-viewer` on CPU Basic failed with HTTP 402 because Hugging Face requires PRO for this configuration. The historical TLOMMP Space was untouched. No subscription was purchased. [Hugging Face Spaces requirements](https://huggingface.co/docs/hub/spaces-overview) On local Python 3.14, Gradio 6.1.0 started and its HTTP review endpoint returned all three synthetic slice images and a volume of 0.502 mL. Synthetic and uploaded fixture rendering matched. Fractional labels were rejected. Space Python is pinned to 3.12 and still needs hosted validation. Inference, new Space startup and remote deployment remain unverified. |
| F · Recovery | Prepared | Version-controlled steps, dependency pins and storage conventions are available. The owner verified recovery of the synthetic R2 input in a second runtime. The owner also recovered the synthetic output from the local writer in Kaggle. Manifest reconstruction in a fresh session and recovery of Kaggle-produced output remain pending. |

## Live R2 receipt

Uploaded at 2026-11-14 12:44:17 UTC using the project AWS profile. The synthetic object remains in the private bucket for cloud readback.

- Object key `novo-nordisk/verification/r2-gate-b-20261114-124417-c454f3dc/synthetic.nii`
- SHA-256 `7a9b9082712814448816bacb582c7faabd0dc3a8037910413e0d7c841d571562`
- Size in bytes `480`

## Local output recovery fixture

The local writer generated and stored this 297-byte synthetic normalized NIfTI in `novo-nordisk-pbl` and verified the stored SHA-256. The owner reported successful checksum verification in a fresh interactive Kaggle CPU session.

- Object key `novo-nordisk/verification/r2-output-recovery-20261115/normalized.nii.gz`
- SHA-256 `77bb3dcb02deb0d1dde3a9b0e716ce53150a2693928bcb3ae7fde159c37ae95f`

## Collaboration audit

GitHub API reported ADMIN access, public visibility, enabled Actions with all actions allowed, no active rules on main and branch protection API response `404 Branch not protected`. Before integration a repository admin should enable a main ruleset requiring PR review and passing applicable validation checks, and restrict branch deletion and force-pushes. Protect summer-archive from deletion and updates. The admin also configures the deployment environment, reviewer access, branch restrictions and scoped Hugging Face credentials. Repository environment rules were configured for the two TCIA jobs. [GitHub branch protection](https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/managing-protected-branches/about-protected-branches)

## Cost and next execution

No paid workload or subscription was initiated. See [architecture](architecture.md) for estimated raw Standard R2 storage and [handoff](handoff.md) for access setup and gate commands.
