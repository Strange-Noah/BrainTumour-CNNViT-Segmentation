# Longitudinal migration acceptance and promotion

`main` becomes the longitudinal project. `summer-archive` preserves the summer project at commit `6d953c65badd06ff2f81109b068ef016bf723a93`, tree `4de354bd7694f12eacf86038c741566978800757`. Start segmentation, lesion tracking and viewer feature branches from reviewed `main` after promotion. A future CNN–ViT baseline belongs in a dedicated experiment PR with longitudinal inputs, patient-level partitions and a documented evaluation protocol.

## Code and architecture acceptance

| Area | Review result |
|---|---|
| Repository structure | Removed the remaining summer application, reports, and figures from the proposed active tree after verifying that GitHub retains the exact archive commit and tree. The longitudinal cloud, visualization and documentation paths remain. |
| Project scope | PR #1 establishes the longitudinal foundation and synthetic/reference-mask viewer. Segmentation training and lesion tracking algorithms remain feature work. The manifest and evaluation conventions preserve patient-level partitioning and distinguish tissue labels from lesion identities. |
| Operational workflows | TCIA is manual from main and uses separate writer and reader environments. Hugging Face is guarded to main and requires explicit enablement and a Space ID. Both use pinned GitHub actions and read-only repository permissions. Kaggle smoke requires a distinct output bucket and credentials. |
| Dependencies and verification | Local fixture checks passed for NIfTI rendering and label validation, manifest sorting and timing, and conditional storage writes. The cloud CLI help commands and dependency consistency check passed. A secret-free PR check exercises the viewer and cloud entry points on Python 3.12 and Ubuntu. |
| Documentation | The handoff and infrastructure audit describe the main-only cutover. The acceptance ledger carries the outstanding real-data and deployment checks. |

PR #1 is the direct promotion path. Promotion places the TCIA workflow on GitHub's default branch, satisfying the manual dispatch registration requirement. The workflow-only registration in PR #2 is superseded. [GitHub manual dispatch requirements](https://docs.github.com/en/actions/how-tos/manage-workflow-runs/manually-run-a-workflow)

## Cutover

1. Review the final PR #1 diff and require a passing `Longitudinal code checks / validate` result for its latest revision. The owner explicitly approves promotion. A reviewer then takes PR #1 out of draft and merges the approved revision.
2. Configure main protection for reviewed PRs, the observed validation status check, dismissal of stale approvals, and restrictions on force-push and deletion. Protect `summer-archive` against updates and deletion. Apply repository protections with the owner or authorized administrator before promotion.
3. Keep `TCIA_PILOT_ENABLED` and `HF_DEPLOY_ENABLED` unset or false during cutover. Before any operational run, replace the migration-branch allowlist in both TCIA environments with exactly the `main` branch. Verify that each environment has its own bucket-scoped credentials, owner review, blocked self-review and no administrator bypass. Configure an equivalent main-only policy for `hugging-face` when deployment is authorized.
4. Verify the promoted commit and archive SHA. A trusted collaborator dispatches the TCIA pilot from main and the designated reviewer approves the exact run SHA at each environment gate. Enable deployment separately once its account requirements are met.

## Runtime acceptance

The persistent 43-file R2 upload and fresh-runner readback follow promotion and the environment cutover. Kaggle output upload and independent recovery need the permanent output bucket and its scoped credentials. Full ingestion needs a complete per-file inventory and validated persistent pilot. Hosted Gradio verification needs an eligible account plan and separate authorization. These are activation gates for their respective workflows. [Runtime verification](verification.md) · [Handoff](handoff.md)
