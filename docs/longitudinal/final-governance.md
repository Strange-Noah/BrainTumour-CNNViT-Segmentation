# Final governance and promotion verification

Verified 2026-11-15, Asia/Shanghai.

## Promotion

PR #1 merged the approved head `7bb43f4862fa22cf307ae9a2874d481c499e9a69` into main as `395c7f92bcd63306b34b2fb3a766b66ca7edd50b`. GitHub attributes the merge to `zhukaile5-crypto`. This merge occurred before the owner-only ruleset was installed. Both approved-head CI and post-merge main CI passed. The promoted tree is `d8864391443fd94d6a0800b2a4b4eb92993656cf`.

- [Merge commit](https://github.com/Strange-Noah/BrainTumour-CNNViT-Segmentation/commit/395c7f92bcd63306b34b2fb3a766b66ca7edd50b)
- [Verified tree](https://github.com/Strange-Noah/BrainTumour-CNNViT-Segmentation/tree/395c7f92bcd63306b34b2fb3a766b66ca7edd50b)
- [Main CI](https://github.com/Strange-Noah/BrainTumour-CNNViT-Segmentation/actions/runs/43696891362)

The full recursive tree was checked. It contains the longitudinal viewer, cloud tooling and project records. It excludes the summer application directory, summer reports and figures, and the original src, scripts, configs, outputs and Kaggle training pipelines.

## Enforcement

GitHub accepted an active main-update restriction with only user `Strange-Noah` (232499089) allowed to bypass that update restriction through a PR. A second independent ruleset requires PRs and a successful, up-to-date GitHub Actions `validate` check, requires zero approving reviews, and blocks deletion and force pushes without a bypass list. The owner-only exception applies to the update restriction and cannot bypass the separate PR/CI ruleset.

The collaborator `zhukaile5-crypto` has repository write access for feature branches and PRs and has no bypass entry for main updates. This configuration is supported by the repository's current GitHub service. Preserve the exact user-only bypass when adding teammates. Repository ownership and ruleset administration remain privileged operations.

- [Owner-only update restriction](https://github.com/Strange-Noah/BrainTumour-CNNViT-Segmentation/rules/27310576)
- [Main PR and CI requirements](https://github.com/Strange-Noah/BrainTumour-CNNViT-Segmentation/rules/27304894)
- [Summer archive freeze](https://github.com/Strange-Noah/BrainTumour-CNNViT-Segmentation/rules/27304893)

Summer archive remains at commit `6d953c65badd06ff2f81109b068ef016bf723a93`, tree `4de354bd7694f12eacf86038c741566978800757`. Its active ruleset blocks updates, deletion and force pushes with an empty bypass list.

## Operational cutover

Both `tcia-ingestion` and `tcia-verification` deployment branch policies now allow only `main`. Separate writer and reader secrets remain in their respective environments. The manual TCIA workflow remains gated on `TCIA_PILOT_ENABLED`.

Owner approval, blocked self-review and blocked administrator bypass still protect both credential environments. Under the current policy a trusted collaborator dispatches the main run and the owner approves its two credential jobs. The automated execution uses the fixed reviewed main SHA.

Persistent pilot upload and independent R2 verification, the complete release inventory, full ingestion, Kaggle-originated output write/recovery and hosted Gradio remain open runtime gates. Feature development can branch from main now.
