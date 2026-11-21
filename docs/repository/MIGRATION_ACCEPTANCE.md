# Lossless shared-main reorganization

## Scope and preservation

Owner-authorized repository restructuring only: no GPU submission, MRI preparation, science/configuration changes, private R2 writes or new credential permissions. Existing shared viewer/deployment and frozen split remain byte-identical. Governance is in [AGENTS.md](../../AGENTS.md).

Original recovery main `56fc2a7767e3f7895aab2dccba3a05a5001dab08` is retained. The separately reviewed GET/HEAD-only campaign audit was integrated in [PR #37](https://github.com/Strange-Noah/BrainTumour-CNNViT-Segmentation/pull/37), advancing the migration base to `68bd4f1fcead6f82b68b79a36c5c9e206f73cce4`; main was not reset to the old anchor.

Before removals, the complete permanent [personal workstream](https://github.com/Strange-Noah/BrainTumour-CNNViT-Segmentation/tree/b9456ba9e668ec2419218b4b76239c6e35e2ae83) was published and its remote SHA verified as `b9456ba9e668ec2419218b4b76239c6e35e2ae83`. All 132 removal candidates have a recorded original blob and preservation destination in the [ledger](preservation-ledger.json). Unchanged source blobs are exact. Five intentional differences are explained individually; their original blobs remain in the workstream's history and `archive/pre-workstream-reorganization-20261121`.

All 23 changed-file blobs from five divergent branches were separately copied and hash-checked before cleanup. [Divergent-file ledger](https://github.com/Strange-Noah/BrainTumour-CNNViT-Segmentation/blob/b9456ba9e668ec2419218b4b76239c6e35e2ae83/docs/archive/repository-migration/divergent-file-ledger.json). Original SVG trailing whitespace is preserved with narrowly scoped formatting exceptions, not rewritten.

## Immutable history references

| Reference | Original commit |
| --- | --- |
| archive/pre-workstream-reorganization-20261121 | 68bd4f1fcead6f82b68b79a36c5c9e206f73cce4 |
| archive/fp32-benchmark-report-20261118 | d066c8dea15b00e5434176d52aba0116af5867bb |
| archive/training-ready-20261119 | d7e977239c67f585422d24adb983a4b8bc1c69a9 |
| archive/training-continuation-20261120 | 2d9153f160e84d40a395a7f6f24472b970c19d83 |
| archive/cnn-dataset-audit-report | 62a3edb9b8568ee9fb7313c9a8c6d9cdc3b5d5a3 |
| archive/initial-tcia-pilot-dispatch | 551932b816bab64990d61bdc45ed0b89d8023db4 |
| unet25d/v1-original-source | e818635a9e7df4d91792afaad7fc9ff3dd4b2a23 |
| unet25d/v1-recovery-source | 56fc2a7767e3f7895aab2dccba3a05a5001dab08 |

PR #36 could not fast-forward after the audit integration. Its five changed files were selectively restored on the personal branch, with exact original and divergent copies preserved; no bulk merge was used. The shared full-dataset audit and aggregate JSON from PR #17 were selectively restored on main; CNN-specific report and all seven divergent report blobs remain preserved on the personal branch.

## Ownership and retained shared components

The eight CNN pilot files were authored in owner PR #14/15 and were inherited unchanged by the teammate branch. Its independently developed Swin files have no imports or active implementation dependency on that pilot; only the inherited pilot workflows/docs reference it. Those workflows move with the pilot to the personal workstream. No teammate ref or file was changed. Common CNN/Swin/nnU-Net adapter examples and the reference smoke remain shared. No pilot exception was necessary.

Protected remote identities: summer-archive `6d953c65badd06ff2f81109b068ef016bf723a93`; swinunetr-baseline `83ecae2ff84e879489ce5178a3d8b3298a527902`. They must match again after remote cleanup.

## Archived campaign evidence

[Read-only CPU audit 44885318566](https://github.com/Strange-Noah/BrainTumour-CNNViT-Segmentation/actions/runs/44885318566) succeeded using existing authorized credentials, verifying the [independent 963-object recovery](https://github.com/Strange-Noah/BrainTumour-CNNViT-Segmentation/actions/runs/44878438761), latest/best checkpoint receipts and two complete validation summaries. Bodies and patient records remain private. [Failed baseline v1](https://github.com/Strange-Noah/BrainTumour-CNNViT-Segmentation/blob/b9456ba9e668ec2419218b4b76239c6e35e2ae83/experiments/unet25d/runs/v1/README.md) · [results](https://github.com/Strange-Noah/BrainTumour-CNNViT-Segmentation/blob/b9456ba9e668ec2419218b4b76239c6e35e2ae83/experiments/unet25d/runs/v1/result.json) · [provenance](https://github.com/Strange-Noah/BrainTumour-CNNViT-Segmentation/blob/b9456ba9e668ec2419218b4b76239c6e35e2ae83/experiments/unet25d/runs/v1/provenance.json). Stale active metadata is preserved, not rewritten to claim a stop.

## Trusted routing and verification

The [registry](../../cloud/workstreams.json) pins the published personal SHA, owner actor, branch, module and private resource names. Both `auto_recover` and `formal_enabled` are false. [Resolver](../../cloud/resolve_workstream.py), [routing](../../cloud/workstream_route.py) and [CPU-only routing tests](../../cloud/test_workstream_route.py) enforce trusted-main authorization before personal checkout, dependencies, imports or secrets. Schedule/recovery exits before credentials. Formal dispatch is refused. GPU budget/quota safeguards are retained in the unchanged personal implementation. No GPU-capable workflow was dispatched for this migration.

Personal control/pipeline/readiness/recovery checks: 33 tests completed, 32 passed and 1 skipped. Default source rendering remains unchanged, with explicit repository overrides supported. Shared checks passed: 14 routing tests, syntax compilation, seven CLI help checks, synthetic viewer (0.502 mL), all 68 installed package dependencies, zero missing relative Markdown links, and unchanged viewer/deployment/split Git blobs. Existing 132 removal records and 23 divergent blobs were verified. Final CI/merge/cleanup identities are recorded in the final acceptance record below.

## Cutover and cleanup record

The authorized cutover removes only 22 already-merged owner task refs and five exactly tagged divergent owner refs, after checking current remote identities and ancestry; PRs #17/#26/#36 are closed only after their work is preserved. The temporary review branch is deleted after normal CI-protected merge. No force push or history rewrite is used. Shared workflow availability is restored only after disabled-by-default routing is integrated, without launching training.

The final post-merge branch inventory, exact merge/personal SHAs, CI links, deletion receipts and final test counts are recorded on the established personal workstream in [FINAL_ACCEPTANCE.md](https://github.com/Strange-Noah/BrainTumour-CNNViT-Segmentation/blob/model/unet25d-kaggle/docs/archive/repository-migration/FINAL_ACCEPTANCE.md). This section specifies the cutover procedure; the linked record is the authority for its completed outcome.

Future v2 requires scientific strategy approval, an explicit GPU budget, distinct experiment identity and separate checkpoint namespace. No future experiment is launched by repository acceptance.
