# Lossless repository reorganization — final acceptance

Completed 2026-11-21T17:04:10+08:00 (Asia/Shanghai).

REPOSITORY_REORGANIZATION: PASS  
FAILED_BASELINE_V1_PRESERVED: true  
TEAMMATE_REFS_UNCHANGED: true  
GPU_SUBMISSIONS_DURING_REORGANIZATION: 0

## Integrated main and permanent workstream

- New main: `453109cb0a00715b8b136cb340931f8370d4ca2b`; [merge commit](https://github.com/Strange-Noah/BrainTumour-CNNViT-Segmentation/commit/453109cb0a00715b8b136cb340931f8370d4ca2b). [PR #38](https://github.com/Strange-Noah/BrainTumour-CNNViT-Segmentation/pull/38), normal merge after required CI; no administrative bypass.
- Exact integrated PR head: `c83e475caaa3c225569a5aca447095c1a0b27f26`.
- Published personal implementation/source pin: `b9456ba9e668ec2419218b4b76239c6e35e2ae83`; [permanent workstream](https://github.com/Strange-Noah/BrainTumour-CNNViT-Segmentation/tree/model/unet25d-kaggle). This final record is a subsequent documentation-only commit on that same branch; the dispatcher pin deliberately remains the immutable implementation SHA. The final record commit is identified by this file's GitHub commit history.
- Original recovery anchor: `56fc2a7767e3f7895aab2dccba3a05a5001dab08`; audit-integrated migration base: `68bd4f1fcead6f82b68b79a36c5c9e206f73cce4`. No reset, force push, bulk workstream merge or history rewrite.

## Branch inventory

Original remote inventory: 30 branches (main + two protected branches + 27 owner task branches). After permanent personal publication: 31. After normal review-branch integration and actual cleanup: **four remote heads**, with no new teammate branches discovered.

| Remaining remote branch | Verified commit before final documentation commit |
| --- | --- |
| main | `453109cb0a00715b8b136cb340931f8370d4ca2b` |
| model/unet25d-kaggle | `b9456ba9e668ec2419218b4b76239c6e35e2ae83` |
| summer-archive | `6d953c65badd06ff2f81109b068ef016bf723a93` |
| swinunetr-baseline | `83ecae2ff84e879489ce5178a3d8b3298a527902` |

Summer and teammate branch SHAs are unchanged, including their entire trees. No teammate branch was merged, renamed, rebased, deleted or modified. Temporary audit/review branches `refactor/audit-saved-campaign` and `refactor/shared-main-layout` were removed after integration. Existing unrelated local historical branches were not forcibly deleted.

## Actually deleted owner refs

All 22 listed merged refs were confirmed ancestral to the new main, with unchanged remote heads and owner authorship. Each deletion rechecked the current SHA immediately before the supported GitHub ref DELETE operation.

- `docs/formal-training-ten-minute-handoff`
- `docs/full-cohort-preparation-record`
- `docs/promotion-governance-record`
- `feat/adaptive-two-phase-training`
- `feat/autonomous-execution-bridge`
- `feat/experiment-protocol-v1`
- `feat/full-audit-cnn-vit`
- `feat/smp-unet-acquisition-audit`
- `feat/team-onboarding`
- `feat/unet25d-benchmark`
- `feat/unet25d-full-cohort-inputs`
- `fix/cache-stream-recovery`
- `fix/formal-canonical-observation`
- `fix/formal-normalization-recovery`
- `fix/formal-state-diagnostic`
- `fix/kaggle-private-processing`
- `fix/private-cache-publication`
- `fix/private-input-version-binding`
- `fix/unet-encoder-range`
- `fix/unet-forward-range`
- `fix/unet-native-export`
- `infra/longitudinal-cloud-migration`

The five divergent refs were deleted only after exact remote tag equality and file-level preservation:

| Deleted branch | Permanent archival tag | Commit |
| --- | --- | --- |
| docs/integrated-fp32-results | `archive/fp32-benchmark-report-20261118` | `d066c8dea15b00e5434176d52aba0116af5867bb` |
| docs/training-ready-20261119 | `archive/training-ready-20261119` | `d7e977239c67f585422d24adb983a4b8bc1c69a9` |
| docs/verified-training-continuation | `archive/training-continuation-20261120` | `2d9153f160e84d40a395a7f6f24472b970c19d83` |
| reports/full-audit-cnn-persistence | `archive/cnn-dataset-audit-report` | `62a3edb9b8568ee9fb7313c9a8c6d9cdc3b5d5a3` |
| infra/register-tcia-pilot-dispatch | `archive/initial-tcia-pilot-dispatch` | `551932b816bab64990d61bdc45ed0b89d8023db4` |

[PR #26](https://github.com/Strange-Noah/BrainTumour-CNNViT-Segmentation/pull/26), [PR #36](https://github.com/Strange-Noah/BrainTumour-CNNViT-Segmentation/pull/36) and [PR #17](https://github.com/Strange-Noah/BrainTumour-CNNViT-Segmentation/pull/17) were closed with explicit preservation explanations. PR #17's full-dataset audit and aggregate JSON are exact original blobs on main; its CNN-only evidence is preserved in this branch and the archive tag. PR #36's five changed files were restored selectively because the audit advanced main; no obsolete fast-forward equality was forced.

## File and science preservation

- All 132 main-removal candidates are recorded in the [main preservation ledger](https://github.com/Strange-Noah/BrainTumour-CNNViT-Segmentation/blob/453109cb0a00715b8b136cb340931f8370d4ca2b/docs/repository/preservation-ledger.json). 127 personal blobs are byte-identical; five deliberate navigation/source-authorization/URL/continuation-record changes retain their original blobs in the pre-migration tag and workstream history.
- All **23 divergent changed-file blobs** match the [divergent-file ledger](divergent-file-ledger.json) exactly. Original SVG formatting is preserved; formatting exceptions are restricted to archived files and the exact original SVG.
- Eight archival/source tags are verified remotely. Original and recovered v1 configurations are exact source blobs; science identities, catalogue, split/data-policy and checkpoint identities are unchanged.
- Shared adapters remain. Eight owner CNN pilot files have no imports from the teammate's independently developed Swin implementation; only inherited pilot workflows/docs referenced them. No shared/teammate exception was necessary. All are retained in this branch/history.
- Viewer source, viewer deployment workflow, TCIA pilot workflow, dataset-audit workflow and frozen split files are unchanged byte-for-byte from the migration base. No environment or main/archival protection policy was loosened.

## Private campaign audit and failed baseline v1

[CPU-only read-only audit 44885318566](https://github.com/Strange-Noah/BrainTumour-CNNViT-Segmentation/actions/runs/44885318566) succeeded through the existing authorized runtime. It used R2 GET/HEAD only and verified collection SHA `4fe1028e398817d1dd84597d120a9803ab31a5f7cac564275fe8cd6c2a804ccd`, the independently recovered 963 receipts, durable checkpoint HEAD sizes/hashes and two complete validations. [Independent collection 44878438761](https://github.com/Strange-Noah/BrainTumour-CNNViT-Segmentation/actions/runs/44878438761).

[Failed baseline v1](../../../experiments/unet25d/runs/v1/README.md) · [aggregate results](../../../experiments/unet25d/runs/v1/result.json) · [exact provenance](../../../experiments/unet25d/runs/v1/provenance.json) · [sanitized audit](private-audit-summary.json).

Final step **165,951**; Phase 1 unique positions 220,384; Phase 2 consumed 111,518 (one 67,200-sample cycle plus 44,318 samples); two complete 45-patient/159-examination validations; no pending validation. Latest checkpoint SHA `e80dac6f547ca3809e24b3efee15e9e876a396f031c504cb1597651cdda8d971`, 302,041,423 bytes. Best positive-Dice checkpoint SHA `c22a8f4109a0d15b71d267c3899a25d6402cc3b7b31555c5f30015429625d2b4`, 300,416,719 bytes. Full private artifact locations remain in the authorized provenance; no weights or patient outputs are published.

At 0.5, later full-validation positive-patient Dice was 0.01268945, sensitivity 0.05505312, FP/scan 3.29259259. This is an unsuccessful baseline, not an accepted or clinically useful model. All 0.3/0.5/0.7 aggregates and lesion count denominators are preserved. BN causality for the poor result remains unproven. Provider statuses are terminal (original ERROR, recovery COMPLETE); the stale `running` active record was preserved unchanged. No acceptable tradeoff checkpoint or scientific target completion is claimed.

## Trusted shared workflow and checks

[Registry](https://github.com/Strange-Noah/BrainTumour-CNNViT-Segmentation/blob/453109cb0a00715b8b136cb340931f8370d4ca2b/cloud/workstreams.json) · [resolver](https://github.com/Strange-Noah/BrainTumour-CNNViT-Segmentation/blob/453109cb0a00715b8b136cb340931f8370d4ca2b/cloud/resolve_workstream.py) · [routing](https://github.com/Strange-Noah/BrainTumour-CNNViT-Segmentation/blob/453109cb0a00715b8b136cb340931f8370d4ca2b/cloud/workstream_route.py) · [governance](https://github.com/Strange-Noah/BrainTumour-CNNViT-Segmentation/blob/453109cb0a00715b8b136cb340931f8370d4ca2b/AGENTS.md) · [shared handoff](https://github.com/Strange-Noah/BrainTumour-CNNViT-Segmentation/blob/453109cb0a00715b8b136cb340931f8370d4ca2b/docs/longitudinal/handoff.md).

The updated shared dispatcher is available again, with **auto_recover=false / formal_enabled=false**. Availability does not launch training. Schedules/recovery exit before secrets or personal imports; formal requests are refused even with stale authorization inputs. Only the actual owner on main can resolve the registered exact personal SHA/branch/resources; shared main-ancestor CPU execution remains supported. Registered branch ancestry is verified before personal dependencies or code run. Secret injection remains confined to the execution step. Existing environments, concurrency, CPU limits, pinned actions and GPU budget/quota guards remain intact.

- Personal CPU tests: 33 completed, **32 passed / 1 skipped**. No GPU tests/submissions.
- Shared routing: **14 passed**, including source/actor/ref/resource rejection, schedule/recovery no-op, disabled formal, shared-source support and workflow secret boundaries.
- Real Git ancestry checks: approved personal pin and integrated shared source both accepted.
- Syntax compilation, seven shared CLI entrypoint checks and synthetic viewer (0.502 mL) passed. All 68 installed dependency packages are compatible.
- Shared Markdown link checker: zero missing relative links outside fenced code; surviving workstream references are explicit immutable/branch links, registry values, historical audit namespaces or trusted-main controls.
- PR required CI: [44886837600](https://github.com/Strange-Noah/BrainTumour-CNNViT-Segmentation/actions/runs/44886837600), SUCCESS.
- Merged main CI: [44886969805](https://github.com/Strange-Noah/BrainTumour-CNNViT-Segmentation/actions/runs/44886969805), SUCCESS.

Git download failed on three bounded attempts (empty reply, ref-negotiation timeout, connection timeout). No fourth Git download retry was made. The working authenticated GitHub API supplied the signed merge commit; its exact object SHA, parents and tree were verified locally before advancing the tracking ref. Branch cleanup used supported API ref operations after exact checks. Final documentation publication uses normal fast-forward Git data/ref operations, never force.

## Privacy, resources and next authorization

**No GPU submissions, paid resources, MRI preparation, inference, model optimization, scientific configuration changes or private R2 artifact mutations occurred during reorganization.** No credentials, signed medical URLs, images, checkpoints or new patient-level outputs were published. Existing free CPU audit/CI only.

Future v2 work remains design-only: owner approval is required for scientific changes, distinct experiment/campaign identity and separate checkpoint namespace, plus explicit GPU budget before any experiment or formal run. Do not accidentally resume or overwrite v1. Continue development on this same logical workstream; do not create version/failure/recovery branches or merge shared cleanup deletions into personal model code unreviewed.
