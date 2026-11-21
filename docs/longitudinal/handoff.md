# Shared project handoff

Main contains common data acquisition, evaluation, deployment and trusted execution controls. Model development continues on existing logical workstreams; see [navigation](active-experiments.md) and [governance](../../AGENTS.md). Do not merge main's model-file deletions into a personal workstream without inspecting the result.

## Dataset and access

Acquisition and the original audit are complete: 300 patients, 1,065 examinations, 11,405 NIfTI files. See [availability](team-data-status.md), [full audit](full-dataset-audit.md), [canonical receipts](../../protocol/v1/final-ingestion.json), [frozen protocol](../../protocol/v1/README.md) and [prediction contracts](model-contracts.md). Do not repeat ingestion/preparation or access the sealed test cohort for development. Teammates receive dataset read-only credentials; outputs have a separate bucket and permissions.

## Execution routing

Workflows run from main, restricted by the existing environments. [Registry](../../cloud/workstreams.json) pins the owner's approved workstream commit. [Trusted resolver](../../cloud/resolve_workstream.py) and [routing controls](../../cloud/workstream_route.py) verify the actual actor, dispatch ref, exact commit and registered branch ancestry before personal dependencies or code execute. Arbitrary branches, SHAs and Kaggle resource names are refused.

[CPU execution](../../.github/workflows/project-execute.yml) retains approved main-ancestor execution and the exact registered personal source alternative. The owner now explicitly authorizes isolated v2 Gate 1 and Phase 1 within currently available free Kaggle quota. The [registry](../../cloud/workstreams.json) enables formal execution and recovery only for that exact v2 source/resources. Archived v1 remains stopped; its active record and namespaces are never reused. v2 requires complete CPU correctness/data/plan receipts before submission, fresh diagnostic/formal initialization, provider and cumulative quota limits, and stops at `AWAIT_PHASE2_DESIGN` after full native validation. No future-week renewal, Phase 2, paid compute or test evaluation is authorized. See the [v2 design and execution records](https://github.com/Strange-Noah/BrainTumour-CNNViT-Segmentation/tree/model/unet25d-kaggle/experiments/unet25d/runs/v2). When registry flags are disabled, schedule/recovery remains a pre-secret no-op and formal launch is refused; the permission boundary is unchanged.

## Archived owner experiment

U-Net 2.5D failed baseline v1, all original code/configurations, benchmark figures and recovery history are preserved on [model/unet25d-kaggle](https://github.com/Strange-Noah/BrainTumour-CNNViT-Segmentation/tree/model/unet25d-kaggle), with [v1 provenance](https://github.com/Strange-Noah/BrainTumour-CNNViT-Segmentation/blob/b9456ba9e668ec2419218b4b76239c6e35e2ae83/experiments/unet25d/runs/v1/provenance.json). It is not an accepted model baseline. The stale private active record was retained unchanged; provider terminal status and durable collection supersede its stale `running` field. No automatic continuation is authorized.

Future v2 needs explicit scientific approval, a distinct campaign identity and separate checkpoint namespace before launch; it must never overwrite or inadvertently resume archived v1. Repository reorganization grants no GPU budget or experiment approval.

## Shared deployment

The longitudinal viewer and its deployment workflow remain unchanged. See [viewer](../../deployment/longitudinal_viewer/README.md) and [architecture](architecture.md). No hosting subscription or new provider is authorized. Summer remains immutable on summer-archive; the historical demo is separate from longitudinal research.

## Verification and repository transition

[Shared verification](verification.md) · [migration evidence and preservation](../repository/MIGRATION_ACCEPTANCE.md) · [owner workflow](autonomous-execution.md). Historical model reports use immutable personal-workstream links, not removed main paths. Remote cleanup is limited to the owner's explicitly enumerated obsolete branches; teammate workstreams are untouched.
