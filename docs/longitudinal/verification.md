# Shared verification record

## Canonical data — complete

The immutable [final ingestion receipt](../../protocol/v1/final-ingestion.json) reconciles 300 patients, 1,065 examinations and 11,405 files (175,705,024,365 bytes), with no missing objects, required modalities or size conflicts. [Full audit](full-dataset-audit.md): 11,333 PASS / 72 WARNING / zero FAIL files. The warnings are 70 empty masks and two explicit unknown-unit masks; raw bytes remain unchanged. [Aggregate audit](../../protocol/v1/full-audit-summary.json). These are existing verified results; reorganization does not rerun ingestion or inspect test images.

## Independent CPU execution and recovery — established

[Execution 44145453007](https://github.com/Strange-Noah/BrainTumour-CNNViT-Segmentation/actions/runs/44145453007) and [independent recovery 44145990561](https://github.com/Strange-Noah/BrainTumour-CNNViT-Segmentation/actions/runs/44145990561) established authenticated CPU execution/persistence. No interactive notebook operation is required. Historical details: [execution record](autonomous-execution.md).

## Archived v1 campaign — durable, scientifically failed baseline

[Collection 44878438761](https://github.com/Strange-Noah/BrainTumour-CNNViT-Segmentation/actions/runs/44878438761) independently read back 963 receipts. [Read-only audit 44885318566](https://github.com/Strange-Noah/BrainTumour-CNNViT-Segmentation/actions/runs/44885318566) verified collection SHA, checkpoint HEAD metadata/size and two complete 45-patient/159-examination validations, using GET/HEAD only. No model weights were loaded or predictions recomputed.

Final completed optimizer step 165,951; Phase 2 cycle index 1, cursor 44,318; 111,518 cumulative Phase 2 samples; no pending validation. Latest checkpoint SHA `e80dac6f547ca3809e24b3efee15e9e876a396f031c504cb1597651cdda8d971`, 302,041,423 bytes. Best nonzero-Dice checkpoint SHA `c22a8f4109a0d15b71d267c3899a25d6402cc3b7b31555c5f30015429625d2b4`, 300,416,719 bytes. Checkpoint bodies remain private.

Phase 1 Dice/sensitivity/FP were zero at all recorded thresholds. The later complete validation had positive-patient macro Dice 0.01268945, lesion sensitivity 0.05505312 and FP/scan 3.29259259 at 0.5. It is not clinically useful or target-achieving. [Complete aggregate threshold grids](https://github.com/Strange-Noah/BrainTumour-CNNViT-Segmentation/blob/b9456ba9e668ec2419218b4b76239c6e35e2ae83/experiments/unet25d/runs/v1/result.json) · [source/configuration and private receipt provenance](https://github.com/Strange-Noah/BrainTumour-CNNViT-Segmentation/blob/b9456ba9e668ec2419218b4b76239c6e35e2ae83/experiments/unet25d/runs/v1/provenance.json).

## Repository verification

See [migration acceptance](../repository/MIGRATION_ACCEPTANCE.md), [preservation ledger](../repository/preservation-ledger.json), [dispatcher tests](../../cloud/test_workstream_route.py) and [handoff](handoff.md). Shared viewer/deployment, frozen split, summer archive and teammate branch identities must remain unchanged. No new GPU allocation, private artifact mutation or scientific change is part of this transition.
