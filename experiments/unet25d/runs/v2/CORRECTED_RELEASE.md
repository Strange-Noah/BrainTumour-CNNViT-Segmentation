# Corrected v2 recovery release — execution handoff

Owner authorized one corrected Gate1 followed immediately by Phase1 in the same execution, only on genuine PASS. No speculative retry, Phase2, test cohort, paid resource or future-week quota. Old s1/s2 committed executions are CANCEL_ACKNOWLEDGED; owner confirms their interactive sessions stopped.

## Scientific identity and frozen data

New science SHA: `61504c0ca515653bd30cfd63c956e80d525b9a46d3bc5ea4270b5d117f5a099a`. Separate private R2 namespace under that hash. Corrected source is pinned by the trusted registry after normal CI review; this report's commit identifies its reviewed implementation.

Architecture/FP32/dual T4/local1/global2, image preparation, 12-channel order, label3, masks, loss, BN momentum, learning rates and patient split are unchanged. Frozen plan: 220,384 mandatory +73,462 replay =146,923 optimizer steps, digest `359433d8d4e5c8bdfa1008b2db26dd85945ff9683e8638fc4d290e3430d9e2a3`. Catalogue `04b59724c5cc6ac654f85d51ad19e8a8a0766944d8211ecfd03a3495a418de1e`. Accepted existing inventories/plan/shards are referenced into the new namespace only after comparing all non-Gate1/transport config keys with frozen 3ddc324. No full MRI preparation or ingestion. Fresh private package binds new source/science, existing data hashes and short-lived input/output leases. Old incompatible v3 package is not silently reused.

## Corrected capacity gate

Native 512 anchor retained. Deterministic plan-order selection prefers an isolated <100 mm³ real 128 component using conservative component-plane bounding boxes. CPU acceptance confirms actual isolation and gradients through production loading before publication. Original three-lesion crop remains a separately measured, untrained stress example. All actual label3 GT is retained.

Positive probability >=0.6, Dice >=0.1, genuine predicted foreground; tiny selected-component recall >=0.1. Both negative classes retain mean probability <=0.4 and predicted fraction <=0.1. Tiny metrics must independently pass; native-only fitting cannot pass. Record component probability/recall/Dice and whole-crop Dice, selected/incident counts, both-rank health and metrics. Final and periodic capacity evaluation uses rank0 BN buffers broadcast to both ranks; local BN training remains unchanged. Four capacity examples receive equal exposure on each rank; stress is measured separately.

Maximum 4,000 updates /600 seconds, measurement every50 and immediate early PASS. Prior throughput 400/55≈7.27 updates/s makes the old ceiling stop after55s; 4,000 permits about550s without increasing the ten-minute deadline. No threshold relaxation, objective or BN optimization. On failure preserve diagnostic tensors, optimizer, both-rank RNG and complete rank-history metrics; never start formal training or retry. On PASS initialize independent formal model/optimizer/RNG immediately in the same process.

## Supported execution route and limits

Official docs: https://www.kaggle.com/docs/mcp and https://www.kaggle.com/docs/notebooks . SDK create_kernel_session allocates an interactive session via a long-running operation; the available public interface does not establish executing Run All/attaching code execution or listing the existing editor session. It was not invoked, since allocation alone would risk an idle/competing GPU. There is no evidence that it bypasses a queue.

Use one supported `kernels push` of a genuine private notebook containing the immutable launcher as one code cell. This changes packaging, not model science; it is NOT claimed to guarantee faster provider scheduling. No equivalent queued retries. Acknowledgment event is V2_JOB_SUBMITTED. Separate state tracks SUBMITTED, QUEUED, GPU_ALLOCATED (real CUDA/world-size startup), GATE1, FORMAL_TRAINING, COLLECTED and PHASE1_UNATTENDED_ACTIVE. Neither an Actions SUCCESS nor acknowledgment establishes allocation. Any manual editor execution must be independently identified; do not create a second run.

## Private durable checkpoints before long execution

Existing CPU publisher issues immutable object-scoped signed PUT/GET/receipt URLs in a private input lease (no permanent R2 secret). Startup transport probe and diagnostic archive are uploaded and SHA-read back in the GPU runtime. CPU observer independently verifies the actual objects. Periodic formal archive includes model, optimizer, per-rank CPU/CUDA/Python RNG, config/source, committed cursor/counters, metrics and health JSONL. Data caches and signed input leases are excluded. Diagnostic weights are not repeatedly reuploaded with formal checkpoints. Each slot is reserved before upload and protected with If-None-Match; errors stop safely rather than silently losing progress.

Independent CPU archive recovery verifies SHA and identity/cursor, rejects rollback, writes an immutable checkpoint object and advances latest.json. Recovery can inspect R2 receipt keys even when provider logs lag. Normal committed-output collection remains available; completed sessions may fall back to verified live pointers. Errors or failed Gate1 never automatically retry. Existing signed lease covers at most18h against <=12h execution plus margin; no permanent GPU credential.

Before GPU, the same trusted CPU publisher runs a real signed PUT/GET plus independent reader SHA probe, real corrected-example loss/mask checks, and private package publication/readback. These are integrated release checks, not GPU smoke campaigns.

## Budget and unattended behavior

Original approved ceiling remains85,680 wall seconds. New campaign remaining allowance is capped by that ceiling minus actual prior account GPU quota consumption and by current free quota with reserve/safety. Original weekly refresh remains binding. Old failed/interactive optimizer steps do not become formal progress. Old reserved sessions are not counted as new free allowance.

Observe ten minutes from first actual formal update; require finite progression, genuine signal, both-rank BN health, GPU utilization/memory and independently R2-verified formal checkpoint before PHASE1_UNATTENDED_ACTIVE. Existing main half-hourly collector continues exact cursor/state across allowed sessions. Native 45-patient/159-exam validation and original early diagnostic retained. Stop AWAIT_PHASE2_DESIGN.

Status at implementation: CPU/unit code prepared; real cloud acceptance, immutable release pin, input version/expiry, provider allocation, Gate1 and Phase1 results must be filled only from actual runs. No GPU is claimed launched by this document.
