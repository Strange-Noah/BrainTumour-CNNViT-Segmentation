# Phase1 transport interruption and bounded recovery

Owner authorized repair and checkpoint continuation on 2026-11-22. No scientific configuration change, Gate1 rerun, Phase2, test cohort or paid resource is included.

## Observed execution

- Actual GPU source: `072b55ee1869d23c3cd4b928833e767c9aee4211`.
- Science SHA: `61504c0ca515653bd30cfd63c956e80d525b9a46d3bc5ea4270b5d117f5a099a`.
- [First corrected execution](https://www.kaggle.com/code/mathewzhou/novo-unet25d-v2-44950328254-s1): provider ERROR, freshly checked 2026-11-22. The terminal failure is `Private shard retrieval failed` in a data worker. The legacy reader suppressed the underlying exception, so the original HTTP/network cause is **UNRESOLVED**. No numerical or memory failure was established.
- Gate1 genuinely passed at 750 diagnostic updates / 111.50 seconds on both ranks. Native positive Dice 0.97959, foreground probability 0.94659, measured component recall 0.96. Isolated tiny-lesion zoom Dice 0.44657, foreground probability 0.99410, recall 0.99671. Both negative controls had zero thresholded foreground. The separately reported difficult three-component stress example remained Dice 0. This is capacity evidence, not cohort generalization.
- Fresh formal model/optimizer followed Gate1 in the same execution. [Initial ten-minute acceptance](https://github.com/Strange-Noah/BrainTumour-CNNViT-Segmentation/actions/runs/44950770290) confirmed 1,400 formal updates and independent R2 persistence. GPU runtime was PyTorch 2.11.0+cu130 / CUDA 13.0 / SMP 0.5.0 on two T4s.
- Last logged formal update: 47,900; mandatory 71,851 / replay 23,949. Last independently retrieved live checkpoint: step 46,784, cursor 93,568, mandatory 70,176, replay 23,392. Complete optimizer steps and both rank RNG states agree. All 210 training patients / 746 examinations had been encountered; mandatory-position coverage was only about 32%. Full 45-patient/159-examination validation has **not** completed.
- Checkpoint SHA `e76428bec9e1d43e8f851fe0703b5c330125ee53c9a710c499d4d2fbf03e380a`, 307,113,167 bytes. Live archive SHA `4ef4b587c4029643e71bdda2d10bf5b3788a7d39682eee35de4f24b544fac0fd`, 330,065,920 bytes. Authorized private output bucket, science namespace, live prefix `1c464cd8eec2430dbdfede96a2a9b4fb`, slot 36. Independent local full SHA and CPU state checks passed; cloud latest-pointer repair is the next operation. Committed provider outputs may contain a newer valid checkpoint; only verified state will be selected.
- CPU re-read of the bounded failure window: 98 objects / 178,694,541 bytes, all SHA-256 correct. This excludes currently persistent bad content in that window, not the original transient cause.

## Confirmed operational defects and correction

1. [Collector failure](https://github.com/Strange-Noah/BrainTumour-CNNViT-Segmentation/actions/runs/44975008104): signed GPU receipts have no prepared-metadata checksum header. Applying the prepared catalogue reader rejected valid receipts. New receipt reader checks exact live-prefix/slot/object binding and schema; full archive hashing remains mandatory. Global metadata integrity checks are unchanged. Only the newest receipt is replayed, avoiding stale-checkpoint rollback and redundant archive downloads.
2. The legacy GET path had no transient retry or safe cause reporting. The v2-only reader now retries up to three attempts for selected temporary service/network failures. Authentication, expiry, integrity, SSL verification and disk faults remain fatal. Signed URLs and patient identifiers never enter diagnostics. Frozen v1 code remains untouched.

## Continuation contract

The CPU repair entry first confirms the old provider job is ERROR, disables its continuation while preserving outputs, independently recovers checkpoint/model/optimizer/RNG and verifies the resume/failure input window. Only an explicitly owner-authorized repair dispatch can adopt the new operational source and resume this exact campaign once. Scheduled recovery cannot silently arm this transition. A second ERROR is preserved and stopped, not automatically retried.

Science, plan digest `359433d8d4e5c8bdfa1008b2db26dd85945ff9683e8638fc4d290e3430d9e2a3`, catalogue, 220,384 mandatory positions, 73,462 replay positions, FP32/DDP and Gate1 acceptance are unchanged. Resume carries the passed Gate1 flag, optimizer and cursor; the uncommitted suffix after the last durable checkpoint is replayed without resetting committed coverage.

The original corrected-campaign budget remains 82,692 seconds, bounded by the original 85,680-second allowance minus previous actual use and the same weekly boundary. Unused reservation is released only after provider termination; actual quota use, including unsaved work and conservative rounding margin, stays charged. At the last read, weekly quota used 6.93h / remaining 23.07h; this is **not** a new 23.07h campaign authorization. Fresh quota is checked before continuation. Input publication reuses existing MRI shards and issues new private, version-verified leases. No preprocessing/ingestion is repeated.

After real resumed updates, observe approximately ten minutes and independently recover a new checkpoint. Healthy continuation returns to the existing collector. Scientific endpoint remains `AWAIT_PHASE2_DESIGN` after complete Phase1 and frozen validation. Source, CPU run, resumed GPU identity and measured health will be appended after actual execution; submission alone is not allocation.
