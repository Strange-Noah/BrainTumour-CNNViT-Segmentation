# Failed baseline v1 — U-Net 2.5D

**Research outcome: FAILED_BASELINE.** This is a preserved unsuccessful research experiment, not a clinically useful or accepted shared model. Continue development on the same `model/unet25d-kaggle` workstream; no v2 campaign is authorized.

[Aggregate results](result.json) · [Exact provenance and private checkpoint receipts](provenance.json) · [Original configuration](config-original.json) · [Recovered configuration](config-recovered.json).

All 220,384 unique training positions were consumed. Final update 165,951; Phase 2 completed one 67,200-example cycle plus 44,318 samples of the next. Two complete native validations cover 45 patients / 159 examinations each. At .5: positive-patient Dice .012689, sensitivity .055053, FP/scan 3.292593; targets not achieved. At .3/.7 sensitivity remains .055053, so small threshold shifts did not recover more lesions. Small-lesion matched counts at .5: 8/992 (<100 mm³), 7/168 (100–<1000), 24/78 (≥1000). Patient-macro metrics and pooled counts are distinct. No test evaluation.

Original provider session ERROR; recovered session COMPLETE. The owner stopped this scientific line before further GPU spending. Latest durable checkpoint SHA `e80dac6f547ca3809e24b3efee15e9e876a396f031c504cb1597651cdda8d971`, 302,041,423 bytes; best positive-Dice SHA `c22a8f4109a0d15b71d267c3899a25d6402cc3b7b31555c5f30015429625d2b4`, 300,416,719 bytes. No acceptable sensitivity/FP checkpoint exists.

[Collection 44878438761](https://github.com/Strange-Noah/BrainTumour-CNNViT-Segmentation/actions/runs/44878438761) independently GET-verified 963 files. [Read-only private audit 44885318566](https://github.com/Strange-Noah/BrainTumour-CNNViT-Segmentation/actions/runs/44885318566) verified the exact collection SHA and durable checkpoint HEAD identities, and extracted only aggregate evidence. Active metadata still says running; that discrepancy is preserved, not interpreted as a live GPU.

For private recovery use the existing authorized CPU runtime, collection receipt and SHA-verifying `cloud.storage.verify`; load only after full GET hash/size agreement. Never publish weights, medical images, signed URLs or patient-level JSON. Collection and immutable source tags remain the primary historical identities. No R2 v1 assets were changed.
