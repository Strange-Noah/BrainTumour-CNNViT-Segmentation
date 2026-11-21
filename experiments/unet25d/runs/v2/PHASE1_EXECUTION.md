# v2 Phase 1 execution

State: GPU job submitted; actual CUDA capacity/ten-minute health results pending. Do not interpret submission alone as actual training or successful learning.

- Approved immutable runtime: `3ddc324e114d0a0cd740234b5ba26ec750238d2a`.
- Science identity: `1d6fa16a689916e4225d561b7f2217e61f0ec423cddb9894b5ac3f8636ee8110`.
- [CPU Gate0 and independent private input readback, 44899144892](https://github.com/Strange-Noah/BrainTumour-CNNViT-Segmentation/actions/runs/44899144892): SUCCESS. Producer `54b34d139e5161be9be26e9e433fdd70cb65b95c`; scientific data/loss/mask/sampling recipe unchanged in the runtime; later changes record private exceptions/framework/zoom and check cache capacity before allocation.
- [Integrated GPU workflow 44906471093](https://github.com/Strange-Noah/BrainTumour-CNNViT-Segmentation/actions/runs/44906471093). Uses current free quota only, provider limit ≤12h, cumulative reservation and no future-week renewal. No additional GPU benchmark.
- [Runtime authorization PR #41](https://github.com/Strange-Noah/BrainTumour-CNNViT-Segmentation/pull/41), [initial authorization #39](https://github.com/Strange-Noah/BrainTumour-CNNViT-Segmentation/pull/39), [private exception bookkeeping #40](https://github.com/Strange-Noah/BrainTumour-CNNViT-Segmentation/pull/40). All normal CI-protected merges; no source/actor/branch restriction loosened.

[Private v2 Kaggle job](https://www.kaggle.com/code/mathewzhou/novo-unet25d-v2-44906471093-s1): submission/version acknowledged by the provider. Initial quota: 30.0 reported hours remaining / 0.0 used; refresh 2026-11-28T00:00:00. Cumulative allowed wall time after reserve/safety: 85,680 seconds (23.8h); first provider session ≤43,200 seconds. These are current free resources, not purchased compute. Initial live log connection returned before training events; observation continues on this same job, not a duplicate GPU submission.

## Actual CPU acceptance

210 training patients / 746 examinations / 9,739 connected label-3 lesions. All 220,384 mandatory positions occur once; 73,462 supplementary positives; 293,846 samples / 146,923 updates. Positive sample fraction 37.649653%; 36,291 both-negative updates, 110,632 mixed updates, no both-positive updates. Negative-only update streak maximum/median/P95 = 1. Both ranks have ≥37.5% positives in every sliding 32-update window. Max two consecutive samples from one patient/exam (the raw audit counts repeated adjacency transitions).

Selected replay caps achieved: max 356/patient, 356/visit, 128/component. Selected 6,563/9,739 lesions for supplementary replay; every lesion remains included through complete mandatory coverage. Do not equate selected replay frequency, optimizer-step foreground exposure and foreground pixel density. Actual scale counts: 512=36,732; 256=22,044; 192=9 escalations; 128=14,677.

Five bounded real training examples passed masked gradients, target/crop alignment and finite FP32 loss. Three positives (128/256/512) retained 3,312/72/50 target pixels. Two negative examples used explicit uncertain-support FOV fallbacks; their losses and negative gradients remained nonzero. Every training component passed geometric FOV audit. MRI-only anatomical support is checked per training occurrence with a recorded GT-preserving fallback; no repeated full MRI preparation was performed. Private input `mathewzhou/novo-unet25d-v2-input` version 1 independently read back exactly.

See [Gate0](GATE0_ACCEPTANCE.json), [sampling](SAMPLING_AUDIT.json), [masks/loss](MASK_AND_LOSS_ACCEPTANCE.json) and [design](DESIGN.md). All patient-level inventories, plans, images, predictions and checkpoints remain private.

## Scientific run boundaries

Gate1 must demonstrate actual positive eval-mode predictions and negative control, then formal model/AdamW/RNG initialize independently. Observe ~ten minutes of actual formal computation. Healthy training continues unattended under the existing main dispatcher; all mandatory coverage and replay precede 45-patient/159-exam native validation. Early fixed diagnostics are not complete validation. Stop at `AWAIT_PHASE2_DESIGN`; Phase2 and test evaluation are not authorized.

Namespace: `novo-nordisk/experiments/unet25d/v2/1d6fa16a689916e4225d561b7f2217e61f0ec423cddb9894b5ac3f8636ee8110/`. Active record `active.json`, recovered checkpoint pointer `latest.json`. No v1 object is reused as training state or overwritten. A live runner checkpoint is not claimed durable before the independent end-of-session R2 collector/readback succeeds.
