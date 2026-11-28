# LesionLocator zero-shot tracking — terminal results and handoff

## Execution and provenance

The experiment uses the frozen UCSD-BMETS validation set of 45 patients, 159 examinations and 114 adjacent pairs. T1c and raw enhancing label 3 define the inputs and reference. Visit order defines the longitudinal steps. The sealed test set is outside the experiment. The original [G0/L0 baseline](CPU_RESULTS.md) covers all 114 pairs and supplies the paired L0 results.

- [Official asset retrieval](https://github.com/Strange-Noah/BrainTumour-CNNViT-Segmentation/actions/runs/46031603433) verified the selected ZIP members, persisted them in private R2 and checked SHA-256 readback. The 1,384,731,741-byte fold 0 checkpoint is `f00f2e71392a88f922fcdfa852b71c9fdbdf9c14c33472a21c104398f2266596`.
- [Loaded checkpoint configuration](https://github.com/Strange-Noah/BrainTumour-CNNViT-Segmentation/actions/runs/46056070058) identifies `LesionLocatorTrainerTrack`, `3d_fullres` and 1,145 named tensors. Working spacing is [3, 0.841796875, 0.841796875] mm in the official reader convention, with a 96×256×256 local patch and 175³ registration array. The latter uses case-specific window and resampling coordinates. Fixed CT normalization clips [−33, 134], subtracts 58.50637769 and divides by 30.87594623. The official image/label/logit routes use orders 3/1/1, with label-specific interpolation and separate-axis order 0 where selected.
- The first GPU job reached TrackNet and stopped at the upstream ROI padding call with CUDA out-of-memory. [The spatial-axis correction](https://github.com/Strange-Noah/BrainTumour-CNNViT-Segmentation/pull/55) preserves official zero padding, weights, geometry targets and working precision. CPU tests confirmed inversion of the corrected spatial padding and exact agreement for valid depth-only padding. The [GPU execution source](https://github.com/Strange-Noah/BrainTumour-CNNViT-Segmentation/commit/9fff1e41eb6a22000c6fd04984dbced6e05ae96f) uses fold 0, native T1c, prev_mask, and no mirroring. The follow-up entered the forward path without target GT. GT-based observations ran independently for scoring.
- [Corrected GPU dispatch](https://github.com/Strange-Noah/BrainTumour-CNNViT-Segmentation/actions/runs/46057677602) independently verified private Kaggle input v2 at package digest `b7043022ebb85608142b369afbc539bff26c3f5337e15e6b3923fe3baa0acc1b`. Its two real smoke pairs passed technical checks before the same session entered the formal queue. [Handoff](EXECUTION.md) records the initial dual-T4 observation and recovery.
- [Terminal independent audit](https://github.com/Strange-Noah/BrainTumour-CNNViT-Segmentation/actions/runs/46123366262) resolves frozen baseline references, verifies committed archive bytes and SHA-256, reconciles patient slots and retains the detailed private inventory. This audit reuses recorded predictions, geometry and traces. It launches no model or new Kaggle compute.

## Completion and independent recovery

| Workstream | Verified patients | Verified exams | Verified adjacent pairs | Remaining to frozen target |
| --- | ---: | ---: | ---: | --- |
| G0/L0 baseline | 45 | 159 | 114 | 0 |
| Formal L1/L2 GPU | 38 | 136 | 98 | 7 patients, 23 exams, 16 pairs |
| Independent G1–G5 CPU | 25 | 88 | 63 | 20 patients, 71 exams, 51 pairs |
| GPU smoke | 2 | 4 | 2 | 0 of the 2 selected smoke pairs |

The smoke pairs reappear at the start of the formal queue. They are excluded from the formal denominator. The independent audit found zero archive/receipt orphans across 65 committed units, compared each archived pair with the frozen L0 record, and independently verified every object's bytes and SHA-256. The private receipt retains the completed slots, data digests and remaining slots. No medically identifying records, arrays or signed URLs are published here.

Kaggle reported `CANCEL_ACKNOWLEDGED` for both terminal jobs. The [GPU execution](https://www.kaggle.com/code/mathewzhou/novo-lesionlocator-t-46057677602) ended at 10,211.85 s of provider log time with a 10,200 s session cap. Its last committed patient was recorded at 10,054.27 s. The [CPU geometry execution](https://www.kaggle.com/code/mathewzhou/novo-lesionlocator-g-46056070058) recorded its last commit at 5,359.11 s with a 5,400 s session cap. The terminal evidence is consistent with both configured limits. It does not identify a separate cancellation reason. No new inference or geometry computation was submitted during terminal recovery.

## Paired formal results

All four arms use the same 98 adjacent pairs, including 87 with target GT foreground, and the same 426 target lesion components. Spatial match uses the shared one-to-one positive-overlap assignment. False positives count unmatched predicted union components. The 11 GT-empty pairs contribute separately. Empty–empty masks score Dice 1 under the shared evaluator, so positive-only Dice is also shown. Patient macro averages the pair Dice within each patient and then across patients.

| Arm | Mean Dice | Positive-only Dice | Patient macro Dice | Matches / 426 | Sensitivity | FP / pair | FP on GT-empty / positive pairs |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| L0 | 0.33575 | 0.32073 | 0.33046 | 268 | 62.91% | 2.255 | 18 / 203 |
| L1-Warp | 0.16457 | 0.12791 | 0.16707 | 171 | 40.14% | 3.704 | 15 / 348 |
| L1-Final | 0.10552 | 0.00392 | 0.09071 | 6 | 1.41% | 0.398 | 1 / 38 |
| L2 | 0.09233 | 0.00055 | 0.07937 | 2 | 0.47% | 0.184 | 2 / 16 |

| Arm | Mean HD95 on defined GT-positive pairs | Defined / 87 | Mean absolute volume error |
| --- | ---: | ---: | ---: |
| L0 | 30.91 mm | 85 | 2,404 mm³ |
| L1-Warp | 33.59 mm | 85 | 2,383 mm³ |
| L1-Final | 80.24 mm | 29 | 2,853 mm³ |
| L2 | 72.67 mm | 12 | 2,952 mm³ |

HD95 denominators differ because the shared metric is undefined for a positive GT with empty prediction. The conservative unique mutual same-grid GT overlap proxy covers 234 historical instances. Mean paired per-instance Dice is 0.39351, 0.10012 and 0.00200 for L0, L1-Warp and L1-Final. These proxy comparisons use spatial overlap references. Independent longitudinal identity annotations would be necessary for identity accuracy or new-versus-resolved lesion claims.

## G1–G5 and normalization

| Stage | Instance count | Disappeared | Native instance mean Dice | Mean absolute volume error | Median centroid shift among surviving instances |
| --- | ---: | ---: | ---: | ---: | ---: |
| G1 official preprocessing round trip | 342 historical | 52 | 0.58677 | 16.88 mm³ | 0.395 mm |
| G2 with official Z-window | 342 historical | 52 | 0.58677 | 16.88 mm³ | 0.395 mm |
| G3 intermediate 175³ array | 342 historical | 54 | — | — | — |
| G4 full geometry round trip | 342 historical | 54 | 0.57819 | 17.59 mm³ | 0.404 mm |
| G5 ideal-logit export | 293 follow-up | 42 | 0.58494 | 16.17 mm³ | 0.412 mm |

G3 reports presence in registration array coordinates. G4 restores native geometry before comparison. G5 uses ideal logits from diagnostic follow-up GT, separately from model inference. The independent CPU G2 used FP16 inputs and the official window procedure. The two GPU smoke diagnostics used the CUDA execution conditions and lost 3/11 historical instances at G1–G4, with no additional loss at G2 in those smoke cases.

| Region | Formal voxels | Median across visit-level raw medians | Above clip ceiling | Below clip floor |
| --- | ---: | ---: | ---: | ---: |
| Whole image | 6,590,169,088 | 0 | 10.7808% | 0% |
| Nonzero MRI | 721,357,173 | 1,918.5 | 98.4910% | 0% |
| Raw label 3 | 1,628,967 | 2,765.0 | 99.9258% | 0% |
| Nonzero perilesional shell | 1,461,946 | 1,873.0 | 98.0459% | 0% |

Whole-image and nonzero calculations cover 136 formal exams. Foreground and shell comparisons cover the 123 visits with valid lesion-region statistics. Median absolute lesion–shell mean contrast was 802.71 raw intensity units and 0.000557 after the official transform. The near-zero transformed contrast is descriptive of these visits. It does not isolate the causal effect of normalization on model performance. Independent CPU geometry showed similar saturation, with 99.8718% of lesion voxels above the upper clip across its 81 lesion-positive visits. Representative overlays and raw distributions remain in the private verified archives.

## Resource use and handoff

The formal results account for 16,261.85 combined worker seconds and 98 completed pairs, or 0.362 pairs per worker-minute. Measured totals were 5,624.96 s preprocessing, 342.49 s CUDA registration, 226.71 s CUDA U-Net, 1,233.11 s native export and 78.29 s MRI reads. Peak allocated GPU memory was 4.63 GiB, peak PyTorch reserved memory was 8.22 GiB and sampled per-worker RSS reached 4.71 GiB. The corrected session's provider log spans about 2 h 50 min with two T4s. The earlier failed attempt consumed about 0.10 h of displayed Kaggle quota. The current quota window has reset to 30.00 h available. A final previous-window quota ledger was not available. No paid workloads or training were started.

The recovered cohort slice shows that direct physical propagation outperformed official learned Warp and Final, that local foreground is sparse despite full observed ROI coverage, and that autoregressive propagation further reduces recall. These are paired descriptive results on the recovered slice. Full-cohort extrapolation, causal normalization claims and scored longitudinal identity accuracy remain open. Further work on missing cases can use the private receipt inventory and the immutable producer sources. Existing scientific artifacts remain untouched.

## T0–T8 — Where foreground was lost

| Stage | Observation in the recovered L1 cohort |
| --- | --- |
| T0/T1 historical prompt and preprocessing | 489 historical instance traces. 61 prompts were empty after preprocessing and the official predictor skipped them. |
| T2 z-window | 428 recorded calls. Their actual windows did not truncate the work tensors. |
| T3 registration-grid prompt | 4/428 empty after the actual registration-grid conversion. |
| T4 learned deformation | All 428 observed fields were finite. The median sampled within-field P95 displacement was 0.07356 in normalized coordinates. The greatest sampled displacement was 0.13861. Representative fields remain private. |
| T5 warp | 5/428 empty in the registration array and 19/428 empty on the recovered follow-up native grid. These are occupancy counts in different spaces. |
| T6 actual ROI | 413 target-positive ROI measurements had mean GT union coverage of 1.0. The 228 proxy-specific ROIs also had mean coverage of 1.0. The observed Z-visible coverage had the same means. All 428 full traces exercised the spatial padding compatibility path. |
| T7 actual local U-Net | 102/428 local argmax masks contained foreground after padding removal. Mean local GT-union Dice was 0.000858 over 414 defined local comparisons. |
| T8 embedding and native export | 102 work-grid argmax masks had foreground, 89 remained nonempty after diagnostic label restoration, and 83 official native predictions were nonempty. 19 nonempty work-grid masks became empty in the official native export. Six of those remained nonempty under diagnostic label restoration. |

The local U-Net already produced empty masks for 326/428 observed instances. The native exporter accounts for additional empty outputs. The measured ROI crop did not exclude the target in the scored cases. Intensity saturation, geometric disappearance and later model-stage losses were each observed. These linked observations do not establish a single causal mechanism.

The [final trace audit](https://github.com/Strange-Noah/BrainTumour-CNNViT-Segmentation/actions/runs/46123366262) checks the history preprocessing event separately from the unlabeled follow-up placeholder and measures working-grid and native emptiness directly. Its independent private receipt has SHA-256 `f0affd740f350820b1049019292283d88138be03b72692502a8b42500403b9f6`. The [first terminal audit](https://github.com/Strange-Noah/BrainTumour-CNNViT-Segmentation/actions/runs/46122809756) verifies the same archive coverage and paired scores. The older bridge collector had a file-length check failure; both terminal audits used file length measured after flushing the downloaded stream and verified hashes against the producer receipts.

## Autoregressive identity and empty states

The first L2 step shares the recorded first L1 result for each of the 38 patients. Two patients had no initial GT instance to initialize. The other 36 initialized 213 stable historical IDs. After the first step, 175 IDs had no nonempty prediction and 38 carried forward. The following 60 pairs used earlier model output as the prompt. Only 5 of those pairs had a nonempty final union. Across the later per-instance records, 298 had no earlier nonempty prediction, 34 ran to empty output, 8 had nonempty output, and 1 was skipped by the official preprocessor. Original IDs and explicit empty/skip states remain in private R2. There was no GT reinitialization on later steps.

## Scope of scientific conclusions

- Compare the four model arms only within the 98 recovered formal pairs. The missing patients come from a fixed dispatch order under time limits, and no missing pair received an imputed score.
- The reference proxies support a paired spatial-overlap analysis. The dataset's frozen records do not establish expert-verified longitudinal lesion identities or reliable inter-visit day intervals.
- GPU trace counts refer to saved histories. ROI and native paired analyses use only their applicable denominators. Ground-truth diagnostic crops and intensity statistics did not enter follow-up inference.
- The collection reads the original private artifacts and preserves model outputs. The recovery manifest enumerates the missing patient slots for bounded continuation if that is separately scheduled.
