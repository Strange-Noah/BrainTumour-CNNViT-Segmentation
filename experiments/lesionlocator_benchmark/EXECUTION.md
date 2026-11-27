# LesionLocator fold-0 execution

[Accepted G0/L0 results](CPU_RESULTS.md) cover the frozen 45 validation patients, 159 examinations and 114 adjacent pairs. They remain the source of the L0 arms for matched comparisons. The target is raw enhancing label 3 in native T1c. The data identifies visit order.

[Official asset acquisition](https://github.com/Strange-Noah/BrainTumour-CNNViT-Segmentation/actions/runs/46031603433) verified CRC32 of the selected tracking ZIP members and independently read them from private R2 to verify SHA-256. The 1,384,731,741-byte fold-0 checkpoint has SHA-256 `f00f2e71392a88f922fcdfa852b71c9fdbdf9c14c33472a21c104398f2266596`. This verification covers the three selected members. The publisher's full-archive MD5 pertains to the complete ZIP.

[Protected source registration](https://github.com/Strange-Noah/BrainTumour-CNNViT-Segmentation/pull/54) binds the inference source to `1574ebb51fb5ca148d1fd985821c81b95bf76548`. The [execution record](https://github.com/Strange-Noah/BrainTumour-CNNViT-Segmentation/actions/runs/46056070058) records the exact checkpoint configuration and private input digest when those steps succeed.

## Recovery

Kaggle uses a private, digest-bound package with expiring grants for specific validation inputs and create-once result slots. The worker loads fold 0 once per GPU and runs the official TrackNet with mirroring disabled. Follow-up preprocessing receives the image and no target annotation. Diagnostic GT enters a separate post-inference scoring path. Native output masks, stable-ID records and image visualizations remain private.

A completed patient yields a private archive and a checksum receipt. The independent Actions collector compares the persisted bytes and SHA-256 with that receipt and verifies source, checkpoint namespace and patient slot before using metrics. An archive without its receipt requires recovery before reuse. The registered bridge's `collect` mode performs readback without submitting GPU jobs.

The first follow-up of a sequence shares the L1/L2 initial-GT inference. Later L2 steps carry the original instance ID with the preceding prediction and record empty or skipped states. Same-grid unique mutual overlap is a reference proxy. Diseased tissue match and longitudinal identity use distinct claims.

## Effective checkpoint configuration

The checkpoint contains 1,145 named weight/buffer tensors and selects `LesionLocatorTrainerTrack` with `3d_fullres`. The checkpoint plans select `CTNormalization` with fixed clipping [−33, 134], mean 58.50637769071138, standard deviation 30.87594622558693, and no normalization mask. The working spacing is [3, 0.841796875, 0.841796875] mm. The U-Net patch is [96, 256, 256]. The registration grid is a 175³ array and its coordinates are relative to the actual work volume and z-window.

The selected official resampler is `resample_data_or_seg_to_shape` for image, segmentation, and logits/probabilities. These routes use orders 3, 1, and 1 respectively, with separate-axis order 0 when anisotropic handling is selected by the official implementation. The segmentation route sets `is_seg=true`. Both transpose arrays are [0, 1, 2] in the official image reader's array convention.

[Independent Kaggle input readback and submissions](https://github.com/Strange-Noah/BrainTumour-CNNViT-Segmentation/actions/runs/46056070058) verified the private package digest and recorded a 10,500-second session cap against 22,956 free quota seconds.

## First allocated GPU execution

[GPU execution](https://www.kaggle.com/code/mathewzhou/novo-lesionlocator-t-46056070058) obtained two Tesla T4s and loaded fold 0. The first real case reached the official ROI padding immediately after registration and warp, then stopped with CUDA out-of-memory. No prediction score was emitted by that GPU execution. The independently submitted [CPU geometry execution](https://www.kaggle.com/code/mathewzhou/novo-lesionlocator-g-46056070058) completed its first real pair and emitted a private readback receipt.

The [official padding implementation](https://github.com/MIC-DKFZ/LesionLocator/blob/ef073baaa0dcba8dd036ec5a29f130db18c9d580/lesionlocator/modules/tracknet.py#L120) puts nonzero in-plane padding into the channel and batch axes of a five-dimensional input. The compatibility fix uses zero padding on the intended spatial dimensions and removes it at the same spatial offsets. It delegates to the original functions on existing valid depth-only inputs. The CPU regression reproduces the unexpected expansion and checks exact depth-only equivalence and in-plane round-trip. The corrected execution has a new source identity and a separate private output namespace.

## First independently recovered CPU geometry batch

[Collector 46057770674](https://github.com/Strange-Noah/BrainTumour-CNNViT-Segmentation/actions/runs/46057770674) independently checked the bytes, SHA-256 and provenance for five patient archives. These cover 15 examinations and 10 adjacent pairs.

| Diagnostic | Instance denominator | Empty after stage | Mean native-grid instance Dice |
| --- | ---: | ---: | ---: |
| G1 official preprocess and native-grid round trip | 55 | 8 | 0.5734 |
| G2 after the official z-window on CPU half-precision inputs | 55 | 8 | 0.5734 |
| G3 intermediate registration-grid mask | 55 | 8 | — |
| G4 complete geometric round trip | 55 | 8 | 0.5648 |
| G5 ideal-logit native export | 59 target instances | 13 | 0.5489 |

The G3 grid counts use array coordinates. G4 restores the native grid before scoring. For CTNormalization on these 15 MRI visits, the upper clipping bound affected 70,372,380 of 614,858,752 full-volume voxels, 70,372,380 of 71,300,771 nonzero voxels, and 119,506 of 119,767 enhancing-label voxels. No sample in those three groups fell below the lower bound. These are pooled voxel fractions from the first five patients. The experiment uses the official normalization. The private archives contain the per-visit distributions, contrast observations, representative image panels and native diagnostic masks.

[Corrected dispatch 46057677602](https://github.com/Strange-Noah/BrainTumour-CNNViT-Segmentation/actions/runs/46057677602) uses source `9fff1e41eb6a22000c6fd04984dbced6e05ae96f`, registered by [PR #55](https://github.com/Strange-Noah/BrainTumour-CNNViT-Segmentation/pull/55). It republishes only expiring grants and reuses the checkpoint in private R2. The original CPU geometry job continues on its fixed source.

## First independently recovered GPU pair

[Collector 46059586976](https://github.com/Strange-Noah/BrainTumour-CNNViT-Segmentation/actions/runs/46059586976) verified the first GPU smoke archive after [the multipart readback correction](https://github.com/Strange-Noah/BrainTumour-CNNViT-Segmentation/pull/56). The producer also independently read back this archive immediately after upload. Both paths accepted the same private artifact.

| Arm | Native union Dice | Matched target lesions | FP components |
| --- | ---: | ---: | ---: |
| L0 | 0.14708 | 2/6 | 4 |
| L1-Warp | 0.12819 | 2/6 | 5 |
| L1-Final | 0 | 0/6 | 0 |

The sole strict spatial-reference proxy had Dice 0.16504, 0.12612, and 0 across the three arms. This is a same-grid overlap proxy. Three of six historical instances remained present after G1; the same three remained through G2–G4. All three scored native ROIs covered the full target union. The final union was empty in this pair. The evidence shows that missed ROI coverage alone cannot explain this pair's failed detection. The private per-instance artifacts preserve local U-Net predictions and work-space/native export comparisons.

The smoke unit used about 492.4 s end-to-end including geometry, observations and output readback. MRI download used 1.89 s, tracked official preprocessing 130.0 s, registration CUDA 10.32 s, local U-Net CUDA 1.71 s, and native export 17.07 s. These are instrumented substage totals for the first pair. Peak PyTorch CUDA allocation was 4.63 GiB and sampled worker RSS reached 4.44 GiB. The results come from one smoke patient.

The same collector recovered 10 independent CPU geometry archives covering 33 examinations and 23 pairs. The cumulative disappeared counts were 17/120 after G1/G2, 18/120 at G3/G4 and 21/127 after G5. Across the 33 examinations, 663,830 of 664,609 lesion voxels exceeded the CT upper bound.

## Formal transition

Both real smoke archives completed in the [corrected dual-T4 Kaggle execution](https://www.kaggle.com/code/mathewzhou/novo-lesionlocator-t-46057677602). The second producer-side readback was recorded at minute 19:24 of notebook execution. Formal inference then began in the same session, with two resident workers processing separate patients. GPU inputs use private Kaggle version 2, independently read at package digest `b7043022ebb85608142b369afbc539bff26c3f5337e15e6b3923fe3baa0acc1b`; the grants expire 2026-11-29 03:30:34 Asia/Shanghai. The session cap is 10,200 seconds. The original CPU geometry job runs independently.

The recovery bridge verifies patient-level artifacts using the executing immutable GPU and geometry SHAs. Its protected collector-only source can advance without changing the producing namespace. The R2 producer verifies completed units as it goes. The Actions `collect` path can independently recover those completed units during or after the Kaggle run.

[Independent formal recovery 46060866987](https://github.com/Strange-Noah/BrainTumour-CNNViT-Segmentation/actions/runs/46060866987) verified both smoke and both formal pilot archives. The two formal pilot pairs repeat the two smoke pairs under the fixed protocol. Across those two pairs, the L0, L1-Warp and L1-Final means were 0.126354, 0.090299 and 0. Spatial matching was 4/14, 4/14 and 0/14; false-positive component counts were 7, 9 and 0. The eight observed ROIs covered the complete target unions. The three strict overlap-proxy Dice means were 0.28147, 0.10482 and 0. Per-instance local scoring and export differences are in the private result archives.

The two formal units took 685.9 s total across workers. Stage totals were 246.6 s for official preprocessing, 6.26 s registration CUDA, 3.84 s local U-Net CUDA, 45.1 s native export, and 1.56 s MRI transfers. Maximum observed CUDA allocation was 4.63 GiB and sampled worker RSS was 4.57 GiB. These observations characterize the pilot inputs. Host-side preprocessing was much longer than the measured CUDA kernels.

The same independent collector verified 13 geometry patients, 41 exams and 28 pairs. G1 and G2 lost 28/151 historical instances, G3/G4 lost 30/151, and G5 lost 23/144 follow-up reference instances. The mean native instance Dice was 0.5454 (G1/G2), 0.5345 (G4) and 0.5675 (G5). Across the 41 MRI visits, upper clipping fractions were 11.0127% of all 1,628,831,744 voxels, 98.6770% of 181,783,189 nonzero voxels, and 99.8542% of 748,848 enhancing-label voxels. Lower clipping was 0% for each pooled group. This shows saturation under the unchanged official CT normalization on this sample. A cause for the final prediction failures requires the per-instance local and native-stage evidence.

## Unattended handoff — 2026-11-28, Asia/Shanghai

At 04:08 both the dual-T4 formal run and the independent CPU geometry run were RUNNING. The initial formal observation covered approximately ten minutes of real inference. Both workers advanced after pilot and continued into subsequent patients. Each worker commits completed patients to private R2 and verifies them. The active GPU session remains unattended. Provider quota showed 24.24 h used and 5.76 h displayed remaining at handoff.

[Independent handoff readback 46061515025](https://github.com/Strange-Noah/BrainTumour-CNNViT-Segmentation/actions/runs/46061515025) verified **5 formal patients, 15 examinations, 10 adjacent pairs**, in addition to the two separate smoke archives. The formal patient count includes the repeat of the designated smoke cases and counts each formal case once.

| Formal arm | Mean union Dice | Target lesions matched | False-positive components |
| --- | ---: | ---: | ---: |
| L0 | 0.202063 | 29/59 | 26 |
| L1-Warp | 0.092062 | 16/59 | 34 |
| L1-Final | 0 | 0/59 | 1 |

The 25 conservative same-grid instance proxies produced mean Dice 0.40667, 0.03335 and 0. Among the observed ROIs with measurable target GT, all 47 covered the target union and all 24 scored proxy-specific ROIs achieved complete matched-target coverage. The formal batch included one false-positive component. The local and native masks allow the stage of foreground loss to be examined in private. The current aggregate does not attribute failure to a particular stage or provide a complete L2 performance estimate.

The 10 formal pairs took 1,392.6 combined worker-seconds or 0.431 pairs per worker-minute. This denominator differs from wall-clock duration on two workers. Stage totals were 504.21 s official preprocessing, 39.84 s registration CUDA, 25.61 s local U-Net CUDA, 106.97 s native export and 7.43 s MRI data reads. Peak PyTorch CUDA allocation was 4.63 GiB and sampled worker RSS was 4.57 GiB. These are early batch measurements.

The same Actions run verified **14 geometry patients, 45 examinations, 31 pairs**. Historical disappearance counts were 29/162 at G1/G2 and 31/162 at G3/G4. G5 ideal-logit restoration lost 24/151 target instances. Across these 45 exams the upper clip affected 10.7646% of full-volume voxels, 98.7921% of nonzero voxels, and 99.8546% of raw-label-3 voxels. Lower clipping was 0% in those pooled groups. The complete 45/159/114 scientific comparison remains in progress. Previously accepted G0/L0 archives remain immutable.
