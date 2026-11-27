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
