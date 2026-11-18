# Full-resolution 2.5D U-Net benchmark implementation

Status: implementation and CPU correctness checks prepared; **GPU campaign not run**. No full training authorized. See [benchmark report and blockers](BENCHMARK.md).

## Frozen initial specification

[config.json](config.json) fixes SMP 0.5.0 U-Net / ResNet34, pinned ImageNet encoder, one raw-logit output, full-network training, modality-major T1/T1c/T2/FLAIR × offsets −1/0/+1, replicated boundaries, center label 3 target, 512×512, independent nonzero-volume z-score, encoder/decoder learning rates 1e-4/3e-4, AdamW decay 1e-4, equally weighted BCE/soft Dice and FP16 GradScaler (initial scale 128). The CPU-verified numerical continuation candidate keeps the encoder and first decoder Conv/BN/ReLU in FP32, with the remaining decoder under FP16 autocast; its runtime precision is checkpointed separately from the unchanged cache configuration. [Actual failed GPU measurements and precision evidence](ATTEMPT_3.md). CUDA stability of this candidate is not yet accepted. Empty-target Dice is smooth/(sum prediction+smooth), so false positives are penalized. Constant/empty channels deterministically become zero; nonfinite inputs fail with a diagnostic rather than silently replacing voxels. No RGB normalization, cropping, resolution reduction, architectural changes or augmentation is enabled.

Encoder BatchNorm statistics remain in eval mode after every model.train() while affine parameters and the whole network receive gradients. Decoder BatchNorm remains in train mode. CPU backward checks pass; decoder stability on T4 batch 1 is still unverified. No scheduler is used. Future model selection requires complete frozen validation-cohort evaluation and shared metrics, not training loss; the implemented one-visit smoke is not model selection.

## Geometry

Align to native T1c through the shared loader and preserve its original mask for evaluation. Orthogonal orientations use explicit RAS axis permutation/flips. Small in-plane arrays are symmetrically zero-padded to 512 without changing spacing or losing FOV; restore removes those exact margins before inverse orientation. Oblique acquisitions use actual physical RAS resampling at mapped native voxel spacing and a world-coordinate bounding box. MRI interpolation is linear and targets use nearest-neighbor. Any full-FOV physical grid requiring more than 512 in either in-plane axis is an explicit `GeometryConflict`; no hidden resize, crop or cohort removal occurs. Predictions use continuous logits, inverse physical mapping to native T1c, then threshold zero (sigmoid 0.5). Original native truth is retained, never reconstructed from a resampled target.

Existing cohort metadata verifies 660 native 512×512, 368 native 256×256, six 640×640, and 31 other non-square exams; 53 are oblique. [Exact evidence](evidence/cohort-geometry.json). The six 640 cases cannot satisfy fixed 512 + original resolution + full FOV simultaneously. This is an unresolved **full-cohort specification conflict**, not justification to drop patients. CPU preparation declares a bounded eight-training-patient, one-validation-visit benchmark subset whose geometry fits the policy. It records eligibility counts and fails on unexpected conflicts. This subset does not change Protocol v1 or authorize exclusions for full training.

## Memory and sampling

One volume at a time is prepared into private disk-backed float32 modality arrays, binary targets and transform metadata; temporary source-cache files are deleted after each visit. Prepared disk storage is capped at 16 GiB. No 12-channel cohort tensor is built. Per-worker memory maps are bounded (default one visit), closed on eviction; each sample copies only 12 slices (12 MiB at 512). Mappings across ranks refer to shared files rather than duplicated decoded-volume arrays. Summed process RSS can double-count shared pages, so monitor PSS and available RAM too. Prepared cache reads avoid NIfTI decompression during training.

Sampler draws patients uniformly, then visits, then center slices. Positive/negative centers alternate ~50/50. Negative centers alternate tumor-bearing and fully negative visits. The deterministic global sequence is divided by rank without assigning all positives to one rank. No validation oversampling is used. Repetition is bounded by benchmark duration; full-training sampling policy must still be evaluated on the complete train index. DDP requires the same world size and local batch for exact continuation; changing either is not silent resume.

## Entry points

Install [requirements.txt](requirements.txt) while retaining the runtime's matching CUDA torch/torchvision pair. Do not install CPU torch wheels into a GPU session.

Authenticated private Kaggle **CPU** session: the self-contained [CPU entry script](kaggle_cpu_entry.py) pins the source commit, reads enabled Secrets in the interactive process, installs dependencies and starts only preparation. It never requests GPUs. Run it in the existing authenticated session; no credential values need to be copied. The saved-run API is currently blocked, so this interactive launch is a real outstanding operation, not a completed step.

For an existing pinned checkout:

```python
# Run from the pinned project checkout. Configure public account endpoint separately.
import os
os.environ['R2_ENDPOINT_URL'] = '<project account S3 endpoint>'
# kaggle_launcher loads the four enabled bucket-scoped Secrets into process memory only.
```

```sh
python -m experiments.unet25d.kaggle_launcher prepare
```

This checks all four Secrets, environment, output-bucket create-once+SHA readback, frozen manifest/split, pretrained initialization, all-cohort header metadata and the bounded cache/index. It writes per-stage records. Preserve the completed cache as **private** Kaggle CPU output and mount it before requesting GPUs. Do not put imaging/cache data in public GitHub or a public Kaggle dataset. A failed CPU stage forbids GPU launch.

In an authenticated dual-T4 session with prepared private CPU artifacts mounted:

```sh
python -m experiments.unet25d.kaggle_launcher campaign --cpu-artifacts /kaggle/input/PRIVATE_CPU_OUTPUT/novo-unet25d
```

The source commit and config/cache receipts must match preparation. The launcher rejects any hardware other than two T4s. A/B short tests have 10 warm-up + up to 40 measured steps. Promising A/B sustained tests target 660 seconds each; acceptance requires >=600 measured steady-state seconds and an RSS plateau. Worker-2 and local-batch-2 short comparisons use the better accepted sustained candidate. Failed candidates stop rather than silently retrying at reduced resolution. No DataParallel or unrestricted sweep is implemented. Parent monitoring enforces a 55-minute campaign deadline; launcher kills the complete process tree by ~58 minutes. No GPU job has yet been submitted, so these guards remain runtime-unverified and must not replace the platform's 60-minute limit.

The sampler/index, train loop, telemetry, native-space validation smoke, state save, periodic output upload, fresh-process restore and two-step continuation are executable components. They are **not an accepted unattended loop** until the real campaign passes. `monitor.py` records per-device nvidia-smi utilization/power/memory, process-tree RSS/PSS/I/O/faults, available RAM and swap. Rank telemetry records data wait, H2D, forward/backward/optimizer times, finite checks, VRAM and worker cache occupancy/hits/misses. Backward time includes synchronization; it is not an isolated NCCL overhead measurement. Timings deliberately synchronize CUDA and therefore include instrumentation overhead.

A complete state contains model, optimizer, scaler, config, step/epoch, rank RNG and deterministic sampler position; no scheduler is used. Periodic checkpoints are coordinated across ranks at intervals of at least 120 seconds and uploaded with GET/SHA verification when output credentials are available. Failure diagnostics are best-effort persisted independently of normal completion. Abrupt platform termination can still lose data since the last completed write: no shutdown-hook guarantee is assumed. Fresh-process recovery uses weights-only loading. R2 recovery, resumed GPU steps and the benchmark's validation smoke remain unexecuted.

## Verified checks and handoff

[CPU checks](evidence/cpu-checks.json): full-network finite gradients, frozen encoder BN running statistics, active decoder BN, deterministic balanced rank sampling, modality context boundaries, zero background, orientation and 256-padding round trips, explicit 640 rejection. [Real cached-visit checks](evidence/real-cpu-checks.json): positive/negative training visits, exact native-target round trip, 512 input and replicated boundaries; zero new image downloads, zero GPU allocation. Compile checks passed. The same synthetic checks also completed in an independent private Kaggle CPU kernel: [Kaggle evidence](evidence/kaggle-cpu-checks.json). These do not substitute for actual FP16/CUDA/DDP tests.

First unblock authenticated interactive execution, complete private CPU cache preparation, then run the bounded campaign and inspect measured late throughput/RSS/cache turnover before recommending a full-training configuration. Resolve 640/oblique grid conflicts before any full-cohort training proposal. Do not start full training automatically.
