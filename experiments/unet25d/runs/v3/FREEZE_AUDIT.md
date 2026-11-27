# V3 training-only freeze

Frozen V2 base: c9369e86ab4c8aaa9189d6cb4737d5ffddf4822d.
This is an unpublished local candidate. The base plus the add-only patch
reconstructs the V3 source. The manifest lists SHA-256 for each relevant file.

## Active execution

- V3 `cohort/v3/train.py`: authenticated-run inputs, PyTorch DDP on two T4s,
  pinned ImageNet SMP ResNet34 U-Net, local batch 1, AdamW, training health,
  bounded Gate 1, fresh formal initialization, exact fixed-plan consumption.
- V3 `loss.py`: normalized positive lesion redistribution, unit negatives,
  per-sample valid-pixel denominator. No V2 50/50 class split, other weights
  or added loss terms. Frozen Soft Dice and 0.5/0.5 coefficients.
- V3 `settings.py`, `inputs.py`, `state.py`: science/config identity,
  frozen input and encoder pinning, safe serialization, matched resume and
  exact terminal coverage. Old V3 and V2 state cannot be resumed.
- V3 `persistence.py`: V3-only lease namespace and existing signed archive
  transport with write, GET/SHA comparison and receipt. Independent V3
  cloud readback has not been exercised.
- Frozen V2 `loading.py`, `geometry.py`, `sampling.py`: original sharded
  12-channel alignment, image support/GT fallback, crop and zoom, lesions,
  normalized volume weights, deterministic pairing and committed indices.
- Frozen V2 `resilient_fetch.py`, `settings.py`, `state.py`,
  `persistence.py`: verified private reads, cohort/split checks, initial
  state and shared transport. Common `cohort/loading.py` provides the
  bounded shard cache; common `cohort/settings.py` provides paths and
  provenance. Shared Unet25d `train.py` supplies FP32 safety and optimizer
  stepping. Its benchmark model, loss and entry point are inactive;
  `data.py` is its transitive import.
- V3 `cpu_gate_acceptance.py` and `test_v3.py` are CPU-only audit/test
  entry points. The V2 balanced BCE in those files is a reference only.

## Scientific preservation and scheduling

The BCE objective is the sole scientific training change. Sampling is the same
frozen 293,846 draws / 146,923 paired updates. No new model, augmentation,
normalization, optimizer schedule, loss, or inference policy. The bounded
difficulty records use detached logits and are passive telemetry; the sampler
does not consume them.

The V3 schedule is recorded in the configuration and therefore changes the
science identifier. It removes both automatic validation branches and the
auxiliary early positive-learning check. Gate 1 retains the original criteria,
up to 4,000 updates and at most 600 seconds for starting updates. An in-flight
batch and final health/measurement/save can finish outside that window.
The terminal Gate health check now also covers a partial 50-step interval.
Gate failure stops. Gate success starts Phase 1 in the same process after
fresh initialization. Completion requires all exact counts and the Gate pass.
Its completion event is emitted only after the final signed checkpoint save
and GET/SHA readback acknowledge; no validation or Phase 2 runs.

The current trusted-main registry pins only the V2 source/entrypoint and
resources. The formal workflow and authorized generic CPU workflow both
resolve source on main before using secrets. A new isolated branch alone
cannot authorize V3. No cloud write, source publication or submission occurred.
