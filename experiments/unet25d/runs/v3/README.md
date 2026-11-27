# V3 class-unweighted masked BCE with lesion redistribution

Local candidate based on V2 commit `c9369e86ab4c8aaa9189d6cb4737d5ffddf4822d`.
V3 starts from the same pinned ImageNet initialization. It cannot resume V2 training.
The V2 catalogue, train-only inventory and exact sample plan are byte-pinned in
`experiments/unet25d/cohort/v3/inputs.py`. The experiment stops after Phase 1
training, at `PHASE1_TRAINING_COMPLETE`, after the final checkpoint is
persisted. This campaign skips the auxiliary early validation and 159-exam
validation. Phase 2 and test evaluation remain disabled.

## Training objective

**Standard class-unweighted masked BCE with lesion-volume-aware redistribution
within the positive class.** For a lesion of volume (V) mm³ represented by
(n) foreground pixels in the current crop, the existing V2 loader assigns
each of those pixels `clip((100 / max(V, 1e-9))**0.25, 1, 4) / n`.
It then rescales all foreground weights in each sample to mean 1.

V3 adds the weighted positive BCE and unit-weight valid negative BCE, divides
by the number of valid pixels, and averages samples. The total coefficients
before multiplying pixel losses are foreground fraction and background
fraction. The outer loss remains 0.5 BCE + 0.5 frozen masked Soft Dice.
Validation and its historical reporting loss are unchanged.

The new configuration changes the science hash and output prefix. Checkpoints
and manifests from the previous local V3 objective are incompatible.

## Local CPU checks

```sh
CUDA_VISIBLE_DEVICES='' python -m unittest experiments.unet25d.cohort.v3.test_v3
CUDA_VISIBLE_DEVICES='' python -m experiments.unet25d.cohort.v3.cpu_gate_acceptance /path/to/approved/private/input
```

The CPU acceptance entry also supports `--examples` with a current read-only
lease. It reads just the five frozen Gate 1 examples and differentiates the loss
at zero logits. It does not instantiate a GPU model or submit jobs.

## Runtime handoff

The training module is `experiments.unet25d.cohort.v3.train`. A future reviewed
dispatcher must pin the integrated source, prepare a new V3 private input with
the frozen objects and offline encoder, bind the existing signed transport to
a unique V3 prefix, enforce the owner and provider budget, and run the entry
point with two T4 ranks. The local manifest entry point is
`python -m experiments.unet25d.cohort.v3.inputs PRIVATE_FOLDER --source-sha SHA`.
Use a separate empty working directory. A V3 resume checkpoint must come
from an independently verified snapshot of that exact source and science.
The current V2 dispatcher is not a V3 dispatcher.

The live archive saves the checkpoint and training telemetry.
The collector must independently verify the SHA-256 and recover the checkpoint
with its model, AdamW, rank RNG and cursor before any continuation. Native
validation is deferred to a separately authorized execution.

V3 has no remote registration, cloud inputs, or GPU acceptance yet.
