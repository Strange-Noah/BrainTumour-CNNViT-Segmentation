# Summer CNN–ViT port and first UCSD experiment

## Recovered implementation

Source is frozen summer commit `6d953c65badd06ff2f81109b068ef016bf723a93`, `src/pretrained_hybrid_vit.py`. The model uses timm's R50–ViT-B/16 hybrid encoder (`vit_base_r50_s16_224.orig_in21k`) and a four-stage bilinear/convolution/GroupNorm/GELU decoder. The summer input is four-channel 240×240 axial slices. Binary and four-class decoder heads were used.

The active port is `experiments/cnn_vit_model.py`. Its architecture and tensor names are retained, with safe tensor-only checkpoint loading and classifier-key filtering for timm initialization. The archive is unchanged.

## Checkpoint recovery and task adaptation

The summer app references public `TLOMMP/route-b-4label-full-train/best_checkpoint.pt`. The experiment runner resolves and records the repository revision and checkpoint SHA-256. It restores the original head, strictly loads the full model and performs a native-head forward on a real UCSD slice. The native prediction is a compatibility diagnostic. The old MSD task is not a valid UCSD enhancing-tissue model.

After compatibility checks, the binary UCSD output head is reinitialized. Compatible summer encoder and decoder features are retained. The first bounded experiment optimizes only the new binary decoder head using BCE against original UCSD label 3. A failed summer checkpoint recovery is recorded by exception type, then the documented timm pretrained backbone is attempted. That initialization adapts the 3-channel stem to four inputs and interpolates positional embeddings from 224 to 240 while excluding the classification head.

Direct local requests for both Hugging Face metadata endpoints timed out. Live weight recovery and tensor compatibility are determined by the actual free-runner execution, not presumed from the source reference.

## UCSD geometry and preprocessing

The shared loader supplies T1, T1c, T2 and FLAIR in that order. Each axial slice uses the summer normalization over the union of nonzero channel foreground, then bilinear resizing to 240×240. Targets are resized with nearest-neighbor interpolation for the small slice optimization. Inference receives images only and predicts every slice. Bilinear interpolation restores logits to original T1c XY dimensions before sigmoid and threshold 0.5. Original Z indexing and the original T1c affine are retained.

The final binary NIfTI and visit-local lesion candidates go through the shared exporter and evaluator. Native preprocessing changes for full training must be recorded in experiment metadata. No held-out test visit is selected. Train and validation examples are selected from the frozen split and currently available objects.

## Execution and interpretation

Manual workflow `cnn-vit-pilot.yml` uses standard free CPU runners and existing read-only dataset credentials. It installs torch 2.8.0, torchvision 0.23.0 and timm 1.0.28 and runs four genuine optimization steps on a train visit, a validation forward, and optional complete validation-volume inference. It reports model parameter counts, losses, wall time and peak Linux RSS, and GPU memory when executed on an already authorized GPU.

```sh
python -m experiments.cnn_vit_real_run --work /private/run --steps 4 --inference
```

This is a task-specific head pilot, not a converged model or final model-selection result. Larger fine-tuning may unfreeze the original supported encoder stages once memory and runtime are measured. Full training requires suitable authorized compute and coordination.

If separate output-bucket credentials are available, checkpoint, prediction and report are persisted under a unique run prefix in `novo-nordisk-pbl-outputs`. Otherwise only aggregate run evidence is logged, and binary outputs remain ephemeral. Dataset objects are read-only throughout this experiment.

The preserved local summer Task01 dataset.json declares FLAIR,T1w,t1gd,T2w. The shared UCSD order is T1,T1c,T2,FLAIR. The port permutes the recovered first-convolution input kernels [1,2,3,0] so learned modality features retain their original meaning. The native-head diagnostic instead reorders its input to summer order. UCSD checkpoints declare their channel order, and experiments.cnn_vit_ucsd:predict loads the fine-tuned head through research.run_visit without resetting it.
