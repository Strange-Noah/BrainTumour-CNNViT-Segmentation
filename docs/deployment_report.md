# Web Deployment Report

## Overview

The final deployment is a Hugging Face Spaces Gradio web application for brain tumour segmentation, volumetry, and interactive 3D/2D visualization. The deployed site is:

```text
https://huggingface.co/spaces/TLOMMP/brain-tumour-segmentation
```

The web application is designed as a research demonstration for the trained Route B CNN-ViT segmentation models. It accepts a single four-channel brain MRI NIfTI volume, runs GPU inference, reports tumour volume statistics, optionally compares the prediction with an uploaded label volume, and renders the prediction in an interactive viewer.

The deployment supports two model branches:

- Binary whole-tumour segmentation.
- 4-label tumour-subclass segmentation.

The binary model is the primary model used for whole-tumour volumetry. The 4-label branch is provided as a separate inference option with the same input and visualization interface.

## User Inputs

The application exposes a deliberately small set of user-facing controls.

### MRI Upload

The required input is a four-channel MRI NIfTI file:

```text
.nii or .nii.gz
```

The loaded data must have exact shape:

```text
(240, 240, 155, 4)
```

This matches the model training protocol:

- 240 x 240 native slice grid.
- 155 axial slices.
- 4 MRI modalities.

Files with other shapes are rejected instead of being resized. This keeps the MVP deployment aligned with the training-time data geometry and avoids silent geometric changes before segmentation.

### Optional Label Upload

The user may also upload a tumour label NIfTI file:

```text
.nii or .nii.gz
```

The label must have exact shape:

```text
(240, 240, 155)
```

If provided, the label is used only for evaluation and visualization. It is not required for inference.

### Segmentation Mode

The model selector has two modes:

```text
Binary tumour
4-label tumour subclasses
```

The generated job ID includes the mode name, so binary and 4-label jobs remain distinguishable in saved browser history and server-side job outputs.

### Job Controls

The interface provides:

- Run segmentation.
- Attach label to current completed job.
- Recent jobs in this browser.
- Previous job ID reload.
- Refresh selected job status.
- Download predicted mask.

The recent-job dropdown is backed by browser storage, allowing a user to reload completed jobs without manually copying job IDs. Manual job ID loading remains available as a fallback.

## Inference Pipeline

The deployment separates CPU preprocessing, GPU inference, and CPU postprocessing.

### CPU Preprocessing

Before GPU inference, the server:

1. Loads the NIfTI image with nibabel.
2. Verifies file type and shape.
3. Extracts each of the 155 slices.
4. Applies the same normalization used during training:

```text
per-slice, per-modality z-score over nonzero brain pixels
```

5. Converts each slice from image layout:

```text
(H, W, 4)
```

to model layout:

```text
(4, H, W)
```

The preprocessed slices are stacked into a batchable array of 155 two-dimensional inputs.

### GPU Segmentation

Only the model forward pass is wrapped in the GPU execution function. On Hugging Face ZeroGPU this is handled through:

```python
@spaces.GPU(duration=10)
```

The function performs:

- model loading or reuse,
- CUDA placement when available,
- automatic mixed precision on CUDA,
- batched slice inference,
- sigmoid thresholding for binary segmentation,
- softmax and argmax for 4-label segmentation.

The model is cached in memory by segmentation mode after first load. It is not reconstructed for every request.

### CPU Postprocessing

After GPU inference, the server:

1. Reconstructs a 3D prediction volume with shape:

```text
(240, 240, 155)
```

2. Saves the predicted mask as:

```text
prediction.nii.gz
```

using the input image affine and header metadata.

3. Saves a probability map for label comparison.
4. Computes brain and tumour volumes.
5. Creates job metadata and visualization assets.

The saved prediction uses the same physical grid as the input MRI, so the downloaded NIfTI mask can be loaded with the original MRI in external viewers.

## Reported Outputs

For each completed segmentation job, the application reports:

- Job ID.
- Segmentation mode.
- Input shape.
- Tumour volume in cubic centimeters.
- Brain volume in cubic centimeters.
- Tumour fraction of brain volume.
- GPU inference time.

Brain volume is computed from the nonzero MRI brain region rather than from the full rectangular image volume. Tumour fraction is therefore:

```text
100 * tumour_volume_cm3 / brain_volume_cm3
```

This is more meaningful than using the entire 240 x 240 x 155 bounding box as the denominator.

## Optional Label Metrics

When a label volume is provided, the application computes whole-tumour 3D metrics:

- Whole-tumour 3D Dice.
- Whole-tumour 3D binary cross entropy.
- Label tumour volume.

For both binary and 4-label segmentation modes, the label comparison is evaluated as a whole-tumour binary comparison:

```text
prediction > 0
label > 0
```

This keeps the online evaluation focused on whole-tumour overlap and volumetry, which are the primary deployment goals.

## Visualization Design

The final viewer combines a 3D overview with three synchronized 2D orthogonal slice views.

### 3D View

The 3D panel is rendered with Plotly inside the web page. It contains:

- A semitransparent brain volume.
- Predicted tumour volume.
- Optional label volume.
- Three colored plane frames.
- Three colored axis lines crossing at the current slice intersection.

The brain is shown as a semitransparent context volume. Tumour predictions are shown as colored overlays. In 4-label mode, tumour subclasses use separate prediction colors. If a label is available, label overlays use corresponding but shifted colors so prediction and label remain visually separable.

The 3D volume uses downsampled arrays for browser responsiveness:

- brain display volume is downsampled for context rendering,
- prediction and label masks are max-pooled before 3D plotting.

This preserves visible tumour regions while keeping the browser workload manageable.

### 2D Orthogonal Views

The viewer includes three 2D slice panels:

- Oyz slice at x.
- Ozx slice at y.
- Oxy slice at z.

The three sliders map directly to these planes:

- x controls Oyz.
- y controls Ozx.
- z controls Oxy.

Each 2D slice is rendered on an HTML canvas from data already embedded in the page. This means slider movement does not call the Python backend. The canvas view updates locally in JavaScript.

### Overlay Color Semantics

The display uses grayscale MRI background with colored overlays.

For binary segmentation:

- prediction uses one tumour color,
- label uses a related but distinct label color,
- overlap is shown as a darker/intensified version of the same colour family rather than a third unrelated colour.

For 4-label segmentation:

- prediction classes use three separated colours,
- label classes use shifted versions of the same color families,
- matching prediction-label overlap is darkened within the corresponding class color.

This avoids visual ambiguity while keeping class identity consistent across prediction, label, and overlap.

### Frontend Responsiveness

The viewer is designed so that most interactions are handled client-side:

- Slider movement updates the corresponding 2D canvas immediately.
- Prediction/label visibility switches update Plotly traces and 2D overlays in the browser.
- The Python backend is not called for routine slice navigation or overlay toggling.

The 3D guide frames are updated in Plotly, while the heavier MRI and tumour volume data are not recomputed on every interaction.

## Job Persistence

Each segmentation request creates a job directory under:

```text
job_outputs/
```

Each job stores:

- copied input NIfTI,
- optional copied label NIfTI,
- prediction NIfTI,
- probability map,
- overlay preview,
- metadata JSON.

A job ID has the form:

```text
YYYYMMDD-HHMMSS-mode-randomid
```

For example:

```text
20260824-153021-binary-a1b2c3d4e5
```

This design separates computation from the browser page state. Once a job is created, it can continue in the background and can be reloaded later through browser history or manual job ID entry.

## Safety and Fool-Proof Design

The deployment includes several safeguards to prevent invalid or misleading outputs.

### File-Type Validation

Only NIfTI files are accepted:

```text
.nii
.nii.gz
```

Other file types are rejected before model execution.

### Shape Validation

Input MRI volumes must match:

```text
(240, 240, 155, 4)
```

Label volumes must match:

```text
(240, 240, 155)
```

The app does not silently crop, pad, or resize data in the MVP version.

### Label-Input Consistency Checks

When a label is uploaded, the app checks:

- label shape,
- voxel spacing,
- affine matrix,
- whether the label contains positive tumour voxels,
- whether positive label voxels mostly lie inside the nonzero brain region.

If the label affine or spacing does not match the MRI, the label is rejected. If more than 1% of positive label voxels fall outside the nonzero brain region, the label is treated as likely misaligned and rejected.

### Checkpoint Validation

Model checkpoints must contain:

```text
model_state_dict
```

The model is loaded with strict state-dict compatibility. This prevents accidentally running inference with a checkpoint from an incompatible architecture.

### Medical Disclaimer

The web application is a research demonstration and not a clinical diagnostic system. It is intended for controlled research data in the expected format, not for arbitrary hospital imaging workflows.

## Runtime Parameters

Most deployment parameters are controlled through environment variables rather than user-facing UI controls:

| Variable | Purpose | Default |
|---|---|---:|
| `SEG_THRESHOLD` | Binary tumour threshold | `0.5` |
| `SEG_BATCH_SIZE` | Slice batch size for inference | `16` |
| `BINARY_CHECKPOINT_PATH` | Local binary checkpoint path | `checkpoints/binary_route_b_best_checkpoint.pt` |
| `MULTICLASS_CHECKPOINT_PATH` | Local 4-label checkpoint path | `checkpoints/route_b_4label_best_checkpoint.pt` |
| `MULTICLASS_CHECKPOINT_REPO_ID` | Remote HF model repo for 4-label checkpoint fallback | `TLOMMP/route-b-4label-full-train` |
| `MULTICLASS_CHECKPOINT_FILENAME` | Remote checkpoint filename | `best_checkpoint.pt` |
| `SEG_JOB_ROOT` | Server-side job output directory | `job_outputs` |
| `SEG_MAX_BACKGROUND_JOBS` | Number of background worker jobs | `1` |

The public UI intentionally keeps these fixed so users cannot accidentally change the trained model protocol.

## Deployment Boundary

The final Hugging Face Space acts as an inference and visualization layer. It does not expose training functionality. Large model checkpoints and raw training data are not stored in the GitHub repository. The deployed application uses prepared checkpoints and expects user-provided NIfTI volumes in the same format as the training data.

This keeps the deployment focused on a single workflow:

```text
upload MRI -> validate -> segment on GPU -> compute volumetry -> visualize -> download mask
```

