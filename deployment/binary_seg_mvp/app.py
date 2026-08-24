from __future__ import annotations

import json
import os
import re
import shutil
import threading
import time
import uuid
import base64
import html
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path

import gradio as gr
import nibabel as nib
import numpy as np
import plotly.graph_objects as go
import torch
from PIL import Image
from huggingface_hub import hf_hub_download

try:
    import spaces
except ImportError:  # Local CPU/GPU fallback outside Hugging Face Spaces.
    class _SpacesFallback:
        @staticmethod
        def GPU(*args, **kwargs):
            def decorator(fn):
                return fn

            return decorator

    spaces = _SpacesFallback()

from src.pretrained_hybrid_vit import PretrainedHybridViTSegmenter


EXPECTED_SHAPE = (240, 240, 155, 4)
THRESHOLD = float(os.getenv("SEG_THRESHOLD", "0.5"))
BATCH_SIZE = int(os.getenv("SEG_BATCH_SIZE", "16"))
BINARY_CHECKPOINT_PATH = Path(os.getenv("BINARY_CHECKPOINT_PATH", "checkpoints/binary_route_b_best_checkpoint.pt"))
MULTICLASS_CHECKPOINT_PATH = Path(
    os.getenv("MULTICLASS_CHECKPOINT_PATH", "checkpoints/route_b_4label_best_checkpoint.pt")
)
MULTICLASS_CHECKPOINT_REPO_ID = os.getenv(
    "MULTICLASS_CHECKPOINT_REPO_ID",
    "TLOMMP/route-b-4label-full-train",
)
MULTICLASS_CHECKPOINT_FILENAME = os.getenv("MULTICLASS_CHECKPOINT_FILENAME", "best_checkpoint.pt")
HF_MODEL_CACHE = Path(os.getenv("HF_MODEL_CACHE", "checkpoints/hf_cache"))
MODEL_CONFIGS = {
    "binary": {
        "label": "Binary tumour",
        "checkpoint": BINARY_CHECKPOINT_PATH,
        "num_classes": 1,
        "repo_id": None,
        "filename": None,
    },
    "4label": {
        "label": "4-label tumour subclasses",
        "checkpoint": MULTICLASS_CHECKPOINT_PATH,
        "num_classes": 4,
        "repo_id": MULTICLASS_CHECKPOINT_REPO_ID,
        "filename": MULTICLASS_CHECKPOINT_FILENAME,
    },
}
JOB_ROOT = Path(os.getenv("SEG_JOB_ROOT", "job_outputs"))
JOB_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{8,80}$")

_MODELS: dict[str, torch.nn.Module] = {}
_MODEL_DEVICES: dict[str, torch.device] = {}
_JOB_EXECUTOR = ThreadPoolExecutor(max_workers=int(os.getenv("SEG_MAX_BACKGROUND_JOBS", "1")))
_JOB_FUTURES: dict[str, Future] = {}
_JOB_LOCK = threading.Lock()


def validate_uploaded_path(uploaded_file) -> Path:
    if uploaded_file is None:
        raise gr.Error("Please upload a .nii or .nii.gz file.")
    path = Path(uploaded_file if isinstance(uploaded_file, str) else uploaded_file.name)
    name = path.name.lower()
    if not (name.endswith(".nii") or name.endswith(".nii.gz")):
        raise gr.Error("Only .nii and .nii.gz files are accepted.")
    if not path.exists():
        raise gr.Error(f"Uploaded file was not found: {path}")
    return path


def make_job_id(segmentation_mode: str = "binary") -> str:
    mode = validate_segmentation_mode(segmentation_mode)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return f"{timestamp}-{mode}-{uuid.uuid4().hex[:10]}"


def validate_job_id(job_id: str | None) -> str:
    cleaned = (job_id or "").strip()
    if not JOB_ID_PATTERN.fullmatch(cleaned):
        raise gr.Error("Invalid job ID. Paste the exact job_id shown after a completed run.")
    return cleaned


def has_valid_job_id(job_id: str | None) -> bool:
    return JOB_ID_PATTERN.fullmatch((job_id or "").strip()) is not None


def get_job_dir(job_id: str) -> Path:
    return JOB_ROOT / validate_job_id(job_id)


def saved_input_path(job_dir: Path, source_path: Path) -> Path:
    name = source_path.name.lower()
    suffix = ".nii.gz" if name.endswith(".nii.gz") else ".nii"
    return job_dir / f"input{suffix}"


def saved_label_path(job_dir: Path, source_path: Path) -> Path:
    name = source_path.name.lower()
    suffix = ".nii.gz" if name.endswith(".nii.gz") else ".nii"
    return job_dir / f"label{suffix}"


def resolve_saved_input(metadata: dict) -> Path:
    input_path = Path(str(metadata.get("input_path", "")))
    if not input_path.exists():
        raise gr.Error("The saved input file for this job is missing, so it cannot be resumed.")
    return input_path


def resolve_saved_label(metadata: dict) -> Path | None:
    label_path = metadata.get("label_path")
    if not label_path:
        return None
    path = Path(str(label_path))
    if not path.exists():
        raise gr.Error("The saved label file for this job is missing.")
    return path


def load_nifti_strict(path: Path):
    try:
        nii = nib.load(str(path), mmap=True)
    except Exception as exc:
        raise gr.Error(f"Could not read this file as NIfTI: {exc}") from exc
    if tuple(int(v) for v in nii.shape) != EXPECTED_SHAPE:
        raise gr.Error(
            "Input rejected. Expected loaded NIfTI shape "
            f"{EXPECTED_SHAPE}, but got {tuple(int(v) for v in nii.shape)}."
        )
    return nii


def validate_label_path(label_file) -> Path | None:
    if label_file is None:
        return None
    path = Path(label_file if isinstance(label_file, str) else label_file.name)
    name = path.name.lower()
    if not (name.endswith(".nii") or name.endswith(".nii.gz")):
        raise gr.Error("Label must be a .nii or .nii.gz file.")
    if not path.exists():
        raise gr.Error(f"Uploaded label file was not found: {path}")
    return path


def load_label_strict(path: Path) -> tuple:
    try:
        nii = nib.load(str(path), mmap=True)
    except Exception as exc:
        raise gr.Error(f"Could not read label as NIfTI: {exc}") from exc
    if tuple(int(v) for v in nii.shape) != EXPECTED_SHAPE[:3]:
        raise gr.Error(
            "Label rejected. Expected loaded NIfTI shape "
            f"{EXPECTED_SHAPE[:3]}, but got {tuple(int(v) for v in nii.shape)}."
        )
    return nii


def load_label_volume(path: Path) -> np.ndarray:
    label = np.asarray(load_label_strict(path).dataobj)
    return label.astype(np.uint8, copy=False)


def assert_label_matches_input(input_path: Path, label_path: Path) -> None:
    input_nii = load_nifti_strict(input_path)
    label_nii = load_label_strict(label_path)
    input_spacing = tuple(float(v) for v in input_nii.header.get_zooms()[:3])
    label_spacing = tuple(float(v) for v in label_nii.header.get_zooms()[:3])
    if not np.allclose(input_spacing, label_spacing, atol=1e-4):
        raise gr.Error(f"Label spacing does not match MRI spacing: label={label_spacing}, MRI={input_spacing}.")
    if not np.allclose(input_nii.affine, label_nii.affine, atol=1e-4):
        raise gr.Error("Label affine does not match MRI affine. Please upload the label from the same case/grid.")
    label = np.asarray(label_nii.dataobj) > 0
    if not np.any(label):
        raise gr.Error("Label contains no positive tumour voxels.")
    outside = 0
    positive = 0
    for z in range(EXPECTED_SHAPE[2]):
        image = np.asarray(input_nii.dataobj[:, :, z, :])
        brain = np.any(image != 0, axis=-1)
        lab = label[:, :, z]
        positive += int(lab.sum())
        outside += int(np.logical_and(lab, ~brain).sum())
    outside_fraction = outside / max(positive, 1)
    if outside_fraction > 0.01:
        raise gr.Error(
            "Label appears misaligned with the MRI: "
            f"{100.0 * outside_fraction:.2f}% of positive label voxels fall outside the nonzero brain region."
        )


def normalize_mri_slice(image: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    image = image.astype(np.float32, copy=True)
    brain_mask = np.any(image != 0, axis=-1)
    for channel in range(image.shape[-1]):
        channel_data = image[:, :, channel]
        values = channel_data[brain_mask]
        if values.size == 0:
            image[:, :, channel] = 0.0
            continue
        mean = float(values.mean())
        std = float(values.std())
        if std < eps:
            std = 1.0
        channel_data = (channel_data - mean) / std
        channel_data[~brain_mask] = 0.0
        image[:, :, channel] = channel_data
    return image


def preferred_model_device() -> torch.device:
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def validate_segmentation_mode(segmentation_mode: str | None) -> str:
    mode = (segmentation_mode or "binary").strip()
    if mode not in MODEL_CONFIGS:
        raise gr.Error(f"Unknown segmentation mode: {mode}.")
    return mode


def resolve_checkpoint_path(mode: str) -> Path:
    config = MODEL_CONFIGS[mode]
    checkpoint_path = Path(config["checkpoint"])
    if checkpoint_path.exists():
        return checkpoint_path
    repo_id = config.get("repo_id")
    filename = config.get("filename")
    if repo_id and filename:
        try:
            downloaded = hf_hub_download(
                repo_id=str(repo_id),
                filename=str(filename),
                repo_type="model",
                cache_dir=str(HF_MODEL_CACHE),
            )
        except Exception as exc:
            raise gr.Error(
                f"{config['label']} checkpoint is not available locally and could not be downloaded "
                f"from {repo_id}/{filename}: {exc}"
            ) from exc
        return Path(downloaded)
    raise gr.Error(
        f"{config['label']} checkpoint not found. Place the file at {checkpoint_path} "
        "or configure a Hugging Face checkpoint repo."
    )


def load_model(segmentation_mode: str | None) -> torch.nn.Module:
    mode = validate_segmentation_mode(segmentation_mode)
    if mode in _MODELS:
        return _MODELS[mode]
    config = MODEL_CONFIGS[mode]
    checkpoint_path = resolve_checkpoint_path(mode)
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise gr.Error("Checkpoint must be a dict containing model_state_dict.")
    model = PretrainedHybridViTSegmenter(weights_path=None, num_classes=int(config["num_classes"]))
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    target_device = preferred_model_device()
    try:
        model = model.to(target_device)
        _MODEL_DEVICES[mode] = target_device
    except Exception as exc:
        print(f"Model placement on {target_device} failed; keeping model on CPU. Reason: {exc}", flush=True)
        model = model.to("cpu")
        _MODEL_DEVICES[mode] = torch.device("cpu")
    _MODELS[mode] = model
    return _MODELS[mode]


def load_binary_model() -> torch.nn.Module:
    return load_model("binary")


def ensure_model_on_device(model: torch.nn.Module, device: torch.device, segmentation_mode: str | None) -> torch.nn.Module:
    mode = validate_segmentation_mode(segmentation_mode)
    if _MODEL_DEVICES.get(mode) != device:
        model = model.to(device)
        _MODEL_DEVICES[mode] = device
        _MODELS[mode] = model
    return model


def autocast_context(use_amp: bool):
    if not use_amp:
        return nullcontext()
    return torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True)


def preprocess_volume(path: Path) -> np.ndarray:
    nii = load_nifti_strict(path)
    batches = []
    for z in range(EXPECTED_SHAPE[2]):
        image = np.asarray(nii.dataobj[:, :, z, :], dtype=np.float32)
        image = normalize_mri_slice(image)
        batches.append(image.transpose(2, 0, 1))
    return np.stack(batches).astype(np.float32, copy=False)


@spaces.GPU(duration=10)
def segment_on_gpu(preprocessed_slices: np.ndarray, segmentation_mode: str = "binary") -> tuple[np.ndarray, np.ndarray, float]:
    mode = validate_segmentation_mode(segmentation_mode)
    model = load_model(mode)
    device = preferred_model_device()
    use_amp = device.type == "cuda"
    model = ensure_model_on_device(model, device, mode)
    model.eval()

    pred_volume = np.zeros(EXPECTED_SHAPE[:3], dtype=np.uint8)
    prob_volume = np.zeros(EXPECTED_SHAPE[:3], dtype=np.float32)
    start = time.perf_counter()
    with torch.inference_mode():
        for z0 in range(0, EXPECTED_SHAPE[2], BATCH_SIZE):
            z1 = min(z0 + BATCH_SIZE, EXPECTED_SHAPE[2])
            batch_tensor = torch.from_numpy(preprocessed_slices[z0:z1]).to(device=device, dtype=torch.float32)
            with autocast_context(use_amp):
                logits = model(batch_tensor)
            logits = logits.detach().float()
            if mode == "binary":
                probs = torch.sigmoid(logits).cpu().numpy()[:, 0]
                preds = (probs >= THRESHOLD).astype(np.uint8)
                prob_volume[:, :, z0:z1] = np.moveaxis(probs.astype(np.float32, copy=False), 0, -1)
            else:
                probs_tensor = torch.softmax(logits, dim=1)
                preds = torch.argmax(probs_tensor, dim=1).cpu().numpy().astype(np.uint8)
                whole_tumour_probs = (1.0 - probs_tensor[:, 0]).cpu().numpy()
                prob_volume[:, :, z0:z1] = np.moveaxis(whole_tumour_probs.astype(np.float32, copy=False), 0, -1)
            pred_volume[:, :, z0:z1] = np.moveaxis(preds, 0, -1)
    elapsed = time.perf_counter() - start
    return pred_volume, prob_volume, elapsed


def save_prediction(pred_volume: np.ndarray, reference_nii, job_dir: Path) -> Path:
    output_path = job_dir / "prediction.nii.gz"
    header = reference_nii.header.copy()
    header.set_data_dtype(np.uint8)
    nib.save(nib.Nifti1Image(pred_volume.astype(np.uint8), reference_nii.affine, header), str(output_path))
    return output_path


def save_probability(prob_volume: np.ndarray, job_dir: Path) -> Path:
    output_path = job_dir / "probability.npy"
    np.save(output_path, prob_volume.astype(np.float16, copy=False))
    return output_path


def load_probability_volume(path: Path) -> np.ndarray:
    if not path.exists():
        raise gr.Error("Probability map is missing for this job. Label metrics need a job created after probability saving was enabled.")
    prob = np.load(path).astype(np.float32, copy=False)
    if tuple(int(v) for v in prob.shape) != EXPECTED_SHAPE[:3]:
        raise gr.Error(f"Probability shape mismatch: expected {EXPECTED_SHAPE[:3]}, got {prob.shape}.")
    return prob


def save_overlay_image(overlay: np.ndarray, job_dir: Path) -> Path:
    output_path = job_dir / "overlay.png"
    Image.fromarray(overlay.astype(np.uint8)).save(output_path)
    return output_path


def metrics_markdown(metadata: dict) -> str:
    tumour_volume = float(metadata.get("tumour_volume_cm3", 0.0))
    brain_volume = metadata.get("brain_volume_cm3")
    tumour_brain_percent = metadata.get("tumour_brain_percent")
    mode = validate_segmentation_mode(str(metadata.get("segmentation_mode", "binary")))
    mode_label = str(MODEL_CONFIGS[mode]["label"])
    brain_line = (
        f"- Brain volume: **{float(brain_volume):.2f} cm^3**\n"
        if brain_volume is not None
        else "- Brain volume: `not available`\n"
    )
    ratio_line = (
        f"- Tumour fraction of brain volume: **{float(tumour_brain_percent):.3f}%**\n"
        if tumour_brain_percent is not None
        else "- Tumour fraction of brain volume: `not available`\n"
    )
    label_lines = ""
    if metadata.get("label_metrics_available"):
        label_lines = (
            "\n### Label comparison\n\n"
            f"- Whole-tumour 3D Dice: **{float(metadata.get('label_dice3d', 0.0)):.4f}**\n"
            f"- Whole-tumour 3D binary cross entropy: **{float(metadata.get('label_bce3d', 0.0)):.4f}**\n"
            f"- Label tumour volume: **{float(metadata.get('label_volume_cm3', 0.0)):.2f} cm^3**\n"
        )
    return (
        f"### Tumour segmentation complete\n\n"
        f"- Job ID: `{metadata['job_id']}`\n"
        f"- Segmentation mode: **{mode_label}**\n"
        f"- Input shape: `{EXPECTED_SHAPE}`\n"
        f"- Tumour volume: **{tumour_volume:.2f} cm^3**\n"
        f"{brain_line}"
        f"{ratio_line}"
        f"- GPU inference time: `{metadata['gpu_inference_seconds']:.2f} s`\n\n"
        f"{label_lines}\n"
        "Keep this Job ID to reload the result if the page is refreshed."
    )


def save_job_metadata(job_dir: Path, metadata: dict) -> Path:
    output_path = job_dir / "metadata.json"
    output_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    return output_path


def load_job_metadata(job_dir: Path) -> dict:
    metadata_path = job_dir / "metadata.json"
    if not metadata_path.exists():
        return {}
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def update_job_metadata(job_id: str, updates: dict) -> dict:
    job_dir = get_job_dir(job_id)
    with _JOB_LOCK:
        metadata = load_job_metadata(job_dir)
        metadata.update(updates)
        save_job_metadata(job_dir, metadata)
    return metadata


def count_brain_voxels(path: Path) -> int:
    nii = load_nifti_strict(path)
    count = 0
    for z in range(EXPECTED_SHAPE[2]):
        image = np.asarray(nii.dataobj[:, :, z, :])
        count += int(np.any(image != 0, axis=-1).sum())
    return count


def compute_label_metrics(pred_volume: np.ndarray, prob_volume: np.ndarray, label_volume: np.ndarray, voxel_volume_mm3: float) -> dict:
    pred = pred_volume.astype(bool)
    label = label_volume.astype(bool)
    intersection = float(np.logical_and(pred, label).sum())
    pred_sum = float(pred.sum())
    label_sum = float(label.sum())
    eps = 1e-6
    dice = (2.0 * intersection + eps) / (pred_sum + label_sum + eps)
    probs = np.clip(prob_volume.astype(np.float32, copy=False), eps, 1.0 - eps)
    target = label.astype(np.float32)
    bce = float(-(target * np.log(probs) + (1.0 - target) * np.log(1.0 - probs)).mean())
    label_volume_cm3 = label_sum * voxel_volume_mm3 / 1000.0
    return {
        "label_metrics_available": True,
        "label_dice3d": float(dice),
        "label_bce3d": bce,
        "label_voxels": int(label_sum),
        "label_volume_cm3": float(label_volume_cm3),
    }


def make_overlay(path: Path, pred_volume: np.ndarray) -> np.ndarray:
    nii = load_nifti_strict(path)
    positive_slices = np.flatnonzero(pred_volume.sum(axis=(0, 1)) > 0)
    if positive_slices.size:
        z = int(positive_slices[np.argmax(pred_volume[:, :, positive_slices].sum(axis=(0, 1)))])
    else:
        z = EXPECTED_SHAPE[2] // 2
    image = np.asarray(nii.dataobj[:, :, z, 0], dtype=np.float32)
    mask = pred_volume[:, :, z].astype(bool)

    brain = image[image != 0]
    if brain.size:
        lo, hi = np.percentile(brain, [1, 99])
    else:
        lo, hi = float(image.min()), float(image.max())
    if hi <= lo:
        hi = lo + 1.0
    gray = np.clip((image - lo) / (hi - lo), 0, 1)
    rgb = np.repeat(gray[..., None], 3, axis=-1)
    overlay = rgb.copy()
    overlay[mask] = 0.35 * overlay[mask] + 0.65 * np.array([1.0, 0.05, 0.02])
    canvas = (overlay * 255).astype(np.uint8)
    return np.flipud(np.swapaxes(canvas, 0, 1))


def empty_3d_figure() -> go.Figure:
    fig = go.Figure()
    fig.update_layout(
        template="plotly_white",
        height=520,
        margin=dict(l=0, r=0, t=30, b=0),
        scene=dict(
            xaxis_title="x",
            yaxis_title="y",
            zaxis_title="z",
            aspectmode="data",
        ),
        title="3D visualization appears after segmentation completes",
    )
    return fig


def load_prediction_volume(path: Path) -> np.ndarray:
    try:
        pred = np.asarray(nib.load(str(path), mmap=True).dataobj, dtype=np.uint8)
    except Exception as exc:
        raise gr.Error(f"Could not read prediction mask: {exc}") from exc
    if tuple(int(v) for v in pred.shape) != EXPECTED_SHAPE[:3]:
        raise gr.Error(f"Prediction shape mismatch: expected {EXPECTED_SHAPE[:3]}, got {pred.shape}.")
    return pred


def dilate_mask(mask: np.ndarray, iterations: int = 1) -> np.ndarray:
    out = np.asarray(mask).astype(bool)
    for _ in range(iterations):
        padded = np.pad(out, 1, mode="constant", constant_values=False)
        out = (
            padded[1:-1, 1:-1]
            | padded[:-2, 1:-1]
            | padded[2:, 1:-1]
            | padded[1:-1, :-2]
            | padded[1:-1, 2:]
            | padded[:-2, :-2]
            | padded[:-2, 2:]
            | padded[2:, :-2]
            | padded[2:, 2:]
        )
    return out


def erode_mask(mask: np.ndarray) -> np.ndarray:
    padded = np.pad(np.asarray(mask).astype(bool), 1, mode="constant", constant_values=False)
    return (
        padded[1:-1, 1:-1]
        & padded[:-2, 1:-1]
        & padded[2:, 1:-1]
        & padded[1:-1, :-2]
        & padded[1:-1, 2:]
        & padded[:-2, :-2]
        & padded[:-2, 2:]
        & padded[2:, :-2]
        & padded[2:, 2:]
    )


def render_overlay_plane(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    image = np.asarray(image, dtype=np.float32)
    mask = np.asarray(mask).astype(bool)
    brain = image[image != 0]
    if brain.size:
        lo, hi = np.percentile(brain, [1, 99])
    else:
        lo, hi = float(image.min()), float(image.max())
    if hi <= lo:
        hi = lo + 1.0
    gray = np.clip((image - lo) / (hi - lo), 0, 1)
    rgb = np.repeat(gray[..., None], 3, axis=-1)
    if np.any(mask):
        display_mask = dilate_mask(mask, iterations=1)
        rgb[display_mask] = 0.12 * rgb[display_mask] + 0.88 * np.array([1.0, 0.02, 0.0])
    canvas = (rgb * 255).astype(np.uint8)
    return np.flipud(np.swapaxes(canvas, 0, 1))


def tumour_center(pred_volume: np.ndarray) -> tuple[int, int, int]:
    coords = np.argwhere(pred_volume > 0)
    if coords.size == 0:
        return EXPECTED_SHAPE[0] // 2, EXPECTED_SHAPE[1] // 2, EXPECTED_SHAPE[2] // 2
    center = np.round(coords.mean(axis=0)).astype(int)
    return (
        int(np.clip(center[0], 0, EXPECTED_SHAPE[0] - 1)),
        int(np.clip(center[1], 0, EXPECTED_SHAPE[1] - 1)),
        int(np.clip(center[2], 0, EXPECTED_SHAPE[2] - 1)),
    )


def make_slice_views(input_path: Path, prediction_path: Path, x: int, y: int, z: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    nii = load_nifti_strict(input_path)
    pred = load_prediction_volume(prediction_path)
    x = int(np.clip(x, 0, EXPECTED_SHAPE[0] - 1))
    y = int(np.clip(y, 0, EXPECTED_SHAPE[1] - 1))
    z = int(np.clip(z, 0, EXPECTED_SHAPE[2] - 1))
    data = nii.dataobj
    oxy = render_overlay_plane(np.asarray(data[:, :, z, 0], dtype=np.float32), pred[:, :, z])
    oyz = render_overlay_plane(np.asarray(data[x, :, :, 0], dtype=np.float32), pred[x, :, :])
    ozx = render_overlay_plane(np.asarray(data[:, y, :, 0], dtype=np.float32), pred[:, y, :])
    return oxy, oyz, ozx


def max_pool_binary_volume(mask: np.ndarray, step: int) -> np.ndarray:
    mask = np.asarray(mask).astype(bool)
    pad = [(0, (-size) % step) for size in mask.shape]
    padded = np.pad(mask, pad, mode="constant", constant_values=False)
    pooled = padded.reshape(
        padded.shape[0] // step,
        step,
        padded.shape[1] // step,
        step,
        padded.shape[2] // step,
        step,
    ).max(axis=(1, 3, 5))
    return pooled.astype(np.float32)


def max_pool_label_volume(volume: np.ndarray, step: int) -> np.ndarray:
    labels = np.asarray(volume).astype(np.uint8, copy=False)
    pad = [(0, (-size) % step) for size in labels.shape]
    padded = np.pad(labels, pad, mode="constant", constant_values=0)
    pooled = padded.reshape(
        padded.shape[0] // step,
        step,
        padded.shape[1] // step,
        step,
        padded.shape[2] // step,
        step,
    ).max(axis=(1, 3, 5))
    return pooled.astype(np.uint8, copy=False)


def make_crosshair_axis(axis: str, x_index: int, y_index: int, z_index: int, color: str) -> go.Scatter3d:
    nx, ny, nz = EXPECTED_SHAPE[:3]
    x0 = float(np.clip(x_index, 0, nx - 1))
    y0 = float(np.clip(y_index, 0, ny - 1))
    z0 = float(np.clip(z_index, 0, nz - 1))
    if axis == "x":
        xs, ys, zs = [0, nx - 1], [y0, y0], [z0, z0]
    elif axis == "y":
        xs, ys, zs = [x0, x0], [0, ny - 1], [z0, z0]
    else:
        xs, ys, zs = [x0, x0], [y0, y0], [0, nz - 1]
    return go.Scatter3d(
        x=xs,
        y=ys,
        z=zs,
        mode="lines",
        line=dict(color=color, width=6),
        hoverinfo="skip",
        name=f"{axis} axis",
        showlegend=True,
    )


def make_3d_visualization(input_path: Path, prediction_path: Path, x_index: int, y_index: int, z_index: int) -> go.Figure:
    nii = load_nifti_strict(input_path)
    pred = load_prediction_volume(prediction_path)
    step = 4
    brain = np.asarray(nii.dataobj[::step, ::step, ::step, 0], dtype=np.float32)
    mask_step = 3
    mask = max_pool_binary_volume(pred, step=mask_step)
    brain_mask = brain != 0
    if np.any(brain_mask):
        values = brain[brain_mask]
        lo, hi = np.percentile(values, [5, 99])
        if hi <= lo:
            hi = lo + 1.0
        brain_norm = np.clip((brain - lo) / (hi - lo), 0, 1)
    else:
        brain_norm = np.zeros_like(brain, dtype=np.float32)

    xs = np.arange(brain.shape[0]) * step
    ys = np.arange(brain.shape[1]) * step
    zs = np.arange(brain.shape[2]) * step
    grid_x, grid_y, grid_z = np.meshgrid(xs, ys, zs, indexing="ij")
    fig = go.Figure()
    fig.add_trace(
        go.Volume(
            x=grid_x.ravel(),
            y=grid_y.ravel(),
            z=grid_z.ravel(),
            value=brain_norm.ravel(),
            isomin=0.07,
            isomax=1.0,
            opacity=0.16,
            surface_count=14,
            colorscale="Gray",
            showscale=False,
            name="Brain",
        )
    )
    if np.any(mask):
        mask_xs = np.arange(mask.shape[0]) * mask_step + mask_step / 2.0
        mask_ys = np.arange(mask.shape[1]) * mask_step + mask_step / 2.0
        mask_zs = np.arange(mask.shape[2]) * mask_step + mask_step / 2.0
        mask_x, mask_y, mask_z = np.meshgrid(mask_xs, mask_ys, mask_zs, indexing="ij")
        fig.add_trace(
            go.Volume(
                x=mask_x.ravel(),
                y=mask_y.ravel(),
                z=mask_z.ravel(),
                value=mask.ravel(),
                isomin=0.1,
                isomax=1.0,
                surface_count=12,
                opacity=0.42,
                colorscale=[[0.0, "rgb(220,0,0)"], [1.0, "rgb(255,20,10)"]],
                showscale=False,
                name="Predicted tumour",
            )
        )
    fig.add_trace(make_crosshair_axis("x", int(x_index), int(y_index), int(z_index), "rgb(40,120,255)"))
    fig.add_trace(make_crosshair_axis("y", int(x_index), int(y_index), int(z_index), "rgb(30,190,90)"))
    fig.add_trace(make_crosshair_axis("z", int(x_index), int(y_index), int(z_index), "rgb(245,180,35)"))
    fig.update_layout(
        template="plotly_white",
        height=560,
        margin=dict(l=0, r=0, t=30, b=0),
        legend=dict(x=0.02, y=0.98),
        scene=dict(
            xaxis=dict(title="x", range=[0, EXPECTED_SHAPE[0] - 1]),
            yaxis=dict(title="y", range=[0, EXPECTED_SHAPE[1] - 1]),
            zaxis=dict(title="z", range=[0, EXPECTED_SHAPE[2] - 1]),
            aspectmode="data",
        ),
        title="3D predicted tumour visualization",
    )
    return fig


def encode_uint8(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array.astype(np.uint8, copy=False))
    return base64.b64encode(contiguous.tobytes()).decode("ascii")


def normalize_display_volume(nii) -> np.ndarray:
    volume = np.asarray(nii.dataobj[:, :, :, 0], dtype=np.float32)
    brain = volume[volume != 0]
    if brain.size:
        lo, hi = np.percentile(brain, [1, 99])
    else:
        lo, hi = float(volume.min()), float(volume.max())
    if hi <= lo:
        hi = lo + 1.0
    return (np.clip((volume - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8)


def make_local_viewer_html(
    input_path: Path,
    prediction_path: Path,
    x_index: int,
    y_index: int,
    z_index: int,
    label_path: Path | None = None,
    segmentation_mode: str = "binary",
) -> str:
    mode = validate_segmentation_mode(segmentation_mode)
    is_multiclass = mode == "4label"
    nii = load_nifti_strict(input_path)
    pred_raw = load_prediction_volume(prediction_path)
    label_raw = load_label_volume(label_path) if label_path is not None else None
    pred = pred_raw if is_multiclass else (pred_raw > 0).astype(np.uint8)
    label_volume = (
        label_raw if is_multiclass else (label_raw > 0).astype(np.uint8)
    ) if label_raw is not None else None
    display = normalize_display_volume(nii)

    brain_step = 5
    brain_small = display[::brain_step, ::brain_step, ::brain_step]
    mask_step = 3
    mask_small = max_pool_label_volume(pred, step=mask_step)
    label_small = (
        max_pool_label_volume(label_volume, step=mask_step)
        if label_volume is not None
        else np.zeros_like(mask_small, dtype=np.uint8)
    )

    nx, ny, nz = EXPECTED_SHAPE[:3]
    payload = {
        "nx": nx,
        "ny": ny,
        "nz": nz,
        "x0": int(np.clip(x_index, 0, nx - 1)),
        "y0": int(np.clip(y_index, 0, ny - 1)),
        "z0": int(np.clip(z_index, 0, nz - 1)),
        "brainStep": brain_step,
        "brainShape": list(brain_small.shape),
        "maskStep": mask_step,
        "maskShape": list(mask_small.shape),
        "display": encode_uint8(display),
        "mask": encode_uint8(pred),
        "hasLabel": label_volume is not None,
        "label": encode_uint8(label_volume) if label_volume is not None else "",
        "isMulticlass": is_multiclass,
        "brainSmall": encode_uint8(brain_small),
        "maskSmall": encode_uint8(mask_small),
        "labelSmall": encode_uint8(label_small),
    }
    payload_json = json.dumps(payload, separators=(",", ":"))
    document = f"""
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <style>
    body {{
      margin: 0;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      color: #111827;
      background: #ffffff;
    }}
    .viewer {{
      display: grid;
      grid-template-columns: 1fr;
      gap: 14px;
    }}
    #plot3d {{
      width: 100%;
      height: 560px;
      border: 1px solid #e5e7eb;
      border-radius: 8px;
      contain: layout paint;
    }}
    .controls {{
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 12px;
      align-items: end;
    }}
    .control {{
      border: 1px solid #e5e7eb;
      border-radius: 8px;
      padding: 10px 12px;
      background: #fafafa;
    }}
    .control label {{
      display: flex;
      justify-content: space-between;
      gap: 8px;
      font-size: 13px;
      font-weight: 650;
      margin-bottom: 6px;
    }}
    .control-x label {{
      color: #2878ff;
    }}
    .control-y label {{
      color: #1ebe5a;
    }}
    .control-z label {{
      color: #d49600;
    }}
    .control input {{
      width: 100%;
    }}
    .toggles {{
      display: flex;
      flex-wrap: wrap;
      gap: 12px;
      align-items: center;
      border: 1px solid #e5e7eb;
      border-radius: 8px;
      padding: 10px 12px;
      background: #ffffff;
      font-size: 14px;
    }}
    .toggles label {{
      display: inline-flex;
      align-items: center;
      gap: 6px;
      font-weight: 650;
    }}
    .legend-dot {{
      display: inline-block;
      width: 10px;
      height: 10px;
      border-radius: 50%;
    }}
    .slices {{
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 12px;
    }}
    .slice-panel {{
      border: 1px solid #e5e7eb;
      border-radius: 8px;
      padding: 10px;
      background: #ffffff;
    }}
    .slice-panel h3 {{
      margin: 0 0 8px;
      font-size: 14px;
    }}
    canvas {{
      width: 100%;
      height: auto;
      image-rendering: pixelated;
      background: #000000;
      border-radius: 4px;
    }}
    @media (max-width: 900px) {{
      .controls, .slices {{
        grid-template-columns: 1fr;
      }}
    }}
  </style>
</head>
<body>
  <div class="viewer">
    <div id="plot3d"></div>
    <div class="toggles">
      <label><input id="showPred" type="checkbox" checked /> <span class="legend-dot" style="background:#ff3b30;"></span><span id="predLegend1">Prediction</span></label>
      <label class="multiclass-legend"><span class="legend-dot" style="background:#22c55e;"></span>Pred 2</label>
      <label class="multiclass-legend"><span class="legend-dot" style="background:#3b82f6;"></span>Pred 3</label>
      <label><input id="showLabel" type="checkbox" checked /> <span class="legend-dot" style="background:#ff8a3d;"></span><span id="labelLegend1">Label</span></label>
      <label class="multiclass-legend"><span class="legend-dot" style="background:#14d9a7;"></span>Label 2</label>
      <label class="multiclass-legend"><span class="legend-dot" style="background:#8b5cf6;"></span>Label 3</label>
    </div>
    <div class="controls">
      <div class="control control-x">
        <label><span>Sagittal x → Oyz</span><span id="xValue"></span></label>
        <input id="xSlider" type="range" min="0" max="{nx - 1}" step="1" />
      </div>
      <div class="control control-y">
        <label><span>Coronal y → Ozx</span><span id="yValue"></span></label>
        <input id="ySlider" type="range" min="0" max="{ny - 1}" step="1" />
      </div>
      <div class="control control-z">
        <label><span>Axial z → Oxy</span><span id="zValue"></span></label>
        <input id="zSlider" type="range" min="0" max="{nz - 1}" step="1" />
      </div>
    </div>
    <div class="slices">
      <div class="slice-panel"><h3>Oyz slice at x</h3><canvas id="oyzCanvas"></canvas></div>
      <div class="slice-panel"><h3>Ozx slice at y</h3><canvas id="ozxCanvas"></canvas></div>
      <div class="slice-panel"><h3>Oxy slice at z</h3><canvas id="oxyCanvas"></canvas></div>
    </div>
  </div>
  <script>
    const payload = {payload_json};
    const nx = payload.nx, ny = payload.ny, nz = payload.nz;
    function decodeBase64(value) {{
      const binary = atob(value);
      const out = new Uint8Array(binary.length);
      for (let i = 0; i < binary.length; i++) out[i] = binary.charCodeAt(i);
      return out;
    }}
    const display = decodeBase64(payload.display);
    const mask = decodeBase64(payload.mask);
    const label = payload.hasLabel ? decodeBase64(payload.label) : new Uint8Array(mask.length);
    const brainSmall = decodeBase64(payload.brainSmall);
    const maskSmall = decodeBase64(payload.maskSmall);
    const labelSmall = decodeBase64(payload.labelSmall);
    const xSlider = document.getElementById("xSlider");
    const ySlider = document.getElementById("ySlider");
    const zSlider = document.getElementById("zSlider");
    const showPred = document.getElementById("showPred");
    const showLabel = document.getElementById("showLabel");
    const isMulticlass = Boolean(payload.isMulticlass);
    xSlider.value = payload.x0;
    ySlider.value = payload.y0;
    zSlider.value = payload.z0;
    showLabel.disabled = !payload.hasLabel;
    if (!payload.hasLabel) showLabel.checked = false;
    document.getElementById("predLegend1").textContent = isMulticlass ? "Pred 1" : "Prediction";
    document.getElementById("labelLegend1").textContent = isMulticlass ? "Label 1" : "Label";
    for (const item of document.querySelectorAll(".multiclass-legend")) {{
      item.style.display = isMulticlass ? "inline-flex" : "none";
    }}

    function idx(x, y, z) {{
      return (x * ny + y) * nz + z;
    }}

    const predColors = {{
      1: [255, 59, 48],
      2: [34, 197, 94],
      3: [59, 130, 246],
    }};
    const labelColors = {{
      1: [255, 138, 61],
      2: [20, 217, 167],
      3: [139, 92, 246],
    }};
    const overlapColors = {{
      1: [180, 24, 20],
      2: [10, 132, 64],
      3: [30, 75, 180],
    }};
    function blendOnGray(gray, color, alpha) {{
      return [
        Math.round((1 - alpha) * gray + alpha * color[0]),
        Math.round((1 - alpha) * gray + alpha * color[1]),
        Math.round((1 - alpha) * gray + alpha * color[2]),
      ];
    }}
    function addColors(a, b) {{
      return [
        Math.min(255, Math.round(0.62 * a[0] + 0.62 * b[0])),
        Math.min(255, Math.round(0.62 * a[1] + 0.62 * b[1])),
        Math.min(255, Math.round(0.62 * a[2] + 0.62 * b[2])),
      ];
    }}
    function classColor(colors, classId) {{
      return colors[classId] || colors[1];
    }}
    function drawPixel(buffer, offset, gray, predClass, labelClass) {{
      predClass = predClass > 0 ? Math.min(3, predClass) : 0;
      labelClass = labelClass > 0 ? Math.min(3, labelClass) : 0;
      const predVisible = showPred.checked && predClass > 0;
      const labelVisible = showLabel.checked && labelClass > 0;
      let rgb;
      if (predVisible && labelVisible && predClass === labelClass) {{
        rgb = blendOnGray(gray, classColor(overlapColors, predClass), 0.92);
      }} else if (predVisible && labelVisible) {{
        rgb = blendOnGray(gray, addColors(classColor(predColors, predClass), classColor(labelColors, labelClass)), 0.86);
      }} else if (predVisible) {{
        rgb = blendOnGray(gray, classColor(predColors, predClass), 0.88);
      }} else if (labelVisible) {{
        rgb = blendOnGray(gray, classColor(labelColors, labelClass), 0.82);
      }} else {{
        rgb = [gray, gray, gray];
      }}
      buffer[offset] = rgb[0];
      buffer[offset + 1] = rgb[1];
      buffer[offset + 2] = rgb[2];
      buffer[offset + 3] = 255;
    }}

    function drawPlane(canvasId, plane, index) {{
      let width, height;
      if (plane === "oyz") {{
        width = ny; height = nz;
      }} else if (plane === "ozx") {{
        width = nx; height = nz;
      }} else {{
        width = nx; height = ny;
      }}
      const canvas = document.getElementById(canvasId);
      canvas.width = width;
      canvas.height = height;
      const ctx = canvas.getContext("2d");
      const image = ctx.createImageData(width, height);
      const data = image.data;
      for (let row = 0; row < height; row++) {{
        for (let col = 0; col < width; col++) {{
          let x, y, z;
          if (plane === "oyz") {{
            x = index; y = col; z = nz - 1 - row;
          }} else if (plane === "ozx") {{
            x = col; y = index; z = nz - 1 - row;
          }} else {{
            x = col; y = ny - 1 - row; z = index;
          }}
          const linear = idx(x, y, z);
          drawPixel(data, (row * width + col) * 4, display[linear], mask[linear], label[linear]);
        }}
      }}
      ctx.putImageData(image, 0, 0);
    }}

    function drawCurrentPlane(axis) {{
      if (axis === "x") {{
        drawPlane("oyzCanvas", "oyz", Number(xSlider.value));
      }} else if (axis === "y") {{
        drawPlane("ozxCanvas", "ozx", Number(ySlider.value));
      }} else if (axis === "z") {{
        drawPlane("oxyCanvas", "oxy", Number(zSlider.value));
      }}
    }}

    function makeGrid(shape, step, offset, values, scale) {{
      const sx = shape[0], sy = shape[1], sz = shape[2];
      const x = [], y = [], z = [], value = [];
      for (let i = 0; i < sx; i++) {{
        for (let j = 0; j < sy; j++) {{
          for (let k = 0; k < sz; k++) {{
            x.push(i * step + offset);
            y.push(j * step + offset);
            z.push(k * step + offset);
            value.push(values[(i * sy + j) * sz + k] / scale);
          }}
        }}
      }}
      return {{x, y, z, value}};
    }}

    function makeClassGrid(shape, step, offset, values, classId) {{
      const sx = shape[0], sy = shape[1], sz = shape[2];
      const x = [], y = [], z = [], value = [];
      for (let i = 0; i < sx; i++) {{
        for (let j = 0; j < sy; j++) {{
          for (let k = 0; k < sz; k++) {{
            x.push(i * step + offset);
            y.push(j * step + offset);
            z.push(k * step + offset);
            value.push(values[(i * sy + j) * sz + k] === classId ? 1 : 0);
          }}
        }}
      }}
      return {{x, y, z, value}};
    }}

    function hasClass(values, classId) {{
      for (let i = 0; i < values.length; i++) {{
        if (values[i] === classId) return true;
      }}
      return false;
    }}

    function tumourTrace(grid, name, color, visible) {{
      return {{
        type: "volume",
        ...grid,
        isomin: 0.1,
        isomax: 1.0,
        opacity: 0.44,
        surface_count: 8,
        colorscale: [[0, color], [1, color]],
        showscale: false,
        name,
        visible,
      }};
    }}

    function appendLine(xs, ys, zs, points) {{
      for (const point of points) {{
        xs.push(point[0]); ys.push(point[1]); zs.push(point[2]);
      }}
      xs.push(null); ys.push(null); zs.push(null);
    }}

    function planeFrame(axis, x, y, z, color) {{
      const xs = [], ys = [], zs = [];
      if (axis === "x") {{
        appendLine(xs, ys, zs, [[x, 0, 0], [x, ny - 1, 0], [x, ny - 1, nz - 1], [x, 0, nz - 1], [x, 0, 0]]);
      }} else if (axis === "y") {{
        appendLine(xs, ys, zs, [[0, y, 0], [nx - 1, y, 0], [nx - 1, y, nz - 1], [0, y, nz - 1], [0, y, 0]]);
      }} else {{
        appendLine(xs, ys, zs, [[0, 0, z], [nx - 1, 0, z], [nx - 1, ny - 1, z], [0, ny - 1, z], [0, 0, z]]);
      }}
      return {{x: xs, y: ys, z: zs, mode: "lines", type: "scatter3d", line: {{color, width: 3}}, opacity: 0.85, hoverinfo: "skip", name: axis + " plane frame", showlegend: false}};
    }}

    function axisLine(axis, x, y, z, color) {{
      let xs, ys, zs;
      if (axis === "x") {{
        xs = [0, nx - 1]; ys = [y, y]; zs = [z, z];
      }} else if (axis === "y") {{
        xs = [x, x]; ys = [0, ny - 1]; zs = [z, z];
      }} else {{
        xs = [x, x]; ys = [y, y]; zs = [0, nz - 1];
      }}
      return {{x: xs, y: ys, z: zs, mode: "lines", type: "scatter3d", line: {{color, width: 6}}, hoverinfo: "skip", name: axis + " axis"}};
    }}

    const brainGrid = makeGrid(payload.brainShape, payload.brainStep, 0, brainSmall, 255);
    const predTraceIndices = [];
    const labelTraceIndices = [];
    const initialX = Number(xSlider.value), initialY = Number(ySlider.value), initialZ = Number(zSlider.value);
    const plotData = [
      {{type: "volume", ...brainGrid, isomin: 0.07, isomax: 1.0, opacity: 0.18, surface_count: 8, colorscale: "Gray", showscale: false, name: "Brain"}},
    ];
    const predTraceColors = {{1: "rgb(255,59,48)", 2: "rgb(34,197,94)", 3: "rgb(59,130,246)"}};
    const labelTraceColors = {{1: "rgb(255,138,61)", 2: "rgb(20,217,167)", 3: "rgb(139,92,246)"}};
    for (const classId of [1, 2, 3]) {{
      const visible = hasClass(maskSmall, classId);
      predTraceIndices.push(plotData.length);
      plotData.push(tumourTrace(
        makeClassGrid(payload.maskShape, payload.maskStep, payload.maskStep / 2, maskSmall, classId),
        "Pred " + classId,
        predTraceColors[classId],
        visible
      ));
    }}
    for (const classId of [1, 2, 3]) {{
      const visible = payload.hasLabel && hasClass(labelSmall, classId);
      labelTraceIndices.push(plotData.length);
      plotData.push(tumourTrace(
        makeClassGrid(payload.maskShape, payload.maskStep, payload.maskStep / 2, labelSmall, classId),
        "Label " + classId,
        labelTraceColors[classId],
        visible
      ));
    }}
    const guideTraceStart = plotData.length;
    plotData.push(
      planeFrame("x", initialX, initialY, initialZ, "rgb(40,120,255)"),
      planeFrame("y", initialX, initialY, initialZ, "rgb(30,190,90)"),
      planeFrame("z", initialX, initialY, initialZ, "rgb(245,180,35)"),
      axisLine("x", initialX, initialY, initialZ, "rgb(40,120,255)"),
      axisLine("y", initialX, initialY, initialZ, "rgb(30,190,90)"),
      axisLine("z", initialX, initialY, initialZ, "rgb(245,180,35)")
    );
    const layout = {{
      title: "3D predicted tumour visualization",
      height: 560,
      margin: {{l: 0, r: 0, t: 30, b: 0}},
      legend: {{x: 0.02, y: 0.98}},
      scene: {{
        xaxis: {{title: "x", range: [0, nx - 1]}},
        yaxis: {{title: "y", range: [0, ny - 1]}},
        zaxis: {{title: "z", range: [0, nz - 1]}},
        aspectmode: "data"
      }}
    }};

    Plotly.newPlot("plot3d", plotData, layout, {{displayModeBar: true, responsive: true}});

    let pending2dFrame = null;
    let pending3dFrame = null;
    function updateValueLabels() {{
      const x = Number(xSlider.value), y = Number(ySlider.value), z = Number(zSlider.value);
      document.getElementById("xValue").textContent = x;
      document.getElementById("yValue").textContent = y;
      document.getElementById("zValue").textContent = z;
    }}
    function updateAllSlices() {{
      const x = Number(xSlider.value), y = Number(ySlider.value), z = Number(zSlider.value);
      drawPlane("oyzCanvas", "oyz", x);
      drawPlane("ozxCanvas", "ozx", y);
      drawPlane("oxyCanvas", "oxy", z);
    }}
    function update3dGuides() {{
      const x = Number(xSlider.value), y = Number(ySlider.value), z = Number(zSlider.value);
      const xFrame = planeFrame("x", x, y, z, "rgb(40,120,255)");
      const yFrame = planeFrame("y", x, y, z, "rgb(30,190,90)");
      const zFrame = planeFrame("z", x, y, z, "rgb(245,180,35)");
      const xWire = axisLine("x", x, y, z, "rgb(40,120,255)");
      const yWire = axisLine("y", x, y, z, "rgb(30,190,90)");
      const zWire = axisLine("z", x, y, z, "rgb(245,180,35)");
      Plotly.restyle("plot3d", {{x: [xFrame.x, yFrame.x, zFrame.x, xWire.x, yWire.x, zWire.x], y: [xFrame.y, yFrame.y, zFrame.y, xWire.y, yWire.y, zWire.y], z: [xFrame.z, yFrame.z, zFrame.z, xWire.z, yWire.z, zWire.z]}}, [guideTraceStart, guideTraceStart + 1, guideTraceStart + 2, guideTraceStart + 3, guideTraceStart + 4, guideTraceStart + 5]);
      Plotly.restyle("plot3d", {{visible: predTraceIndices.map((_, i) => showPred.checked && hasClass(maskSmall, i + 1))}}, predTraceIndices);
      Plotly.restyle("plot3d", {{visible: labelTraceIndices.map((_, i) => showLabel.checked && payload.hasLabel && hasClass(labelSmall, i + 1))}}, labelTraceIndices);
    }}
    function updateTraceVisibility() {{
      Plotly.restyle("plot3d", {{visible: predTraceIndices.map((_, i) => showPred.checked && hasClass(maskSmall, i + 1))}}, predTraceIndices);
      Plotly.restyle("plot3d", {{visible: labelTraceIndices.map((_, i) => showLabel.checked && payload.hasLabel && hasClass(labelSmall, i + 1))}}, labelTraceIndices);
    }}
    function scheduleSliceUpdate(axis) {{
      if (pending2dFrame !== null) cancelAnimationFrame(pending2dFrame);
      pending2dFrame = requestAnimationFrame(() => {{
        pending2dFrame = null;
        updateValueLabels();
        drawCurrentPlane(axis);
      }});
    }}
    function scheduleFullUpdate() {{
      if (pending3dFrame !== null) cancelAnimationFrame(pending3dFrame);
      pending3dFrame = requestAnimationFrame(() => {{
        pending3dFrame = null;
        updateValueLabels();
        updateAllSlices();
        update3dGuides();
      }});
    }}
    function scheduleVisibilityUpdate() {{
      if (pending3dFrame !== null) cancelAnimationFrame(pending3dFrame);
      pending3dFrame = requestAnimationFrame(() => {{
        pending3dFrame = null;
        updateAllSlices();
        updateTraceVisibility();
      }});
    }}
    xSlider.addEventListener("input", () => scheduleSliceUpdate("x"));
    ySlider.addEventListener("input", () => scheduleSliceUpdate("y"));
    zSlider.addEventListener("input", () => scheduleSliceUpdate("z"));
    xSlider.addEventListener("change", scheduleFullUpdate);
    ySlider.addEventListener("change", scheduleFullUpdate);
    zSlider.addEventListener("change", scheduleFullUpdate);
    showPred.addEventListener("change", scheduleVisibilityUpdate);
    showLabel.addEventListener("change", scheduleVisibilityUpdate);
    updateValueLabels();
    updateAllSlices();
    update3dGuides();
  </script>
</body>
</html>
"""
    return f'<iframe srcdoc="{html.escape(document, quote=True)}" style="width:100%; height:1120px; border:0;" loading="lazy"></iframe>'


def empty_viewer_html() -> str:
    return (
        "<div style='height:520px;border:1px solid #e5e7eb;border-radius:8px;"
        "display:flex;align-items:center;justify-content:center;color:#6b7280;'>"
        "3D/2D viewer appears after segmentation completes."
        "</div>"
    )


def start_job(uploaded_file, label_file, segmentation_mode, history):
    mode = validate_segmentation_mode(segmentation_mode)
    path = validate_uploaded_path(uploaded_file)
    label_source = validate_label_path(label_file)
    nii = load_nifti_strict(path)
    if label_source is not None:
        load_label_strict(label_source)
        assert_label_matches_input(path, label_source)
    job_id = make_job_id(mode)
    job_dir = get_job_dir(job_id)
    job_dir.mkdir(parents=True, exist_ok=False)
    input_path = saved_input_path(job_dir, path)
    shutil.copyfile(path, input_path)
    copied_label_path = None
    if label_source is not None:
        copied_label_path = saved_label_path(job_dir, label_source)
        shutil.copyfile(label_source, copied_label_path)
    spacing = tuple(float(v) for v in nii.header.get_zooms()[:3])
    metadata = {
        "job_id": job_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_filename": path.name,
        "input_path": str(input_path),
        "label_filename": label_source.name if label_source is not None else None,
        "label_path": str(copied_label_path) if copied_label_path is not None else None,
        "input_shape": list(EXPECTED_SHAPE),
        "spacing_mm": list(spacing),
        "segmentation_mode": mode,
        "segmentation_mode_label": MODEL_CONFIGS[mode]["label"],
        "num_output_classes": MODEL_CONFIGS[mode]["num_classes"],
        "checkpoint_path": str(MODEL_CONFIGS[mode]["checkpoint"]),
        "status": "queued",
        "threshold": THRESHOLD,
    }
    save_job_metadata(job_dir, metadata)
    submit_background_job(job_id)
    updated_history, dropdown_update = remember_job(job_id, history)
    status = (
        f"### Segmentation job created\n\n"
        f"- Job ID: `{job_id}`\n"
        f"- Status: `queued`\n\n"
        f"- Segmentation mode: **{MODEL_CONFIGS[mode]['label']}**\n"
        f"- Label: `{'provided' if copied_label_path is not None else 'not provided'}`\n\n"
        "This job is now saved in this browser's recent jobs and is running in the background. "
        "The result panel will refresh automatically when segmentation completes."
    )
    center_x, center_y, center_z = EXPECTED_SHAPE[0] // 2, EXPECTED_SHAPE[1] // 2, EXPECTED_SHAPE[2] // 2
    return (
        status,
        empty_viewer_html(),
        None,
        job_id,
        updated_history,
        dropdown_update,
    )


def process_job(job_id: str):
    cleaned_job_id = validate_job_id(job_id)
    job_dir = get_job_dir(cleaned_job_id)
    job_dir.mkdir(parents=True, exist_ok=True)
    metadata = load_job_metadata(job_dir)
    if metadata.get("status") == "completed":
        return load_job(cleaned_job_id)
    mode = validate_segmentation_mode(str(metadata.get("segmentation_mode", "binary")))
    path = resolve_saved_input(metadata)
    nii = load_nifti_strict(path)
    metadata.update(
        {
            "job_id": cleaned_job_id,
            "input_shape": list(EXPECTED_SHAPE),
            "status": "running",
            "started_at_utc": datetime.now(timezone.utc).isoformat(),
            "segmentation_mode": mode,
            "segmentation_mode_label": MODEL_CONFIGS[mode]["label"],
            "num_output_classes": MODEL_CONFIGS[mode]["num_classes"],
            "checkpoint_path": str(MODEL_CONFIGS[mode]["checkpoint"]),
            "threshold": THRESHOLD,
        }
    )
    save_job_metadata(job_dir, metadata)
    load_model(mode)
    preprocessed_slices = preprocess_volume(path)
    pred_volume, prob_volume, elapsed = segment_on_gpu(preprocessed_slices, mode)

    tumour_voxels = int((pred_volume > 0).sum())
    brain_voxels = count_brain_voxels(path)
    spacing = tuple(float(v) for v in nii.header.get_zooms()[:3])
    voxel_volume_mm3 = float(np.prod(spacing))
    tumour_volume_cm3 = tumour_voxels * voxel_volume_mm3 / 1000.0
    brain_volume_cm3 = brain_voxels * voxel_volume_mm3 / 1000.0
    tumour_brain_percent = 100.0 * tumour_volume_cm3 / brain_volume_cm3 if brain_volume_cm3 > 0 else 0.0
    label_metrics = {}
    label_path = resolve_saved_label(metadata)
    if label_path is not None:
        label_volume = load_label_volume(label_path)
        label_metrics = compute_label_metrics(pred_volume, prob_volume, label_volume, voxel_volume_mm3)

    prediction_path = save_prediction(pred_volume, nii, job_dir)
    probability_path = save_probability(prob_volume, job_dir)
    overlay = make_overlay(path, pred_volume)
    overlay_path = save_overlay_image(overlay, job_dir)
    metadata.update(
        {
            "status": "completed",
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "spacing_mm": list(spacing),
            "tumour_voxels": tumour_voxels,
            "brain_voxels": brain_voxels,
            "tumour_volume_cm3": tumour_volume_cm3,
            "brain_volume_cm3": brain_volume_cm3,
            "tumour_brain_percent": tumour_brain_percent,
            "gpu_inference_seconds": elapsed,
            "prediction_path": str(prediction_path),
            "probability_path": str(probability_path),
            "overlay_path": str(overlay_path),
            **label_metrics,
        }
    )
    save_job_metadata(job_dir, metadata)
    return metrics_markdown(metadata), overlay, str(prediction_path), cleaned_job_id


def process_job_background(job_id: str) -> None:
    try:
        process_job(job_id)
    except Exception as exc:
        update_job_metadata(
            job_id,
            {
                "status": "failed",
                "failed_at_utc": datetime.now(timezone.utc).isoformat(),
                "error": str(exc),
            },
        )
        print(f"Background job {job_id} failed: {exc}", flush=True)


def submit_background_job(job_id: str) -> None:
    cleaned_job_id = validate_job_id(job_id)
    with _JOB_LOCK:
        future = _JOB_FUTURES.get(cleaned_job_id)
        if future is not None and not future.done():
            return
        _JOB_FUTURES[cleaned_job_id] = _JOB_EXECUTOR.submit(process_job_background, cleaned_job_id)


def load_job(job_id: str):
    if not has_valid_job_id(job_id):
        return (
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
        )
    cleaned = validate_job_id(job_id)
    job_dir = get_job_dir(cleaned)
    metadata_path = job_dir / "metadata.json"
    prediction_path = job_dir / "prediction.nii.gz"
    overlay_path = job_dir / "overlay.png"
    if not metadata_path.exists():
        raise gr.Error("Could not find a job with that ID.")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("status") != "completed":
        status = metadata.get("status", "unknown")
        error = metadata.get("error")
        error_text = f"\n\nError: `{error}`" if error else ""
        message = (
            f"### Segmentation job found\n\n"
            f"- Job ID: `{cleaned}`\n"
            f"- Status: `{status}`\n\n"
            "The completed mask is not available yet. Use **Refresh selected job status** "
            "to check again in a moment."
            f"{error_text}"
        )
        center_x, center_y, center_z = EXPECTED_SHAPE[0] // 2, EXPECTED_SHAPE[1] // 2, EXPECTED_SHAPE[2] // 2
        return (
            message,
            empty_viewer_html(),
            None,
            cleaned,
        )
    missing = [str(path) for path in (prediction_path, overlay_path) if not path.exists()]
    if missing:
        raise gr.Error("The job metadata exists, but completed result files are missing.")
    input_path = resolve_saved_input(metadata)
    if metadata.get("brain_volume_cm3") is None or metadata.get("tumour_brain_percent") is None:
        brain_voxels = count_brain_voxels(input_path)
        spacing = metadata.get("spacing_mm") or list(load_nifti_strict(input_path).header.get_zooms()[:3])
        voxel_volume_mm3 = float(np.prod([float(v) for v in spacing]))
        brain_volume_cm3 = brain_voxels * voxel_volume_mm3 / 1000.0
        tumour_volume_cm3 = float(metadata.get("tumour_volume_cm3", 0.0))
        metadata.update(
            {
                "brain_voxels": brain_voxels,
                "brain_volume_cm3": brain_volume_cm3,
                "tumour_brain_percent": 100.0 * tumour_volume_cm3 / brain_volume_cm3 if brain_volume_cm3 > 0 else 0.0,
            }
        )
        save_job_metadata(job_dir, metadata)
    pred = load_prediction_volume(prediction_path)
    center_x, center_y, center_z = tumour_center(pred)
    label_path = resolve_saved_label(metadata)
    mode = validate_segmentation_mode(str(metadata.get("segmentation_mode", "binary")))
    viewer = make_local_viewer_html(
        input_path,
        prediction_path,
        center_x,
        center_y,
        center_z,
        label_path=label_path,
        segmentation_mode=mode,
    )
    return (
        metrics_markdown(metadata),
        viewer,
        str(prediction_path),
        cleaned,
    )


def attach_label_to_job(job_id: str, label_file):
    if not has_valid_job_id(job_id):
        raise gr.Error("Select or load a completed job before attaching a label.")
    label_source = validate_label_path(label_file)
    if label_source is None:
        raise gr.Error("Please upload a label NIfTI before attaching it to a job.")
    cleaned = validate_job_id(job_id)
    job_dir = get_job_dir(cleaned)
    metadata = load_job_metadata(job_dir)
    if metadata.get("status") != "completed":
        raise gr.Error("A label can be attached only after segmentation completes.")
    input_path = resolve_saved_input(metadata)
    prediction_path = job_dir / "prediction.nii.gz"
    probability_path = Path(str(metadata.get("probability_path") or (job_dir / "probability.npy")))
    if not prediction_path.exists():
        raise gr.Error("Prediction mask is missing for this job.")
    assert_label_matches_input(input_path, label_source)
    copied_label_path = saved_label_path(job_dir, label_source)
    shutil.copyfile(label_source, copied_label_path)
    pred_volume = load_prediction_volume(prediction_path)
    prob_volume = load_probability_volume(probability_path)
    label_volume = load_label_volume(copied_label_path)
    spacing = metadata.get("spacing_mm") or list(load_nifti_strict(input_path).header.get_zooms()[:3])
    voxel_volume_mm3 = float(np.prod([float(v) for v in spacing]))
    label_metrics = compute_label_metrics(pred_volume, prob_volume, label_volume, voxel_volume_mm3)
    metadata.update(
        {
            "label_filename": label_source.name,
            "label_path": str(copied_label_path),
            "label_attached_at_utc": datetime.now(timezone.utc).isoformat(),
            **label_metrics,
        }
    )
    save_job_metadata(job_dir, metadata)
    center_x, center_y, center_z = tumour_center(pred_volume)
    mode = validate_segmentation_mode(str(metadata.get("segmentation_mode", "binary")))
    viewer = make_local_viewer_html(
        input_path,
        prediction_path,
        center_x,
        center_y,
        center_z,
        label_path=copied_label_path,
        segmentation_mode=mode,
    )
    return metrics_markdown(metadata), viewer, str(prediction_path), cleaned


def wait_for_job_result(job_id: str, timeout_seconds: int = 300):
    if not has_valid_job_id(job_id):
        return load_job(job_id)
    cleaned = validate_job_id(job_id)
    start = time.perf_counter()
    while time.perf_counter() - start < timeout_seconds:
        metadata = load_job_metadata(get_job_dir(cleaned))
        status = metadata.get("status")
        if status in {"completed", "failed"}:
            return load_job(cleaned)
        time.sleep(2.0)
    return load_job(cleaned)


def poll_job(job_id: str):
    if not has_valid_job_id(job_id):
        return gr.update(), gr.update()
    cleaned = validate_job_id(job_id)
    metadata = load_job_metadata(get_job_dir(cleaned))
    status = metadata.get("status")
    if status == "completed":
        return (
            f"### Segmentation job found\n\n"
            f"- Job ID: `{cleaned}`\n"
            f"- Status: `completed`\n\n"
            "Loading visualization...",
            cleaned,
        )
    if status in {"queued", "running", "failed"}:
        error = metadata.get("error")
        error_text = f"\n\nError: `{error}`" if error else ""
        message = (
            f"### Segmentation job found\n\n"
            f"- Job ID: `{cleaned}`\n"
            f"- Status: `{status}`\n\n"
            "This panel refreshes automatically while the job is active."
            f"{error_text}"
        )
        return message, cleaned
    return gr.update(), gr.update()


def normalize_job_history(history) -> list[str]:
    if not isinstance(history, list):
        return []
    normalized = []
    seen = set()
    for item in history:
        if not isinstance(item, str):
            continue
        cleaned = item.strip()
        if not JOB_ID_PATTERN.fullmatch(cleaned) or cleaned in seen:
            continue
        normalized.append(cleaned)
        seen.add(cleaned)
    return normalized[:20]


def remember_job(job_id: str, history) -> tuple[list[str], gr.Dropdown]:
    cleaned = validate_job_id(job_id)
    updated = [cleaned] + [item for item in normalize_job_history(history) if item != cleaned]
    updated = updated[:20]
    return updated, gr.update(choices=updated, value=cleaned)


def refresh_recent_jobs(history) -> gr.Dropdown:
    normalized = normalize_job_history(history)
    return gr.update(choices=normalized, value=None)


for mode_name, mode_config in MODEL_CONFIGS.items():
    checkpoint_path = Path(mode_config["checkpoint"])
    if checkpoint_path.exists():
        status = f"available locally: {checkpoint_path}"
    elif mode_config.get("repo_id") and mode_config.get("filename"):
        status = f"will be downloaded from {mode_config['repo_id']}/{mode_config['filename']}"
    else:
        status = f"missing: {checkpoint_path}"
    print(f"{mode_config['label']} checkpoint {status}", flush=True)
if BINARY_CHECKPOINT_PATH.exists():
    try:
        load_binary_model()
        print(f"Binary model loaded from {BINARY_CHECKPOINT_PATH} on {_MODEL_DEVICES.get('binary')}.", flush=True)
    except Exception as exc:
        print(f"Startup binary model load failed: {exc}", flush=True)


with gr.Blocks(title="Brain Tumour Segmentation MVP", delete_cache=(3600, 3600)) as demo:
    job_history = gr.BrowserState(
        default_value=[],
        storage_key="brain_tumour_segmentation_job_history_v1",
        secret="brain-tumour-segmentation-history-v1",
    )
    gr.Markdown(
        """
        # Brain Tumour Segmentation MVP

        Upload a single `.nii` or `.nii.gz` file with exact loaded shape `(240, 240, 155, 4)`.
        Optionally upload a binary/label NIfTI with shape `(240, 240, 155)` to compute whole-tumour 3D Dice and 3D binary cross entropy.
        Files with any other shape are rejected. This is a research demonstration, not a clinical tool.
        """
    )
    with gr.Row():
        upload = gr.File(label="Four-channel MRI NIfTI", type="filepath")
        label_upload = gr.File(label="Optional tumour label NIfTI", type="filepath")
    segmentation_mode = gr.Radio(
        choices=[
            ("Binary tumour", "binary"),
            ("4-label tumour subclasses", "4label"),
        ],
        value="binary",
        label="Segmentation mode",
    )
    run_button = gr.Button("Run segmentation", variant="primary")
    attach_label_button = gr.Button("Attach label to current completed job")
    job_id_out = gr.Textbox(label="Job ID", interactive=False)
    metrics_out = gr.Markdown(label="Tumour volume")
    viewer_out = gr.HTML(label="Interactive 3D/2D viewer", value=empty_viewer_html())
    download_out = gr.File(label="Download predicted binary mask (.nii.gz)")

    gr.Markdown("### Reload a completed result")
    recent_jobs = gr.Dropdown(label="Recent jobs in this browser", choices=[], interactive=True)
    with gr.Row():
        job_id_in = gr.Textbox(label="Previous Job ID", placeholder="Paste a completed job_id")
        load_button = gr.Button("Load previous job")

    run_button.click(
        fn=start_job,
        inputs=[upload, label_upload, segmentation_mode, job_history],
        outputs=[
            metrics_out,
            viewer_out,
            download_out,
            job_id_out,
            job_history,
            recent_jobs,
        ],
    ).then(
        fn=wait_for_job_result,
        inputs=[job_id_out],
        outputs=[
            metrics_out,
            viewer_out,
            download_out,
            job_id_out,
        ],
    )

    refresh_button = gr.Button("Refresh selected job status", elem_id="refresh-job-status-button")
    recent_jobs.change(
        fn=load_job,
        inputs=[recent_jobs],
        outputs=[
            metrics_out,
            viewer_out,
            download_out,
            job_id_out,
        ],
    )
    refresh_button.click(
        fn=load_job,
        inputs=[job_id_out],
        outputs=[
            metrics_out,
            viewer_out,
            download_out,
            job_id_out,
        ],
    )
    load_button.click(
        fn=load_job,
        inputs=[job_id_in],
        outputs=[
            metrics_out,
            viewer_out,
            download_out,
            job_id_out,
        ],
    )
    attach_label_button.click(
        fn=attach_label_to_job,
        inputs=[job_id_out, label_upload],
        outputs=[
            metrics_out,
            viewer_out,
            download_out,
            job_id_out,
        ],
    )
    demo.load(
        fn=refresh_recent_jobs,
        inputs=[job_history],
        outputs=[recent_jobs],
    )


if __name__ == "__main__":
    demo.launch()
