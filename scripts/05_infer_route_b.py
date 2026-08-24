from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import sys
import time
from contextlib import nullcontext
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / "outputs/.matplotlib"))
sys.path.insert(0, str(PROJECT_ROOT))

import nibabel as nib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from src.pretrained_hybrid_vit import PretrainedHybridViTSegmenter


def normalize_mri_slice(image: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    image = image.copy()
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


def optional_case_id(value: str | None) -> str | None:
    if value is None:
        return None
    value = str(value).strip()
    if value.lower() in {"", "none", "null"}:
        return None
    return value


def resolve_nifti_path(directory: Path, case_id: str) -> Path:
    candidates = [
        directory / f"{case_id}.nii.gz",
        directory / f"{case_id}.nii",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"NIfTI file not found for {case_id}. Tried: {candidates}")


def resolve_case_paths(row, data_root: Path) -> tuple[Path, Path | None]:
    case_id = str(row.case_id)
    image_path = Path(str(getattr(row, "image_path", "")))
    label_path = Path(str(getattr(row, "label_path", "")))
    if not image_path.exists():
        image_path = resolve_nifti_path(data_root / "imagesTr", case_id)
    if not label_path.exists():
        try:
            label_path = resolve_nifti_path(data_root / "labelsTr", case_id)
        except FileNotFoundError:
            label_path = None
    return image_path, label_path


def load_nifti(path: Path):
    try:
        return nib.load(str(path), mmap=True, keep_file_open=True)
    except TypeError:
        return nib.load(str(path), mmap=True)


class CaseSliceDataset(Dataset):
    def __init__(self, *, case_id: str, image_path: Path):
        self.case_id = case_id
        self.image_path = image_path
        self._image_nii = None
        image_nii = load_nifti(image_path)
        self.spatial_shape = tuple(int(v) for v in image_nii.shape[:3])
        self.modalities = int(image_nii.shape[3]) if len(image_nii.shape) == 4 else 1
        if self.spatial_shape != (240, 240, 155):
            raise RuntimeError(f"{case_id}: expected spatial shape (240, 240, 155), got {self.spatial_shape}.")
        if self.modalities != 4:
            raise RuntimeError(f"{case_id}: expected 4 MRI modalities, got {self.modalities}.")

    def __len__(self) -> int:
        return self.spatial_shape[2]

    def _image(self):
        if self._image_nii is None:
            self._image_nii = load_nifti(self.image_path)
        return self._image_nii

    def __getitem__(self, index: int) -> dict[str, object]:
        z = int(index)
        image = np.asarray(self._image().dataobj[:, :, z, :], dtype=np.float32)
        image = normalize_mri_slice(image)
        image_tensor = torch.from_numpy(image.transpose(2, 0, 1).copy())

        return {
            "image": image_tensor,
            "slice_index": z,
        }


class SeedWorker:
    def __init__(self, seed: int):
        self.seed = seed

    def __call__(self, worker_id: int) -> None:
        worker_seed = self.seed + worker_id
        random.seed(worker_seed)
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)


def parse_gpu_ids(value: str) -> list[int]:
    gpu_ids = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not gpu_ids:
        raise ValueError("At least one GPU id must be provided.")
    return gpu_ids


def configure_device_and_parallelism(*, data_parallel: bool, gpu_ids: list[int]) -> tuple[torch.device, dict[str, object]]:
    if data_parallel:
        if not torch.cuda.is_available():
            raise RuntimeError("DataParallel requested but CUDA is unavailable.")
        device_count = torch.cuda.device_count()
        if len(gpu_ids) < 2:
            raise RuntimeError("DataParallel requested but fewer than two GPU ids were provided.")
        invalid_ids = [gpu_id for gpu_id in gpu_ids if gpu_id < 0 or gpu_id >= device_count]
        if invalid_ids:
            raise RuntimeError(f"Invalid GPU ids {invalid_ids}; CUDA device count is {device_count}.")
        return torch.device(f"cuda:{gpu_ids[0]}"), {
            "mode": "DataParallel",
            "gpu_ids": gpu_ids,
            "gpu_count_used": len(gpu_ids),
        }
    if torch.cuda.is_available():
        return torch.device(f"cuda:{gpu_ids[0]}"), {
            "mode": "single_cuda",
            "gpu_ids": [gpu_ids[0]],
            "gpu_count_used": 1,
        }
    return torch.device("cpu"), {
        "mode": "cpu",
        "gpu_ids": [],
        "gpu_count_used": 0,
    }


def maybe_wrap_data_parallel(model: torch.nn.Module, *, data_parallel: bool, gpu_ids: list[int]) -> torch.nn.Module:
    if not data_parallel:
        return model
    return torch.nn.DataParallel(model, device_ids=gpu_ids, output_device=gpu_ids[0])


def autocast_context(use_amp: bool):
    if not use_amp:
        return nullcontext()
    return torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True)


def torch_load_trusted(path: Path) -> dict[str, object]:
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Expected checkpoint dict, got {type(checkpoint)!r}")
    if "model_state_dict" not in checkpoint:
        raise KeyError("Checkpoint is missing model_state_dict.")
    return checkpoint


def build_model_from_checkpoint(checkpoint_path: Path) -> tuple[PretrainedHybridViTSegmenter, dict[str, object]]:
    checkpoint = torch_load_trusted(checkpoint_path)
    model = PretrainedHybridViTSegmenter(weights_path=None)
    missing, unexpected = model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    if missing or unexpected:
        raise RuntimeError(f"Strict checkpoint load mismatch: missing={missing}, unexpected={unexpected}")
    model.eval()
    metadata = checkpoint.get("metadata", {})
    if isinstance(metadata, dict):
        train_summary = metadata.get("train_summary", {})
        val_summary = metadata.get("val_summary", {})
        if train_summary and int(train_summary.get("unique_patients", 0)) != 339:
            print(f"[checkpoint warning] unexpected train patient count in metadata: {train_summary}", flush=True)
        if val_summary and int(val_summary.get("unique_patients", 0)) != 72:
            print(f"[checkpoint warning] unexpected val patient count in metadata: {val_summary}", flush=True)
    return model, checkpoint


def dice_and_rve(pred: np.ndarray, gt: np.ndarray, *, voxel_volume_mm3: float, eps: float = 1e-6) -> dict[str, float]:
    pred_bool = pred.astype(bool)
    gt_bool = gt.astype(bool)
    intersection = float(np.logical_and(pred_bool, gt_bool).sum())
    pred_sum = float(pred_bool.sum())
    gt_sum = float(gt_bool.sum())
    dice = (2.0 * intersection + eps) / (pred_sum + gt_sum + eps)
    pred_volume_mm3 = pred_sum * voxel_volume_mm3
    gt_volume_mm3 = gt_sum * voxel_volume_mm3
    rve = abs(pred_volume_mm3 - gt_volume_mm3) / gt_volume_mm3 if gt_volume_mm3 > 0 else float("nan")
    return {
        "dice3d": dice,
        "gt_volume_mm3": gt_volume_mm3,
        "pred_volume_mm3": pred_volume_mm3,
        "gt_volume_cm3": gt_volume_mm3 / 1000.0,
        "pred_volume_cm3": pred_volume_mm3 / 1000.0,
        "rve": rve,
    }


def save_nifti(volume: np.ndarray, *, reference_nii, path: Path, dtype) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = reference_nii.header.copy()
    header.set_data_dtype(dtype)
    nib.save(nib.Nifti1Image(volume.astype(dtype), reference_nii.affine, header), str(path))


def infer_case(
    *,
    model: torch.nn.Module,
    device: torch.device,
    use_amp: bool,
    case_id: str,
    image_path: Path,
    label_path: Path | None,
    batch_size: int,
    num_workers: int,
    threshold: float,
    output_dir: Path,
    save_nifti_masks: bool,
    save_probability_maps: bool,
    seed: int,
) -> dict[str, object]:
    start = time.perf_counter()
    image_nii = load_nifti(image_path)
    spatial_shape = tuple(int(v) for v in image_nii.shape[:3])
    spacing = tuple(float(v) for v in image_nii.header.get_zooms()[:3])
    voxel_volume_mm3 = float(spacing[0] * spacing[1] * spacing[2])

    dataset = CaseSliceDataset(case_id=case_id, image_path=image_path)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
        worker_init_fn=SeedWorker(seed),
        persistent_workers=num_workers > 0,
    )

    pred_volume = np.zeros(spatial_shape, dtype=np.uint8)
    probability_volume = np.zeros(spatial_shape, dtype=np.float32) if save_probability_maps else None

    with torch.no_grad():
        for batch in loader:
            image = batch["image"].to(device=device, dtype=torch.float32)
            if image.shape[1:] != (4, 240, 240):
                raise RuntimeError(f"{case_id}: unexpected batch image shape {tuple(image.shape)}")
            with autocast_context(use_amp):
                logits = model(image)
            probs = torch.sigmoid(logits.detach().float()).cpu().numpy()[:, 0]
            preds = (probs >= threshold).astype(np.uint8)
            slice_indices = [int(idx) for idx in batch["slice_index"]]
            for batch_idx, z in enumerate(slice_indices):
                pred_volume[:, :, z] = preds[batch_idx]
                if probability_volume is not None:
                    probability_volume[:, :, z] = probs[batch_idx]

    metrics: dict[str, float]
    if label_path is not None and label_path.exists():
        label_nii = load_nifti(label_path)
        gt_volume = np.asarray(label_nii.dataobj) > 0
        metrics = dice_and_rve(pred_volume, gt_volume.astype(np.uint8), voxel_volume_mm3=voxel_volume_mm3)
    else:
        tumour_voxels = float(pred_volume.sum())
        pred_volume_mm3 = tumour_voxels * voxel_volume_mm3
        metrics = {
            "dice3d": float("nan"),
            "gt_volume_mm3": float("nan"),
            "pred_volume_mm3": pred_volume_mm3,
            "gt_volume_cm3": float("nan"),
            "pred_volume_cm3": pred_volume_mm3 / 1000.0,
            "rve": float("nan"),
        }

    if save_nifti_masks:
        save_nifti(
            pred_volume,
            reference_nii=image_nii,
            path=output_dir / "nifti_masks" / f"{case_id}_pred_mask.nii.gz",
            dtype=np.uint8,
        )
    if probability_volume is not None:
        save_nifti(
            probability_volume,
            reference_nii=image_nii,
            path=output_dir / "probability_maps" / f"{case_id}_prob.nii.gz",
            dtype=np.float32,
        )

    elapsed = time.perf_counter() - start
    return {
        "case_id": case_id,
        "image_path": str(image_path),
        "label_path": str(label_path) if label_path is not None else "",
        "spacing_x_mm": spacing[0],
        "spacing_y_mm": spacing[1],
        "spacing_z_mm": spacing[2],
        "voxel_volume_mm3": voxel_volume_mm3,
        "pred_voxels": int(pred_volume.sum()),
        "seconds": elapsed,
        **metrics,
    }


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def choose_cases(split_df: pd.DataFrame, *, mode: str, case_id: str | None) -> pd.DataFrame:
    if mode == "single":
        if case_id is None:
            raise ValueError("--case-id is required when --mode single.")
        rows = split_df[split_df["case_id"].astype(str) == case_id].copy()
        if rows.empty:
            raise ValueError(f"case_id={case_id!r} was not found in split manifest.")
        return rows
    if case_id is not None:
        raise ValueError("--case-id should only be provided when --mode single.")
    rows = split_df[split_df["split"] == "test"].copy().sort_values("case_id")
    if len(rows) != 73:
        raise RuntimeError(f"Expected 73 frozen test patients, got {len(rows)}.")
    return rows


def write_case_metrics(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "case_id",
        "dice3d",
        "gt_volume_cm3",
        "pred_volume_cm3",
        "rve",
        "gt_volume_mm3",
        "pred_volume_mm3",
        "pred_voxels",
        "voxel_volume_mm3",
        "spacing_x_mm",
        "spacing_y_mm",
        "spacing_z_mm",
        "seconds",
        "image_path",
        "label_path",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def summarize(rows: list[dict[str, object]]) -> dict[str, object]:
    dice_values = np.array([float(row["dice3d"]) for row in rows], dtype=np.float64)
    rve_values = np.array([float(row["rve"]) for row in rows], dtype=np.float64)
    return {
        "n_patients": len(rows),
        "dice3d_mean": float(np.nanmean(dice_values)) if np.isfinite(dice_values).any() else float("nan"),
        "dice3d_median": float(np.nanmedian(dice_values)) if np.isfinite(dice_values).any() else float("nan"),
        "rve_mean": float(np.nanmean(rve_values)) if np.isfinite(rve_values).any() else float("nan"),
        "rve_median": float(np.nanmedian(rve_values)) if np.isfinite(rve_values).any() else float("nan"),
        "mean_seconds_per_patient": float(np.mean([float(row["seconds"]) for row in rows])),
        "total_seconds": float(np.sum([float(row["seconds"]) for row in rows])),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Route B CNN-ViT tumour-segmentation inference.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, default=PROJECT_ROOT / "outputs/splits/frozen_patient_split_seed42.csv")
    parser.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "data/processed/Task01_BrainTumour")
    parser.add_argument("--mode", choices=["single", "test"], default="test")
    parser.add_argument("--case-id", type=optional_case_id, default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--data-parallel", action="store_true")
    parser.add_argument("--gpu-ids", type=str, default="0,1")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "run_outputs/route_b_test")
    parser.add_argument("--save-nifti", action="store_true")
    parser.add_argument("--save-probability-maps", action="store_true")
    parser.add_argument("--save-case-metrics", action="store_true")
    parser.add_argument("--seed", type=int, default=731)
    args = parser.parse_args()

    if not 0.0 <= args.threshold <= 1.0:
        raise ValueError(f"threshold must be in [0, 1], got {args.threshold}.")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    gpu_ids = parse_gpu_ids(args.gpu_ids)
    device, parallelism = configure_device_and_parallelism(data_parallel=args.data_parallel, gpu_ids=gpu_ids)
    use_amp = bool(args.amp and device.type == "cuda")

    split_df = pd.read_csv(args.split_manifest)
    case_rows = choose_cases(split_df, mode=args.mode, case_id=args.case_id)

    model, checkpoint = build_model_from_checkpoint(args.checkpoint)
    model = model.to(device)
    model = maybe_wrap_data_parallel(model, data_parallel=args.data_parallel, gpu_ids=gpu_ids)
    model.eval()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_metadata = {
        "mode": args.mode,
        "case_id": args.case_id,
        "n_cases": int(len(case_rows)),
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": file_sha256(args.checkpoint),
        "split_manifest": str(args.split_manifest),
        "split_manifest_sha256": file_sha256(args.split_manifest),
        "data_root": str(args.data_root),
        "threshold": args.threshold,
        "device": str(device),
        "parallelism": parallelism,
        "amp": {"enabled": use_amp, "dtype": "float16" if use_amp else None},
        "save_nifti_masks": args.save_nifti,
        "save_probability_maps": args.save_probability_maps,
        "checkpoint_metadata": checkpoint.get("metadata", {}),
        "args": vars(args) | {
            "checkpoint": str(args.checkpoint),
            "split_manifest": str(args.split_manifest),
            "data_root": str(args.data_root),
            "output_dir": str(args.output_dir),
        },
    }
    (args.output_dir / "inference_metadata.json").write_text(
        json.dumps(run_metadata, indent=2, default=str),
        encoding="utf-8",
    )

    rows: list[dict[str, object]] = []
    total_start = time.perf_counter()
    for idx, row in enumerate(case_rows.itertuples(index=False), start=1):
        case_id = str(row.case_id)
        image_path, label_path = resolve_case_paths(row, args.data_root)
        print(f"[{idx}/{len(case_rows)}] infer {case_id}", flush=True)
        case_metrics = infer_case(
            model=model,
            device=device,
            use_amp=use_amp,
            case_id=case_id,
            image_path=image_path,
            label_path=label_path,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            threshold=args.threshold,
            output_dir=args.output_dir,
            save_nifti_masks=args.save_nifti,
            save_probability_maps=args.save_probability_maps,
            seed=args.seed,
        )
        rows.append(case_metrics)
        print(
            f"{case_id}: dice3d={float(case_metrics['dice3d']):.4f} "
            f"rve={float(case_metrics['rve']):.4f} "
            f"pred_cm3={float(case_metrics['pred_volume_cm3']):.2f} "
            f"time={float(case_metrics['seconds']):.1f}s",
            flush=True,
        )

    if args.save_case_metrics:
        write_case_metrics(args.output_dir / "test_case_metrics.csv", rows)
    summary = summarize(rows)
    summary["wall_seconds"] = time.perf_counter() - total_start
    (args.output_dir / "test_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(json.dumps(summary, indent=2, default=str), flush=True)


if __name__ == "__main__":
    main()
