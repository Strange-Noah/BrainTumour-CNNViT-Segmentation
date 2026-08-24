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
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / "outputs/.matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, Sampler, WeightedRandomSampler

sys.path.insert(0, str(PROJECT_ROOT))

from src.losses import MulticlassForegroundDiceCELoss  # noqa: E402
from src.pretrained_hybrid_vit import PretrainedHybridViTSegmenter, count_parameters  # noqa: E402


def optional_positive_int(value: str | None) -> int | None:
    if value is None:
        return None
    normalized = str(value).strip().lower()
    if normalized in {"", "none", "null", "-1", "0"}:
        return None
    parsed = int(normalized)
    if parsed <= 0:
        return None
    return parsed


@dataclass(frozen=True)
class SliceSample:
    case_id: str
    image_path: Path
    label_path: Path
    slice_index: int
    positive: bool


class BrainTumourSliceDataset(Dataset):
    def __init__(self, samples: list[SliceSample]):
        self.samples = samples
        self._nifti_cache = {}

    def __len__(self) -> int:
        return len(self.samples)

    def _load_nifti(self, path: Path):
        path = Path(path)
        cached = self._nifti_cache.get(path)
        if cached is None:
            try:
                cached = nib.load(str(path), mmap=True, keep_file_open=True)
            except TypeError:
                cached = nib.load(str(path), mmap=True)
            self._nifti_cache[path] = cached
        return cached

    def __getitem__(self, index: int) -> dict[str, object]:
        sample = self.samples[index]
        image_nii = self._load_nifti(sample.image_path)
        label_nii = self._load_nifti(sample.label_path)

        image = np.asarray(image_nii.dataobj[:, :, sample.slice_index, :], dtype=np.float32)
        label = np.asarray(label_nii.dataobj[:, :, sample.slice_index], dtype=np.int64)
        unique = set(np.unique(label).astype(int).tolist())
        if not unique.issubset({0, 1, 2, 3}):
            raise RuntimeError(f"Unexpected labels for {sample.case_id} slice {sample.slice_index}: {sorted(unique)}")

        image = normalize_mri_slice(image)
        image = torch.from_numpy(image.transpose(2, 0, 1).copy())
        label = torch.from_numpy(label.copy()).long()

        return {
            "image": image,
            "label": label,
            "case_id": sample.case_id,
            "slice_index": sample.slice_index,
            "positive": sample.positive,
        }


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


def build_slice_samples(
    split_df: pd.DataFrame,
    split: str,
    *,
    data_root: Path,
    max_patients: int | None = None,
    max_slices_per_patient: int | None = None,
    seed: int = 731,
) -> list[SliceSample]:
    rows = split_df[split_df["split"] == split].copy()
    rows = rows.sort_values("case_id")
    if max_patients is not None:
        rows = rows.head(max_patients)

    rng = random.Random(seed)
    samples: list[SliceSample] = []
    for row in rows.itertuples(index=False):
        image_path, label_path = resolve_case_paths(row, data_root)
        label = np.asarray(nib.load(str(label_path)).dataobj)
        positive_by_slice = np.any(label > 0, axis=(0, 1))
        slice_indices = list(range(label.shape[2]))
        if max_slices_per_patient is not None and max_slices_per_patient < len(slice_indices):
            positives = [idx for idx in slice_indices if positive_by_slice[idx]]
            negatives = [idx for idx in slice_indices if not positive_by_slice[idx]]
            rng.shuffle(positives)
            rng.shuffle(negatives)
            half = max_slices_per_patient // 2
            chosen = positives[:half] + negatives[: max_slices_per_patient - half]
            if len(chosen) < max_slices_per_patient:
                remaining = [idx for idx in slice_indices if idx not in set(chosen)]
                rng.shuffle(remaining)
                chosen += remaining[: max_slices_per_patient - len(chosen)]
            slice_indices = sorted(chosen)

        for slice_index in slice_indices:
            samples.append(
                SliceSample(
                    case_id=str(row.case_id),
                    image_path=image_path,
                    label_path=label_path,
                    slice_index=int(slice_index),
                    positive=bool(positive_by_slice[slice_index]),
                )
            )
    return samples


def resolve_case_paths(row, data_root: Path) -> tuple[Path, Path]:
    image_path = Path(str(row.image_path))
    label_path = Path(str(row.label_path))
    if image_path.exists() and label_path.exists():
        return image_path, label_path

    case_id = str(row.case_id)
    fallback_image = resolve_nifti_path(data_root / "imagesTr", case_id)
    fallback_label = resolve_nifti_path(data_root / "labelsTr", case_id)
    if fallback_image is not None and fallback_label is not None:
        return fallback_image, fallback_label

    raise FileNotFoundError(
        "Could not resolve image/label paths for "
        f"{case_id}. Manifest paths were {image_path} and {label_path}; "
        f"fallback directory was {data_root}."
    )


def resolve_nifti_path(directory: Path, case_id: str) -> Path | None:
    candidates = [
        directory / f"{case_id}.nii.gz",
        directory / f"{case_id}.nii",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


class PatientLocalitySampler(Sampler[int]):
    def __init__(self, samples: list[SliceSample], *, seed: int):
        self.samples = samples
        self.seed = seed
        self.epoch = 0
        self.patient_to_indices: dict[str, list[int]] = {}
        for index, sample in enumerate(samples):
            self.patient_to_indices.setdefault(sample.case_id, []).append(index)

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        self.epoch += 1
        patient_ids = list(self.patient_to_indices)
        rng.shuffle(patient_ids)
        for case_id in patient_ids:
            indices = list(self.patient_to_indices[case_id])
            rng.shuffle(indices)
            yield from indices

    def __len__(self) -> int:
        return len(self.samples)


def make_train_loader(
    samples: list[SliceSample],
    *,
    batch_size: int,
    num_workers: int,
    seed: int,
    balanced: bool,
    sampling_mode: str,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    dataset = BrainTumourSliceDataset(samples)
    worker_init_fn = make_worker_init_fn(seed)
    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": False,
        "worker_init_fn": worker_init_fn,
        "persistent_workers": num_workers > 0,
    }
    if sampling_mode not in {"patient_local", "random"}:
        raise ValueError(f"Unknown train sampling mode: {sampling_mode}")
    if not balanced:
        if sampling_mode == "patient_local":
            sampler = PatientLocalitySampler(samples, seed=seed)
            return DataLoader(
                dataset,
                sampler=sampler,
                **loader_kwargs,
            )
        return DataLoader(
            dataset,
            shuffle=True,
            generator=generator,
            **loader_kwargs,
        )

    positives = sum(sample.positive for sample in samples)
    negatives = len(samples) - positives
    weights = []
    for sample in samples:
        if sample.positive:
            weights.append(0.5 / max(positives, 1))
        else:
            weights.append(0.5 / max(negatives, 1))
    sampler = WeightedRandomSampler(
        weights=torch.DoubleTensor(weights),
        num_samples=len(samples),
        replacement=True,
        generator=generator,
    )
    return DataLoader(
        dataset,
        sampler=sampler,
        **loader_kwargs,
    )


class SeedWorker:
    def __init__(self, seed: int):
        self.seed = seed

    def __call__(self, worker_id: int) -> None:
        worker_seed = self.seed + worker_id
        random.seed(worker_seed)
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)


def make_worker_init_fn(seed: int):
    return SeedWorker(seed)


def sample_summary(samples: list[SliceSample]) -> dict[str, object]:
    patient_ids = {sample.case_id for sample in samples}
    positives = sum(sample.positive for sample in samples)
    negatives = len(samples) - positives
    class_positive_slices = {class_index: 0 for class_index in range(1, 4)}
    class_voxels = {class_index: 0 for class_index in range(1, 4)}
    by_label_path: dict[Path, list[int]] = {}
    for sample in samples:
        by_label_path.setdefault(sample.label_path, []).append(sample.slice_index)
    for label_path, slice_indices in by_label_path.items():
        label_nii = nib.load(str(label_path))
        for slice_index in slice_indices:
            label = np.asarray(label_nii.dataobj[:, :, slice_index], dtype=np.int16)
            for class_index in range(1, 4):
                voxels = int(np.sum(label == class_index))
                class_voxels[class_index] += voxels
                if voxels > 0:
                    class_positive_slices[class_index] += 1
    return {
        "samples": len(samples),
        "unique_patients": len(patient_ids),
        "positive_slices": positives,
        "negative_slices": negatives,
        "positive_fraction": positives / len(samples) if samples else 0.0,
        "class_positive_slices": {str(k): int(v) for k, v in class_positive_slices.items()},
        "class_voxels": {str(k): int(v) for k, v in class_voxels.items()},
    }


def assert_frozen_split_integrity(split_df: pd.DataFrame) -> None:
    split_to_ids = {
        split: set(split_df.loc[split_df["split"] == split, "case_id"].astype(str))
        for split in ["train", "val", "test"]
    }
    missing = [split for split, ids in split_to_ids.items() if not ids]
    if missing:
        raise RuntimeError(f"Frozen split is missing patient ids for: {missing}")
    if not split_to_ids["train"].isdisjoint(split_to_ids["val"]):
        raise RuntimeError("Train/val patient leakage detected.")
    if not split_to_ids["train"].isdisjoint(split_to_ids["test"]):
        raise RuntimeError("Train/test patient leakage detected.")
    if not split_to_ids["val"].isdisjoint(split_to_ids["test"]):
        raise RuntimeError("Val/test patient leakage detected.")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_model_state_checkpoint(path: Path) -> dict[str, object]:
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Expected checkpoint dict at {path}, got {type(checkpoint)!r}.")
    if "model_state_dict" not in checkpoint:
        raise KeyError(f"Checkpoint {path} is missing model_state_dict.")
    return checkpoint


def load_initial_checkpoint(model: torch.nn.Module, checkpoint_path: Path) -> dict[str, object]:
    checkpoint = load_model_state_checkpoint(checkpoint_path)
    missing, unexpected = model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    if missing or unexpected:
        raise RuntimeError(
            "Initial checkpoint strict load mismatch: "
            f"missing={missing}, unexpected={unexpected}"
        )
    return checkpoint


def finite_float_or_default(value, default: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if math.isfinite(parsed) else default


def parse_stage_indices(value: str) -> list[int]:
    value = value.strip()
    if not value:
        return []
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_gpu_ids(value: str) -> list[int]:
    gpu_ids = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not gpu_ids:
        raise ValueError("At least one GPU id must be provided.")
    return gpu_ids


def configure_device_and_parallelism(
    *,
    data_parallel: bool,
    gpu_ids: list[int],
) -> tuple[torch.device, dict[str, object]]:
    if data_parallel:
        if not torch.cuda.is_available():
            raise RuntimeError("DataParallel requested but CUDA is unavailable.")
        device_count = torch.cuda.device_count()
        if len(gpu_ids) < 2:
            raise RuntimeError("DataParallel requested but fewer than two GPU ids were provided.")
        invalid_ids = [gpu_id for gpu_id in gpu_ids if gpu_id < 0 or gpu_id >= device_count]
        if invalid_ids:
            raise RuntimeError(f"Invalid GPU ids {invalid_ids}; CUDA device count is {device_count}.")
        device = torch.device(f"cuda:{gpu_ids[0]}")
        return device, {
            "mode": "DataParallel",
            "gpu_ids": gpu_ids,
            "gpu_count_used": len(gpu_ids),
        }

    if torch.cuda.is_available():
        device = torch.device(f"cuda:{gpu_ids[0]}")
        return device, {
            "mode": "single_cuda",
            "gpu_ids": [gpu_ids[0]],
            "gpu_count_used": 1,
        }

    return torch.device("cpu"), {
        "mode": "cpu",
        "gpu_ids": [],
        "gpu_count_used": 0,
    }


def maybe_wrap_data_parallel(
    model: torch.nn.Module,
    *,
    data_parallel: bool,
    gpu_ids: list[int],
) -> torch.nn.Module:
    if not data_parallel:
        return model
    return torch.nn.DataParallel(
        model,
        device_ids=gpu_ids,
        output_device=gpu_ids[0],
    )


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if isinstance(model, torch.nn.DataParallel) else model


def make_optimizer(
    model: PretrainedHybridViTSegmenter,
    *,
    lr_new: float,
    lr_pretrained: float,
    lr_pos_embed: float,
    weight_decay: float,
) -> torch.optim.Optimizer:
    groups = []
    used: set[int] = set()

    def add_group(name: str, params, lr: float) -> None:
        selected = [param for param in params if param.requires_grad and id(param) not in used]
        if not selected:
            return
        used.update(id(param) for param in selected)
        groups.append({"name": name, "params": selected, "lr": lr})

    add_group("decoder", model.decoder.parameters(), lr_new)
    add_group("stem", model.encoder.patch_embed.backbone.stem.parameters(), lr_new)
    add_group("pos_embed", [model.encoder.pos_embed], lr_pos_embed)
    for stage_idx, stage in enumerate(model.encoder.patch_embed.backbone.stages):
        add_group(f"resnet_stage{stage_idx}", stage.parameters(), lr_pretrained)
    add_group("vit_blocks", model.encoder.blocks.parameters(), lr_pretrained)
    add_group("encoder_norm", model.encoder.norm.parameters(), lr_pretrained)

    trainable_ids = {id(param) for param in model.parameters() if param.requires_grad}
    if used != trainable_ids:
        missing = len(trainable_ids - used)
        extra = len(used - trainable_ids)
        raise RuntimeError(f"Optimizer param groups mismatch: missing={missing}, extra={extra}")
    return torch.optim.AdamW(groups, weight_decay=weight_decay)


def batch_metrics(logits: torch.Tensor, label: torch.Tensor) -> dict[str, float]:
    pred = torch.argmax(logits, dim=1)
    truth = label.long()
    metrics = {}
    foreground_dice_values = []
    for class_index in range(1, 4):
        pred_c = pred == class_index
        truth_c = truth == class_index
        intersection = torch.sum(pred_c & truth_c).float()
        denominator = torch.sum(pred_c).float() + torch.sum(truth_c).float()
        dice = (2.0 * intersection + 1e-6) / (denominator + 1e-6)
        metrics[f"dice_class_{class_index}"] = float(dice.detach().cpu().item())
        metrics[f"pred_fraction_class_{class_index}"] = float(pred_c.float().mean().detach().cpu().item())
        metrics[f"gt_fraction_class_{class_index}"] = float(truth_c.float().mean().detach().cpu().item())
        foreground_dice_values.append(dice)
    pred_fg = pred > 0
    truth_fg = truth > 0
    intersection = torch.sum(pred_fg & truth_fg).float()
    denominator = torch.sum(pred_fg).float() + torch.sum(truth_fg).float()
    whole_dice = (2.0 * intersection + 1e-6) / (denominator + 1e-6)
    metrics["mean_foreground_dice"] = float(torch.stack(foreground_dice_values).mean().detach().cpu().item())
    metrics["whole_tumour_dice"] = float(whole_dice.detach().cpu().item())
    metrics["pred_foreground_fraction"] = float(pred_fg.float().mean().detach().cpu().item())
    metrics["gt_foreground_fraction"] = float(truth_fg.float().mean().detach().cpu().item())
    return metrics


def empty_patient_3d_metrics(status: str) -> dict[str, float | str]:
    metrics: dict[str, float | str] = {
        "patient_3d_metrics_status": status,
        "patient_mean_fg_dice3d": float("nan"),
        "patient_whole_dice3d_mean": float("nan"),
        "patient_whole_dice3d_median": float("nan"),
        "patient_whole_rve_mean": float("nan"),
        "patient_whole_rve_median": float("nan"),
        "patient_3d_count": 0.0,
    }
    for class_index in range(1, 4):
        metrics[f"patient_dice3d_class_{class_index}_mean"] = float("nan")
        metrics[f"patient_rve_class_{class_index}_mean"] = float("nan")
        metrics[f"patient_gt_present_class_{class_index}_count"] = 0.0
    return metrics


def compute_patient_3d_metrics(
    patient_stats: dict[str, dict[str, object]],
    *,
    expected_patient_ids: set[str] | None = None,
    expected_slices: int = 155,
    eps: float = 1e-6,
) -> dict[str, float | str]:
    if not patient_stats:
        return empty_patient_3d_metrics("skipped_no_predictions")

    required_slices = set(range(expected_slices))
    observed_patient_ids = set(patient_stats)
    if expected_patient_ids is not None and observed_patient_ids != expected_patient_ids:
        return empty_patient_3d_metrics("skipped_incomplete_patient_set")

    for stats in patient_stats.values():
        if set(stats["slice_ids"]) != required_slices:
            return empty_patient_3d_metrics("skipped_incomplete_patient_volumes")

    whole_dice_values = []
    whole_rve_values = []
    class_dice_values = {class_index: [] for class_index in range(1, 4)}
    class_rve_values = {class_index: [] for class_index in range(1, 4)}
    for case_id in sorted(patient_stats):
        stats = patient_stats[case_id]
        whole_intersection = float(stats["whole_intersection"])
        whole_pred_sum = float(stats["whole_pred_sum"])
        whole_gt_sum = float(stats["whole_gt_sum"])
        whole_dice = (2.0 * whole_intersection + eps) / (whole_pred_sum + whole_gt_sum + eps)
        whole_rve = abs(whole_pred_sum - whole_gt_sum) / whole_gt_sum if whole_gt_sum > 0 else float("nan")
        whole_dice_values.append(float(whole_dice))
        whole_rve_values.append(float(whole_rve))
        for class_index in range(1, 4):
            intersection = float(stats[f"intersection_{class_index}"])
            pred_sum = float(stats[f"pred_sum_{class_index}"])
            gt_sum = float(stats[f"gt_sum_{class_index}"])
            if gt_sum > 0:
                dice = (2.0 * intersection + eps) / (pred_sum + gt_sum + eps)
                rve = abs(pred_sum - gt_sum) / gt_sum
                class_dice_values[class_index].append(float(dice))
                class_rve_values[class_index].append(float(rve))
            elif pred_sum > 0:
                class_dice_values[class_index].append(0.0)

    whole_dice_array = np.asarray(whole_dice_values, dtype=np.float64)
    whole_rve_array = np.asarray(whole_rve_values, dtype=np.float64)
    class_means = []
    metrics: dict[str, float | str] = {
        "patient_3d_metrics_status": "computed",
        "patient_whole_dice3d_mean": float(np.nanmean(whole_dice_array)),
        "patient_whole_dice3d_median": float(np.nanmedian(whole_dice_array)),
        "patient_whole_rve_mean": float(np.nanmean(whole_rve_array)),
        "patient_whole_rve_median": float(np.nanmedian(whole_rve_array)),
        "patient_3d_count": float(len(whole_dice_values)),
    }
    for class_index in range(1, 4):
        dice_array = np.asarray(class_dice_values[class_index], dtype=np.float64)
        rve_array = np.asarray(class_rve_values[class_index], dtype=np.float64)
        dice_mean = float(np.nanmean(dice_array)) if dice_array.size else float("nan")
        rve_mean = float(np.nanmean(rve_array)) if rve_array.size else float("nan")
        metrics[f"patient_dice3d_class_{class_index}_mean"] = dice_mean
        metrics[f"patient_rve_class_{class_index}_mean"] = rve_mean
        metrics[f"patient_gt_present_class_{class_index}_count"] = float(dice_array.size)
        if math.isfinite(dice_mean):
            class_means.append(dice_mean)
    metrics["patient_mean_fg_dice3d"] = float(np.mean(class_means)) if class_means else float("nan")
    return metrics


def grad_norm(model: torch.nn.Module) -> float:
    total = 0.0
    for param in model.parameters():
        if param.grad is None:
            continue
        value = param.grad.detach().data.norm(2).item()
        total += value * value
    return math.sqrt(total)


def make_grad_scaler(use_amp: bool):
    init_scale = 2**10
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        try:
            return torch.amp.GradScaler("cuda", enabled=use_amp, init_scale=init_scale)
        except TypeError:
            return torch.amp.GradScaler(enabled=use_amp, init_scale=init_scale)
    return torch.cuda.amp.GradScaler(enabled=use_amp, init_scale=init_scale)


def autocast_context(use_amp: bool):
    if not use_amp:
        return nullcontext()
    return torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True)


def run_epoch(
    *,
    model: PretrainedHybridViTSegmenter,
    loader: DataLoader,
    criterion: MulticlassForegroundDiceCELoss,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    phase: str,
    max_steps: int | None,
    scaler=None,
    use_amp: bool = False,
    log_every: int | None = None,
    expected_patient_ids: set[str] | None = None,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    totals: dict[str, float] = {}
    count = 0
    samples_seen = 0
    patient_stats: dict[str, dict[str, object]] = {}
    phase_start = time.perf_counter()
    planned_steps = len(loader) if max_steps is None else min(len(loader), max_steps)

    for step, batch in enumerate(loader, start=1):
        image = batch["image"].to(device=device, dtype=torch.float32)
        label = batch["label"].to(device=device, dtype=torch.long)
        if image.shape[1:] != (4, 240, 240) or label.shape[1:] != (240, 240):
            raise RuntimeError(f"Unexpected image/label shapes: {tuple(image.shape)}, {tuple(label.shape)}")
        if not torch.isfinite(image).all():
            raise RuntimeError("Non-finite values found in input image batch.")
        if not torch.isfinite(label).all():
            raise RuntimeError("Non-finite values found in label batch.")

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            with autocast_context(use_amp):
                logits = model(image)
                if logits.shape[0] != label.shape[0] or logits.shape[1:] != (4, 240, 240):
                    raise RuntimeError(f"Logits/label mismatch: {tuple(logits.shape)} vs {tuple(label.shape)}")
                losses = criterion(logits, label)
                loss = losses["loss"]
            if not torch.isfinite(loss.detach()):
                raise RuntimeError(f"Non-finite loss at {phase} step {step}: {float(loss.detach().cpu().item())}")

            if training:
                if use_amp:
                    if scaler is None:
                        raise RuntimeError("AMP training requested but GradScaler was not provided.")
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    gnorm = grad_norm(model)
                    if not math.isfinite(gnorm):
                        print(
                            f"[AMP warning] non-finite grad norm at {phase} step {step}; "
                            f"scale={float(scaler.get_scale()):.1f}. "
                            "GradScaler will skip/back off this update.",
                            flush=True,
                        )
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    gnorm = grad_norm(model)
                    if not math.isfinite(gnorm):
                        raise RuntimeError(f"Non-finite grad norm: {gnorm}")
                    optimizer.step()
            else:
                gnorm = float("nan")

        logits_for_metrics = logits.detach().float()
        metrics = batch_metrics(logits_for_metrics, label.detach())
        if not training:
            preds = torch.argmax(logits_for_metrics, dim=1)
            gts = label.detach().long()
            case_ids = [str(case_id) for case_id in batch["case_id"]]
            slice_indices = [int(idx) for idx in batch["slice_index"]]
            for batch_idx, case_id in enumerate(case_ids):
                z = slice_indices[batch_idx]
                stats = patient_stats.setdefault(
                    case_id,
                    {
                        "whole_intersection": 0.0,
                        "whole_pred_sum": 0.0,
                        "whole_gt_sum": 0.0,
                        "intersection_1": 0.0,
                        "pred_sum_1": 0.0,
                        "gt_sum_1": 0.0,
                        "intersection_2": 0.0,
                        "pred_sum_2": 0.0,
                        "gt_sum_2": 0.0,
                        "intersection_3": 0.0,
                        "pred_sum_3": 0.0,
                        "gt_sum_3": 0.0,
                        "slice_ids": set(),
                    },
                )
                pred_slice = preds[batch_idx]
                gt_slice = gts[batch_idx]
                pred_whole = pred_slice > 0
                gt_whole = gt_slice > 0
                stats["whole_intersection"] = float(stats["whole_intersection"]) + float(torch.sum(pred_whole & gt_whole).detach().cpu().item())
                stats["whole_pred_sum"] = float(stats["whole_pred_sum"]) + float(torch.sum(pred_whole).detach().cpu().item())
                stats["whole_gt_sum"] = float(stats["whole_gt_sum"]) + float(torch.sum(gt_whole).detach().cpu().item())
                for class_index in range(1, 4):
                    pred_c = pred_slice == class_index
                    gt_c = gt_slice == class_index
                    stats[f"intersection_{class_index}"] = float(stats[f"intersection_{class_index}"]) + float(torch.sum(pred_c & gt_c).detach().cpu().item())
                    stats[f"pred_sum_{class_index}"] = float(stats[f"pred_sum_{class_index}"]) + float(torch.sum(pred_c).detach().cpu().item())
                    stats[f"gt_sum_{class_index}"] = float(stats[f"gt_sum_{class_index}"]) + float(torch.sum(gt_c).detach().cpu().item())
                stats["slice_ids"].add(z)

        row = {
            "loss": float(loss.detach().cpu().item()),
            "ce": float(losses["ce"].detach().cpu().item()),
            "dice_loss": float(losses["dice_loss"].detach().cpu().item()),
            "mean_foreground_dice": metrics["mean_foreground_dice"],
            "whole_tumour_dice": metrics["whole_tumour_dice"],
            "pred_foreground_fraction": metrics["pred_foreground_fraction"],
            "gt_foreground_fraction": metrics["gt_foreground_fraction"],
            "dice_class_1": metrics["dice_class_1"],
            "dice_class_2": metrics["dice_class_2"],
            "dice_class_3": metrics["dice_class_3"],
            "pred_fraction_class_1": metrics["pred_fraction_class_1"],
            "pred_fraction_class_2": metrics["pred_fraction_class_2"],
            "pred_fraction_class_3": metrics["pred_fraction_class_3"],
            "gt_fraction_class_1": metrics["gt_fraction_class_1"],
            "gt_fraction_class_2": metrics["gt_fraction_class_2"],
            "gt_fraction_class_3": metrics["gt_fraction_class_3"],
            "grad_norm": gnorm,
            "amp_scale": float(scaler.get_scale()) if use_amp and scaler is not None else float("nan"),
        }
        for key, value in row.items():
            if math.isfinite(value):
                totals[key] = totals.get(key, 0.0) + value
        count += 1
        samples_seen += int(image.shape[0])
        if log_every is not None and log_every > 0 and step % log_every == 0:
            if torch.cuda.is_available():
                torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - phase_start
            throughput = samples_seen / elapsed if elapsed > 0 else float("nan")
            running_loss = totals.get("loss", 0.0) / count
            print(
                f"[{phase}] "
                f"step {step}/{planned_steps} "
                f"samples={samples_seen} "
                f"elapsed={elapsed / 60:.1f}m "
                f"throughput={throughput:.2f} slices/s "
                f"loss={running_loss:.4f} "
                f"grad_norm={gnorm:.4f} "
                f"amp_scale={row['amp_scale']:.1f}",
                flush=True,
            )
        if max_steps is not None and step >= max_steps:
            break

    if count == 0:
        raise RuntimeError("No batches were processed.")
    averaged = {key: value / count for key, value in totals.items()} | {
        "steps": float(count),
        "samples_seen": float(samples_seen),
    }
    if training:
        return averaged | {
            "patient_3d_metrics_status": "not_applicable_train",
            "patient_mean_fg_dice3d": float("nan"),
            "patient_whole_dice3d_mean": float("nan"),
            "patient_whole_dice3d_median": float("nan"),
            "patient_whole_rve_mean": float("nan"),
            "patient_whole_rve_median": float("nan"),
            "patient_dice3d_class_1_mean": float("nan"),
            "patient_dice3d_class_2_mean": float("nan"),
            "patient_dice3d_class_3_mean": float("nan"),
            "patient_rve_class_1_mean": float("nan"),
            "patient_rve_class_2_mean": float("nan"),
            "patient_rve_class_3_mean": float("nan"),
            "patient_3d_count": float("nan"),
        }
    return averaged | compute_patient_3d_metrics(
        patient_stats,
        expected_patient_ids=expected_patient_ids,
    )


def append_history(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "epoch",
        "phase",
        "steps",
        "samples_seen",
        "loss",
        "ce",
        "dice_loss",
        "mean_foreground_dice",
        "whole_tumour_dice",
        "dice_class_1",
        "dice_class_2",
        "dice_class_3",
        "pred_foreground_fraction",
        "gt_foreground_fraction",
        "pred_fraction_class_1",
        "pred_fraction_class_2",
        "pred_fraction_class_3",
        "gt_fraction_class_1",
        "gt_fraction_class_2",
        "gt_fraction_class_3",
        "grad_norm",
        "amp_scale",
        "patient_3d_metrics_status",
        "patient_mean_fg_dice3d",
        "patient_whole_dice3d_mean",
        "patient_whole_dice3d_median",
        "patient_whole_rve_mean",
        "patient_whole_rve_median",
        "patient_dice3d_class_1_mean",
        "patient_dice3d_class_2_mean",
        "patient_dice3d_class_3_mean",
        "patient_rve_class_1_mean",
        "patient_rve_class_2_mean",
        "patient_rve_class_3_mean",
        "patient_gt_present_class_1_count",
        "patient_gt_present_class_2_count",
        "patient_gt_present_class_3_count",
        "patient_3d_count",
        "train_seconds",
        "val_seconds",
        "epoch_seconds",
        "train_samples_per_sec",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def plot_history(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    train = [row for row in rows if row["phase"] == "train"]
    val = [row for row in rows if row["phase"] == "val"]

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    panels = [
        ("loss", "Total loss"),
        ("ce", "Cross entropy"),
        ("dice_loss", "Dice loss"),
        ("mean_foreground_dice", "Mean foreground Dice"),
    ]
    for ax, (key, title) in zip(axes.flat, panels):
        for phase_rows, label in [(train, "train"), (val, "val")]:
            if phase_rows:
                ax.plot([r["epoch"] for r in phase_rows], [r[key] for r in phase_rows], marker="o", label=label)
        ax.set_title(title)
        ax.set_xlabel("epoch")
        ax.grid(True, alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train/sanity-check Route B 4-label pretrained hybrid ViT.")
    parser.add_argument("--split-manifest", type=Path, default=PROJECT_ROOT / "outputs/splits/frozen_patient_split_seed42.csv")
    parser.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "data/processed/Task01_BrainTumour")
    parser.add_argument("--weights", type=Path, default=PROJECT_ROOT / "models/pretrained/timm/vit_base_r50_s16_224.orig_in21k/model.safetensors")
    parser.add_argument("--init-source", choices=["pretrained", "checkpoint"], default="pretrained")
    parser.add_argument("--init-checkpoint", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs/training/route_b_4label_sanity")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-train-steps-per-epoch", type=optional_positive_int, default=200)
    parser.add_argument("--max-val-steps-per-epoch", type=optional_positive_int, default=25)
    parser.add_argument("--max-train-patients", type=optional_positive_int, default=None)
    parser.add_argument("--max-val-patients", type=optional_positive_int, default=8)
    parser.add_argument("--max-slices-per-val-patient", type=optional_positive_int, default=8)
    parser.add_argument("--require-full-val", action="store_true")
    parser.add_argument("--seed", type=int, default=731)
    parser.add_argument("--balanced-train-sampler", dest="balanced_train_sampler", action="store_true", default=True)
    parser.add_argument("--no-balanced-train-sampler", dest="balanced_train_sampler", action="store_false")
    parser.add_argument("--train-sampling-mode", choices=["patient_local", "random"], default="patient_local")
    parser.add_argument("--trainable-vit-blocks", type=int, default=4)
    parser.add_argument("--trainable-resnet-stages", type=str, default="2")
    parser.add_argument("--train-stem", dest="train_stem", action="store_true", default=True)
    parser.add_argument("--no-train-stem", dest="train_stem", action="store_false")
    parser.add_argument("--train-pos-embed", dest="train_pos_embed", action="store_true", default=True)
    parser.add_argument("--no-train-pos-embed", dest="train_pos_embed", action="store_false")
    parser.add_argument("--lr-new", type=float, default=3e-4)
    parser.add_argument("--lr-pretrained", type=float, default=3e-5)
    parser.add_argument("--lr-pos-embed", type=float, default=3e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--data-parallel", action="store_true")
    parser.add_argument("--gpu-ids", type=str, default="0,1")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--train-log-every", type=int, default=500)
    parser.add_argument("--val-log-every", type=int, default=200)
    parser.add_argument("--build-only", action="store_true")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    if args.init_source == "checkpoint" and args.init_checkpoint is None:
        raise ValueError("--init-checkpoint is required when --init-source checkpoint.")
    if args.init_source == "pretrained" and args.init_checkpoint is not None:
        raise ValueError("--init-checkpoint should only be provided with --init-source checkpoint.")
    if args.init_source == "pretrained" and not args.weights.exists():
        raise FileNotFoundError(f"Pretrained weights not found: {args.weights}")
    if args.init_checkpoint is not None and not args.init_checkpoint.exists():
        raise FileNotFoundError(f"Initial checkpoint not found: {args.init_checkpoint}")

    gpu_ids = parse_gpu_ids(args.gpu_ids)
    device, parallelism = configure_device_and_parallelism(
        data_parallel=args.data_parallel,
        gpu_ids=gpu_ids,
    )
    use_amp = bool(args.amp and device.type == "cuda")
    split_df = pd.read_csv(args.split_manifest)
    assert_frozen_split_integrity(split_df)
    split_patient_counts = {
        split: int((split_df["split"] == split).sum())
        for split in ["train", "val", "test"]
    }
    train_samples = build_slice_samples(
        split_df,
        "train",
        data_root=args.data_root,
        max_patients=args.max_train_patients,
        seed=args.seed,
    )
    val_samples = build_slice_samples(
        split_df,
        "val",
        data_root=args.data_root,
        max_patients=args.max_val_patients,
        max_slices_per_patient=args.max_slices_per_val_patient,
        seed=args.seed,
    )
    if args.require_full_val:
        expected_val_patients = int((split_df["split"] == "val").sum())
        observed_val_patients = len({sample.case_id for sample in val_samples})
        if observed_val_patients != expected_val_patients:
            raise RuntimeError(
                "Full validation was requested but validation is subsetted: "
                f"observed {observed_val_patients}, expected {expected_val_patients}."
            )
        if expected_val_patients != 72:
            raise RuntimeError(f"Expected frozen validation patient count 72, got {expected_val_patients}.")
        if len(val_samples) != expected_val_patients * 155:
            raise RuntimeError(
                f"Expected full validation samples {expected_val_patients * 155}, got {len(val_samples)}."
            )
    train_loader = make_train_loader(
        train_samples,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        balanced=args.balanced_train_sampler,
        sampling_mode=args.train_sampling_mode,
    )
    val_loader = DataLoader(
        BrainTumourSliceDataset(val_samples),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False,
        worker_init_fn=make_worker_init_fn(args.seed),
        persistent_workers=args.num_workers > 0,
    )

    trainable_resnet_stages = parse_stage_indices(args.trainable_resnet_stages)
    model = PretrainedHybridViTSegmenter(
        weights_path=args.weights if args.init_source == "pretrained" else None,
        num_classes=4,
        trainable_vit_blocks=args.trainable_vit_blocks,
        trainable_resnet_stages=trainable_resnet_stages,
        train_stem=args.train_stem,
        train_pos_embed=args.train_pos_embed,
    )
    init_checkpoint = None
    init_checkpoint_metadata = {}
    if args.init_source == "checkpoint":
        init_checkpoint = load_initial_checkpoint(model, args.init_checkpoint)
        init_checkpoint_metadata = {
            "epoch": init_checkpoint.get("epoch"),
            "checkpoint_selection_metric": init_checkpoint.get("checkpoint_selection_metric"),
            "best_val_loss": init_checkpoint.get("best_val_loss"),
            "best_patient_mean_fg_dice3d": init_checkpoint.get("best_patient_mean_fg_dice3d"),
            "val_metrics": init_checkpoint.get("val_metrics", {}),
        }
    criterion = MulticlassForegroundDiceCELoss(ce_weight=0.5, dice_weight=0.5)
    optimizer = make_optimizer(
        model,
        lr_new=args.lr_new,
        lr_pretrained=args.lr_pretrained,
        lr_pos_embed=args.lr_pos_embed,
        weight_decay=args.weight_decay,
    )
    model = model.to(device)
    model = maybe_wrap_data_parallel(
        model,
        data_parallel=args.data_parallel,
        gpu_ids=gpu_ids,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "device": str(device),
        "parallelism": parallelism,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "effective_batch": {
            "batch_size": args.batch_size,
            "per_gpu_batch_size": (
                args.batch_size / max(int(parallelism["gpu_count_used"]), 1)
                if int(parallelism["gpu_count_used"]) > 0
                else None
            ),
        },
        "amp": {
            "enabled": use_amp,
            "dtype": "float16" if use_amp else None,
            "init_scale": 2**10 if use_amp else None,
        },
        "logging": {
            "train_log_every": args.train_log_every,
            "val_log_every": args.val_log_every,
        },
        "data_loading": {
            "train_sampling_mode": args.train_sampling_mode,
            "balanced_train_sampler": args.balanced_train_sampler,
            "effective_train_sampler": (
                "weighted_random"
                if args.balanced_train_sampler
                else args.train_sampling_mode
            ),
            "nifti_proxy_cache": True,
            "persistent_workers": args.num_workers > 0,
        },
        "label_semantics": {
            "num_classes": 4,
            "class_0": "background",
            "class_1": "label 1",
            "class_2": "label 2",
            "class_3": "label 3",
            "class_names_are_placeholder": True,
        },
        "checkpoint_selection_metric": "patient_mean_fg_dice3d",
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "split_patient_counts": split_patient_counts,
        "split_manifest_sha256": file_sha256(args.split_manifest),
        "initialization": {
            "source": args.init_source,
            "pretrained_weights": str(args.weights) if args.init_source == "pretrained" else None,
            "pretrained_weights_sha256": file_sha256(args.weights) if args.init_source == "pretrained" else None,
            "checkpoint": str(args.init_checkpoint) if args.init_checkpoint is not None else None,
            "checkpoint_sha256": file_sha256(args.init_checkpoint) if args.init_checkpoint is not None else None,
            "checkpoint_summary": init_checkpoint_metadata,
        },
        "checkpoint_selection_baseline": {
            "best_val_loss": (
                finite_float_or_default(init_checkpoint.get("best_val_loss"), float("inf"))
                if init_checkpoint is not None
                else float("inf")
            ),
            "best_patient_mean_fg_dice3d": (
                finite_float_or_default(init_checkpoint.get("best_patient_mean_fg_dice3d"), -float("inf"))
                if init_checkpoint is not None
                else -float("inf")
            ),
            "inherited_from_init_checkpoint": init_checkpoint is not None,
        },
        "train_summary": sample_summary(train_samples),
        "val_summary": sample_summary(val_samples),
        "parameter_counts": count_parameters(model),
        "optimizer_groups": [
            {"name": group["name"], "lr": group["lr"], "params": len(group["params"])}
            for group in optimizer.param_groups
        ],
        "args": vars(args)
        | {
            "split_manifest": str(args.split_manifest),
            "data_root": str(args.data_root),
            "weights": str(args.weights),
            "init_checkpoint": str(args.init_checkpoint) if args.init_checkpoint is not None else None,
            "output_dir": str(args.output_dir),
        },
    }
    (args.output_dir / "training_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    if args.build_only:
        print("Build-only check complete.")
        print(f"train_summary={metadata['train_summary']}")
        print(f"val_summary={metadata['val_summary']}")
        return

    history: list[dict[str, object]] = []
    best_val_loss = float(metadata["checkpoint_selection_baseline"]["best_val_loss"])
    best_patient_mean_fg_dice3d = float(metadata["checkpoint_selection_baseline"]["best_patient_mean_fg_dice3d"])
    history_csv = args.output_dir / "loss_history.csv"
    plot_path = args.output_dir / "loss_curves.png"
    scaler = make_grad_scaler(use_amp)

    for epoch in range(1, args.epochs + 1):
        train_start = time.perf_counter()
        train_metrics = run_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            device=device,
            optimizer=optimizer,
            scaler=scaler,
            use_amp=use_amp,
            phase="train",
            log_every=args.train_log_every,
            max_steps=args.max_train_steps_per_epoch,
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize(device)
        train_seconds = time.perf_counter() - train_start

        val_start = time.perf_counter()
        val_metrics = run_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            optimizer=None,
            scaler=scaler,
            use_amp=use_amp,
            phase="val",
            log_every=args.val_log_every,
            max_steps=args.max_val_steps_per_epoch,
            expected_patient_ids=(
                set(split_df.loc[split_df["split"] == "val", "case_id"].astype(str))
                if args.require_full_val
                else None
            ),
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize(device)
        val_seconds = time.perf_counter() - val_start
        epoch_seconds = train_seconds + val_seconds
        train_samples_per_sec = (
            float(train_metrics["samples_seen"]) / train_seconds
            if train_seconds > 0
            else float("nan")
        )

        history.append(
            {
                "epoch": epoch,
                "phase": "train",
                **train_metrics,
                "train_seconds": train_seconds,
                "val_seconds": float("nan"),
                "epoch_seconds": epoch_seconds,
                "train_samples_per_sec": train_samples_per_sec,
            }
        )
        history.append(
            {
                "epoch": epoch,
                "phase": "val",
                **val_metrics,
                "train_seconds": float("nan"),
                "val_seconds": val_seconds,
                "epoch_seconds": epoch_seconds,
                "train_samples_per_sec": train_samples_per_sec,
            }
        )
        append_history(history_csv, history)
        plot_history(plot_path, history)

        has_patient_dice3d = math.isfinite(float(val_metrics.get("patient_mean_fg_dice3d", float("nan"))))
        current_val_loss = float(val_metrics["loss"])
        current_patient_dice3d = float(val_metrics.get("patient_mean_fg_dice3d", float("nan")))
        previous_best_val_loss = best_val_loss
        previous_best_patient_dice3d = best_patient_mean_fg_dice3d

        best_val_loss = min(best_val_loss, current_val_loss)
        if has_patient_dice3d:
            best_patient_mean_fg_dice3d = max(best_patient_mean_fg_dice3d, current_patient_dice3d)

        should_save = current_patient_dice3d > previous_best_patient_dice3d if has_patient_dice3d else current_val_loss < previous_best_val_loss
        if should_save:
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": unwrap_model(model).state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "checkpoint_selection_metric": "patient_mean_fg_dice3d" if has_patient_dice3d else "val_loss",
                    "best_val_loss": best_val_loss,
                    "best_patient_mean_fg_dice3d": best_patient_mean_fg_dice3d,
                    "val_metrics": val_metrics,
                    "metadata": metadata,
                },
                args.output_dir / "best_checkpoint.pt",
            )

        patient_status = str(val_metrics.get("patient_3d_metrics_status", "unknown"))
        patient_dice = float(val_metrics.get("patient_mean_fg_dice3d", float("nan")))
        patient_whole_dice = float(val_metrics.get("patient_whole_dice3d_mean", float("nan")))
        patient_rve = float(val_metrics.get("patient_whole_rve_mean", float("nan")))
        print(
            f"epoch={epoch} "
            f"train_loss={train_metrics['loss']:.4f} val_loss={val_metrics['loss']:.4f} "
            f"train_fg_dice={train_metrics['mean_foreground_dice']:.4f} "
            f"val_fg_dice={val_metrics['mean_foreground_dice']:.4f} "
            f"val_whole_dice={val_metrics['whole_tumour_dice']:.4f} "
            f"train_fg={train_metrics['pred_foreground_fraction']:.4f} "
            f"val_fg={val_metrics['pred_foreground_fraction']:.4f} "
            f"pred_cls=({val_metrics['pred_fraction_class_1']:.4f},"
            f"{val_metrics['pred_fraction_class_2']:.4f},"
            f"{val_metrics['pred_fraction_class_3']:.4f}) "
            f"patient3d_status={patient_status} "
            f"patient_mean_fg_dice3d={patient_dice:.4f} "
            f"patient_whole_dice3d={patient_whole_dice:.4f} "
            f"patient_whole_rve={patient_rve:.4f} "
            f"timing: train={train_seconds:.1f}s "
            f"val={val_seconds:.1f}s "
            f"total={epoch_seconds:.1f}s "
            f"throughput: train={train_samples_per_sec:.2f} slices/s"
        )


if __name__ == "__main__":
    main()
