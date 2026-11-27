"""Train a patch-based Swin UNETR baseline on the frozen BMETS v1 split."""
import argparse
import hashlib
import json
import os
import random
import subprocess
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import nibabel as nib
from monai.inferers import sliding_window_inference
from monai.losses import DiceCELoss
from scipy import ndimage

from experiments.download_manifest import MANIFEST_SHA256
from experiments.swin_unetr_common import build_model, normalize_modalities
from experiments.swin_sampling import (
    acquisition_bounds,
    cached_acquisition_support,
    CONNECTIVITY_26,
    component_index,
    patient_balanced_draw_counts,
    sample_training_patch,
)
from research.dataset import Dataset as BMETSDataset
from research.geometry import mask_geometry, require_mm


def random_patch(images, target, roi, rng):
    pads = [(0, max(0, roi - size)) for size in target.shape]
    target = np.pad(target, pads, mode="constant")
    images = np.pad(images, [(0, 0), *pads], mode="constant")

    positive = np.argwhere(target > 0)
    if len(positive) and rng.random() < 0.7:
        center = positive[rng.randrange(len(positive))]
    else:
        center = np.asarray([rng.randrange(size) for size in target.shape])

    start = [
        max(0, min(int(center[d]) - roi // 2, target.shape[d] - roi))
        for d in range(3)
    ]
    slices = tuple(slice(start[d], start[d] + roi) for d in range(3))
    x = images[(slice(None), *slices)]
    y = target[slices]
    return np.ascontiguousarray(x), np.ascontiguousarray(y)


def patient_macro_dice(score_by_patient):
    return float(np.mean([np.mean(scores) for scores in score_by_patient.values()]))


def validation_indices(visits, limit):
    """Round-robin visits across patients so a small validation subset is patient-balanced."""
    by_patient = defaultdict(list)
    for index, visit in enumerate(visits):
        by_patient[visit["patient_id"]].append(index)
    patients = sorted(by_patient)
    chosen = []
    depth = 0
    while True:
        added = False
        for patient in patients:
            indices = by_patient[patient]
            if depth < len(indices):
                chosen.append(indices[depth])
                added = True
                if limit and len(chosen) >= limit:
                    return chosen
        if not added:
            return chosen
        depth += 1


def validate(model, dataset, device, roi, max_visits=0, support_mode="none", support_cache=None):
    model.eval()
    scores = defaultdict(list)
    indices = validation_indices(dataset.visits, max_visits)
    count = len(indices)

    with torch.inference_mode():
        for number, index in enumerate(indices, start=1):
            sample = dataset.load(index, "enhancing")
            images = normalize_modalities(sample["images"])
            support = (
                cached_acquisition_support(
                    sample["images"], support_cache, sample["visit"]["exam_id"]
                )
                if support_mode in ("input", "output") else None
            )
            if support is not None and np.any(np.asarray(sample["target"], dtype=bool) & ~support):
                raise ValueError(
                    f"MRI support mask excludes reference enhancing voxels in "
                    f"{sample['visit']['exam_id']}; do not use this mask for checkpoint selection"
                )
            if support_mode == "input" and support is not None:
                images[:, ~support] = 0
            # Keep the full volume on host memory; sliding windows move to CUDA.
            image = torch.from_numpy(images[None])
            # Keep the full-volume output accumulator on CPU to reduce VRAM use.
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = sliding_window_inference(
                    image,
                    roi_size=(roi, roi, roi),
                    sw_batch_size=1,
                    predictor=model,
                    overlap=0.25,
                    sw_device=device,
                    device=torch.device("cpu"),
                )
            prediction = logits.argmax(dim=1)[0].numpy().astype(np.uint8)
            if support_mode == "output" and support is not None:
                prediction[~support] = 0
            truth = np.asarray(sample["target"], dtype=np.uint8)
            denom = int(prediction.sum() + truth.sum())
            dice = (2.0 * int(np.logical_and(prediction, truth).sum()) / denom) if denom else 1.0
            patient = sample["visit"]["patient_id"]
            scores[patient].append(float(dice))
            print(
                f"val {number}/{count} {sample['visit']['exam_id']} dice={dice:.4f}",
                flush=True,
            )
            del image, logits, sample

    if not scores:
        raise RuntimeError("Validation selected zero visits")
    return patient_macro_dice(scores), len(scores)


def make_loss(logits, label, component_ids, component_volumes, support, mode,
              alpha, beta, baseline_loss, weight_reference_mm3=100.0,
              weight_normalizer=1.0):
    """Compute the unchanged baseline loss or one explicitly selected ablation."""
    if mode == "baseline" and support is None:
        return baseline_loss(logits, label)

    probabilities = torch.softmax(logits.float(), dim=1)[:, 1]
    truth = label[:, 0].float()
    if support is None:
        valid = torch.ones_like(truth)
    else:
        valid = support.float()
    p = probabilities * valid
    y = truth * valid
    tp = (p * y).sum()
    fp = (p * (1.0 - truth) * valid).sum()
    fn = ((1.0 - probabilities) * y).sum()
    eps = 1e-5

    if mode == "tversky":
        overlap_loss = 1.0 - (tp + eps) / (tp + alpha * fp + beta * fn + eps)
    else:
        overlap_loss = 1.0 - (2.0 * tp + eps) / (p.sum() + y.sum() + eps)

    ce = F.cross_entropy(logits.float(), label[:, 0], reduction="none")
    weights = torch.ones_like(ce)
    if mode == "lesion_weighted_ce":
        lookup = np.ones(max(component_volumes.keys(), default=0) + 1, dtype=np.float32)
        for component_id, volume_mm3 in component_volumes.items():
            lookup[component_id] = float(np.clip(
                np.sqrt(weight_reference_mm3 / max(volume_mm3, 1e-6)), 0.5, 2.0
            ) / weight_normalizer)
        lookup_tensor = torch.as_tensor(lookup, device=component_ids.device)
        indices = component_ids.long().clamp(min=0, max=len(lookup) - 1)
        weights = lookup_tensor[indices]
        weights = torch.where(truth > 0, weights, 1.0)
    ce_loss = (ce * weights * valid).sum() / valid.sum().clamp_min(1.0)
    return overlap_loss + ce_loss


def training_lesion_weight_stats(dataset, indices, cache, manifest_sha, split_sha):
    """Compute a fixed train-only positive-voxel weight normalizer and cache it."""
    stats_dir = Path(cache) / "derived"
    stats_dir.mkdir(parents=True, exist_ok=True)
    stats_path = stats_dir / f"lesion_weight_stats_{manifest_sha[:12]}_{split_sha[:12]}.json"
    if stats_path.exists():
        cached = json.loads(stats_path.read_text(encoding="utf-8"))
        if cached.get("manifest_sha256") == manifest_sha and cached.get("split_sha256") == split_sha:
            return cached

    volumes_mm3 = []
    voxel_counts = []
    for number, index in enumerate(indices, start=1):
        visit = dataset.visits[index]
        reference = nib.load(dataset.fetch(visit["modalities"]["t1c"]))
        require_mm(reference, "T1c reference")
        segmentation = nib.load(dataset.fetch(visit["segmentation"]))
        mask_geometry(segmentation, reference, visit["exam_id"])
        raw = np.asanyarray(segmentation.dataobj)
        if not np.isfinite(raw).all() or not np.isin(raw, [0, 1, 2, 3, 4]).all():
            raise ValueError(f"Unexpected segmentation labels in {visit['exam_id']}")
        labels, count = ndimage.label(raw == 3, structure=CONNECTIVITY_26)
        if count:
            counts = np.bincount(labels.ravel(), minlength=count + 1)[1:]
            voxel_volume = abs(float(np.linalg.det(reference.affine[:3, :3])))
            voxel_counts.extend(int(value) for value in counts if value)
            volumes_mm3.extend(float(value * voxel_volume) for value in counts if value)
        print(f"lesion-weight statistics {number}/{len(indices)} {visit['exam_id']}", flush=True)
        del raw, labels, segmentation, reference

    if not volumes_mm3:
        raise RuntimeError("No label-3 lesions found in the training partition")
    volumes = np.asarray(volumes_mm3, dtype=np.float64)
    counts = np.asarray(voxel_counts, dtype=np.float64)
    reference_volume = float(np.median(volumes))
    raw_weights = np.clip(np.sqrt(reference_volume / np.maximum(volumes, 1e-6)), 0.5, 2.0)
    global_positive_mean = float(np.dot(raw_weights, counts) / counts.sum())
    result = {
        "manifest_sha256": manifest_sha,
        "split_sha256": split_sha,
        "training_visits": len(indices),
        "training_lesions": int(len(volumes)),
        "reference_median_volume_mm3": reference_volume,
        "global_positive_voxel_weight_mean": global_positive_mean,
        "weight_clip": [0.5, 2.0],
    }
    stats_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def save_checkpoint(model, output, epoch, args, split_sha, manifest_sha,
                    checkpoint_name="best_checkpoint.pt", metadata_name="run_metadata.json"):
    """Write a checkpoint and its exact configuration provenance."""
    torch.save(model.state_dict(), output / checkpoint_name)
    try:
        import monai
        monai_version = monai.__version__
    except Exception:
        monai_version = "unknown"
    source_files = [
        Path("experiments/train_swin_unetr.py"),
        Path("experiments/swin_sampling.py"),
        Path("experiments/swin_unetr_common.py"),
        Path("experiments/swin_unetr.py"),
        Path("experiments/run_swin_partition.py"),
        Path("experiments/analyze_swin_lesion_sizes.py"),
    ]
    source_hash = hashlib.sha256()
    for source_path in source_files:
        source_hash.update(source_path.as_posix().encode("utf-8"))
        source_hash.update(source_path.read_bytes())
    try:
        source_git_sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
        source_git_dirty = bool(subprocess.check_output(
            ["git", "status", "--porcelain"], text=True, stderr=subprocess.DEVNULL
        ).strip())
    except (OSError, subprocess.CalledProcessError):
        source_git_sha, source_git_dirty = None, None
    metadata = {
        "checkpoint_ref": str((output / checkpoint_name).resolve()),
        "framework_version": f"torch={torch.__version__}; monai={monai_version}",
        "source": {
            "git_sha": source_git_sha,
            "working_tree_dirty": source_git_dirty,
            "experiment_code_sha256": source_hash.hexdigest(),
        },
        "model_config": {
            "architecture": "MONAI SwinUNETR",
            "initialization": (
                f"warm-start from {args.init_checkpoint}"
                if args.init_checkpoint else
                (f"resumed from {args.resume}" if args.resume else "random initialization")
            ),
            "in_channels": 4,
            "out_channels": 2,
            "feature_size": args.feature_size,
            "roi": args.roi,
            "target": "enhancing tissue, original label 3",
            "input_order": ["T1", "T1c", "T2", "FLAIR"],
            "normalization": "per-volume z-score over nonzero voxels per modality",
            "sampling": args.sampling,
            "sampling_details": {
                "patient_draws_per_epoch": "uniform patient allocation; full training set yields two or three crops per patient",
                "visit_rotation": "round-robin through each patient's visits by epoch",
                "lesion_connectivity": 26,
                "lesion_weight": "clip(sqrt(median visit lesion volume / lesion volume), 0.5, 2.0)",
                "without_replacement": "per visit within epoch until all components have been sampled",
                "negative_crop_probability": 0.25,
            },
            "local_crop_scale": args.local_crop_scale,
            "local_crop_probability": args.local_crop_probability,
            "support_mask": args.support_mask,
            "support_mask_definition": "largest 26-connected component of the union of nonzero input voxels",
            "loss": args.loss,
            "lesion_weight_statistics": getattr(args, "lesion_weight_stats", None),
            "lesion_weight_rule": "clip(sqrt(training median lesion volume / lesion volume), 0.5, 2.0), normalized to mean 1 over training positive voxels",
            "tversky_alpha_fp": args.tversky_alpha,
            "tversky_beta_fn": args.tversky_beta,
            "validation_selection_visits": args.max_val_visits,
            "seed": args.seed,
            "optimizer": "AdamW lr=1e-4 weight_decay=1e-5",
            "amp": "CUDA float16",
            "split_sha256": split_sha,
            "manifest_sha256": manifest_sha,
            "epoch": epoch,
        },
    }
    (output / metadata_name).write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def resume_config(args):
    """Settings that must remain fixed when continuing an optimizer run."""
    return {
        "roi": args.roi,
        "feature_size": args.feature_size,
        "max_train_visits": args.max_train_visits,
        "max_val_visits": args.max_val_visits,
        "sampling": args.sampling,
        "local_crop_scale": args.local_crop_scale,
        "local_crop_probability": args.local_crop_probability,
        "support_mask": args.support_mask,
        "loss": args.loss,
        "tversky_alpha": args.tversky_alpha,
        "tversky_beta": args.tversky_beta,
        "seed": args.seed,
    }


def save_training_state(path, model, optimizer, scaler, epoch, best_dice,
                        history, rng, args, split_sha, manifest_sha):
    """Atomically save everything needed to continue after a completed epoch."""
    state = {
        "format_version": 1,
        "epoch": epoch,
        "best_dice": best_dice,
        "history": history,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "python_rng_state": random.getstate(),
        "numpy_rng_state": np.random.get_state(),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state_all": torch.cuda.get_rng_state_all(),
        "sampler_rng_state": rng.getstate(),
        "run_config": resume_config(args),
        "split_sha256": split_sha,
        "manifest_sha256": manifest_sha,
    }
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.stem}_", suffix=".tmp", dir=path.parent
    )
    os.close(fd)
    try:
        torch.save(state, temporary_name)
        os.replace(temporary_name, path)
    finally:
        Path(temporary_name).unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    recovery = parser.add_mutually_exclusive_group()
    recovery.add_argument(
        "--resume", type=Path,
        help="Resume a full last_training_state.pt checkpoint at the next epoch",
    )
    recovery.add_argument(
        "--init-checkpoint", type=Path,
        help="Warm-start from model weights only; optimizer and epoch restart",
    )
    parser.add_argument("--epochs", type=int, default=21)
    parser.add_argument("--roi", type=int, default=96)
    parser.add_argument("--feature-size", type=int, default=24)
    parser.add_argument("--max-train-visits", type=int, default=0,
                        help="Debug only; 0 uses every visit in the frozen train partition")
    parser.add_argument("--max-val-visits", type=int, default=45,
                        help="Checkpoint-selection visits; 45 selects one fixed visit from each validation patient; 0 uses all visits")
    parser.add_argument("--sampling", choices=("voxel", "lesion_balanced"), default="voxel",
                        help="voxel reproduces the initial sampler; lesion_balanced balances patients and samples components")
    parser.add_argument("--local-crop-scale", type=float, default=0.75,
                        help="Physical-axis voxel crop fraction for local magnification before resize to --roi")
    parser.add_argument("--local-crop-probability", type=float, default=0.0,
                        help="Probability of a local crop for a selected lesion; 0.5 is a useful ablation")
    parser.add_argument("--support-mask", choices=("none", "input", "loss", "output"), default="none",
                        help="Use the MRI-derived dominant nonzero support on input, loss, or output")
    parser.add_argument("--loss", choices=("baseline", "lesion_weighted_ce", "tversky"), default="baseline")
    parser.add_argument("--tversky-alpha", type=float, default=0.6,
                        help="Tversky false-positive penalty coefficient")
    parser.add_argument("--tversky-beta", type=float, default=0.4,
                        help="Tversky false-negative penalty coefficient")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.epochs < 1 or args.roi < 32 or args.roi % 32:
        parser.error("epochs must be positive and --roi must be a positive multiple of 32")
    if args.feature_size not in (12, 24, 48):
        parser.error("MONAI Swin UNETR feature size must be one of 12, 24, or 48")
    if not 0.0 <= args.local_crop_probability <= 1.0:
        parser.error("--local-crop-probability must be between 0 and 1")
    if not 0.0 < args.local_crop_scale <= 1.0:
        parser.error("--local-crop-scale must be in (0, 1]")
    if args.tversky_alpha <= 0 or args.tversky_beta <= 0:
        parser.error("Tversky alpha and beta must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; this run expects an NVIDIA GPU")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda")
    print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)

    actual_manifest_sha = hashlib.sha256(args.manifest.read_bytes()).hexdigest()
    if actual_manifest_sha != MANIFEST_SHA256:
        raise ValueError("Manifest is not the frozen canonical BMETS v1 manifest")
    split = Path("protocol/v1/patients.csv")
    split_sha = hashlib.sha256(split.read_bytes()).hexdigest()
    if split_sha != "57f5af55141e16ecb0b85fdda63cc355175d9b35271126784eb22766698b36f0":
        raise ValueError("Patient split does not match frozen Protocol v1")

    support_cache = args.cache / "derived" / "support" / actual_manifest_sha[:12]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.resume:
        if not args.resume.is_file():
            parser.error(f"resume checkpoint not found: {args.resume}")
        if args.resume.parent.resolve() != args.output.resolve():
            parser.error("--resume checkpoint must be inside the --output run folder")
        args.output.mkdir(exist_ok=True)
    else:
        args.output.mkdir(exist_ok=False)
    if args.init_checkpoint and not args.init_checkpoint.is_file():
        parser.error(f"initialization checkpoint not found: {args.init_checkpoint}")
    train_data = BMETSDataset(args.manifest, split, args.cache / "raw", "train")
    val_data = BMETSDataset(args.manifest, split, args.cache / "raw", "validation")
    train_indices = list(range(len(train_data.visits)))
    if args.max_train_visits:
        train_indices = train_indices[:args.max_train_visits]
    if not train_indices:
        raise RuntimeError("No training visits found in the frozen train partition")
    args.lesion_weight_stats = (
        training_lesion_weight_stats(
            train_data, list(range(len(train_data.visits))), args.cache, actual_manifest_sha, split_sha
        )
        if args.loss == "lesion_weighted_ce" else None
    )

    model = build_model(args.feature_size, use_checkpoint=True).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    loss_fn = DiceCELoss(include_background=False, to_onehot_y=True, softmax=True)
    scaler = torch.cuda.amp.GradScaler()
    best_dice = -1.0
    history = []
    rng = random.Random(args.seed)
    device_reported = False
    start_epoch = 1

    if args.resume:
        # This is a trusted local checkpoint written by this script; unrestricted
        # deserialization is required to restore NumPy and Python RNG state.
        state = torch.load(args.resume, map_location="cpu", weights_only=False)
        if state.get("format_version") != 1 or "optimizer_state_dict" not in state:
            raise ValueError(
                "This is a weights-only checkpoint, not a resumable training state. "
                "Use --init-checkpoint to warm-start it, or resume from "
                "last_training_state.pt."
            )
        if state.get("manifest_sha256") != actual_manifest_sha or state.get("split_sha256") != split_sha:
            raise ValueError("Resume checkpoint manifest/split does not match this run")
        saved_config = state.get("run_config", {})
        current_config = resume_config(args)
        mismatches = [
            key for key, value in current_config.items()
            if saved_config.get(key) != value
        ]
        if mismatches:
            raise ValueError(
                "Resume settings differ from checkpoint: " + ", ".join(mismatches)
            )
        if args.epochs <= int(state["epoch"]):
            raise ValueError(
                f"--epochs ({args.epochs}) must be greater than the saved epoch "
                f"({state['epoch']})"
            )
        model.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        scaler.load_state_dict(state["scaler_state_dict"])
        best_dice = float(state["best_dice"])
        history = list(state["history"])
        random.setstate(state["python_rng_state"])
        np.random.set_state(state["numpy_rng_state"])
        torch.set_rng_state(state["torch_rng_state"])
        torch.cuda.set_rng_state_all(state["cuda_rng_state_all"])
        rng.setstate(state["sampler_rng_state"])
        start_epoch = int(state["epoch"]) + 1
        print(
            f"Resuming after epoch {state['epoch']}; next epoch is "
            f"{start_epoch}/{args.epochs}",
            flush=True,
        )
    elif args.init_checkpoint:
        state = torch.load(args.init_checkpoint, map_location="cpu", weights_only=False)
        weights = state.get("model_state_dict", state)
        model.load_state_dict(weights)
        print(
            "Warm-started model weights; optimizer, sampler, and epoch start fresh.",
            flush=True,
        )

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        losses = []
        if args.sampling == "voxel":
            visit_draw_counts = {index: 1 for index in train_indices}
            epoch_indices = list(train_indices)
            rng.shuffle(epoch_indices)
        else:
            visit_draw_counts = patient_balanced_draw_counts(
                train_data.visits, train_indices, rng, epoch
            )
            epoch_indices = list(visit_draw_counts)
            rng.shuffle(epoch_indices)

        completed_steps = 0
        used_components = defaultdict(set)
        for index in epoch_indices:
            sample = train_data.load(index, "enhancing")
            support = (
                cached_acquisition_support(
                    sample["images"], support_cache, sample["visit"]["exam_id"]
                )
                if args.support_mask in ("input", "loss") else None
            )
            support_bounds = acquisition_bounds(sample["images"]) if args.sampling == "lesion_balanced" else None
            images = normalize_modalities(sample["images"])
            target = np.asarray(sample["target"], dtype=np.uint8)
            if args.sampling == "lesion_balanced" or args.loss == "lesion_weighted_ce":
                component_ids, components = component_index(target, sample["reference"].affine)
            else:
                component_ids = np.zeros_like(target, dtype=np.int32)
                components = {}
            if args.support_mask in ("input", "loss") and support is not None and np.any(target.astype(bool) & ~support):
                raise ValueError(
                    f"MRI support mask excludes labelled enhancing voxels in "
                    f"{sample['visit']['exam_id']}; inspect the mask before training"
                )
            if args.support_mask == "input" and support is not None:
                images[:, ~support] = 0

            for repeat in range(visit_draw_counts[index]):
                x_patch, y_patch, id_patch, support_patch, selected_id = sample_training_patch(
                    images,
                    target,
                    args.roi,
                    rng,
                    sampling=args.sampling,
                    labels=component_ids,
                    components=components,
                    used_component_ids=used_components[index],
                    support=support,
                    support_bounds=support_bounds,
                    local_scale=args.local_crop_scale,
                    local_probability=args.local_crop_probability,
                )
                image = torch.from_numpy(x_patch[None]).to(device)
                label = torch.from_numpy(y_patch[None, None].astype(np.int64)).to(device)
                id_tensor = torch.from_numpy(id_patch[None]).to(device)
                support_tensor = (
                    torch.from_numpy(support_patch[None]).to(device)
                    if args.support_mask == "loss" else None
                )

                optimizer.zero_grad(set_to_none=True)
                if not device_reported:
                    print(
                        f"Training devices: model={next(model.parameters()).device}, "
                        f"image={image.device}",
                        flush=True,
                    )
                    device_reported = True
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    logits = model(image)
                    loss = make_loss(
                        logits, label, id_tensor, components, support_tensor,
                        args.loss, args.tversky_alpha, args.tversky_beta, loss_fn,
                        weight_reference_mm3=(args.lesion_weight_stats or {}).get(
                            "reference_median_volume_mm3", 100.0
                        ),
                        weight_normalizer=(args.lesion_weight_stats or {}).get(
                            "global_positive_voxel_weight_mean", 1.0
                        ),
                    )
                if not torch.isfinite(loss):
                    raise RuntimeError(f"Non-finite loss at epoch {epoch}, visit {index}")
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                losses.append(float(loss.detach().cpu()))
                completed_steps += 1
                print(
                    f"epoch={epoch}/{args.epochs} update={completed_steps}/"
                    f"{len(train_indices)} visit={sample['visit']['exam_id']} "
                    f"selected_lesion={selected_id} loss={losses[-1]:.4f}",
                    flush=True,
                )
                del image, label, id_tensor, support_tensor, logits, loss
            del sample

        val_dice, val_patients = validate(
            model, val_data, device, args.roi, args.max_val_visits,
            args.support_mask, support_cache,
        )
        record = {
            "epoch": epoch,
            "train_loss_mean": float(np.mean(losses)),
            "validation_patient_macro_dice": val_dice,
            "validation_patients_seen": val_patients,
            "training_updates": completed_steps,
            "sampling": args.sampling,
            "loss": args.loss,
            "support_mask": args.support_mask,
            "local_crop_probability": args.local_crop_probability,
        }
        history.append(record)
        (args.output / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        print(json.dumps(record), flush=True)

        # Keep the final-epoch checkpoint separately from the validation-selected
        # checkpoint for controlled epoch-matched ablations.
        save_checkpoint(
            model, args.output, epoch, args, split_sha, actual_manifest_sha,
            checkpoint_name="last_checkpoint.pt", metadata_name="last_run_metadata.json",
        )

        if val_dice > best_dice:
            best_dice = val_dice
            save_checkpoint(model, args.output, epoch, args, split_sha, actual_manifest_sha)

        save_training_state(
            args.output / "last_training_state.pt",
            model, optimizer, scaler, epoch, best_dice, history, rng,
            args, split_sha, actual_manifest_sha,
        )
        print(f"Resumable training state saved after epoch {epoch}.", flush=True)

    print(f"Best validation patient-macro Dice: {best_dice:.4f}")
    print(f"Checkpoint: {args.output / 'best_checkpoint.pt'}")


if __name__ == "__main__":
    main()
