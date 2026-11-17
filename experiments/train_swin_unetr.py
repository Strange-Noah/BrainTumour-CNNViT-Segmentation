"""Train a patch-based Swin UNETR baseline on the frozen BMETS v1 split."""
import argparse
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from monai.inferers import sliding_window_inference
from monai.losses import DiceCELoss

from experiments.download_manifest import MANIFEST_SHA256
from experiments.swin_unetr_common import build_model, normalize_modalities
from research.dataset import Dataset as BMETSDataset


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


def validate(model, dataset, device, roi, max_visits=0):
    model.eval()
    scores = defaultdict(list)
    indices = validation_indices(dataset.visits, max_visits)
    count = len(indices)

    with torch.inference_mode():
        for number, index in enumerate(indices, start=1):
            sample = dataset.load(index, "enhancing")
            images = normalize_modalities(sample["images"])
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--roi", type=int, default=96)
    parser.add_argument("--feature-size", type=int, default=24)
    parser.add_argument("--max-train-visits", type=int, default=0,
                        help="Debug only; 0 uses every visit in the frozen train partition")
    parser.add_argument("--max-val-visits", type=int, default=12,
                        help="Visits per epoch for patient-balanced checkpoint selection; 0 uses all validation visits")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.epochs < 1 or args.roi < 32 or args.roi % 32:
        parser.error("epochs must be positive and --roi must be a positive multiple of 32")
    if args.feature_size not in (12, 24, 48):
        parser.error("MONAI Swin UNETR feature size must be one of 12, 24, or 48")
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

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.mkdir(exist_ok=False)
    train_data = BMETSDataset(args.manifest, split, args.cache / "raw", "train")
    val_data = BMETSDataset(args.manifest, split, args.cache / "raw", "validation")
    train_indices = list(range(len(train_data.visits)))
    if args.max_train_visits:
        train_indices = train_indices[:args.max_train_visits]
    if not train_indices:
        raise RuntimeError("No training visits found in the frozen train partition")

    model = build_model(args.feature_size, use_checkpoint=True).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    loss_fn = DiceCELoss(include_background=False, to_onehot_y=True, softmax=True)
    scaler = torch.cuda.amp.GradScaler()
    best_dice = -1.0
    history = []
    rng = random.Random(args.seed)

    for epoch in range(1, args.epochs + 1):
        model.train()
        rng.shuffle(train_indices)
        losses = []
        for step, index in enumerate(train_indices, start=1):
            sample = train_data.load(index, "enhancing")
            images = normalize_modalities(sample["images"])
            target = np.asarray(sample["target"], dtype=np.uint8)
            x_patch, y_patch = random_patch(images, target, args.roi, rng)
            image = torch.from_numpy(x_patch[None]).to(device)
            label = torch.from_numpy(y_patch[None, None].astype(np.int64)).to(device)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = model(image)
                loss = loss_fn(logits, label)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss at epoch {epoch}, visit {index}")
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach().cpu()))
            print(
                f"epoch={epoch}/{args.epochs} visit={step}/{len(train_indices)} "
                f"loss={losses[-1]:.4f}",
                flush=True,
            )
            del image, label, logits, loss, sample

        val_dice, val_patients = validate(
            model, val_data, device, args.roi, args.max_val_visits
        )
        record = {
            "epoch": epoch,
            "train_loss_mean": float(np.mean(losses)),
            "validation_patient_macro_dice": val_dice,
            "validation_patients_seen": val_patients,
        }
        history.append(record)
        (args.output / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        print(json.dumps(record), flush=True)

        if val_dice > best_dice:
            best_dice = val_dice
            # The inference adapter intentionally expects a raw state_dict.
            torch.save(model.state_dict(), args.output / "best_checkpoint.pt")
            try:
                import monai
                monai_version = monai.__version__
            except Exception:
                monai_version = "unknown"
            metadata = {
                "checkpoint_ref": str((args.output / "best_checkpoint.pt").resolve()),
                "framework_version": f"torch={torch.__version__}; monai={monai_version}",
                "model_config": {
                    "architecture": "MONAI SwinUNETR",
                    "initialization": "random initialization",
                    "in_channels": 4,
                    "out_channels": 2,
                    "feature_size": args.feature_size,
                    "roi": args.roi,
                    "target": "enhancing tissue, original label 3",
                    "input_order": ["T1", "T1c", "T2", "FLAIR"],
                    "normalization": "per-volume z-score over nonzero voxels per modality",
                    "loss": "MONAI DiceCELoss; foreground Dice plus softmax cross-entropy",
                    "optimizer": "AdamW lr=1e-4 weight_decay=1e-5",
                    "amp": "CUDA float16",
                    "split_sha256": split_sha,
                    "manifest_sha256": actual_manifest_sha,
                },
            }
            (args.output / "run_metadata.json").write_text(
                json.dumps(metadata, indent=2), encoding="utf-8"
            )

    print(f"Best validation patient-macro Dice: {best_dice:.4f}")
    print(f"Checkpoint: {args.output / 'best_checkpoint.pt'}")


if __name__ == "__main__":
    main()
