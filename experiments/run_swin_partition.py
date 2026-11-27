"""Run the Swin predictor over one frozen BMETS partition and use the shared evaluator."""
import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path

import numpy as np

from experiments.swin_unetr_common import cached_acquisition_support
from research.dataset import Dataset as BMETSDataset
from research.evaluate import evaluate, export, patient_aggregates


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--partition", choices=["validation", "test"], required=True)
    parser.add_argument("--model-version", required=True)
    parser.add_argument("--git-sha", required=True)
    args = parser.parse_args()

    if not args.checkpoint.is_file():
        parser.error(f"Checkpoint not found: {args.checkpoint}")
    provenance = json.loads(args.metadata.read_text(encoding="utf-8"))
    required = {"checkpoint_ref", "framework_version", "model_config"}
    if not required <= provenance.keys():
        parser.error(f"Metadata must contain {sorted(required)}")
    config = provenance["model_config"]
    manifest_sha = hashlib.sha256(args.manifest.read_bytes()).hexdigest()
    support_cache = args.cache / "derived" / "support" / manifest_sha[:12]
    os.environ["SWIN_CHECKPOINT"] = str(args.checkpoint.resolve())
    os.environ["SWIN_FEATURE_SIZE"] = str(config["feature_size"])
    os.environ["SWIN_ROI_SIZE"] = str(config["roi"])
    os.environ["SWIN_SUPPORT_MASK"] = str(config.get("support_mask", "none"))
    os.environ["SWIN_SUPPORT_CACHE"] = str(support_cache.resolve())

    split_path = Path("protocol/v1/patients.csv")
    split_sha = hashlib.sha256(split_path.read_bytes()).hexdigest()
    predict = getattr(importlib.import_module("experiments.swin_unetr"), "predict")
    dataset = BMETSDataset(args.manifest, split_path, args.cache / "raw", args.partition)
    if not dataset.visits:
        raise RuntimeError(f"No visits found for {args.partition}")
    if args.output.exists():
        raise FileExistsError(f"Output exists; choose a new run directory: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.mkdir()
    prediction_root = args.output / "visits"
    prediction_root.mkdir()

    visit_rows = []
    for index, row in enumerate(dataset.visits):
        sample = dataset.load(index, "enhancing")
        # Keep raw labels and target out of the model's inference input.
        predictor_input = {
            "images": sample["images"],
            "reference": sample["reference"],
            "visit": sample["visit"],
        }
        prediction = np.asarray(predict(predictor_input))
        target = np.asarray(sample["target"], dtype=np.uint8)
        reference = sample["reference"]
        if config.get("support_mask") in ("input", "output"):
            support = cached_acquisition_support(
                sample["images"], support_cache, row["exam_id"]
            )
            if np.any(target.astype(bool) & ~support):
                raise ValueError(
                    f"MRI support mask excludes reference enhancing voxels in "
                    f"{row['exam_id']}; refusing a potentially truncated evaluation"
                )
        if prediction.shape != target.shape or not np.isin(prediction, [0, 1]).all():
            raise ValueError(f"Invalid binary/original-grid prediction for {row['exam_id']}")

        metadata = {
            **provenance,
            "patient_id": row["patient_id"],
            "exam_id": row["exam_id"],
            "model_version": args.model_version,
            "git_sha": args.git_sha,
            "task": "enhancing",
            "split_sha256": split_sha,
        }
        metrics = evaluate(prediction, target, reference.affine)
        export(prediction, reference, prediction_root / row["exam_id"], metadata)
        visit_rows.append({
            "patient_id": row["patient_id"],
            "exam_id": row["exam_id"],
            **metrics,
        })
        print(
            f"{args.partition} {index + 1}/{len(dataset.visits)} "
            f"{row['exam_id']} dice={metrics['dice']:.4f} "
            f"hd95_mm={metrics['hd95_mm']}",
            flush=True,
        )
        del sample, predictor_input, prediction, target

    patients = patient_aggregates(visit_rows)
    metric_names = [
        "dice", "hd95_mm", "lesion_sensitivity", "false_positives",
        "absolute_volume_error_mm3",
    ]
    macro = {}
    for name in metric_names:
        values = [patient[name] for patient in patients if patient[name] is not None]
        macro[name] = float(np.mean(values)) if values else None
        macro[name + "_defined_patients"] = len(values)

    report = {
        "partition": args.partition,
        "task": "enhancing",
        "visit_count": len(visit_rows),
        "patient_count": len(patients),
        "split_sha256": split_sha,
        "manifest_sha256": manifest_sha,
        "model_version": args.model_version,
        "git_sha": args.git_sha,
        "patient_macro": macro,
        "patients": patients,
        "visits": visit_rows,
    }
    (args.output / "metrics.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"completed": True, "partition": args.partition,
                      "patients": len(patients), "visits": len(visit_rows),
                      "patient_macro": macro}, indent=2))


if __name__ == "__main__":
    main()
