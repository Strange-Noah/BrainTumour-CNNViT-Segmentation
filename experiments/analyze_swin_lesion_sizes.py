"""Supplementary validation analysis by reference lesion volume (mm^3)."""
import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.optimize import linear_sum_assignment

from research.dataset import Dataset as BMETSDataset
from research.evaluate import components


def parse_edges(value):
    try:
        edges = [float(item) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Use comma-separated positive mm^3 values") from exc
    if not edges or any(item <= 0 for item in edges) or edges != sorted(set(edges)):
        raise argparse.ArgumentTypeError("Edges must be unique, positive, and ascending")
    return edges


def size_bin(volume, edges):
    for i, edge in enumerate(edges):
        if volume < edge:
            return i
    return len(edges)


def one_visit(prediction, truth, affine, edges):
    pred_components, pred_records = components(prediction, affine)
    true_components, true_records = components(truth, affine)
    # Use a signed or floating type: the assignment below minimizes -overlap.
    # Negating uint8 wraps 1 to 255, which incorrectly favors unmatched pairs.
    overlap = np.zeros((len(true_records), len(pred_records)), dtype=np.int8)
    intersection = np.asarray(prediction, bool) & np.asarray(truth, bool)
    pairs = np.stack([true_components[intersection], pred_components[intersection]], axis=1)
    pairs, counts = np.unique(pairs, axis=0, return_counts=True)
    for (truth_id, pred_id), count in zip(pairs, counts):
        if truth_id and pred_id and count:
            overlap[truth_id - 1, pred_id - 1] = 1
    matched_truth = set()
    if overlap.size:
        rows, columns = linear_sum_assignment(-overlap)
        matched_truth = {int(row + 1) for row, column in zip(rows, columns) if overlap[row, column]}

    totals = np.zeros(len(edges) + 1, dtype=np.int64)
    matched = np.zeros_like(totals)
    for record in true_records:
        bin_index = size_bin(record["volume_mm3"], edges)
        totals[bin_index] += 1
        if record["candidate_id"] in matched_truth:
            matched[bin_index] += 1
    return totals, matched, len(pred_records) - len(matched_truth)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True,
                        help="Partition output directory containing visits/<exam_id>/seg.nii.gz")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--partition", choices=("validation", "test"), default="validation")
    parser.add_argument("--bins-mm3", type=parse_edges, required=True,
                        help="Locked ascending lesion-volume edges, for example 27,200")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Output exists; choose a new path: {args.output}")

    split = Path("protocol/v1/patients.csv")
    dataset = BMETSDataset(args.manifest, split, args.cache / "raw", args.partition)
    names = [f"<{edge:g}" if i == 0 else f"{args.bins_mm3[i-1]:g}-{edge:g}"
             for i, edge in enumerate(args.bins_mm3)] + [f">={args.bins_mm3[-1]:g}"]
    per_patient = defaultdict(lambda: {
        "visits": 0,
        "false_positives": [],
        "lesions": np.zeros(len(names), dtype=np.int64),
        "matched": np.zeros(len(names), dtype=np.int64),
        "visits_with_bin_lesions": np.zeros(len(names), dtype=np.int64),
    })

    for index, visit in enumerate(dataset.visits, start=1):
        sample = dataset.load(index - 1, "enhancing")
        prediction_path = args.predictions / "visits" / visit["exam_id"] / "seg.nii.gz"
        predicted_img = nib.load(prediction_path)
        if predicted_img.shape != sample["reference"].shape or not np.allclose(
            predicted_img.affine, sample["reference"].affine, atol=1e-4, rtol=0
        ):
            raise ValueError(f"Prediction is not in original T1c geometry: {visit['exam_id']}")
        prediction = predicted_img.get_fdata(dtype=np.float32) > 0.5
        truth = np.asarray(sample["target"], dtype=bool)
        totals, matched, false_positives = one_visit(
            prediction, truth, sample["reference"].affine, args.bins_mm3
        )
        record = per_patient[visit["patient_id"]]
        record["visits"] += 1
        record["false_positives"].append(int(false_positives))
        record["lesions"] += totals
        record["matched"] += matched
        record["visits_with_bin_lesions"] += (totals > 0).astype(np.int64)
        print(f"{args.partition} size-analysis {index}/{len(dataset.visits)} {visit['exam_id']}", flush=True)

    patient_rows = []
    for patient_id, record in sorted(per_patient.items()):
        sensitivity = [
            float(record["matched"][i] / record["lesions"][i]) if record["lesions"][i] else None
            for i in range(len(names))
        ]
        patient_rows.append({
            "patient_id": patient_id,
            "visits": record["visits"],
            "false_positives_mean_per_visit": float(np.mean(record["false_positives"])),
            "bins": [
                {"range_mm3": names[i], "lesions": int(record["lesions"][i]),
                 "matched": int(record["matched"][i]),
                 "sensitivity": sensitivity[i]}
                for i in range(len(names))
            ],
        })

    bins = []
    for bin_index, name in enumerate(names):
        values = [row["bins"][bin_index]["sensitivity"] for row in patient_rows
                  if row["bins"][bin_index]["sensitivity"] is not None]
        total_lesions = sum(row["bins"][bin_index]["lesions"] for row in patient_rows)
        matched_lesions = sum(row["bins"][bin_index]["matched"] for row in patient_rows)
        bins.append({
            "range_mm3": name,
            "lesions": total_lesions,
            "matched": matched_lesions,
            "pooled_sensitivity": matched_lesions / total_lesions if total_lesions else None,
            "patient_macro_sensitivity": float(np.mean(values)) if values else None,
            "patients_with_bin_lesions": len(values),
        })

    report = {
        "partition": args.partition,
        "visit_count": len(dataset.visits),
        "patient_count": len(patient_rows),
        "bins_mm3": args.bins_mm3,
        "split_sha256": hashlib.sha256(split.read_bytes()).hexdigest(),
        "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        "matching": "26-connected components; one-to-one match on any positive voxel overlap",
        "bins": bins,
        "patients": patient_rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "bins": bins}, indent=2))


if __name__ == "__main__":
    main()
