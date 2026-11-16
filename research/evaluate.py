"""Protocol v1 binary metrics, components, matching and prediction export."""
import hashlib
import json
from pathlib import Path

from research.geometry import require_mm
import nibabel as nib
import numpy as np
from scipy import ndimage
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree

TASKS = {"enhancing": (3,), "core": (1, 3)}
STRUCTURE = np.ones((3, 3, 3), dtype=bool)


def target(raw, task="enhancing"):
    return np.isin(raw, TASKS[task])


def components(mask, affine):
    labels, count = ndimage.label(mask, structure=STRUCTURE)
    voxel_mm3 = abs(np.linalg.det(affine[:3, :3]))
    records = []
    for index, region in enumerate(ndimage.find_objects(labels), 1):
        coordinates = np.argwhere(labels[region] == index)
        coordinates += np.array([s.start for s in region])
        center = nib.affines.apply_affine(affine, coordinates.mean(axis=0))
        records.append({"candidate_id": index, "volume_mm3": len(coordinates) * voxel_mm3,
                        "centroid_world_mm": center.tolist()})
    return labels, records


def surface(mask, affine):
    boundary = mask & ~ndimage.binary_erosion(mask, structure=STRUCTURE, border_value=0)
    return nib.affines.apply_affine(affine, np.argwhere(boundary))


def evaluate(prediction, truth, affine):
    prediction, truth = np.asarray(prediction, bool), np.asarray(truth, bool)
    if prediction.shape != truth.shape or prediction.ndim != 3:
        raise ValueError("Expected matching 3D masks")
    if not np.isfinite(affine).all() or abs(np.linalg.det(affine[:3, :3])) <= 0:
        raise ValueError("Invalid reference geometry")
    p, candidates = components(prediction, affine)
    t, lesions = components(truth, affine)
    overlap = np.zeros((len(lesions), len(candidates)), dtype=np.int64)
    # A maximum-cardinality one-to-one match uses any positive voxel overlap.
    intersection = truth & prediction
    pairs = np.stack([t[intersection], p[intersection]], axis=1)
    pairs, counts = np.unique(pairs, axis=0, return_counts=True)
    for (a, b), count in zip(pairs, counts):
        if a and b:
            overlap[a - 1, b - 1] = count
    matched = 0
    if overlap.size:
        rows, columns = linear_sum_assignment(-(overlap > 0).astype(int))
        matched = int((overlap[rows, columns] > 0).sum())
    p_size, t_size = int(prediction.sum()), int(truth.sum())
    if p_size and t_size:
        a, b = surface(prediction, affine), surface(truth, affine)
        hd95 = float(max(np.percentile(cKDTree(a).query(b)[0], 95),
                         np.percentile(cKDTree(b).query(a)[0], 95)))
    else:
        hd95 = 0.0 if p_size == t_size else None
    signed_error = (p_size - t_size) * abs(np.linalg.det(affine[:3, :3]))
    return {"dice": float(2 * (prediction & truth).sum() / (p_size + t_size)) if p_size + t_size else 1.0,
            "hd95_mm": hd95, "hd95_defined": hd95 is not None,
            "lesion_sensitivity": matched / len(lesions) if lesions else None,
            "false_positives": len(candidates) - matched,
            "truth_lesions": len(lesions), "matched_lesions": matched,
            "signed_volume_error_mm3": signed_error, "absolute_volume_error_mm3": abs(signed_error)}


def patient_aggregates(visits):
    groups = {}
    for row in visits:
        groups.setdefault(row["patient_id"], []).append(row)
    result = []
    for patient, rows in sorted(groups.items()):
        record = {"patient_id": patient, "visits": len(rows)}
        for metric in ("dice", "hd95_mm", "lesion_sensitivity", "false_positives", "absolute_volume_error_mm3"):
            values = [row[metric] for row in rows if row[metric] is not None]
            record[metric] = float(np.mean(values)) if values else None
            record[metric + "_defined_visits"] = len(values)
        result.append(record)
    return result


def export(mask, reference, destination, metadata):
    required = {"patient_id", "exam_id", "model_version", "split_sha256", "task", "git_sha"}
    if not required <= metadata.keys() or metadata["task"] not in TASKS:
        raise ValueError("Missing prediction provenance")
    mask = np.asarray(mask)
    if mask.shape != reference.shape or not np.isin(mask, [0, 1]).all():
        raise ValueError("Restore a binary task prediction to original T1c geometry before export")
    require_mm(reference, "prediction export reference")
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    header = reference.header.copy()
    header.set_data_dtype(np.uint8)
    nib.save(nib.Nifti1Image(mask.astype(np.uint8), reference.affine, header), destination / "seg.nii.gz")
    _, candidates = components(mask, reference.affine)
    record = {**metadata, "raw_positive_labels": TASKS[metadata["task"]],
              "prediction_labels": {"0": "background", "1": metadata["task"]},
              "candidates": candidates, "candidate_identity_scope": "visit",
              "prediction_sha256": hashlib.sha256((destination / "seg.nii.gz").read_bytes()).hexdigest(),
              "reference_shape": list(reference.shape), "reference_affine": reference.affine.tolist(),
              "spatial_units": "mm"}
    (destination / "prediction.json").write_text(json.dumps(record, indent=2) + "\n")
