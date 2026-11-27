"""G0 and L0 on the frozen validation partition, executed by trusted cloud CPU."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import tempfile

import nibabel as nib
from nibabel.processing import resample_from_to
import numpy as np
from scipy import ndimage

from cloud.storage import connect
from research.dataset import Dataset
from research.evaluate import evaluate
from research.geometry import mask_geometry, require_mm

MANIFEST_SHA = "040ec6c523d5beae43a30b64fede2bee226e1a08e8c14b2cca8043b01b9c39f4"
MANIFEST_KEY = f"novo-nordisk/datasets/ucsd-bmets/v1/manifests/{MANIFEST_SHA}/visits.jsonl"
SPLIT = Path(__file__).resolve().parents[2] / "protocol/v1/patients.csv"
PREFIX = "novo-nordisk/experiments/lesionlocator/baseline-v1"


def sequences(visits):
    groups = defaultdict(list)
    seen = set()
    for row in visits:
        if row["exam_id"] in seen:
            raise ValueError("Duplicate examination")
        seen.add(row["exam_id"])
        groups[row["patient_id"]].append(row)
    for rows in groups.values():
        rows.sort(key=lambda r: r["visit_order"])
        order = [row["visit_order"] for row in rows]
        days = [row["relative_days"] for row in rows if row["relative_days"] is not None]
        if len(set(order)) != len(rows) or days != sorted(days):
            raise ValueError("Invalid visit sequence")
    return dict(sorted(groups.items()))


def instances(raw):
    if raw.ndim != 3 or not np.isin(raw, [0, 1, 2, 3, 4]).all():
        raise ValueError("Invalid canonical raw labels")
    fg = raw == 3
    labels, _ = ndimage.label(fg, structure=np.ones((3, 3, 3), dtype=bool))
    if not np.array_equal(labels > 0, fg):
        raise AssertionError("G0 foreground identity")
    return labels


def describe(labels, affine):
    voxel_mm3 = abs(np.linalg.det(affine[:3, :3]))
    rows = []
    for identity, box in enumerate(ndimage.find_objects(labels), 1):
        points = np.argwhere(labels[box] == identity)
        center = points.mean(axis=0) + [b.start for b in box]
        rows.append(dict(instance=identity, voxels=len(points),
                         volume_mm3=float(len(points) * voxel_mm3),
                         centroid_mm=nib.affines.apply_affine(affine, center).tolist(),
                         bbox=[[b.start, b.stop] for b in box]))
    return rows


def propagate(labels, historical_affine, followup):
    # Discrete labels mapped using the native affines, without estimated registration.
    image = nib.Nifti1Image(labels.astype(np.int32), historical_affine)
    moved = np.asarray(resample_from_to(image, followup, order=0).dataobj)
    if not set(np.unique(moved)) <= set(np.unique(labels)) | {0}:
        raise ValueError("L0 introduced a novel label")
    return moved


def load_visit(dataset, row):
    t1c = nib.load(dataset.fetch(row["modalities"]["t1c"]))
    require_mm(t1c)
    seg = nib.load(dataset.fetch(row["segmentation"]))
    warnings = mask_geometry(seg, t1c, row["exam_id"])
    return t1c, instances(np.asarray(seg.dataobj)), warnings


def save_once(client, bucket, key, record):
    data = (json.dumps(record, sort_keys=True, allow_nan=False) + "\n").encode()
    digest = hashlib.sha256(data).hexdigest()
    try:
        client.put_object(Bucket=bucket, Key=key, Body=data, IfNoneMatch="*",
                          Metadata={"sha256": digest})
    except client.exceptions.ClientError as error:
        if error.response["ResponseMetadata"]["HTTPStatusCode"] != 412:
            raise
    with client.get_object(Bucket=bucket, Key=key)["Body"] as body:
        if hashlib.sha256(body.read()).hexdigest() != digest:
            raise ValueError("Private output readback mismatch")


def run(first, count):
    reader, bucket = connect()
    writer, output = connect(output=True)
    if output != "novo-nordisk-pbl-outputs" or bucket != "novo-nordisk-pbl":
        raise ValueError("Canonical input and private output buckets required")
    with tempfile.TemporaryDirectory(prefix="novo-lesionlocator-") as folder:
        root = Path(folder)
        manifest = root / "visits.jsonl"
        reader.download_file(bucket, MANIFEST_KEY, str(manifest))
        ds = Dataset(manifest, SPLIT, root / "images", partition="validation")
        groups = list(sequences(ds.visits).values())
        if (len(groups), len(ds.visits), sum(len(g) - 1 for g in groups)) != (45, 159, 114):
            raise ValueError("Frozen validation coverage")
        selected = groups[first:first + count]
        if len(selected) != count:
            raise ValueError("Requested patient range outside validation queue")
        for index, group in enumerate(selected, first):
            g0, pairs = [], []
            previous = None
            for row in group:
                t1c, labels, warnings = load_visit(ds, row)
                g0.append(dict(exam=row["exam_id"], instances=describe(labels, t1c.affine),
                               shape=t1c.shape, affine=t1c.affine.tolist(),
                               spacing=t1c.header.get_zooms()[:3], warning=warnings))
                if previous is not None:
                    ref, prompt, exam = previous
                    moved = propagate(prompt, ref.affine, t1c)
                    pairs.append(dict(historical=exam, followup=row["exam_id"],
                                      metrics=evaluate(moved > 0, labels > 0, t1c.affine)))
                previous = t1c, labels, row["exam_id"]
            save_once(writer, output, f"{PREFIX}/patient-{index:02}.json",
                      dict(manifest_sha256=MANIFEST_SHA, visit_geometry=g0, direct_l0=pairs))
        print(json.dumps(dict(patients=len(selected), completed=True, private_readback=True)))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--first", type=int, required=True)
    p.add_argument("--count", type=int, required=True)
    args = p.parse_args()
    if not 0 <= args.first < 45 or not 1 <= args.count <= 5:
        p.error("Use bounded CPU batches within the frozen validation cohort")
    run(args.first, args.count)
