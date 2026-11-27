"""G0 and L0 on the frozen validation partition, executed by trusted cloud CPU."""
import argparse
from collections import defaultdict
import hashlib
import json
import os
import re
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
PREFIX = "novo-nordisk/experiments/lesionlocator"
CONFIG = {"condition": "L0", "native": "T1c", "target": 3,
          "connectivity": 26, "interpolation_order": 0,
          "identity_reference": "same-grid conservative unique mutual overlap of native GT only"}



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


def overlap_counts(a, b):
    common = (a > 0) & (b > 0)
    pairs, counts = np.unique(np.stack((a[common], b[common]), axis=1),
                              axis=0, return_counts=True)
    result = defaultdict(dict)
    for (x, y), count in zip(pairs, counts):
        result[int(x)][int(y)] = int(count)
    return result


def instance_scores(prompt, moved, truth, source, followup):
    historical = describe(prompt, source.affine)
    # Preserve original label values even when their label set has gaps.
    labels_present = np.unique(moved)
    propagated = {}
    for ident, box in enumerate(ndimage.find_objects(moved), 1):
        if box is None:
            continue
        points = np.argwhere(moved[box] == ident)
        center = nib.affines.apply_affine(followup.affine, points.mean(axis=0) + [b.start for b in box])
        propagated[ident] = dict(voxels=len(points), centroid_mm=center.tolist(),
            volume_mm3=float(len(points)*abs(np.linalg.det(followup.affine[:3, :3]))))
    assert set(propagated) == set(labels_present) - {0}
    observed = overlap_counts(moved, truth)
    same_grid = prompt.shape == truth.shape and np.allclose(source.affine, followup.affine, rtol=0, atol=1e-4)
    # Nominate only unambiguous same-grid GT overlap proxies. These are
    # independent of any model, but favor spatial alignment and are not identity GT.
    proxy = overlap_counts(prompt, truth) if same_grid else {}
    reverse = overlap_counts(truth, prompt) if same_grid else {}
    target = {r["instance"]: r for r in describe(truth, followup.affine)}
    result = []
    for h in historical:
        ident = h["instance"]
        p = propagated.get(ident, dict(voxels=0, volume_mm3=0., centroid_mm=None))
        displacement = float(np.linalg.norm(np.subtract(p["centroid_mm"], h["centroid_mm"]))) if p["voxels"] else None
        entry = dict(instance_id=ident, historical=h, propagated=p, exists=bool(p["voxels"]),
                     volume_change_mm3=p["volume_mm3"]-h["volume_mm3"],
                     centroid_displacement_mm=displacement, overlap_voxels=observed.get(ident, {}))
        candidates = proxy.get(ident, {})
        ref = next(iter(candidates)) if len(candidates) == 1 else None
        if ref is not None and len(reverse[ref]) == 1:
            t = target[ref]
            entry["reference_proxy"] = dict(target_id=ref,
                dice=2*observed.get(ident, {}).get(ref, 0)/(p["voxels"]+t["voxels"]),
                centroid_error_mm=float(np.linalg.norm(np.subtract(p["centroid_mm"], t["centroid_mm"]))) if p["voxels"] else None,
                volume_error_mm3=p["volume_mm3"]-t["volume_mm3"])
        result.append(entry)
    return result


def load_visit(dataset, row):
    ds_image_path = dataset.fetch(row["modalities"]["t1c"])
    t1c = nib.load(ds_image_path)
    require_mm(t1c)
    ds_label_path = dataset.fetch(row["segmentation"])
    seg = nib.load(ds_label_path)
    warnings = mask_geometry(seg, t1c, row["exam_id"])
    raw = np.asarray(seg.dataobj)
    labels = instances(raw)
    def digest(path):
        with path.open("rb") as stream:
            return hashlib.file_digest(stream, "sha256").hexdigest()
    return t1c, labels, warnings, dict(
        t1c_path=row["modalities"]["t1c"], labels_path=row["segmentation"],
        t1c_sha256=digest(ds_image_path), label_sha256=digest(ds_label_path),
        native_shape=t1c.shape, native_affine=t1c.affine.tolist())


def json_scalar(value):
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Unsupported record value: {type(value).__name__}")


def save_once(client, bucket, key, record):
    data = (json.dumps(record, sort_keys=True, allow_nan=False, default=json_scalar) + "\n").encode()
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
    source_sha = os.environ["NOVO_SOURCE_SHA"]
    if not re.fullmatch(r"[0-9a-f]{40}", source_sha):
        raise ValueError("Expected immutable trusted source")
    config_sha = hashlib.sha256(json.dumps(CONFIG, sort_keys=True).encode()).hexdigest()
    prefix = f"{PREFIX}/{MANIFEST_SHA}/{source_sha}/{config_sha}"
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
        all_metrics = []
        successful = 0
        for index, group in enumerate(selected, first):
            g0, pairs = [], []
            previous = None
            for row in group:
                t1c, labels, warnings, origin = load_visit(ds, row)
                g0.append(dict(exam=row["exam_id"], instances=describe(labels, t1c.affine),
                               shape=t1c.shape, affine=t1c.affine.tolist(),
                               spacing=t1c.header.get_zooms()[:3], warning=warnings, source=origin))
                if previous is not None:
                    ref, prompt, exam = previous
                    moved = propagate(prompt, ref.affine, t1c)
                    pairs.append(dict(historical=exam, followup=row["exam_id"],
                                      metrics=evaluate(moved > 0, labels > 0, t1c.affine),
                                      instances=instance_scores(prompt, moved, labels, ref, t1c)))
                previous = t1c, labels, row["exam_id"]
            save_once(writer, output, f"{prefix}/patient-{index:02}.json",
                      dict(manifest_sha256=MANIFEST_SHA, source_sha=source_sha, config=CONFIG,
                           config_sha256=config_sha, visit_geometry=g0, direct_l0=pairs))
            successful += len(g0)
            all_metrics.extend(pair["metrics"] for pair in pairs)
        print(json.dumps(dict(patients=len(selected), visits=successful, pairs=len(all_metrics),
                             dice_sum=sum(r["dice"] for r in all_metrics),
                             matched=sum(r["matched_lesions"] for r in all_metrics),
                             truth=sum(r["truth_lesions"] for r in all_metrics),
                             fp=sum(r["false_positives"] for r in all_metrics),
                             completed=True, private_readback=True)))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--first", type=int, required=True)
    p.add_argument("--count", type=int, required=True)
    args = p.parse_args()
    if not 0 <= args.first < 45 or not 1 <= args.count <= 5:
        p.error("Use bounded CPU batches within the frozen validation cohort")
    run(args.first, args.count)
