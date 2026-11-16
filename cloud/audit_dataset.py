"""Audit bounded UCSD patient groups from canonical R2, preserving source bytes."""
import argparse
import csv
from collections import Counter
from contextlib import closing
import hashlib
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

import nibabel as nib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cloud.storage import DATASET, connect, put_once
from botocore.exceptions import ClientError
from research.geometry import mask_geometry, require_mm, same_grid

MANIFEST_SHA = "040ec6c523d5beae43a30b64fede2bee226e1a08e8c14b2cca8043b01b9c39f4"
SPLIT_SHA = "57f5af55141e16ecb0b85fdda63cc355175d9b35271126784eb22766698b36f0"
AUDIT_VERSION = "full-v1"


def status(record):
    return "FAIL" if record["failures"] else "WARNING" if record["warnings"] else "PASS"


def inspect_file(path, key, reference, exam_id, is_mask):
    record = {"key": key, "exam_id": exam_id, "warnings": [], "failures": []}
    try:
        image = nib.load(path)
        qform, qcode = image.get_qform(coded=True)
        sform, scode = image.get_sform(coded=True)
        record.update(
            shape=list(image.shape), dtype=str(image.get_data_dtype()),
            spatial_units=image.header.get_xyzt_units()[0], temporal_units=image.header.get_xyzt_units()[1],
            spacing=list(map(float, image.header.get_zooms())),
            orientation=list(nib.aff2axcodes(image.affine)),
            effective_affine=image.affine.tolist(), qform_code=int(qcode), sform_code=int(scode),
            qform=qform.tolist() if qform is not None else None,
            sform=sform.tolist() if sform is not None else None,
            matches_T1c_grid=bool(same_grid(image, reference)) if reference is not None else None,
        )
        if not np.isfinite(image.affine).all() or abs(np.linalg.det(image.affine[:3, :3])) <= 0:
            record["failures"].append("invalid_effective_affine")
        if not all(np.isfinite(record["spacing"])) or any(v <= 0 for v in record["spacing"][:3]):
            record["failures"].append("invalid_voxel_spacing")
        if qform is None and sform is None:
            record["warnings"].append("qform_sform_unset_fallback_affine_recorded")
        if qform is not None and not np.isfinite(qform).all():
            record["failures"].append("invalid_qform")
        if sform is not None and not np.isfinite(sform).all():
            record["failures"].append("invalid_sform")
        if qform is not None and sform is not None and not np.allclose(qform, sform, atol=1e-3, rtol=0):
            record["warnings"].append("qform_sform_disagree_effective_affine_recorded")
        if len(image.shape) != 3:
            record["warnings"].append("non_3d_optional_sequence")
        data = image.get_fdata(dtype=np.float32, caching="unchanged")
        record["finite_voxels"] = bool(np.isfinite(data).all())
        if not record["finite_voxels"]:
            record["failures"].append("nonfinite_voxels")
        if is_mask:
            if reference is None:
                record["failures"].append("T1c_reference_unavailable_for_geometry")
            else:
                record["warnings"].extend(mask_geometry(image, reference, exam_id))
            if not np.isin(data, [0, 1, 2, 3, 4]).all():
                record["failures"].append("unexpected_segmentation_labels")
            labels, counts = np.unique(data, return_counts=True)
            record["label_voxels"] = {str(float(k)): int(v) for k, v in zip(labels, counts)}
            record["empty_mask"] = bool(not np.any(data != 0))
            if record["empty_mask"]:
                record["warnings"].append("empty_mask")
        else:
            require_mm(image, "MRI")
            if reference is not None and not record["matches_T1c_grid"]:
                record["warnings"].append("MRI_grid_differs_loader_resampling_required")
    except Exception as error:
        record["failures"].append(type(error).__name__ + ":" + str(error)[:180])
    record["status"] = status(record)
    # Keep failure evidence serializable even when a source header contains NaN.
    def finite_json(value):
        if isinstance(value, dict):
            return {k: finite_json(v) for k, v in value.items()}
        if isinstance(value, list):
            return [finite_json(v) for v in value]
        if isinstance(value, float) and not np.isfinite(value):
            return None
        return value
    return finite_json(record)


def download(client, bucket, key, path):
    response = client.get_object(Bucket=bucket, Key=key)
    expected = response.get("Metadata", {}).get("sha256")
    digest = hashlib.sha256()
    with closing(response["Body"]) as stream, path.open("wb") as output:
        for chunk in stream.iter_chunks(chunk_size=1024 * 1024):
            digest.update(chunk)
            output.write(chunk)
    if not expected or digest.hexdigest() != expected:
        raise ValueError("Missing or mismatched ingestion checksum")
    return expected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.start < 0 or not 1 <= args.count <= 20:
        parser.error("choose a bounded group of 1–20 patients")
    client, bucket = connect()
    response = client.get_object(Bucket=bucket, Key=f"{DATASET}/manifests/{MANIFEST_SHA}/visits.jsonl")
    with closing(response["Body"]) as body:
        content = body.read()
    if hashlib.sha256(content).hexdigest() != MANIFEST_SHA:
        raise ValueError("Canonical manifest mismatch")
    visits = [json.loads(line) for line in content.decode().splitlines()]
    patients = sorted({row["patient_id"] for row in visits})
    if len(patients) != 300 or len(visits) != 1065:
        raise ValueError("Canonical cohort differs")
    split = Path(__file__).resolve().parent.parent / "protocol/v1/patients.csv"
    if hashlib.sha256(split.read_bytes()).hexdigest() != SPLIT_SHA:
        raise ValueError("Frozen split changed")
    prefix = f"{DATASET}/reports/audit/{AUDIT_VERSION}/{args.start:03d}-{args.start+args.count:03d}"
    if args.count == 1:
        # Raw bytes and acquisition receipts are create-once for the version.
        patient = patients[args.start]
        receipt_key = f"{DATASET}/manifests/acquisition/{patient}.json"
        source_head = client.head_object(Bucket=bucket, Key=receipt_key)
        source_sha = source_head.get("Metadata", {}).get("sha256")
        if not source_sha:
            raise ValueError("Patient acquisition receipt must carry SHA-256")
        try:
            previous = client.get_object(Bucket=bucket, Key=f"{prefix}/summary.json")
        except ClientError as error:
            if error.response["Error"]["Code"] not in ("NoSuchKey", "404"):
                raise
        else:
            with closing(previous["Body"]) as body:
                completed = json.load(body)
            if completed.get("source_receipt_sha256") != source_sha:
                raise ValueError("Patient source differs from existing audit")
            print(json.dumps({"resumed": True, "patients": 1, "exams": completed["exams"],
                              "files": completed["files"], "exam_status": completed["exam_status"]}))
            return
    else:
        source_sha = None
    split_rows = list(csv.DictReader(split.read_text().splitlines()))
    partitions = Counter(row["partition"] for row in split_rows)
    if partitions != Counter({"train": 210, "validation": 45, "test": 45}):
        raise ValueError("Frozen partition sizes differ")
    if {row["patient_id"] for row in split_rows} != set(patients) or len(split_rows) != 300:
        raise ValueError("Frozen split patient coverage differs")
    args.output.mkdir(parents=True, exist_ok=True)
    file_path, exam_path = args.output / "files.jsonl", args.output / "exams.jsonl"
    selected = set(patients[args.start:args.start + args.count])
    summary = {"audit_version": AUDIT_VERSION, "patients": len(selected), "exams": 0,
               "files": 0, "file_status": Counter(), "exam_status": Counter(), "unknown_units": [],
               "split_sha256": SPLIT_SHA, "manifest_sha256": MANIFEST_SHA,
               "source_receipt_sha256": source_sha, "split_counts": dict(partitions)}
    with file_path.open("w") as file_output, exam_path.open("w") as exam_output:
        for visit in visits:
            if visit["patient_id"] not in selected:
                continue
            exam = visit["exam_id"]
            prefix = f"{DATASET}/raw/{exam}/"
            expected = {prefix + Path(p).name for p in [*visit["modalities"].values(), visit["segmentation"]]}
            keys = []
            for page in client.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix):
                keys.extend(row["Key"] for row in page.get("Contents", []))
            record = {"exam_id": exam, "patient_id": visit["patient_id"], "warnings": [], "failures": [],
                      "missing_required": sorted(expected - set(keys)), "optional_files": sorted(set(keys) - expected)}
            if record["missing_required"]:
                record["failures"].append("missing_required_files")
            reference_key = f"{DATASET}/raw/{visit['modalities']['t1c']}"
            with TemporaryDirectory() as folder:
                root = Path(folder)
                reference = None
                try:
                    download(client, bucket, reference_key, root / "reference.nii.gz")
                    reference = nib.load(root / "reference.nii.gz")
                    require_mm(reference, "T1c")
                except Exception as error:
                    record["failures"].append("reference_invalid:" + type(error).__name__)
                for key in sorted(keys):
                    if not key.endswith((".nii", ".nii.gz")):
                        record["warnings"].append("unexpected_non_nifti_object")
                        continue
                    path = root / "audit.nii.gz"
                    try:
                        digest = download(client, bucket, key, path)
                        file_record = inspect_file(path, key, reference, exam, key.endswith("-seg.nii.gz"))
                        file_record["sha256"] = digest
                        if key in expected and len(file_record.get("shape", [])) != 3:
                            file_record["failures"].append("required_sequence_not_3d")
                            file_record["status"] = "FAIL"
                    except Exception as error:
                        file_record = {"key": key, "exam_id": exam, "status": "FAIL",
                                       "failures": [type(error).__name__], "warnings": []}
                    summary["files"] += 1
                    summary["file_status"][file_record["status"]] += 1
                    if file_record.get("spatial_units") == "unknown":
                        summary["unknown_units"].append({"exam_id": exam, "key": key})
                    if file_record["status"] == "FAIL":
                        record["failures"].append("file_failed:" + key.rsplit("/", 1)[1])
                    elif file_record["status"] == "WARNING":
                        record["warnings"].append("file_warning:" + key.rsplit("/", 1)[1])
                    file_output.write(json.dumps(file_record, allow_nan=False) + "\n")
                    path.unlink(missing_ok=True)
            record["status"] = status(record)
            summary["exams"] += 1
            summary["exam_status"][record["status"]] += 1
            exam_output.write(json.dumps(record) + "\n")
    summary_file = args.output / "summary.json"
    summary_file.write_text(json.dumps(summary, indent=2) + "\n")
    prefix = f"{DATASET}/reports/audit/{AUDIT_VERSION}/{args.start:03d}-{args.start+args.count:03d}"
    for path in (file_path, exam_path, summary_file):
        put_once(client, bucket, f"{prefix}/{path.name}", path)
    print(json.dumps({k: summary[k] for k in ("patients", "exams", "files", "file_status", "exam_status")}))


if __name__ == "__main__":
    main()
