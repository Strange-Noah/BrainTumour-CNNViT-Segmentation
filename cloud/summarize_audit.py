"""Collect immutable per-patient audit records into a complete cohort report."""
import argparse
from collections import Counter
from contextlib import closing
import csv
import hashlib
import json
from pathlib import Path

from botocore.exceptions import ClientError
from storage import DATASET, connect, put_once


def read_record(client, bucket, key):
    response = client.get_object(Bucket=bucket, Key=key)
    with closing(response["Body"]) as body:
        content = body.read()
    if hashlib.sha256(content).hexdigest() != response.get("Metadata", {}).get("sha256"):
        raise ValueError("Audit record checksum mismatch")
    return content


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--publish", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    client, bucket = connect()
    base = f"{DATASET}/reports/audit/full-v1"
    pending, files, visits, index = [], [], [], []
    for number in range(300):
        prefix = f"{base}/{number:03d}-{number+1:03d}"
        try:
            summary = json.loads(read_record(client, bucket, f"{prefix}/summary.json"))
        except ClientError as error:
            if error.response["Error"]["Code"] not in ("NoSuchKey", "404"):
                raise
            pending.append(number)
            continue
        patient_files = read_record(client, bucket, f"{prefix}/files.jsonl")
        patient_visits = read_record(client, bucket, f"{prefix}/exams.jsonl")
        files.extend(json.loads(line) for line in patient_files.splitlines())
        visits.extend(json.loads(line) for line in patient_visits.splitlines())
        index.append({"prefix": prefix, "files_sha256": hashlib.sha256(patient_files).hexdigest(),
                      "exams_sha256": hashlib.sha256(patient_visits).hexdigest(),
                      "source_receipt_sha256": summary["source_receipt_sha256"]})
    unknown = [row["key"] for row in files if row.get("spatial_units") == "unknown"]
    quarantine = [row["exam_id"] for row in visits if row["status"] == "FAIL"]
    report = {"patients_audited": 300-len(pending), "pending_patient_indices": pending,
              "examinations": len(visits), "unique_examinations": len({r["exam_id"] for r in visits}),
              "files": len(files), "unique_files": len({r["key"] for r in files}),
              "file_status": dict(Counter(r["status"] for r in files)),
              "exam_status": dict(Counter(r["status"] for r in visits)),
              "unknown_spatial_unit_keys": unknown, "quarantine_examinations": quarantine,
              "warning_reasons": dict(Counter(reason for row in files for reason in row["warnings"])),
              "failure_reasons": dict(Counter(reason for row in files for reason in row["failures"])),
              "empty_mask_examinations": [row["exam_id"] for row in files if row.get("empty_mask")],
              "mask_labels_seen": sorted({label for row in files for label in row.get("label_voxels", {})}),
              "complete": not pending and len(visits) == 1065 and len(files) == 11405
                  and len({r["exam_id"] for r in visits}) == 1065
                  and len({r["key"] for r in files}) == 11405}
    (args.output/"summary.json").write_text(json.dumps(report, indent=2)+"\n")
    (args.output/"index.json").write_text(json.dumps(index, indent=2)+"\n")
    with (args.output/"examination-status.csv").open("w", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["exam_id", "status", "warnings", "failures"])
        for row in sorted(visits, key=lambda r: r["exam_id"]):
            writer.writerow([row["exam_id"], row["status"], len(row["warnings"]), len(row["failures"])])
    if args.publish:
        if not report["complete"]:
            raise ValueError("Publish a final report only when every patient audit is complete")
        for path in args.output.iterdir():
            put_once(client, bucket, f"{base}/final/{path.name}", path)
    print(json.dumps({key:value for key,value in report.items() if not isinstance(value,list)}))


if __name__ == "__main__":
    main()
