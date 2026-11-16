"""Reconcile acquisition receipts, canonical examinations and R2 object listing."""
import argparse
from contextlib import closing
import hashlib
import json
from pathlib import Path

from botocore.exceptions import ClientError
from storage import DATASET, connect, put_once

MANIFEST_SHA = "040ec6c523d5beae43a30b64fede2bee226e1a08e8c14b2cca8043b01b9c39f4"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--publish-key")
    args = parser.parse_args()
    client, bucket = connect()
    response = client.get_object(Bucket=bucket, Key=f"{DATASET}/manifests/{MANIFEST_SHA}/visits.jsonl")
    with closing(response["Body"]) as body:
        data = body.read()
    if hashlib.sha256(data).hexdigest() != MANIFEST_SHA:
        raise ValueError("Manifest checksum differs")
    visits = [json.loads(line) for line in data.splitlines()]
    patients = sorted({row["patient_id"] for row in visits})
    expected = {}
    missing_patients = []
    invalid = []
    for patient in patients:
        try:
            response = client.get_object(Bucket=bucket, Key=f"{DATASET}/manifests/acquisition/{patient}.json")
        except ClientError as error:
            if error.response["Error"]["Code"] not in ("NoSuchKey", "404"):
                raise
            missing_patients.append(patient)
            continue
        with closing(response["Body"]) as body:
            receipt = json.load(body)
        for row in receipt:
            key = row["key"]
            if key in expected or not key.startswith(f"{DATASET}/raw/{patient}_"):
                invalid.append(key)
            expected[key] = row
    actual = {}
    for page in client.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=f"{DATASET}/raw/"):
        for row in page.get("Contents", []):
            actual[row["Key"]] = row["Size"]
    absent = []
    for row in visits:
        for path in [*row["modalities"].values(), row["segmentation"]]:
            key = f"{DATASET}/raw/{path}"
            if key not in expected:
                absent.append(key)
    missing_objects = sorted(set(expected) - set(actual))
    unexpected = sorted(set(actual) - set(expected))
    wrong_sizes = sorted(key for key in expected.keys() & actual.keys() if expected[key]["bytes"] != actual[key])
    report = {
        "manifest_sha256": MANIFEST_SHA, "patients": len(patients), "examinations": len(visits),
        "completed_patients": len(patients) - len(missing_patients), "objects": len(actual),
        "bytes": sum(actual.values()), "receipt_objects": len(expected),
        "missing_patients": missing_patients, "missing_required": absent,
        "missing_objects": missing_objects, "unexpected_objects": unexpected,
        "wrong_sizes": wrong_sizes, "invalid_receipts": invalid,
        "complete": not any((missing_patients, absent, missing_objects, unexpected, wrong_sizes, invalid)),
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    if args.publish_key:
        put_once(client, bucket, args.publish_key, args.output)
    print(json.dumps({key: value for key, value in report.items() if not isinstance(value, list)}))


if __name__ == "__main__":
    main()
