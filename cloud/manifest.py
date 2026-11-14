"""Build a deterministic pending manifest from official TCIA TSV metadata."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import re

EXAM = re.compile(r"(?P<patient>BMETS[0-9]{4})_(?P<order>[0-9]{2})")
CORE = ("t1n", "t1c", "t2", "t2f")


def read_tsv(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        if not reader.fieldnames or len(set(reader.fieldnames)) != len(reader.fieldnames):
            raise ValueError(f"Missing or ambiguous TSV headers in {path}")
        for row in reader:
            if None in row or None in row.values():
                raise ValueError(f"Malformed TSV row {reader.line_num} in {path}")
            yield {key: value.strip() for key, value in row.items()}


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def build(mri, clinical, exam_column, patient_column, relative_days_column, treatment_columns):
    clinical_by_patient = {}
    for row in read_tsv(clinical):
        patient = row[patient_column]
        if not re.fullmatch(r"BMETS[0-9]{4}", patient) or patient in clinical_by_patient:
            raise ValueError(f"Invalid or duplicate clinical ID: {patient}")
        clinical_by_patient[patient] = {name: row[name] or None for name in treatment_columns}

    provenance = {"mri_sha256": sha256(mri), "clinical_sha256": sha256(clinical)}
    visits = {}
    for row in read_tsv(mri):
        exam = row[exam_column]
        match = EXAM.fullmatch(exam)
        if not match or exam in visits or int(match["order"]) < 1:
            raise ValueError(f"Invalid or duplicate examination ID: {exam}")
        patient, order = match["patient"], int(match["order"])
        if patient not in clinical_by_patient:
            raise ValueError(f"No matching clinical row for {exam}")
        days = row[relative_days_column] if relative_days_column else ""
        visits[exam] = {
            "dataset_version": "1",
            "patient_id": patient,
            "exam_id": exam,
            "visit_order": order,
            "relative_days": int(days) if days else None,
            "timing_precision": "day" if days else "ordinal",
            "modalities": {modality: f"{exam}/{exam}-{modality}.nii.gz" for modality in CORE},
            "segmentation": f"{exam}/{exam}-seg.nii.gz",
            "treatment": clinical_by_patient[patient],
            "spatial_reference": {"dataset_reference": f"{patient}_01-t1c.nii.gz", "header_verified": False},
            "availability": "expected",
            "integrity": "pending",
            "provenance": provenance,
        }
    if not visits:
        raise ValueError("MRI metadata has no visits")
    return [visits[key] for key in sorted(visits)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mri_tsv", type=Path)
    parser.add_argument("clinical_tsv", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--exam-column", required=True)
    parser.add_argument("--patient-column", required=True)
    parser.add_argument("--relative-days-column")
    parser.add_argument("--treatment-column", action="append", default=[])
    args = parser.parse_args()
    visits = build(
        args.mri_tsv, args.clinical_tsv, args.exam_column, args.patient_column,
        args.relative_days_column, args.treatment_column,
    )
    with args.output.open("x") as output:
        for row in visits:
            print(json.dumps(row, sort_keys=True), file=output)
    print(f"{len(visits)} visits, {len({row['patient_id'] for row in visits})} patients")


if __name__ == "__main__":
    main()
