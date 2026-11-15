"""Freeze stratified 210/45/45 patients using deterministic SHA-256 ranking."""
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
import sys


def freeze(metadata, output):
    with Path(metadata).open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    counts = Counter(row["Study ID"].rsplit("_", 1)[0] for row in rows)
    if len(counts) != 300 or len({row["Study ID"] for row in rows}) != len(rows):
        raise ValueError("Expected 300 patients with unique exams")
    # Largest-remainder allocation by visit strata with fixed column totals.
    allocation = {2: (34, 8, 7), 3: (26, 5, 6), 4: (150, 32, 32)}
    if Counter(counts.values()) != Counter({2: 49, 3: 37, 4: 214}):
        raise ValueError("Release strata differ from audited v1")
    assignments = {}
    for visits, quotas in allocation.items():
        patients = sorted((p for p in counts if counts[p] == visits),
                          key=lambda p: hashlib.sha256(("ucsd-bmets-v1-seed42:" + p).encode()).hexdigest())
        start = 0
        for partition, number in zip(("train", "validation", "test"), quotas):
            for patient in patients[start:start + number]:
                assignments[patient] = partition
            start += number
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    path = output / "patients.csv"
    with path.open("x", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["patient_id", "partition"])
        writer.writerows(sorted(assignments.items()))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    spec = {"protocol": "v1", "seed": 42, "ranking": "sha256(ucsd-bmets-v1-seed42:patient_id)",
            "partition_counts": dict(Counter(assignments.values())), "stratum_allocation": allocation,
            "metadata_sha256": hashlib.sha256(Path(metadata).read_bytes()).hexdigest(),
            "split_sha256": digest}
    (output / "split.json").write_text(json.dumps(spec, indent=2) + "\n")
    print(digest)


if __name__ == "__main__":
    freeze(*sys.argv[1:])
