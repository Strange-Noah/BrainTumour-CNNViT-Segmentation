"""Receive and verify the official four-visit UCSD-BMETS pilot on cloud Linux."""

import argparse
import hashlib
from html.parser import HTMLParser
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import time
from urllib.parse import urljoin, urlsplit
from urllib.request import urlopen

from audit_visit import audit
from storage import DATASET, connect, put_once, verify

PATIENT = "BMETS0001"
COLLECTION = "/UCSD-BMETS-Longitudinal"
INVENTORY_SHA = "da12d84652774206df710220176c8e92c3bdf40588513c1c12977e444bb571d7"
PILOT_SHA = "2d3fb187783099d98269bdece35d73eab9d970588d11346bc16bb197598446da"
RECEIPT_KEY = f"{DATASET}/manifests/pilots/{PILOT_SHA}.json"
SOURCE_PAGE = "https://www.cancerimagingarchive.net/collection/ucsd-bmets-longitudinal/"


def official_link():
    class Links(HTMLParser):
        def __init__(self):
            super().__init__()
            self.matches = set()

        def handle_starttag(self, tag, attrs):
            href = dict(attrs).get("href")
            if tag == "a" and href:
                candidate = urljoin(SOURCE_PAGE, href)
                url = urlsplit(candidate)
                if url.scheme == "https" and url.netloc == "faspex.cancerimagingarchive.net":
                    self.matches.add(candidate)

    with urlopen(SOURCE_PAGE, timeout=30) as response:
        page = response.read(3_000_001)
    if len(page) > 3_000_000:
        raise ValueError("Unexpected TCIA collection page size.")
    parser = Links()
    parser.feed(page.decode("utf-8"))
    if len(parser.matches) != 1:
        raise ValueError("The official collection page did not identify one Faspex link.")
    return parser.matches.pop()


def aspera(cli, link, *args):
    child_env = {
        name: value for name, value in os.environ.items()
        if not name.startswith(("R2_", "AWS_")) and name != "TCIA_FASPEX_URL"
    }
    for attempt in range(3):
        try:
            result = subprocess.run(
                [cli, "faspex5", "packages", *args, f"--url={link}", "--format=json"],
                capture_output=True, timeout=1200, env=child_env,
            )
        except subprocess.TimeoutExpired:
            raise RuntimeError(f"Aspera {args[0]} timed out.") from None
        if result.returncode == 0:
            break
        diagnostic = (result.stdout + result.stderr).decode(errors="replace").lower()
        indicators = [word for word in (
            "timeout", "timed out", "connection reset", "connection refused", "ssl",
            "certificate", "unauthorized", "forbidden", "rate limit", "503", "502", "504",
            "not found", "expired", "401", "403", "429", "unable to resolve", "error"
        ) if word in diagnostic]
        transient = any(word in indicators for word in (
            "timeout", "timed out", "connection reset", "503", "502", "504", "429", "rate limit"
        ))
        if not transient or attempt == 2:
            raise RuntimeError(
                f"Aspera {args[0]} failed (exit {result.returncode}, indicators={indicators})."
            )
        time.sleep(5 * (attempt + 1))
    if args[0] == "browse":
        return json.loads(result.stdout)


def inventory(cli, link):
    files = {}
    for visit in range(1, 5):
        exam = f"{PATIENT}_{visit:02d}"
        folder = f"{COLLECTION}/{exam}"
        for entry in aspera(cli, link, "browse", folder):
            path = PurePosixPath(entry["path"])
            size = entry["size"]
            if (
                str(path.parent) != folder or entry["type"] not in ("file", "symbolic_link")
                or not re.fullmatch(rf"{exam}-[A-Za-z0-9_]+[.]nii[.]gz", path.name)
                or type(size) is not int or size <= 0 or str(path) in files
            ):
                raise ValueError("Unexpected official inventory entry.")
            files[str(path)] = size
    encoded = json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
    if hashlib.sha256(encoded).hexdigest() != INVENTORY_SHA:
        raise ValueError("Official pilot inventory differs from the verified listing.")
    return files


def ingest(cli, link, stage):
    index = inventory(cli, link)
    if shutil.disk_usage(stage).free < 2_000_000_000:
        raise RuntimeError("The pilot needs 2 GB of free staging space.")
    # Checkpoints and partial files can survive retries on a persistent Linux disk.
    aspera(cli, link, "receive", "--sources=@json:" + json.dumps(sorted(index)),
           f"--to-folder={stage}")
    staged = {}
    for remote, size in index.items():
        name = PurePosixPath(remote).name
        matches = [p for p in stage.rglob(name) if p.is_file() and not p.is_symlink()]
        if len(matches) != 1 or matches[0].stat().st_size != size:
            raise ValueError("Pilot file missing, duplicated, or wrong size.")
        staged[name] = matches[0]

    pairs = []
    for name, path in sorted(staged.items()):
        with path.open("rb") as handle:
            pairs.append((name, hashlib.file_digest(handle, "sha256").hexdigest()))
    if hashlib.sha256(json.dumps(pairs).encode()).hexdigest() != PILOT_SHA:
        raise ValueError("Downloaded pilot differs from the earlier complete transfer.")
    for name, path in staged.items():
        if name.endswith("-seg.nii.gz"):
            result = audit(staged[name.replace("-seg.nii.gz", "-t1c.nii.gz")], path)
            if not result["same_grid"] or not result["integer_labels"] or set(result["labels"]) != {0, 2, 3}:
                raise ValueError("Pilot mask audit failed.")

    client, bucket = connect()
    receipts = []
    for name in sorted(staged):
        key = f"{DATASET}/raw/{name.split('-', 1)[0]}/{name}"
        receipts.append(put_once(client, bucket, key, staged[name]))
    receipt = stage / "pilot-receipt.json"
    receipt.write_text(json.dumps(receipts, sort_keys=True, separators=(",", ":")))
    result = put_once(client, bucket, RECEIPT_KEY, receipt)
    print(json.dumps({"files": len(receipts), "bytes": sum(row["bytes"] for row in receipts),
                      "receipt_sha256": result["sha256"]}))


def readback():
    client, bucket = connect()
    response = client.get_object(Bucket=bucket, Key=RECEIPT_KEY)
    with response["Body"] as stream:
        receipts = json.loads(stream.read(100_000))
    if len(receipts) != 43 or len({r["key"] for r in receipts}) != 43:
        raise ValueError("Invalid pilot receipt.")
    pairs = sorted((PurePosixPath(row["key"]).name, row["sha256"]) for row in receipts)
    if hashlib.sha256(json.dumps(pairs).encode()).hexdigest() != PILOT_SHA:
        raise ValueError("Receipt does not match the independently observed pilot digest.")
    for row in receipts:
        name = PurePosixPath(row["key"]).name
        if row["key"] != f"{DATASET}/raw/{name.split('-', 1)[0]}/{name}":
            raise ValueError("Unexpected pilot object key.")
        verify(client, bucket, row["key"], row["sha256"])
    print(json.dumps({"verified": True, "files": len(receipts), "pilot_sha256": PILOT_SHA}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["ingest", "readback"])
    parser.add_argument("--ascli", default="ascli")
    parser.add_argument("--stage", type=Path)
    args = parser.parse_args()
    if args.mode == "readback":
        readback()
        return
    link = os.environ.get("TCIA_FASPEX_URL") or official_link()
    url = urlsplit(link)
    if url.scheme != "https" or url.netloc != "faspex.cancerimagingarchive.net":
        parser.error("use the official Faspex public download link")
    if args.stage is None:
        parser.error("ingest requires --stage")
    args.stage.mkdir(parents=True, exist_ok=True)
    ingest(args.ascli, link, args.stage.resolve())


if __name__ == "__main__":
    main()
