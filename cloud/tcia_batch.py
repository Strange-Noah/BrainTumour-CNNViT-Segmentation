"""Receive bounded patient batches, publish raw objects, and resume by receipt."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
from tempfile import TemporaryDirectory

from botocore.exceptions import ClientError
import nibabel as nib
import numpy as np

from storage import DATASET, connect, put_once, verify
from tcia_pilot import aspera, official_link, COLLECTION


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ascli", default="ascli")
    parser.add_argument("--stage", type=Path, required=True)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--count", type=int, default=5)
    args = parser.parse_args()
    if args.start < 0 or not 1 <= args.count <= 10:
        parser.error("choose 1 to 10 patients per bounded job")
    import os
    link = os.environ.get("TCIA_FASPEX_URL") or official_link()
    client, bucket = connect()
    rows = aspera(args.ascli, link, "browse", COLLECTION)
    folders = sorted(row["path"] for row in rows if row["type"] == "directory")
    if len(folders) != 1065 or len(set(folders)) != 1065:
        raise ValueError("Unexpected release exam inventory")
    patients = sorted({PurePosixPath(path).name.split("_")[0] for path in folders})
    if len(patients) != 300:
        raise ValueError("Unexpected patient cohort")
    args.stage.mkdir(parents=True, exist_ok=True)
    for patient in patients[args.start:args.start + args.count]:
        receipt_key = f"{DATASET}/manifests/acquisition/{patient}.json"
        try:
            response = client.get_object(Bucket=bucket, Key=receipt_key)
        except ClientError as exc:
            if exc.response["Error"]["Code"] not in ("NoSuchKey", "404"):
                raise
        else:
            with closing(response["Body"]) as body:
                receipt = json.load(body)
            for record in receipt:
                verify(client, bucket, record["key"], record["sha256"])
            print(json.dumps({"patient_complete": True, "files": len(receipt), "resumed": True}))
            continue
        index = {}
        for folder in folders:
            if PurePosixPath(folder).name.split("_")[0] != patient:
                continue
            for row in aspera(args.ascli, link, "browse", folder):
                path = PurePosixPath(row["path"])
                if str(path.parent) != folder or not path.name.endswith(".nii.gz") or row["size"] <= 0:
                    raise ValueError("Unexpected source object")
                index[str(path)] = row["size"]
        if shutil.disk_usage(args.stage).free < sum(index.values()) * 1.2 + 500_000_000:
            raise RuntimeError("Insufficient patient staging capacity")
        with TemporaryDirectory(dir=args.stage) as temp:
            stage = Path(temp)
            aspera(args.ascli, link, "receive", "--sources=@json:" + json.dumps(sorted(index)),
                   "--to-folder=" + str(stage))
            validated = []
            for remote, size in sorted(index.items()):
                name = PurePosixPath(remote).name
                paths = list(stage.rglob(name))
                if len(paths) != 1 or not paths[0].is_file() or paths[0].is_symlink() or paths[0].stat().st_size != size:
                    raise ValueError("Missing or truncated patient object")
                image = nib.load(paths[0])
                if len(image.shape) != 3 or not np.isfinite(image.affine).all():
                    raise ValueError("Unreadable source geometry")
                # Force decompression so truncated gzip streams fail before completion.
                data = image.get_fdata(dtype=np.float32, caching="unchanged")
                if not np.isfinite(data).all():
                    raise ValueError("Nonfinite source voxels")
                key = f"{DATASET}/raw/{name.split('-',1)[0]}/{name}"
                validated.append((key, paths[0]))
            with ThreadPoolExecutor(max_workers=4) as pool:
                receipt = list(pool.map(lambda item: put_once(client, bucket, *item), validated))
            path = stage / "receipt.json"
            path.write_text(json.dumps(receipt, sort_keys=True, separators=(",", ":")))
            put_once(client, bucket, receipt_key, path)
            print(json.dumps({"patient_complete": True, "files": len(receipt), "resumed": False}))


if __name__ == "__main__":
    main()
