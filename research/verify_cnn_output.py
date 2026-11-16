"""Independently recover a persisted CNN pilot checkpoint and predictions."""
import argparse
from contextlib import closing
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import nibabel as nib
import numpy as np

from cloud.storage import PREFIX, connect, verify


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_id")
    args = parser.parse_args()
    if not args.run_id.isdecimal():
        parser.error("use the numerical experiment GitHub run ID")
    client, bucket = connect(output=True)
    prefix = f"{PREFIX}/experiments/cnn-vit/ucsd-head-pilot-{args.run_id}"
    response = client.get_object(Bucket=bucket, Key=f"{prefix}/report.json")
    with closing(response["Body"]) as body:
        data = body.read()
    report_sha = hashlib.sha256(data).hexdigest()
    if report_sha != response["Metadata"]["sha256"]:
        raise ValueError("Report checksum mismatch")
    report = json.loads(data)
    receipts = report["persisted"]
    if len(receipts) != 3 or len({r["key"] for r in receipts}) != 3:
        raise ValueError("Expected checkpoint, prediction and prediction metadata")
    with TemporaryDirectory() as name:
        folder = Path(name)
        recovered = {}
        for row in receipts:
            if not row["key"].startswith(prefix + "/"):
                raise ValueError("Unexpected output key")
            path = folder / Path(row["key"]).name
            with path.open("wb") as output:
                verify(client, bucket, row["key"], row["sha256"], output)
            if path.stat().st_size != row["bytes"]:
                raise ValueError("Output size mismatch")
            recovered[path.name] = path
        image = nib.load(recovered["seg.nii.gz"])
        metadata = json.loads(recovered["prediction.json"].read_text())
        if image.header.get_xyzt_units()[0] != "mm" or not np.isin(image.get_fdata(), [0, 1]).all():
            raise ValueError("Invalid binary prediction or units")
        if list(image.shape) != metadata["reference_shape"] or not np.allclose(image.affine, metadata["reference_affine"], atol=1e-4, rtol=0):
            raise ValueError("Prediction differs from its T1c reference geometry")
        checkpoint = next(r for r in receipts if r["key"].endswith(".pt"))
        print(json.dumps({"verified": True, "run_id": args.run_id, "artifacts": 3,
                          "checkpoint_bytes": checkpoint["bytes"],
                          "checkpoint_sha256": checkpoint["sha256"],
                          "report_sha256": report_sha, "git_sha": report["git_sha"]}))


if __name__ == "__main__":
    main()
