"""Persist a synthetic NIfTI, then recover it in an independent runtime."""

import argparse
import json
from pathlib import Path
import re
from tempfile import TemporaryDirectory

import nibabel as nib
import numpy as np

from storage import PREFIX, connect, put_once, verify


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["write", "read"])
    parser.add_argument("run_id")
    parser.add_argument("--sha256")
    args = parser.parse_args()
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{5,63}", args.run_id):
        parser.error("use a unique lowercase run ID with letters, digits and hyphens")
    if args.mode == "read" and not re.fullmatch(r"[0-9a-f]{64}", args.sha256 or ""):
        parser.error("read requires --sha256 from the write result")
    client, bucket = connect()
    key = f"{PREFIX}/verification/{args.run_id}/synthetic.nii"
    expected = np.arange(64, dtype=np.int16).reshape(4, 4, 4)
    with TemporaryDirectory() as temp:
        path = Path(temp) / "synthetic.nii"
        if args.mode == "write":
            image = nib.Nifti1Image(expected, np.eye(4))
            image.header.set_xyzt_units("mm")
            nib.save(image, path)
            print(json.dumps(put_once(client, bucket, key, path)))
        else:
            with path.open("wb") as handle:
                verify(client, bucket, key, args.sha256, handle)
            image = nib.load(path)
            np.testing.assert_array_equal(np.asarray(image.dataobj), expected)
            np.testing.assert_array_equal(image.affine, np.eye(4))
            print(json.dumps({"key": key, "sha256": args.sha256, "verified": True}))


if __name__ == "__main__":
    main()
