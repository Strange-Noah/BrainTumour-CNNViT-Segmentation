"""CPU-only, read-only NIfTI smoke for an S3-compatible object store.

Uses the standard AWS credential chain (profile, environment, or runtime role).
"""

import argparse
from contextlib import closing
import hashlib
import json
import os
from pathlib import Path
import re
from tempfile import TemporaryDirectory
from urllib.parse import urlsplit

import boto3
from botocore.config import Config
import nibabel as nib
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", default=os.environ.get("AWS_ENDPOINT_URL_S3"))
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--key", required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--profile")
    parser.add_argument("--region", default=os.environ.get("AWS_DEFAULT_REGION", "us-east-1"))
    parser.add_argument("--max-bytes", type=int, default=16 * 1024 * 1024)
    args = parser.parse_args()
    if not re.fullmatch(r"[0-9a-f]{64}", args.sha256) or args.max_bytes <= 0:
        parser.error("provide a lowercase SHA-256 and a positive byte limit")
    if args.endpoint:
        url = urlsplit(args.endpoint)
        if url.scheme != "https" or not url.hostname or url.username or url.password or url.query or url.fragment:
            parser.error("use an HTTPS S3 endpoint without credentials or query parameters")
    session = boto3.Session(profile_name=args.profile)
    client = session.client(
        "s3", endpoint_url=args.endpoint, region_name=args.region,
        config=Config(connect_timeout=10, read_timeout=30, retries={"max_attempts": 2},
                      s3={"addressing_style": "path"}),
    )
    with TemporaryDirectory() as folder:
        path = Path(folder) / ("input.nii.gz" if args.key.endswith(".gz") else "input.nii")
        response = client.get_object(Bucket=args.bucket, Key=args.key)
        digest = hashlib.sha256()
        total = 0
        with closing(response["Body"]) as body, path.open("wb") as output:
            for chunk in body.iter_chunks(chunk_size=1024 * 1024):
                total += len(chunk)
                if total > args.max_bytes:
                    raise ValueError("Object exceeds the configured byte limit")
                digest.update(chunk)
                output.write(chunk)
        if digest.hexdigest() != args.sha256:
            raise ValueError("Object SHA-256 differs from the receipt")
        image = nib.load(path)
        if not 0 < np.prod(image.shape, dtype=object) <= 256**3:
            raise ValueError("Unexpected image dimensions")
        if not np.isfinite(image.get_fdata(caching="unchanged")).all():
            raise ValueError("NIfTI contains non-finite voxels")
        print(json.dumps({"verified": True, "bytes": total, "sha256": digest.hexdigest(),
                          "shape": list(image.shape)}))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        # SDK errors may contain endpoint details. Keep notebook and CI logs safe.
        raise SystemExit("Read verification failed. Check endpoint, reader access, key, receipt and file size.") from None
