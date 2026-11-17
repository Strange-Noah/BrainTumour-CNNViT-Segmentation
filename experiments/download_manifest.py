"""Download and verify the canonical UCSD-BMETS v1 manifest from R2."""
import argparse
import hashlib
from pathlib import Path

from cloud.storage import DATASET, connect

MANIFEST_SHA256 = "040ec6c523d5beae43a30b64fede2bee226e1a08e8c14b2cca8043b01b9c39f4"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    client, bucket = connect()
    key = f"{DATASET}/manifests/{MANIFEST_SHA256}/visits.jsonl"
    client.download_file(bucket, key, str(args.output))

    digest = hashlib.sha256(args.output.read_bytes()).hexdigest()
    if digest != MANIFEST_SHA256:
        args.output.unlink(missing_ok=True)
        raise SystemExit(f"Manifest checksum mismatch: expected {MANIFEST_SHA256}, got {digest}")
    print(f"Verified canonical manifest: {args.output}")
    print(f"SHA-256: {digest}")


if __name__ == "__main__":
    main()
