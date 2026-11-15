"""Verify and publish explicitly inventoried TCIA files from cloud Linux staging."""

import argparse
import json
from pathlib import Path

from storage import DATASET, connect, key_under, put_once


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", type=Path)
    parser.add_argument("inventory", type=Path, help="JSON with official relative path to expected byte count")
    parser.add_argument("receipt", type=Path, help="New JSONL receipt path")
    parser.add_argument("--area", choices=["raw", "metadata"], default="raw")
    args = parser.parse_args()
    root = args.stage.resolve(strict=True)
    inventory = json.loads(args.inventory.read_text())
    prefix = f"{DATASET}/{args.area}"
    files = {}
    if not isinstance(inventory, dict) or not inventory:
        raise ValueError("Expected a nonempty inventory.")
    for name, expected_bytes in sorted(inventory.items()):
        key = key_under(prefix, name)
        path = (root / name).resolve(strict=True)
        if (
            not path.is_relative_to(root) or not path.is_file() or key in files
            or type(expected_bytes) is not int or expected_bytes <= 0
            or path.stat().st_size != expected_bytes
        ):
            raise ValueError(f"Staging mismatch: {name}")
        files[key] = path

    client, bucket = connect()
    with args.receipt.open("x") as receipt:
        for key, path in files.items():
            receipt.write(json.dumps(put_once(client, bucket, key, path)) + "\n")
            receipt.flush()


if __name__ == "__main__":
    main()
