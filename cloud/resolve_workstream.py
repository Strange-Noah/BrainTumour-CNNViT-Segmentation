"""Resolve an explicitly approved immutable workstream source."""

import argparse
import json
import os
from pathlib import Path
import re
import subprocess


def resolve(name, source, actor):
    registry = json.loads(
        Path("cloud/workstreams.json").read_text()
    )
    record = registry[name]
    sha = record["source_sha"]
    branch = record["branch"]

    if source != sha or not re.fullmatch(r"[0-9a-f]{40}", sha):
        raise PermissionError("Source is not explicitly approved")

    if actor != record["authorized_actor"]:
        raise PermissionError("Unauthorized workstream actor")

    if not re.fullmatch(r"[A-Za-z0-9_/-]+", branch):
        raise ValueError("Invalid registered branch")

    subprocess.run(
        ["git", "merge-base", "--is-ancestor",
         sha, f"origin/{branch}"],
        check=True,
    )
    return record


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("name")
    p.add_argument("source")
    p.add_argument("actor")
    a = p.parse_args()
    print(json.dumps(resolve(a.name, a.source, a.actor)))
