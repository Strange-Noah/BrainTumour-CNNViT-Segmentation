"""Inspect the verified official fold on cloud CPU with a credential-free child."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

from cloud.storage import connect, verify
from .assets import MEMBERS, PREFIX, UPSTREAM
from .baseline import save_once


def main():
    client, bucket = connect(output=True)
    if bucket != "novo-nordisk-pbl-outputs":
        raise ValueError("Expected private output bucket")
    with client.get_object(Bucket=bucket, Key=f"{PREFIX}/receipt.json")["Body"] as stream:
        asset = json.load(stream)
    if asset["upstream"] != UPSTREAM or asset["record"] != 15174217:
        raise ValueError("Unexpected official asset identity")
    with tempfile.TemporaryDirectory(prefix="novo-ll-checkpoint-") as d:
        root = Path(d)
        for name in MEMBERS:
            receipt = asset["members"][name]
            if receipt["key"] != f"{PREFIX}/{name}":
                raise ValueError("Unexpected asset key")
            dest = root / Path(name).name
            with dest.open("xb") as out:
                verify(client, bucket, receipt["key"], receipt["sha256"], out)
        code = """
import json, torch
c = torch.load("checkpoint_final.pth", map_location="cpu", weights_only=True)
print(json.dumps(dict(configuration=c["init_args"]["configuration"],
                      trainer=c["trainer_name"],
                      parameter_tensors=len(c["network_weights"]),
                      torch=torch.__version__)))
"""
        clean = {"PATH": os.defpath, "HOME": str(root), "CUDA_VISIBLE_DEVICES": "",
                 "PYTHONNOUSERSITE": "1"}
        raw = subprocess.check_output([sys.executable, "-I", "-c", code],
                                      cwd=root, env=clean, text=True, timeout=180)
        data = json.loads(raw)
        if data["trainer"] != "LesionLocatorTrainerTrack":
            raise ValueError("Unexpected official trainer for 175³ geometry")
        data["registration_shape"] = [175, 175, 175]
        plans = json.loads((root/"plans.json").read_text())
        configs = plans["configurations"]
        conf = configs[data["configuration"]]
        # The concrete option takes precedence over inherited fields.
        resolved = dict(conf)
        seen = {data["configuration"]}
        while conf.get("inherits_from"):
            parent = conf["inherits_from"]
            if parent in seen:
                raise ValueError("Configuration inheritance cycle")
            seen.add(parent)
            conf = configs[parent]
            resolved = dict(conf, **resolved)
        with (root/"checkpoint_final.pth").open("rb") as f:
            data["checkpoint_sha256"] = hashlib.file_digest(f, "sha256").hexdigest()
        data["configuration_fields"] = {k: resolved.get(k) for k in (
            "patch_size", "spacing", "normalization_schemes", "resampling_fn_data",
            "resampling_fn_data_kwargs", "resampling_fn_seg", "resampling_fn_seg_kwargs",
            "resampling_fn_probabilities", "resampling_fn_probabilities_kwargs")}
        data["dataset"] = json.loads((root/"dataset.json").read_text())
        data["transpose_forward"] = plans["transpose_forward"]
        data["transpose_backward"] = plans["transpose_backward"]
        save_once(client, bucket, f"{PREFIX}/runtime.json", data)
        print(json.dumps(data, allow_nan=False))


if __name__ == "__main__":
    main()
