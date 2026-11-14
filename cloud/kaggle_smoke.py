"""Run from a private Kaggle session cloned at an immutable GitHub commit."""

import argparse
import json
import re
import shutil
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory

import nibabel as nib
import numpy as np
import torch

from storage import DATASET, PREFIX, connect, put_once, verify


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_id")
    parser.add_argument("input_key")
    parser.add_argument("sha256")
    args = parser.parse_args()
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{5,63}", args.run_id):
        parser.error("invalid run ID")
    if not args.input_key.startswith((f"{DATASET}/raw/", f"{PREFIX}/verification/")):
        parser.error("input must be a raw dataset or verification object")
    if not re.fullmatch(r"[0-9a-f]{64}", args.sha256):
        parser.error("invalid SHA-256")
    repo = Path(__file__).resolve().parent.parent
    sha = subprocess.check_output(["git", "-C", repo, "rev-parse", "HEAD"], text=True).strip()
    if subprocess.check_output(["git", "-C", repo, "status", "--porcelain"]):
        raise RuntimeError("Use a clean checkout for reproducible results.")
    if not torch.cuda.is_available():
        raise RuntimeError("Kaggle GPU accelerator is unavailable.")
    client, bucket = connect()

    with TemporaryDirectory() as folder:
        temporary = Path(folder)
        source = temporary / ("visit.nii.gz" if args.input_key.endswith(".gz") else "visit.nii")
        with source.open("wb") as stream:
            verify(client, bucket, args.input_key, args.sha256, stream)
        image = nib.load(source)
        if len(image.shape) != 3:
            raise ValueError("Expected a 3D MRI.")
        data = image.get_fdata(dtype=np.float32, caching="unchanged")
        if not np.isfinite(data).all():
            raise ValueError("Expected finite voxel values.")
        signal = torch.from_numpy(data).to("cuda")
        # This kernel checks the preprocessing path without training a model.
        normal = (signal - signal.mean()) / signal.std(unbiased=False).clamp_min(1e-6)
        if not torch.isfinite(normal).all().item():
            raise RuntimeError("Preprocessing returned non-finite values.")
        output = temporary / "normalized.nii.gz"
        header = image.header.copy()
        header.set_data_dtype(np.float32)
        nib.save(nib.Nifti1Image(normal.cpu().numpy(), image.affine, header), output)

        prefix = f"{PREFIX}/experiments/smoke/{args.run_id}"
        result = put_once(client, bucket, f"{prefix}/{output.name}", output)
        report = {
            "run_id": args.run_id, "git_sha": sha, "input_sha256": args.sha256,
            "dataset": "ucsd-bmets/v1" if args.input_key.startswith(DATASET) else "synthetic",
            "output": result, "shape": list(image.shape),
            "gpu": torch.cuda.get_device_name(), "gpu_bytes": torch.cuda.get_device_properties(0).total_memory,
            "free_disk_bytes": shutil.disk_usage(folder).free, "torch": torch.__version__,
        }
        report_file = temporary / "report.json"
        report_file.write_text(json.dumps(report, indent=2) + "\n")
        receipt = put_once(client, bucket, f"{prefix}/report.json", report_file)
        print(json.dumps(receipt))
        print(report_file.read_text(), end="")


if __name__ == "__main__":
    main()
