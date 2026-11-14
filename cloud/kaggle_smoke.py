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

from storage import DATASET, PREFIX, connect, put_once, verify


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_id")
    parser.add_argument("input_key")
    parser.add_argument("sha256")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--separate-output", action="store_true")
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
    client, bucket = connect()
    output_client, output_bucket = connect(output=True) if args.separate_output else (client, bucket)
    if args.separate_output and output_bucket == bucket:
        parser.error("the output bucket must differ from the dataset bucket")

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
        runtime = {"device": args.device, "numpy": np.__version__}
        if args.device == "cuda":
            import torch

            if not torch.cuda.is_available():
                raise RuntimeError("Kaggle GPU accelerator is unavailable.")
            signal = torch.from_numpy(data).to("cuda")
            normal = ((signal - signal.mean()) / signal.std(unbiased=False).clamp_min(1e-6)).cpu().numpy()
            runtime.update(
                gpu=torch.cuda.get_device_name(),
                gpu_bytes=torch.cuda.get_device_properties(0).total_memory,
                torch=torch.__version__,
            )
        else:
            normal = (data - data.mean()) / max(data.std(), 1e-6)
        if not np.isfinite(normal).all():
            raise RuntimeError("Preprocessing returned non-finite values.")
        output = temporary / "normalized.nii.gz"
        header = image.header.copy()
        header.set_data_dtype(np.float32)
        nib.save(nib.Nifti1Image(normal, image.affine, header), output)

        prefix = f"{PREFIX}/experiments/smoke/{args.run_id}"
        result = put_once(output_client, output_bucket, f"{prefix}/{output.name}", output)
        report = {
            "run_id": args.run_id, "git_sha": sha, "input_sha256": args.sha256,
            "dataset": "ucsd-bmets/v1" if args.input_key.startswith(DATASET) else "synthetic",
            "output": result, "shape": list(image.shape),
            "free_disk_bytes": shutil.disk_usage(folder).free, **runtime,
        }
        report_file = temporary / "report.json"
        report_file.write_text(json.dumps(report, indent=2) + "\n")
        receipt = put_once(output_client, output_bucket, f"{prefix}/report.json", report_file)
        print(json.dumps(receipt))
        print(report_file.read_text(), end="")


if __name__ == "__main__":
    main()
