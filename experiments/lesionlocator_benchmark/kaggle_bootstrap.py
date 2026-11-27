"""Kaggle entrypoint template; submission prepends immutable package identity."""
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

UPSTREAM = "ef073baaa0dcba8dd036ec5a29f130db18c9d580"


def run():
    package_paths = [p for p in Path("/kaggle/input").rglob("lesionlocator.json")
                     if sha256(p.read_bytes()).hexdigest() == EXPECTED_PACKAGE]
    if len(package_paths) != 1:
        raise ValueError("Expected the exact private input package")
    package = json.loads(package_paths[0].read_text())
    if package["source_sha"] != SOURCE_SHA:
        raise ValueError("Wrong private package source revision")
    from time import time
    if package["expires"] < time()+package["session_seconds"]+180:
        raise ValueError("Private grants do not cover the intended session")
    with tempfile.TemporaryDirectory(prefix="lesionlocator-", dir="/kaggle/working") as d:
        root = Path(d)
        env = {k: os.environ[k] for k in ("PATH", "LD_LIBRARY_PATH", "CUDA_VISIBLE_DEVICES",
                                         "LANG", "LC_ALL") if k in os.environ}
        env.update(HOME=str(root), MPLBACKEND="Agg", PYTHONUNBUFFERED="1",
                   OMP_NUM_THREADS="2", MKL_NUM_THREADS="2")
        subprocess.run(["git","clone","--quiet","--filter=blob:none","--no-checkout",
                        "https://github.com/Strange-Noah/BrainTumour-CNNViT-Segmentation.git", "project"],
                       cwd=root, env=env, check=True)
        checkout = root/"project"
        subprocess.run(["git","checkout","--quiet","--detach", SOURCE_SHA],
                       cwd=checkout, env=env, check=True)
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--disable-pip-version-check",
                        "git+https://github.com/MIC-DKFZ/LesionLocator.git@"+UPSTREAM],
                       cwd=root, env=env, check=True)
        sys.path.insert(0, str(checkout))
        from experiments.lesionlocator_benchmark.kaggle_worker import fetch
        weights = root/"weights"
        (weights/"fold_0").mkdir(parents=True)
        for name, grant in package["assets"].items():
            if MODE == "gpu" or not name.endswith(".pth"):
                fetch(grant, weights/name)
        gpu_count = 1
        if MODE == "gpu":
            import torch
            gpu_count = torch.cuda.device_count()
            if gpu_count not in (1, 2):
                raise RuntimeError("Expected one or two T4 devices")
        children = []
        try:
            for device in range(gpu_count):
                args = [sys.executable, "-m", "experiments.lesionlocator_benchmark.kaggle_worker",
                        str(package_paths[0]), MODE, str(device), str(weights)]
                children.append(subprocess.Popen(args, cwd=checkout, env=env))
            from time import sleep
            while any(child.poll() is None for child in children):
                if any(child.poll() not in (None, 0) for child in children):
                    raise RuntimeError("Resident worker failed")
                sleep(2)
            if any(child.returncode for child in children):
                raise RuntimeError("Resident worker failed")
        finally:
            for child in children:
                if child.poll() is None:
                    child.terminate()
            for child in children:
                child.wait(timeout=20)


if __name__ == "__main__":
    try:
        run()
    except Exception as exc:
        import traceback
        for frame in traceback.extract_tb(exc.__traceback__):
            print(f"{Path(frame.filename).name}:{frame.lineno} in {frame.name}", file=sys.stderr)
        print(type(exc).__name__, file=sys.stderr)
        sys.exit(1)
