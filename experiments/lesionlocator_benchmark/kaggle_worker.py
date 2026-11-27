"""Credential-free worker. Temporary grants bind individual inputs and output slots."""
from concurrent.futures import ThreadPoolExecutor
from hashlib import file_digest, sha256
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
from urllib.request import Request, urlopen


def fetch(grant, destination):
    with urlopen(grant["url"], timeout=120) as response, destination.open("wb") as writer:
        shutil.copyfileobj(response, writer, 1024*1024)
    with destination.open("rb") as reader:
        if file_digest(reader, "sha256").hexdigest() != grant["sha256"]:
            raise ValueError("Private input digest mismatch")


def publish(folder, grant):
    blob = folder.parent/"result.tar.gz"
    with tarfile.open(blob, "w:gz", compresslevel=2) as archive:
        for path in sorted(folder.rglob("*")):
            if path.is_file():
                archive.add(path, arcname=path.relative_to(folder))
    with blob.open("rb") as reader:
        digest = file_digest(reader, "sha256").hexdigest()
        reader.seek(0)
        request = Request(grant["put"], reader, method="PUT", headers={
            "If-None-Match": "*", "Content-Length": str(blob.stat().st_size)})
        with urlopen(request, timeout=180) as response:
            if response.status != 200:
                raise RuntimeError("Private result persistence failed")
    with urlopen(grant["get"], timeout=180) as response:
        check = sha256()
        for block in iter(lambda: response.read(1024*1024), b""):
            check.update(block)
    if digest != check.hexdigest():
        raise ValueError("Independent HTTP result readback mismatch")
    receipt = json.dumps(dict(sha256=digest, bytes=blob.stat().st_size)).encode()
    with urlopen(Request(grant["receipt"], receipt, method="PUT",
                         headers={"If-None-Match": "*"}), timeout=45) as response:
        if response.status != 200:
            raise RuntimeError("Private readback receipt persistence failed")
    return digest


def process(package, indexed_jobs, device, weights, deadline, smoke=False, model=None):
    import numpy as np
    import torch
    from .inference import predictor, process_patient
    from .baseline import json_scalar
    if model is None:
        model = predictor(weights, package["runtime"], device) if device.type == "cuda" else weights
    for job in indexed_jobs:
        if not smoke and time.time()+240 >= deadline:
            break
        started = time.monotonic()
        if device.type == "cuda":
            torch.cuda.set_device(device)
        seed = job["index"]
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        with tempfile.TemporaryDirectory(prefix="ll-patient-", dir="/kaggle/working") as tmp:
            root = Path(tmp)
            inp, out = root/"input", root/"output"
            inp.mkdir()
            out.mkdir()
            visits, transfers = [], []
            for j, row in enumerate(job["visits"][:2] if smoke else job["visits"]):
                visit = {"exam": row["exam"]}
                for k in ("image", "seg"):
                    grant = row[k]
                    path = inp/f"{j}-{k}{grant['suffix']}"
                    transfers.append((grant, path))
                    visit[k] = path
                visits.append(visit)
            with ThreadPoolExecutor(max_workers=3) as pool:
                list(pool.map(lambda args: fetch(*args), transfers))
            net_seconds = time.monotonic() - started
            fetch(job["l0"], inp/"baseline.json")
            report = process_patient(model, package["runtime"], visits,
                                      json.loads((inp/"baseline.json").read_text()),
                                      out, device, device.type == "cpu", smoke)
            report.update(source_sha=package["source_sha"], campaign=package["campaign"],
                          checkpoint_sha256=package["runtime"]["checkpoint_sha256"],
                          index=job["index"], smoke=smoke, io_seconds=net_seconds,
                          total_seconds=time.monotonic()-started, device=str(device),
                          versions=package.get("versions"))
            (out/"report.json").write_text(json.dumps(report, allow_nan=False, default=json_scalar))
            if smoke and not any(item.get("T7") is not None
                                 for pair in report["pairs"] for item in pair["L1"]["instances"]):
                raise RuntimeError("Smoke produced no observable U-Net forward")
            digest = publish(out, job["output"])
            print(json.dumps({"event":"private_readback", "smoke":smoke, "pairs":len(report["pairs"]),
                              "session_seconds":round(time.monotonic()-started, 2),
                              "sha256":digest}), flush=True)
    return model


def main():
    import importlib.metadata
    import torch
    package = json.loads(Path(sys.argv[1]).read_text())
    mode = sys.argv[2]
    started = time.time()
    deadline = min(started+package["session_seconds"]-120, package["expires"]-240)
    if deadline - started < 420:
        raise RuntimeError("Private input lease is too short for safe startup")
    torch.set_num_threads(2)
    package["versions"] = {p: importlib.metadata.version(p) for p in (
        "torch", "numpy", "lesionlocator", "acvl-utils", "unigradicon", "icon-registration",
        "dynamic-network-architectures", "batchgenerators", "blosc2", "SimpleITK")}
    device = torch.device("cpu" if mode == "geometry" else "cuda:"+sys.argv[3])
    if device.type == "cuda":
        if not torch.cuda.is_available() or any("T4" not in torch.cuda.get_device_name(i)
                                               for i in range(torch.cuda.device_count())):
            raise RuntimeError("Expected free T4 accelerator")
        torch.cuda.set_device(device)
        print(json.dumps({"event":"gpu_allocated","device":torch.cuda.get_device_name(device),
                          "count":torch.cuda.device_count()}), flush=True)
    model = Path(sys.argv[4])
    indices = package["geometry"] if mode == "geometry" else package["formal"]
    resident = None
    if mode == "gpu" and device.index == 0:
        resident = process(package, package["smoke"], device, model, deadline, smoke=True)
        (model.parent/"smoke-pass").touch()
    elif mode == "gpu" and not package["smoke"]:
        pass
    elif mode == "gpu":
        while not (model.parent/"smoke-pass").exists():
            if time.time()+300 >= deadline:
                raise RuntimeError("Smoke did not complete")
            time.sleep(5)
    selected = indices if device.type == "cpu" else indices[device.index::torch.cuda.device_count()]
    process(package, selected, device, model, deadline, model=resident)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        # urllib errors include the full signed URL. Report the type and step only.
        import traceback
        for frame in traceback.extract_tb(exc.__traceback__):
            print(f"{Path(frame.filename).name}:{frame.lineno} in {frame.name}", file=sys.stderr)
        print(type(exc).__name__, file=sys.stderr)
        sys.exit(1)
