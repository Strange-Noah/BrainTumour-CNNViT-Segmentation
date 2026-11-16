"""Genuine CPU/GPU summer-checkpoint forward and small UCSD label-3 fine-tune."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
import time

import numpy as np
import torch
from huggingface_hub import HfApi, hf_hub_download

from cloud.storage import DATASET, connect, put_once
from research.dataset import Dataset
from research.evaluate import evaluate, export
from experiments.cnn_vit_ucsd import infer, load_model, preprocess, slice_target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--inference", action="store_true", help="Export complete validation-visit prediction")
    args = parser.parse_args()
    torch.set_num_threads(2)
    args.work.mkdir(parents=True, exist_ok=True)
    sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    client, bucket = connect()
    manifest_sha = "040ec6c523d5beae43a30b64fede2bee226e1a08e8c14b2cca8043b01b9c39f4"
    manifest = args.work / "manifest.jsonl"
    client.download_file(bucket, f"{DATASET}/manifests/{manifest_sha}/visits.jsonl", str(manifest))
    train = Dataset(manifest, "protocol/v1/patients.csv", args.work / "cache", "train")
    validation = Dataset(manifest, "protocol/v1/patients.csv", args.work / "cache", "validation")
    # Check availability from object listing without copying patient data through a Mac.
    available = set()
    for page in client.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=DATASET+"/raw/"):
        available.update(record["Key"].split("/")[-2] for record in page.get("Contents", []))
    train_index = next(i for i, row in enumerate(train.visits) if row["exam_id"] in available)
    val_index = next(i for i, row in enumerate(validation.visits) if row["exam_id"] in available)
    sample, heldout = train.load(train_index), validation.load(val_index)
    report = {"git_sha": sha, "task": "enhancing", "raw_positive_labels": [3],
              "train_patient": sample["visit"]["patient_id"],
              "validation_patient": heldout["visit"]["patient_id"], "test_evaluated": False,
              "torch": torch.__version__, "device": "cuda" if torch.cuda.is_available() else "cpu"}
    model_checkpoint = None
    try:
        info = HfApi().model_info("TLOMMP/route-b-4label-full-train")
        report["summer_revision"] = info.sha
        model_checkpoint = hf_hub_download(info.id, "best_checkpoint.pt", revision=info.sha)
        model, compatibility = load_model(model_checkpoint, diagnostic_input=preprocess(sample["images"], sample["images"].shape[-1] // 2))
    except Exception as error:
        report["summer_recovery_error_type"] = type(error).__name__
        info = HfApi().model_info("timm/vit_base_r50_s16_224.orig_in21k")
        backbone = hf_hub_download(info.id, "model.safetensors", revision=info.sha)
        report["backbone_revision"] = info.sha
        model, compatibility = load_model(None, backbone)
    report["compatibility"] = compatibility
    if model_checkpoint:
        with Path(model_checkpoint).open("rb") as stream:
            report["summer_sha256"] = hashlib.file_digest(stream, "sha256").hexdigest()
    device = torch.device(report["device"])
    model.to(device)
    report["parameters"] = sum(p.numel() for p in model.parameters())
    report["trainable_parameters"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
    positives = np.flatnonzero(sample["target"].any(axis=(0, 1)))
    selected = positives.tolist() or list(range(sample["target"].shape[-1]))
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=1e-3)
    model.eval()  # Keep pretrained dropout/normalization fixed in this head-only pilot.
    losses = []
    start = time.perf_counter()
    for step in range(args.steps):
        z = selected[step % len(selected)]
        optimizer.zero_grad()
        logits = model(preprocess(sample["images"], z).to(device))
        labels = slice_target(sample["target"], z).to(device)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels)
        if not torch.isfinite(loss):
            raise RuntimeError("Nonfinite real training loss")
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    report["training_seconds"] = time.perf_counter()-start
    report["training_losses"] = losses
    model.eval()
    val_z = heldout["images"].shape[-1] // 2  # Independent of validation labels.
    start = time.perf_counter()
    with torch.inference_mode():
        logits = model(preprocess(heldout["images"], val_z).to(device))
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            logits, slice_target(heldout["target"], val_z).to(device))
    report["validation_slice_loss"] = float(loss.cpu())
    report["forward_seconds"] = time.perf_counter()-start
    checkpoint = args.work/"ucsd-enhancing-head-pilot.pt"
    torch.save({"model_state_dict": model.state_dict(), "task": "enhancing",
                "raw_positive_labels": [3], "input_modalities": ["t1n", "t1c", "t2", "t2f"], "git_sha": sha}, checkpoint)
    if args.inference:
        start = time.perf_counter()
        prediction = infer(model, heldout["images"], device)
        report["full_validation_inference_seconds"] = time.perf_counter()-start
        report["validation_metrics"] = evaluate(prediction, heldout["target"], heldout["reference"].affine)
        export(prediction, heldout["reference"], args.work/"prediction", {
            "patient_id":heldout["visit"]["patient_id"], "exam_id":heldout["visit"]["exam_id"],
            "model_version":"summer-initialized-ucsd-head-pilot", "git_sha":sha,
            "split_sha256":"57f5af55141e16ecb0b85fdda63cc355175d9b35271126784eb22766698b36f0",
            "task":"enhancing", "checkpoint_ref": checkpoint.name})
    report["peak_rss_kib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if device.type == "cuda":
        report["peak_gpu_bytes"] = torch.cuda.max_memory_allocated()
    (args.work/"report.json").write_text(json.dumps(report, indent=2)+"\n")
    run_id = os.environ.get("GITHUB_RUN_ID", str(int(time.time())))
    prefix = f"novo-nordisk/experiments/cnn-vit/ucsd-head-pilot-{run_id}"
    report["persisted"] = []
    if all(os.environ.get(name) for name in ("R2_OUTPUT_BUCKET","R2_OUTPUT_ACCESS_KEY_ID","R2_OUTPUT_SECRET_ACCESS_KEY")):
        output_client, output_bucket = connect(output=True)
        if output_bucket == bucket:
            raise ValueError("Model outputs require the separate output bucket")
        for path in sorted(args.work.glob("prediction/*")) + [checkpoint]:
            receipt = put_once(output_client,output_bucket,prefix+"/"+path.relative_to(args.work).as_posix(),path)
            report["persisted"].append(receipt)
        report_file = args.work/"report.json"
        report_file.write_text(json.dumps(report,indent=2)+"\n")
        put_once(output_client,output_bucket,prefix+"/report.json",report_file)
    else:
        report["persistence_status"] = "output credentials unavailable; checkpoint and prediction remain ephemeral"
    (args.work/"report.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps({"initialization": compatibility["initialization"],
                      "summer_strict_load": compatibility.get("summer_strict_load"),
                      "summer_native_forward_shape": compatibility.get("summer_native_forward_shape"),
                      "summer_revision": report.get("summer_revision"),
                      "backbone_revision": report.get("backbone_revision"),
                      "recovery_error_type": report.get("summer_recovery_error_type"),
                      "persistence_status": report.get("persistence_status", "persisted")}))
    # Aggregate report is suitable for workflow logs. Keep paths/keys in private records.
    print(json.dumps({key:report[key] for key in (
        "task","device","training_seconds","forward_seconds","training_losses",
        "validation_slice_loss","peak_rss_kib","parameters","trainable_parameters")}))
    if "validation_metrics" in report:
        print(json.dumps(report["validation_metrics"]))


if __name__=="__main__":
    main()
