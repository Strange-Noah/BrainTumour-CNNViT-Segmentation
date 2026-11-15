"""Run an owner's predictor on a real cached visit and export shared metrics."""
import argparse
import importlib
import json
from pathlib import Path
import hashlib

import numpy as np

from research.dataset import Dataset
from research.evaluate import evaluate, export


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--split", default="protocol/v1/patients.csv")
    parser.add_argument("--cache", required=True)
    parser.add_argument("--partition", default="train", choices=["train", "validation", "test"])
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--task", choices=["enhancing", "core"], default="enhancing")
    parser.add_argument("--predictor", required=True, help="module:function accepting the loaded sample")
    parser.add_argument("--model-version", required=True)
    parser.add_argument("--git-sha", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    dataset = Dataset(args.manifest, args.split, args.cache, args.partition)
    sample = dataset.load(args.index, args.task)
    module, function = args.predictor.split(":")
    predictor_input = sample if args.predictor == "experiments.reference_smoke:predict" else {
        "images": sample["images"], "reference": sample["reference"], "visit": sample["visit"]
    }
    prediction = np.asarray(getattr(importlib.import_module(module), function)(predictor_input))
    if prediction.shape != sample["target"].shape or not np.isin(prediction,[0,1]).all():
        raise ValueError("Predictor must return original T1c-grid binary labels")
    visit = sample["visit"]
    metadata = {"patient_id":visit["patient_id"],"exam_id":visit["exam_id"],
                "model_version":args.model_version,"git_sha":args.git_sha,"task":args.task,
                "split_sha256":hashlib.sha256(Path(args.split).read_bytes()).hexdigest()}
    metrics = evaluate(prediction, sample["target"], sample["reference"].affine)
    export(prediction, sample["reference"], args.output, metadata)
    Path(args.output,"metrics.json").write_text(json.dumps({**metadata,**metrics},indent=2)+"\n")
    print(json.dumps({"completed":True,"dice":metrics["dice"]}))


if __name__ == "__main__":
    main()
