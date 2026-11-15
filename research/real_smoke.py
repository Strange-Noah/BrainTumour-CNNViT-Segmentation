"""Run the first persisted visit through the actual loader and reference-mask export."""
from pathlib import Path
from tempfile import TemporaryDirectory
import hashlib
import json

from cloud.storage import DATASET, connect
from research.dataset import Dataset
from research.evaluate import evaluate, export


def main():
    client,bucket=connect()
    manifest_sha="040ec6c523d5beae43a30b64fede2bee226e1a08e8c14b2cca8043b01b9c39f4"
    split_sha="57f5af55141e16ecb0b85fdda63cc355175d9b35271126784eb22766698b36f0"
    with TemporaryDirectory() as temp:
        root=Path(temp)
        manifest=root/"manifest.jsonl"
        client.download_file(bucket,f"{DATASET}/manifests/{manifest_sha}/visits.jsonl",str(manifest))
        if hashlib.sha256(manifest.read_bytes()).hexdigest()!=manifest_sha:
            raise ValueError("Canonical manifest checksum mismatch")
        split=Path("protocol/v1/patients.csv")
        if hashlib.sha256(split.read_bytes()).hexdigest()!=split_sha:
            raise ValueError("Frozen split mismatch")
        dataset=Dataset(manifest,split,root/"cache","train")
        index=next(i for i,r in enumerate(dataset.visits) if r["exam_id"]=="BMETS0001_01")
        sample=dataset.load(index,"enhancing")
        results={}
        for task in ("enhancing","core"):
            from research.evaluate import target
            expected=target(sample["raw_labels"],task)
            metrics=evaluate(expected,expected,sample["reference"].affine)
            export(expected,sample["reference"],root/task,{
                "patient_id":sample["visit"]["patient_id"],"exam_id":sample["visit"]["exam_id"],
                "model_version":"reference-mask-smoke","split_sha256":split_sha,
                "task":task,"git_sha":__import__("subprocess").check_output(["git","rev-parse","HEAD"],text=True).strip()})
            results[task]=metrics
        print(json.dumps({"verified":True,"images_shape":list(sample["images"].shape),
                          "observed_labels":list(map(int,__import__("numpy").unique(sample["raw_labels"]))),
                          "reference_mask_metrics":results,"learned_inference":False}))


if __name__=="__main__":
    main()
