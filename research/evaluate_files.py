"""Evaluate original-T1c binary task predictions against raw UCSD NIfTI masks."""
import argparse
import csv
import json
from pathlib import Path
import nibabel as nib
import numpy as np
from research.evaluate import evaluate, patient_aggregates, target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cases", help="CSV with patient_id,exam_id,prediction,truth")
    parser.add_argument("output")
    parser.add_argument("--task", choices=["enhancing", "core"], default="enhancing")
    args = parser.parse_args()
    rows = []
    keys = set()
    with Path(args.cases).open(newline="") as source:
        for case in csv.DictReader(source):
            key = (case["patient_id"], case["exam_id"])
            if key in keys:
                raise ValueError("Duplicate visit in evaluation")
            keys.add(key)
            prediction, truth = nib.load(case["prediction"]), nib.load(case["truth"])
            if prediction.shape != truth.shape or not np.allclose(prediction.affine, truth.affine, atol=1e-4):
                raise ValueError("Restore predictions to the source reference grid")
            if truth.header.get_xyzt_units()[0] != "mm":
                raise ValueError("Reference grid must use millimetres")
            raw_prediction = prediction.get_fdata(caching="unchanged")
            raw_truth = truth.get_fdata(caching="unchanged")
            if not np.isin(raw_prediction,[0,1]).all() or not np.isin(raw_truth,[0,1,2,3,4]).all():
                raise ValueError("Unexpected prediction or source label map")
            rows.append({"patient_id":key[0],"exam_id":key[1],
                         **evaluate(raw_prediction > 0,target(raw_truth,args.task),truth.affine)})
    if not rows:
        raise ValueError("No evaluation visits supplied")
    patients = patient_aggregates(rows)
    means = {}
    for metric in ("dice","hd95_mm","lesion_sensitivity","false_positives","absolute_volume_error_mm3"):
        values = [r[metric] for r in patients if r[metric] is not None]
        means[metric] = float(np.mean(values)) if values else None
        means[metric+"_defined_patients"] = len(values)
    with Path(args.output).open("x") as output:
        json.dump({"task":args.task,"visits":rows,"patients":patients,"patient_macro":means},output,indent=2)


if __name__ == "__main__":
    main()
