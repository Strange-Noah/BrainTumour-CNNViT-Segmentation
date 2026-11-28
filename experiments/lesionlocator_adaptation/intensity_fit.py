"""Exact raw label-3 intensity fit on the frozen BMETS training partition."""
import argparse
from hashlib import sha256
from io import BytesIO
import json
import os
from pathlib import Path
import tempfile

import nibabel as nib
import numpy as np
from scipy import ndimage

from cloud.storage import connect, put_once
from experiments.lesionlocator_benchmark.assets import PREFIX as OFFICIAL
from experiments.lesionlocator_benchmark.baseline import MANIFEST_KEY, MANIFEST_SHA, SPLIT, save_once, sequences
from research.dataset import Dataset
from research.geometry import mask_geometry

PREFIX = "novo-nordisk/experiments/lesionlocator-bmets/adaptation/v1"
FIT = {"manifest": MANIFEST_SHA, "partition": "train", "modality": "t1c",
       "label": 3, "connectivity": 26, "quantiles": "exact NumPy linear",
       "moments": "unclipped population, float64"}
IDENTITY = sha256(json.dumps(FIT, sort_keys=True).encode()).hexdigest()


def describe(x):
    if not x.size:
        return {"voxels": 0}
    if not np.isfinite(x).all():
        raise ValueError("Nonfinite raw enhancing intensity")
    lo, md, hi = np.quantile(x, [.005, .5, .995])
    return dict(voxels=int(x.size), mean=float(np.mean(x,dtype=np.float64)),
                std=float(np.std(x,dtype=np.float64)), median=float(md),
                percentile_00_5=float(lo), percentile_99_5=float(hi))


def visit(ds,row):
    t1c=nib.load(ds.fetch(row["modalities"]["t1c"]))
    gt=nib.load(ds.fetch(row["segmentation"]))
    mask_geometry(gt,t1c,row["exam_id"])
    labels=np.asarray(gt.dataobj)
    if not np.isin(labels,(0,1,2,3,4)).all():
        raise ValueError("Unexpected label values")
    mask=labels==3
    intensity = t1c.get_fdata(dtype=np.float64,caching="unchanged")
    values=intensity[mask]
    record=describe(values)
    foreground,count=ndimage.label(mask,structure=np.ones((3,3,3),dtype=bool))
    ids=foreground[mask]
    ordering=np.argsort(ids,kind="stable")
    sorted_values=values[ordering]
    boundaries=np.cumsum(np.bincount(ids,minlength=count+1)[1:-1])
    record["lesions"]=[describe(x) for x in np.split(sorted_values,boundaries)] if count else []
    record["exam"]=row["exam_id"]
    # Each exam contributes once and the shared input cache only holds its two files.
    for filename in ds.cache.iterdir():
        filename.unlink()
    return record,values


def key(index):
    return f"{PREFIX}/intensity/{IDENTITY}/patient-{index:03d}.npz"


def read_private(client,bucket,key_name):
    response=client.get_object(Bucket=bucket,Key=key_name)
    with response["Body"] as body:
        data=body.read()
    if sha256(data).hexdigest()!=response["Metadata"].get("sha256"):
        raise ValueError("Independent R2 result verification failed")
    return data


def measure(ds,groups,client,bucket,first,count,root):
    selected=groups[first:first+count]
    if len(selected)!=count or count<=0:
        raise ValueError("Invalid Train batch bounds")
    examined=voxels=0
    for index,rows in enumerate(selected,first):
        inventory=[r["exam_id"] for r in rows]
        try:
            existing=read_private(client,bucket,key(index))
        except client.exceptions.NoSuchKey:
            existing=None
        if existing is not None:
            with np.load(BytesIO(existing),allow_pickle=False) as item:
                record=json.loads(item["record"].item())
                if record["fit"]!=IDENTITY or [v["exam"] for v in record["visits"]]!=inventory:
                    raise ValueError("Existing partial fit uses a different train inventory")
                examined+=len(record["visits"])
                voxels+=record["patient"]["voxels"]
            continue
        records, chunks=[],[]
        for row in rows:
            record,v=visit(ds,row)
            records.append(record)
            chunks.append(v)
        values=np.concatenate(chunks)
        metadata=dict(fit=IDENTITY,source=os.environ["NOVO_SOURCE_SHA"],
                      visits=records,patient=describe(values))
        path=root/f"patient-{index:03d}.npz"
        np.savez_compressed(path,values=values,record=json.dumps(metadata,allow_nan=False))
        put_once(client,bucket,key(index),path)
        path.unlink()
        examined+=len(records)
        voxels+=len(values)
    print(json.dumps(dict(event="n1_batch",patients=count,exams=examined,voxels=voxels,readback=True)))


def summarize(values):
    return {k:np.quantile([m[k] for m in values if m["voxels"]],[0,.05,.5,.95,1]).tolist()
            for k in ("mean","std","median","voxels")}


def reduce(groups,client,bucket):
    chunks,patients,visits,lesions,sources=[],[],[],[],set()
    for index, rows in enumerate(groups):
        with np.load(BytesIO(read_private(client,bucket,key(index))),allow_pickle=False) as item:
            record=json.loads(item["record"].item())
            if record["fit"]!=IDENTITY or [v["exam"] for v in record["visits"]]!=[r["exam_id"] for r in rows]:
                raise ValueError("Frozen Train fit inventory mismatch")
            chunks.append(item["values"])
            patients.append(record["patient"])
            visits.extend(record["visits"])
            lesions.extend(m for v in record["visits"] for m in v["lesions"])
            sources.add(record["source"])
    values=np.concatenate(chunks)
    params=describe(values)
    del chunks,values
    if params["std"]<=0:
        raise ValueError("Invalid Train foreground standard deviation")
    active=json.loads(read_private(client,bucket,OFFICIAL+"/runtime.json"))
    plans_entry=json.loads(read_private(client,bucket,OFFICIAL+"/receipt.json"))["members"]["LesionLocatorTrack/plans.json"]
    plans=json.loads(read_private(client,bucket,plans_entry["key"]))
    fields=("mean","std","percentile_00_5","percentile_99_5")
    new={k:params[k] for k in fields}
    old={k:active["foreground_intensity_properties_per_channel"]["0"][k] for k in fields}
    def zero(v):
        return (float(np.clip(0,v["percentile_00_5"],v["percentile_99_5"]))-v["mean"])/v["std"]
    plans["foreground_intensity_properties_per_channel"]["0"].update(new)
    payload=json.dumps(plans,sort_keys=True,separators=(",",":")).encode()
    digest=sha256(payload).hexdigest()
    with tempfile.NamedTemporaryFile() as artifact:
        artifact.write(payload)
        artifact.flush()
        put_once(client,bucket,f"{PREFIX}/intensity/{IDENTITY}/plans-{digest}.json",Path(artifact.name))
    aggregate=dict(event="n1_fit",science_sha256=IDENTITY,plans_sha256=digest,
         official_checkpoint_sha256=active["checkpoint_sha256"],source_shas=sorted(sources),
         patients=len(patients),exams=len(visits),lesions=len(lesions),
         positive_visits=sum(v["voxels"]>0 for v in visits),pooled=params,
         official=old,transferred=new,normalized_zero=dict(official=zero(old),transferred=zero(new)),
         patient_distribution=summarize(patients),visit_distribution=summarize(visits),
         lesion_distribution=summarize(lesions),distribution_quantiles=[0,.05,.5,.95,1])
    save_once(client,bucket,f"{PREFIX}/intensity/{IDENTITY}/fit.json",aggregate)
    print(json.dumps(aggregate,allow_nan=False))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode",choices=("batch","reduce"))
    p.add_argument("--first",type=int,default=0)
    p.add_argument("--count",type=int,default=0)
    args=p.parse_args()
    if args.first<0:
        p.error("Batch offset must be nonnegative")
    raw,source=connect()
    writer,destination=connect(output=True)
    if (source,destination)!=("novo-nordisk-pbl","novo-nordisk-pbl-outputs"):
        raise PermissionError("Expected canonical project buckets")
    with tempfile.TemporaryDirectory(prefix="novo-n1-") as d:
        root=Path(d)
        manifest=root/"manifest.jsonl"
        raw.download_file(source,MANIFEST_KEY,str(manifest))
        ds=Dataset(manifest,SPLIT,root/"cache",partition="train")
        groups=list(sequences(ds.visits).values())
        if len(groups)!=210:
            raise ValueError("Unexpected train inventory")
        if args.mode=="reduce":
            reduce(groups,writer,destination)
        else:
            measure(ds,groups,writer,destination,args.first,args.count,root)


if __name__=="__main__":
    main()
