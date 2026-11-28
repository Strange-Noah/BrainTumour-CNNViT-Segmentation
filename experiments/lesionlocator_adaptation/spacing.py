"""Train-only spacing comparison and frozen independent validation."""
import argparse
from copy import deepcopy
from hashlib import sha256
from io import BytesIO
import json
import os
from pathlib import Path
import tempfile

import nibabel as nib
import numpy as np
from scipy.stats import wasserstein_distance
import torch

from cloud.storage import connect,put_once
from experiments.lesionlocator_benchmark.baseline import MANIFEST_KEY,SPLIT,save_once,sequences,instances
from research.dataset import Dataset
from research.geometry import mask_geometry
from .intensity_fit import PREFIX,IDENTITY,key as n1_key,read_private
from .fidelity import summarize as summarize_intensity
from .source_ct import official
from .spatial import CANDIDATES,evaluate_pair

PAIR_COUNT=12
DESIGN="three equal Train strata by earliest positive visit's median lesion voxels; four stable SHA ranks per stratum"
VOLUMES=(100,1000)


def q(x,quantiles=(.05,.5,.95)):
    return np.quantile(x,quantiles).tolist() if len(x) else None


def make_calibration(ds,groups,writer,bucket):
    eligible=[]
    for i,rows in enumerate(groups):
        with np.load(BytesIO(read_private(writer,bucket,n1_key(i))),allow_pickle=False) as payload:
            history=json.loads(payload["record"].item())["visits"]
        for ordinal,v in enumerate(history[:-1]):
            if v["lesions"]:
                median=np.median([x["voxels"] for x in v["lesions"]])
                patient_rank=sha256(("novo-s2-v1/"+rows[0]["patient_id"]).encode()).hexdigest()
                eligible.append((float(median),patient_rank,i,ordinal))
                break
    eligible.sort()
    ranked=[b.tolist() for b in np.array_split(np.arange(len(eligible)),3)]
    selected=[r for indexes in ranked for r in sorted((eligible[i] for i in indexes),key=lambda r:r[1])[:4]]
    if len(selected)!=PAIR_COUNT:
        raise ValueError("Train design underfilled")
    return [(row[2],row[3]) for row in selected]


def load(ds,r,root,ordinal):
    image_path=ds.fetch(r["modalities"]["t1c"])
    gt=ds.fetch(r["segmentation"])
    ref,label=map(nib.load,(image_path,gt))
    mask_geometry(label,ref,r["exam_id"])
    mask=instances(np.asarray(label.dataobj))
    header=ref.header.copy()
    header.set_data_dtype(np.int32)
    destination=root/f"{ordinal}-instances.nii.gz"
    nib.save(nib.Nifti1Image(mask.astype(np.int32),ref.affine,header),destination)
    return dict(image=image_path,ref=ref,labels=mask,instances=destination)


def sweep(ds,groups,writer,bucket,fit,params,plans,data,args):
    root=ds.cache.parent
    design=make_calibration(ds,groups,writer,bucket) if args.mode=="calibrate" else None
    if args.mode=="calibrate":
        rows=[(i,[groups[j][k:k+2]]) for i,(j,k) in enumerate(design)]
        choices=CANDIDATES
    else:
        frozen=json.loads(read_private(writer,bucket,f"{PREFIX}/spatial/{fit['plans_sha256']}/selection.json"))
        rows=[(i,[rows]) for i,rows in enumerate(groups)]
        choices={frozen["selected"]:tuple(frozen["spacing"])}
    chosen=rows[args.first:args.first+args.count]
    if not chosen or len(chosen)!=args.count:
        raise ValueError("Invalid geometry batch limits")
    prefix=f"{PREFIX}/spatial/{fit['plans_sha256']}/{args.mode}"
    for index,units in chosen:
        records=[]
        for group in units:
            a=load(ds,group[0],root,0)
            for j,row in enumerate(group[1:],1):
                b=load(ds,row,root,j)
                record={name:evaluate_pair(plans,params["configuration"],data,sp,a,b,root,fit["transferred"])
                        for name,sp in choices.items()}
                records.append(record)
                a=b
        save_once(writer,bucket,f"{prefix}/unit-{index:02}.json",
                  dict(results=records,science_sha=os.environ["NOVO_SOURCE_SHA"],intensity_sha256=fit["plans_sha256"],
                       cohort="train" if args.mode=="calibrate" else "validation",sampling=DESIGN if design else "all adjacent"))
        for f in ds.cache.iterdir(): f.unlink()
    print(json.dumps(dict(event="s2_batch",units=len(chosen),partition=args.mode,readback=True)))


def distribution(rows,space):
    entries=[r[space] for case in rows for r in case["history"] if r[space].get("voxels")]
    out={"log_n":[np.log(r["voxels"]) for r in entries]}
    for i,k in enumerate(("log_d","log_h","log_w")):
        out[k]=[np.log(r["bbox"][i]) for r in entries]
    if space=="registration":
        out["lesion_fraction"]=[r["voxels"]/175**3 for r in entries]
    else:
        for k in ("pad_fraction","lesion_fraction","anatomy_fraction"):
            out[k]=[r[k] for r in entries]
    return out


def comparison(rows,sources):
    out={}
    for space in ("registration","history_roi"):
        target=distribution(rows,space)
        src=distribution(sources,space)
        out[space]={}
        for name,y in target.items():
            x=src[name]
            w=wasserstein_distance(x,y) if x and y else None
            scale=np.diff(np.quantile(x,[.25,.75]))[0] if x else None
            out[space][name]=dict(n_source=len(x),n_bmets=len(y),source_q=q(x),bmets_q=q(y),
                                    wasserstein=w,scaled_wasserstein=w/scale if w is not None and scale>0 else None)
    return out


def error_group(entries,stage):
    rows=[(r.get(stage,r.get("registration")),r.get("native",{}).get("volume_mm3")) for r in entries]
    if stage=="G5": rows=[(r,r["native_volume_mm3"]) for r,_ in rows]
    def stats(col):
        return dict(n=len(col),lost=sum(not r.get("exists",r.get("voxels",0)) for r in col),
          dice=q([r["dice"] for r in col if r.get("dice") is not None]),
          relative_volume_error=q([r["relative_volume_error"] for r in col if "relative_volume_error" in r]),
          centroid_mm=q([np.linalg.norm(r["centroid_delta_mm"]) for r in col if r.get("centroid_delta_mm") is not None]))
    return {"all":stats([r for r,_ in rows]),
            **{tag:stats([r for r,v in rows if lo<=v<hi]) for tag,(lo,hi) in
               zip(("under_100","100_to_1000","at_least_1000"),zip((0,*VOLUMES),(*VOLUMES,np.inf))) }}


def arm_summary(rows,sources):
    history=[r for case in rows for r in case["history"]]
    target=[r for case in rows for r in case["target_oracle"]]
    roi={}
    for tag in ("history_roi","direct_l0_roi","target_oracle_roi"):
        available=[e[tag] for e in history if e[tag] and e[tag]["available"]]
        measured=[r["target_coverage"] for r in available if r["target_coverage"] is not None]
        roi[tag]=dict(n=len(available),coverage=q(measured),covered_n=len(measured),
            full_coverage=sum(v==1 for v in measured),pad=q([r["pad_fraction"] for r in available]),
            anatomy=q([r["anatomy_fraction"] for r in available]))
    return dict(pairs=len(rows),work_vox=q([np.prod(r["history_work_shape"]) for r in rows]),
        actual_reg_fov=[q([r["history_175_nominal_fov_mm"][i] for r in rows]) for i in range(3)],
        patch_fov=(np.array((96,256,256))*rows[0]["spacing"]).tolist(),
        window_disagreements=sum(r["windows"]!=r["fixed_voxel_windows"] for r in rows),
        stages={**{key:error_group(history,key) for key in ("G1","G2","G3","G4")},
                  "G5":error_group(target,"G5")},
        rois=roi,distance=comparison(rows,sources))


def choose(arms):
    def limits(k):
        d=arms[k]
        return (d["stages"]["G4"]["all"]["lost"],d["stages"]["G3"]["all"]["lost"],
                (d["rois"]["direct_l0_roi"]["coverage"] or [0.])[0])
    def costs(k):
        g4,g3,c=limits(k)
        d=arms[k]["distance"]
        return [g4,g3,1-c,*[x["scaled_wasserstein"] if x["scaled_wasserstein"] is not None else x["wasserstein"]
               for stage in d.values() for x in stage.values()]]
    k0=limits("official")
    eligible=[k for k in arms if limits(k)[0]<=k0[0] and limits(k)[1]<=k0[1] and limits(k)[2]>=k0[2]]
    def dominates(a,b):
        pairs=[(x,y) for x,y in zip(costs(a),costs(b)) if x is not None and y is not None]
        return all(x<=y for x,y in pairs) and any(x<y for x,y in pairs)
    front=[k for k in eligible if not any(dominates(j,k) for j in eligible if j!=k)]
    # When reference feature distances disagree, prefer information preservation
    # and then preserve as much official physical ROI extent as possible.
    selected=min(front,key=lambda k:(limits(k)[0],limits(k)[1],-limits(k)[2],
                                    abs(CANDIDATES[k][0]-CANDIDATES["official"][0])))
    return front,selected


def reduce(writer,bucket,fit,params,plans,args):
    folder=f"{PREFIX}/spatial/{fit['plans_sha256']}"
    sources=[json.loads(read_private(writer,bucket,f"{PREFIX}/source-ct/unit-{i}.json"))["spatial"] for i in range(6)]
    mode="calibrate" if args.mode=="select" else "validate"
    output={}
    n=PAIR_COUNT if mode=="calibrate" else 45
    for i in range(n):
        record=json.loads(read_private(writer,bucket,f"{folder}/{mode}/unit-{i:02}.json"))
        for row in record["results"]:
            for name,values in row.items(): output.setdefault(name,[]).append(values)
    if mode=="validate" and set(map(len,output.values()))!={114}:
        raise ValueError("Expected full frozen validation coverage")
    arms={k:arm_summary(values,sources) for k,values in output.items()}
    stats=dict(event="s2_"+args.mode,partition=mode,patients=n,arms=arms)
    if mode=="calibrate":
        front,selected=choose(arms)
        spacing=CANDIDATES[selected]
        config=deepcopy(plans)
        config["configurations"][params["configuration"]]["spacing"]=list(spacing)
        config["foreground_intensity_properties_per_channel"]["0"].update(fit["transferred"])
        payload=json.dumps(config,sort_keys=True,separators=(",",":")).encode()
        digest=sha256(payload).hexdigest()
        with tempfile.NamedTemporaryFile() as f:
            f.write(payload)
            f.flush()
            put_once(writer,bucket,f"{folder}/combined-{digest}.json",Path(f.name))
        stats.update(selected=selected,spacing=spacing,pareto=front,plans_sha256=digest,
                     intensity=fit["transferred"],z_threshold_mm=60.,model_sha256=params["checkpoint_sha256"],
                     rule="nonworse G4/G3 losses and L0 p05 coverage, Pareto, preservation then patch-context tie-break")
        save_once(writer,bucket,f"{folder}/selection.json",stats)
    else:
        save_once(writer,bucket,f"{folder}/validation-summary.json",stats)
    first={k:{a:d[a] for a in ("pairs","stages","rois","patch_fov","work_vox","window_disagreements","distance","actual_reg_fov")} for k,d in arms.items()}
    print(json.dumps(dict(stats,arms=first),allow_nan=False))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode",choices=("calibrate","select","validate","report"))
    p.add_argument("--first",type=int,default=0)
    p.add_argument("--count",type=int,default=0)
    args=p.parse_args()
    torch.set_num_threads(2)
    writer,bucket=connect(output=True)
    reader,canonical=connect()
    fit=json.loads(read_private(writer,bucket,f"{PREFIX}/intensity/{IDENTITY}/fit.json"))
    params,plans,data=official(writer,bucket)
    if args.mode in ("select","report"):
        reduce(writer,bucket,fit,params,plans,args)
        return
    if args.mode=="calibrate":
        read_private(writer,bucket,f"{PREFIX}/source-ct/summary.json")
    partition="train" if args.mode=="calibrate" else "validation"
    with tempfile.TemporaryDirectory(prefix="novo-s2-") as d:
        root=Path(d)
        manifest=root/"manifest.jsonl"
        reader.download_file(canonical,MANIFEST_KEY,str(manifest))
        ds=Dataset(manifest,SPLIT,root/"cache",partition=partition)
        groups=list(sequences(ds.visits).values())
        if len(groups)!={"train":210,"validation":45}[partition]:
            raise ValueError("Frozen patient inventory mismatch")
        sweep(ds,groups,writer,bucket,fit,params,plans,data,args)


if __name__=="__main__":
    main()
