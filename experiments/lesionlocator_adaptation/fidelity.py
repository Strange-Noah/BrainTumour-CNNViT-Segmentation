"""Shared source/target CPU intensity diagnostics for the fixed affine transfer."""
from collections import defaultdict
from hashlib import sha256
import argparse
import json
import os
from pathlib import Path
import tempfile

import nibabel as nib
import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

from cloud.storage import connect, put_once
from experiments.lesionlocator_benchmark.baseline import MANIFEST_KEY, SPLIT, save_once, sequences
from research.dataset import Dataset
from research.geometry import mask_geometry
from .intensity_fit import IDENTITY,PREFIX,read_private

RADIUS_MM=5.
HIST_EDGES=np.array([-np.inf,*np.linspace(-8,8,65),np.inf])
FIELDS=("mean","std","percentile_00_5","percentile_99_5")


def shell(lesion,affine,radius=RADIUS_MM):
    out=np.zeros(lesion.shape,dtype=bool)
    box=ndimage.find_objects(lesion.astype(np.uint8))
    if not box:
        return out
    matrix=affine[:3,:3]
    halo=np.ceil(radius*np.linalg.norm(np.linalg.inv(matrix),axis=1)).astype(int)
    region=tuple(slice(max(0,b.start-h),min(n,b.stop+h))
                 for b,h,n in zip(box[0],halo,lesion.shape))
    fg=lesion[region]
    gram=matrix.T@matrix
    if np.allclose(gram,np.diag(np.diag(gram)),rtol=0,atol=1e-6):
        out[region]=(ndimage.distance_transform_edt(~fg,sampling=np.sqrt(np.diag(gram)))<=radius)&~fg
    else:
        points=np.argwhere(fg)
        tree=cKDTree(points@matrix.T)
        view=out[region]
        for k in range(fg.shape[0]):
            coords=np.column_stack((np.full(fg[k].size,k),np.indices(fg[k].shape).reshape(2,-1).T))
            view[k]=(tree.query(coords@matrix.T,distance_upper_bound=radius+1e-9)[0]
                     .reshape(fg[k].shape)<=radius)&~fg[k]
    return out


def regions(image,label,affine,modality):
    lesion=label==3 if modality=="t1c" else label>0
    anatomy=image!=0 if modality=="t1c" else image>-500
    normal=anatomy & (label==0)
    near=shell(lesion,affine)&normal
    return {"full":None,"anatomy":anatomy,"lesion":lesion,"normal_proxy":normal,"perilesional":near}


def moments(a):
    if not a.size:
        return {"n":0}
    return dict(n=int(a.size),mean=float(np.mean(a,dtype=np.float64)),
                std=float(np.std(a,dtype=np.float64)),
                quantiles=np.quantile(a,[.005,.05,.5,.95,.995]).tolist())


def fidelity(image, masks, params):
    low, high=params["percentile_00_5"],params["percentile_99_5"]
    fitted={}
    for group,mask in masks.items():
        x=image.ravel() if mask is None else image[mask]
        if not x.size:
            fitted[group]={"n":0}
            continue
        if not np.isfinite(x).all():
            raise ValueError("Nonfinite image values")
        clamped=np.clip(x,low,high)
        normalized=(clamped-params["mean"])/params["std"]
        fitted[group]=dict(n=int(x.size),raw=moments(x),norm=moments(normalized),
            low=int((x<low).sum()),high=int((x>high).sum()),
            at_low=int((x<=low).sum()),at_high=int((x>=high).sum()),
            clipped_mean=float(np.mean(clamped,dtype=np.float64)),
            histogram=np.histogram(normalized,bins=HIST_EDGES)[0].tolist())
    l,p=(fitted[k] for k in ("lesion","perilesional"))
    if l["n"] and p["n"]:
        def compare(which):
            a,b=l[which],p[which]
            difference=a["mean"]-b["mean"]
            scale=np.sqrt((a["std"]**2+b["std"]**2)/2)
            return dict(delta=difference,cnr=float(difference/scale) if scale>0 else None)
        raw,norm=compare("raw"),compare("norm")
        clip_delta=l["clipped_mean"]-p["clipped_mean"]
        fitted["contrast"]=dict(raw=raw,norm=norm,clip_delta=clip_delta,
            retained=abs(clip_delta/raw["delta"]) if raw["delta"] else None)
    return fitted


def summarize(visits):
    groups=defaultdict(list)
    contrasts=[]
    for v in visits:
        for k,r in v.items():
            if k=="contrast":
                contrasts.append(r)
            elif r["n"]:
                groups[k].append(r)
    def q(values):
        return np.quantile(values,[.05,.5,.95]).tolist() if values else None
    result={}
    for k,r in groups.items():
        n=sum(v["n"] for v in r)
        result[k]=dict(visits=len(r),voxels=n,
            rates={f:sum(v[f] for v in r)/n for f in ("low","high","at_low","at_high")},
            median_scan_normalized_quantiles=np.median([v["norm"]["quantiles"] for v in r],axis=0).tolist(),
            scan_means=q([v["norm"]["mean"] for v in r]),scan_std=q([v["norm"]["std"] for v in r]),
            histogram=np.sum([v["histogram"] for v in r],axis=0).tolist())
    result["contrast"]=dict(visits=len(contrasts),raw_delta=q([c["raw"]["delta"] for c in contrasts]),
        normalized_delta=q([c["norm"]["delta"] for c in contrasts]),
        raw_cnr=q([c["raw"]["cnr"] for c in contrasts if c["raw"]["cnr"] is not None]),
        normalized_cnr=q([c["norm"]["cnr"] for c in contrasts if c["norm"]["cnr"] is not None]),
        undefined_normalized_cnr=sum(c["norm"]["cnr"] is None for c in contrasts),
        retained=q([c["retained"] for c in contrasts if c["retained"] is not None]),
        raw_lesion_brighter=sum(c["raw"]["delta"]>0 for c in contrasts),
        normalized_lesion_brighter=sum(c["norm"]["delta"]>0 for c in contrasts))
    return result


def visual(image,lesion,official,transferred,path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    section=int(lesion.sum(axis=(0,1)).argmax())
    raw=image[:,:,section]
    gt=lesion[:,:,section]
    fig,axes=plt.subplots(1,3,figsize=(11,4))
    display=np.quantile(image[image!=0],[.01,.99])
    arms=[("Raw MRI",raw,display)]
    for title,p in (("Official CT",official),("BMETS transfer",transferred)):
        clipped=np.clip(raw,p["percentile_00_5"],p["percentile_99_5"])
        arms.append((title,(clipped-p["mean"])/p["std"],(-3,3)))
    for ax,(title,values,(lo,hi)) in zip(axes,arms):
        ax.imshow(values.T,origin="lower",cmap="gray",vmin=lo,vmax=hi)
        if gt.any() and not gt.all(): ax.contour(gt.T,levels=[.5],colors="yellow",linewidths=.6)
        ax.set_title(title)
        ax.axis("off")
    fig.savefig(path,dpi=110)
    plt.close(fig)


def reduce(writer,bucket,fit):
    path=f"{PREFIX}/fidelity/{fit['plans_sha256']}"
    records=[json.loads(read_private(writer,bucket,f"{path}/validation-{i:02d}.json")) for i in range(45)]
    ct=json.loads(read_private(writer,bucket,f"{PREFIX}/source-ct/summary.json"))
    visits=[v for patient in records for v in patient["visits"]]
    if len(visits)!=159 or len({v["exam"] for v in visits})!=159:
        raise ValueError("Frozen validation coverage mismatch")
    def longitudinal(arm):
        pairs=[d for p in records for d in p["differences"][arm]]
        result={}
        for region in ("anatomy","normal_proxy","lesion","perilesional"):
            rows=[d[region] for d in pairs if region in d]
            result[region]=dict(n=len(rows),absolute_delta={
                k:np.quantile([abs(r[k]) for r in rows],[.05,.5,.95]).tolist()
                for k in ("raw","normalized")}) if rows else {"n":0}
        return result
    result=dict(event="n2_summary",patients=45,exams=len(visits),plans_sha256=fit["plans_sha256"],
                source=ct["intensity"],
                arms={k:summarize([v[k] for v in visits]) for k in ("old","new")},
                longitudinal={k:longitudinal(k) for k in ("old","new")},shell_mm=RADIUS_MM)
    save_once(writer,bucket,f"{path}/summary.json",result)
    print(json.dumps(result,allow_nan=False))


def validation(ds,groups,db,bucket,fit,first,count):
    if not 0<=first<len(groups) or len(groups[first:first+count])!=count or count<=0:
        raise ValueError("Invalid validation batch")
    fields=("old","new")
    reported={key:[] for key in fields}
    plotted=first!=0
    for i, rows in enumerate(groups[first:first+count],first):
        output=[]
        for row in rows:
            t1c=nib.load(ds.fetch(row["modalities"]["t1c"]))
            gt=nib.load(ds.fetch(row["segmentation"]))
            mask_geometry(gt,t1c,row["exam_id"])
            image=t1c.get_fdata(dtype=np.float32,caching="unchanged")
            label=np.asarray(gt.dataobj)
            masks=regions(image,label,t1c.affine,"t1c")
            arms=dict(old=fidelity(image,masks,fit["official"]),new=fidelity(image,masks,fit["transferred"]))
            if not plotted and masks["lesion"].any():
                panel=ds.cache.parent/"intensity-comparison.png"
                visual(image,masks["lesion"],fit["official"],fit["transferred"],panel)
                put_once(db,bucket,f"{PREFIX}/fidelity/{fit['plans_sha256']}/comparison.png",panel)
                plotted=True
            for k in fields: reported[k].append(arms[k])
            output.append(dict(exam=row["exam_id"],**arms))
            for cached in ds.cache.iterdir(): cached.unlink()
        differences={}
        for k in fields:
            differences[k]=[{region:dict(raw=b[k][region]["raw"]["mean"]-a[k][region]["raw"]["mean"],
                      normalized=b[k][region]["norm"]["mean"]-a[k][region]["norm"]["mean"])
                      for region in ("anatomy","normal_proxy","lesion","perilesional")
                      if a[k][region]["n"] and b[k][region]["n"]}
                      for a,b in zip(output,output[1:])]
        prefix=f"{PREFIX}/fidelity/{fit['plans_sha256']}"
        save_once(db,bucket,f"{prefix}/validation-{i:02d}.json",
                  dict(source=os.environ["NOVO_SOURCE_SHA"],visits=output,differences=differences))
    print(json.dumps(dict(event="n2_validation",patients=count,exams=len(reported["new"]),
                          plans_sha256=fit["plans_sha256"],arms={k:summarize(v) for k,v in reported.items()})))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--first",type=int,default=0)
    p.add_argument("--count",type=int,default=0)
    p.add_argument("--reduce",action="store_true")
    args=p.parse_args()
    reader,canonical=connect()
    writer,output=connect(output=True)
    fit=json.loads(read_private(writer,output,f"{PREFIX}/intensity/{IDENTITY}/fit.json"))
    if fit["patients"]!=210 or fit["science_sha256"]!=IDENTITY:
        raise ValueError("Complete frozen N1 fit required")
    if args.reduce:
        reduce(writer,output,fit)
        return
    with tempfile.TemporaryDirectory(prefix="novo-n2-") as d:
        root=Path(d)
        manifest=root/"manifest.jsonl"
        reader.download_file(canonical,MANIFEST_KEY,str(manifest))
        ds=Dataset(manifest,SPLIT,root/"cache",partition="validation")
        groups=list(sequences(ds.visits).values())
        if len(groups)!=45 or len(ds.visits)!=159:
            raise ValueError("Unexpected validation inventory")
        validation(ds,groups,writer,output,fit,args.first,args.count)


if __name__=="__main__":
    main()
