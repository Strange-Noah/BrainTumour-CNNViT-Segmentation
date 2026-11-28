"""Bounded CPU reference measurements from the authors' released CT archive."""
import argparse
from collections import Counter
import csv
from hashlib import sha256
from io import RawIOBase
import json
import os
from pathlib import Path
import shutil
import tempfile
from urllib.request import Request,urlopen
from zipfile import ZipFile

import nibabel as nib
import numpy as np
import torch

from cloud.storage import connect,put_once,verify
from experiments.lesionlocator_benchmark.assets import PREFIX as OFFICIAL
from experiments.lesionlocator_benchmark.baseline import save_once
from .fidelity import fidelity,regions,summarize
from .intensity_fit import PREFIX,read_private
from .spatial import CANDIDATES,evaluate_pair

URL="https://doi.dkfz.de/10.6097/DKFZ/IR/E230/20250324_1.zip"
LENGTH=709050950963
CATEGORIES=("KiTS23","LIDC-IDRI","NSCLC-Radiogenomics")
STEMS=("Baseline/images/{name}_0000.nii.gz","Baseline/labels/{name}.nii.gz",
       "Synthetic_Follow_Up/images/{name}_0000.nii.gz","Synthetic_Follow_Up/labels/{name}.nii.gz")


class RangeArchive(RawIOBase):
    def __init__(self):
        self.position=0
        self.bytes_read=0

    def seek(self,offset,whence=0):
        self.position=(0,self.position,LENGTH)[whence]+offset
        return self.position

    def tell(self):
        return self.position

    def seekable(self):
        return True

    def readable(self):
        return True

    def read(self,size=-1):
        start=self.position
        count=min(size if size>=0 else LENGTH-start,LENGTH-start)
        if count<=0:
            return b""
        if count>16*1024**2 or self.bytes_read+count>800*1024**2:
            raise ValueError("Archive request exceeds the bounded sample")
        end=start+count-1
        request=Request(URL,headers={"Range":f"bytes={start}-{end}","Accept-Encoding":"identity"})
        with urlopen(request,timeout=45) as response:
            if response.status!=206 or response.headers["Content-Range"]!=f"bytes {start}-{end}/{LENGTH}":
                raise ValueError("Official archive range rejected")
            data=response.read(count+1)
        if len(data)!=count:
            raise ValueError("Incomplete archive range")
        self.position+=count
        self.bytes_read+=count
        return data


def selections(archive):
    rows=list(csv.DictReader(archive.read("naming.csv").decode("utf-8-sig").splitlines()))
    paths=set(archive.namelist())
    groups={name:[] for name in CATEGORIES}
    for row in rows:
        group=next((n for n in CATEGORIES if row["Source"].startswith(n)),None)
        if group is None: continue
        names=[p.format(name=row["File Name"]) for p in STEMS]
        if set(names)<=paths and sum(archive.getinfo(n).file_size for n in names)<=144*1024**2:
            groups[group].append((row["Source"],names))
    return [(group,*member) for group in CATEGORIES for member in
            sorted(groups[group],key=lambda r:sha256(("novo-bmets-source-v1/"+r[0]).encode()).digest())[:2]]


def official(client,bucket):
    runtime=json.loads(read_private(client,bucket,OFFICIAL+"/runtime.json"))
    receipt=json.loads(read_private(client,bucket,OFFICIAL+"/receipt.json"))
    plans,data=[json.loads(read_private(client,bucket,receipt["members"][f"LesionLocatorTrack/{name}"]["key"]))
               for name in ("plans.json","dataset.json")]
    return runtime,plans,data


def acquire(archive,client,bucket,prefix,member,root):
    item=archive.getinfo(member)
    dest=root/member.replace("/","-")
    key=prefix+"/"+dest.name
    try:
        head=client.head_object(Bucket=bucket,Key=key)
    except client.exceptions.ClientError as error:
        if error.response["ResponseMetadata"]["HTTPStatusCode"]!=404:
            raise
        head=None
    if head is None:
        with archive.open(item) as body,dest.open("xb") as out:
            shutil.copyfileobj(body,out,8*1024**2)
        if dest.stat().st_size!=item.file_size:
            raise ValueError("Released source member size mismatch")
        record=put_once(client,bucket,key,dest)
    else:
        if head["ContentLength"]!=item.file_size:
            raise ValueError("Saved source member size mismatch")
        with dest.open("xb") as out:
            verify(client,bucket,key,head["Metadata"]["sha256"],out)
        record=dict(key=key,sha256=head["Metadata"]["sha256"],bytes=item.file_size)
    record.update(member=member,crc32=f"{item.CRC:08x}")
    return dest,record


def show(image,lesion,stats,params,destination):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    z=int(lesion.sum(axis=(0,1)).argmax())
    section=image[:,:,z]
    foreground=lesion[:,:,z]
    masked=(np.clip(section,params["percentile_00_5"],params["percentile_99_5"])-params["mean"])/params["std"]
    norm_low,norm_high=-3,3
    fig,axs=plt.subplots(1,2,figsize=(8,4))
    lo,hi=np.quantile(image[image>-500],[.01,.99])
    for ax,values,bounds,title in zip(axs,(section,masked),((lo,hi),(norm_low,norm_high)),("Raw CT","Official CTNormalization")):
        ax.imshow(values.T,origin="lower",cmap="gray",vmin=bounds[0],vmax=bounds[1])
        if foreground.any(): ax.contour(foreground.T,levels=[.5],colors="yellow",linewidths=.6)
        ax.set_title(title)
        ax.axis("off")
    fig.savefig(destination,dpi=110)
    plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--first",type=int)
    p.add_argument("--count",type=int,default=1)
    p.add_argument("--reduce",action="store_true")
    args=p.parse_args()
    torch.set_num_threads(2)
    client,bucket=connect(output=True)
    if bucket!="novo-nordisk-pbl-outputs":
        raise PermissionError("Expected private output bucket")
    params,plans,data=official(client,bucket)
    prefix=f"{PREFIX}/source-ct"
    if args.reduce:
        items=[json.loads(read_private(client,bucket,f"{prefix}/unit-{i}.json")) for i in range(6)]
        result=dict(event="s1_source_summary",source_counts=Counter(r["group"] for r in items),
                    origins=6,exams=12,intensity=summarize([v for r in items for v in r["fidelity"]]))
        save_once(client,bucket,f"{prefix}/summary.json",result)
        print(json.dumps(result,allow_nan=False))
        return
    with ZipFile(RangeArchive()) as archive:
        chosen=selections(archive)
        if len(chosen)!=6 or len(chosen[args.first:args.first+args.count])!=args.count:
            raise ValueError("Expected a bounded six-patient source reference")
        for index,(group,original,members) in enumerate(chosen[args.first:args.first+args.count],args.first):
            with tempfile.TemporaryDirectory(prefix="ll-source-") as d:
                root=Path(d)
                targets, receipts=[],[]
                for name in members:
                    target,receipt=acquire(archive,client,bucket,f"{prefix}/unit-{index}",name,root)
                    targets.append(target)
                    receipts.append(receipt)
                visits,results=[],[]
                ct=params["foreground_intensity_properties_per_channel"]["0"]
                for i in (0,2):
                    image,label=map(nib.load,targets[i:i+2])
                    if image.shape!=label.shape or not np.allclose(image.affine,label.affine,rtol=0,atol=1e-4):
                        raise ValueError("Source native CT and labels differ")
                    labels=np.asarray(label.dataobj)
                    img=image.get_fdata(dtype=np.float32,caching="unchanged")
                    masks=regions(img,labels,image.affine,"ct")
                    result=fidelity(img,masks,ct)
                    visits.append(dict(image=targets[i],ref=image,labels=labels,instances=targets[i+1],modality="ct"))
                    results.append(result)
                    if i==0:
                        plot=root/"reference.png"
                        show(img,masks["lesion"],result,ct,plot)
                        put_once(client,bucket,f"{prefix}/unit-{index}/reference.png",plot)
                    del img,masks
                geometry=evaluate_pair(plans,params["configuration"],data,CANDIDATES["official"],*visits,root,metrics=False)
                save_once(client,bucket,f"{prefix}/unit-{index}.json",dict(group=group,source_case=original,
                       provenance=receipts,science_source=os.environ["NOVO_SOURCE_SHA"],
                       fidelity=results,spatial=geometry,reference_type="author release baseline and synthetic follow-up"))
                print(json.dumps(dict(event="s1_source_unit",series=2,group=group,
                    original_lesions=len(geometry["history"]),readback=True),allow_nan=False))


if __name__=="__main__":
    main()
