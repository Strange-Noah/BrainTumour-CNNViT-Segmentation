"""Lightweight scores from the normal training forward; 3D GT volume properties."""
import gzip,json
from pathlib import Path
import nibabel as nib
import numpy as np
from scipy import ndimage
import torch

def lesion_properties(visit,fetch,root):
    m=visit['metadata'];root=Path(root);root.mkdir(parents=True,exist_ok=True)
    path=root/(m['exam_id']+'.json')
    if path.exists():return json.loads(path.read_text())
    source=root/(m['exam_id']+'.npy.gz')
    fetch(visit['native_labels']['key'],visit['native_labels']['sha256'],source)
    try:
        with gzip.open(source,'rb') as f:mask=np.load(f,allow_pickle=False)==3
    finally:source.unlink(missing_ok=True)
    mask=np.pad(nib.orientations.apply_orientation(mask,np.asarray(m['orientation_transform'])),m['padding'])
    labels,n=ndimage.label(mask,structure=np.ones((3,3,3),bool));del mask
    volumes=np.bincount(labels.ravel())*abs(np.linalg.det(np.asarray(m['native_affine'])[:3,:3]));out={}
    for tile in m['tiles']:
        x,y=tile['origin']
        for z in tile['positive']:
            ids=np.unique(labels[x:x+512,y:y+512,z]);ids=ids[ids!=0]
            if not len(ids):raise ValueError('Catalogue/3D GT positive mismatch')
            out[str((x,y,z))]=float(volumes[ids].min())
    path.write_text(json.dumps(out));return out

def scores(logits,y,identity,step,rank):
    """Peak/top-k background error retains small confident FP structures."""
    with torch.no_grad():
        p=logits.detach().sigmoid();positive=y.bool();bg=p[~positive]
        negative=float((.5*bg.max()+.5*bg.topk(min(256,bg.numel())).values.mean()).cpu()) if bg.numel() else 0.
        base=dict(exam=identity['exam'],tile=identity['tile'],z=identity['z'],step=step,rank=rank)
        result=[dict(base,kind='negative',score=negative,min_volume_mm3=identity['min_volume_mm3'])]
        if bool(positive.any()):
            fn=(1-p[positive]).mean();dice=1-(2*(p*y).sum()+1)/(p.sum()+y.sum()+1)
            # Inner and outer GT boundary, without changing the training loss.
            dilation=torch.nn.functional.max_pool2d(y,3,1,1)
            erosion=1-torch.nn.functional.max_pool2d(1-y,3,1,1)
            edge=(dilation-erosion).bool();boundary=(p-y).abs()[edge].mean() if bool(edge.any()) else fn
            volume=identity['min_volume_mm3'];small=1. if 0<volume<100 else .5 if 0<volume<1000 else 0.
            result.append(dict(base,kind='positive',score=float((fn+dice+boundary).cpu())+small,min_volume_mm3=volume))
        return result
