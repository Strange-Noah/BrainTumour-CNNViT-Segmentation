"""File-backed 2.5D cache; strict geometry and bounded per-worker mappings."""
import json
from collections import OrderedDict
from pathlib import Path
import nibabel as nib
import numpy as np
import torch

class GeometryConflict(ValueError): pass

def prepare_visit(sample, folder):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    ref=sample['reference']; affine=ref.affine
    orientation=nib.orientations.io_orientation(affine)
    transform=nib.orientations.ornt_transform(orientation,nib.orientations.axcodes2ornt(('R','A','S')))
    canonical_affine=affine @ nib.orientations.inv_ornt_aff(transform,ref.shape)
    # Oblique grids need actual physical resampling, never just axis permutation.
    oblique=bool(np.max(np.abs(nib.affines.obliquity(affine)))>1e-4)
    native_shape=ref.shape
    axes=np.argsort(orientation[:,0]).astype(int)
    base_shape=tuple(map(int,np.asarray(native_shape)[axes]));spacing=np.asarray(nib.affines.voxel_sizes(affine))[axes]
    if oblique:
        import itertools
        corners=np.array(list(itertools.product(*[[-.5,n-.5] for n in native_shape])))
        world=nib.affines.apply_affine(affine,corners);low=world.min(0);high=world.max(0)
        needed=np.ceil((high-low)/spacing).astype(int)
        if np.any(needed[:2]>512):raise GeometryConflict(f'Oblique full-FOV native-spacing RAS grid needs {needed.tolist()}; cannot fit 512 without resolution/FOV change')
        shape=(512,512,int(needed[2]));canonical_affine=np.diag([*spacing,1.])
        canonical_affine[:3,3]=[(low[0]+high[0])/2-255.5*spacing[0],(low[1]+high[1])/2-255.5*spacing[1],low[2]+.5*spacing[2]]
        padding=[[0,0],[0,0],[0,0]]
    else:
        if any(n>512 for n in base_shape[:2]):raise GeometryConflict(f'Native RAS shape {base_shape} exceeds 512; cannot preserve both resolution and full FOV')
        padding=[[(512-n)//2,512-n-(512-n)//2] for n in base_shape[:2]]+[[0,0]]
        shift=np.eye(4);shift[:3,3]=[-p[0] for p in padding];canonical_affine=canonical_affine@shift
        shape=(512,512,base_shape[2])
    norm=np.lib.format.open_memmap(folder/'images.npy',mode='w+',dtype='float32',shape=(4,*shape))
    stats=[]
    for c,source in enumerate(sample['images']):
        if oblique:
            from nibabel.processing import resample_from_to
            arr=resample_from_to(nib.Nifti1Image(source,affine),(shape,canonical_affine),order=1).get_fdata(dtype=np.float32)
        else:arr=np.pad(nib.orientations.apply_orientation(source,transform),padding)
        if not np.isfinite(arr).all(): raise ValueError('Nonfinite channel: reject visit; no silent replacement')
        foreground=arr!=0; values=arr[foreground];mean=float(values.mean(dtype=np.float64)) if values.size else 0.;std=float(values.std(dtype=np.float64)) if values.size else 0.
        norm[c]=0
        if std>1e-8: norm[c][foreground]=(values-mean)/std
        stats.append(dict(count=int(values.size),mean=mean,std=std,degenerate=std<=1e-8))
    norm.flush(); del norm
    np.save(folder/'native-target.npy',sample['target'].astype(np.uint8))
    if oblique:
        target=resample_from_to(nib.Nifti1Image(sample['target'].astype(np.uint8),affine),(shape,canonical_affine),order=0).get_fdata().astype(np.uint8)
    else:target=np.pad(nib.orientations.apply_orientation(sample['target'],transform),padding).astype(np.uint8)
    np.save(folder/'target.npy',target)
    positives=np.flatnonzero(target.any(axis=(0,1))).tolist();negatives=np.flatnonzero(~target.any(axis=(0,1))).tolist()
    meta=dict(exam_id=sample['visit']['exam_id'],patient_id=sample['visit']['patient_id'],shape=list(shape),native_shape=list(ref.shape),native_affine=affine.tolist(),canonical_affine=canonical_affine.tolist(),orientation_transform=transform.tolist(),oblique_resampled=oblique,padding=padding,normalization=stats,positive=positives,negative=negatives)
    (folder/'visit.json').write_text(json.dumps(meta));return meta

def restore(prediction,meta,order=0):
    if meta.get('oblique_resampled'):
        from nibabel.processing import resample_from_to
        return resample_from_to(nib.Nifti1Image(np.asarray(prediction,dtype=np.float32),np.asarray(meta['canonical_affine'])),(tuple(meta['native_shape']),np.asarray(meta['native_affine'])),order=order).get_fdata(dtype=np.float32)
    slices=tuple(slice(p[0],prediction.shape[i]-p[1] if p[1] else None) for i,p in enumerate(meta.get('padding',[[0,0]]*3)))
    prediction=prediction[slices]
    # Inverse permutation/flips after removing recorded padding.
    orig=nib.orientations.io_orientation(np.asarray(meta['native_affine']))
    inverse=nib.orientations.ornt_transform(nib.orientations.axcodes2ornt(('R','A','S')),orig)
    result=nib.orientations.apply_orientation(prediction,inverse)
    if list(result.shape)!=meta['native_shape']: raise ValueError('Native restoration shape mismatch')
    return result

class SliceCache(torch.utils.data.Dataset):
    def __init__(self,root,draws,capacity=2,max_mapped_bytes=3*1024**3):
        if capacity<1 or max_mapped_bytes<1:raise ValueError('Positive cache bounds required')
        self.root=Path(root);self.draws=draws;self.capacity=capacity;self.max_mapped_bytes=max_mapped_bytes
        self.cache=OrderedDict();self.hits=0;self.misses=0;self.mapped_bytes=0
    def __len__(self):return len(self.draws)
    def _evict(self):
        _,(arrays,size)=self.cache.popitem(last=False)
        for a in arrays:a._mmap.close()
        self.mapped_bytes-=size
    def close(self):
        while self.cache:self._evict()
    def __getitem__(self,i):
        visit,z=self.draws[i];folder=self.root/visit
        if visit in self.cache:self.hits+=1;self.cache.move_to_end(visit)
        else:
            self.misses+=1;paths=[folder/'images.npy',folder/'target.npy'];size=sum(p.stat().st_size for p in paths)
            if size>self.max_mapped_bytes:raise MemoryError('Visit exceeds per-worker mapping byte bound')
            while self.cache and (len(self.cache)>=self.capacity or self.mapped_bytes+size>self.max_mapped_bytes):self._evict()
            arrays=[]
            try:
                for p in paths:arrays.append(np.load(p,mmap_mode='r'))
            except Exception:
                for a in arrays:a._mmap.close()
                raise
            self.cache[visit]=(arrays,size);self.mapped_bytes+=size
        (images,target),_=self.cache[visit];indices=np.clip([z-1,z,z+1],0,images.shape[-1]-1)
        x=np.stack([images[c,:,:,k] for c in range(4) for k in indices])
        y=np.array(target[:,:,z],dtype=np.float32,copy=True)[None]
        return torch.from_numpy(x),torch.from_numpy(y),dict(hit=self.hits,miss=self.misses,occupancy=len(self.cache),mapped_bytes=self.mapped_bytes)

def sample_draws(records,count,seed,world=1):
    rng=np.random.default_rng(seed); pools={True:{},False:{}}
    for r in records:
        for flag,key in [(True,'positive'),(False,'negative')]:
            if r[key]:pools[flag].setdefault(r['patient_id'],[]).append(r)
    if not all(pools.values()):raise ValueError('Need positive and negative patient pools')
    result=[]
    for i in range(count):
        phase=(i//world)%4;flag=phase in (0,2)
        # Alternate negative centers from tumor-bearing and wholly negative visits.
        candidates=pools[flag]
        if not flag:
            wanted=phase==1
            candidates={p:[r for r in visits if bool(r['positive'])==wanted] for p,visits in candidates.items()}
            candidates={p:v for p,v in candidates.items() if v}
            if not candidates:raise ValueError('Need both tumor-bearing and negative-visit negative centers')
        patients=sorted(candidates);patient=patients[int(rng.integers(len(patients)))];visits=candidates[patient];r=visits[int(rng.integers(len(visits)))];zs=r['positive' if flag else 'negative'];result.append((r['exam_id'],int(zs[int(rng.integers(len(zs)))])))
    return result


def locality_draws(records,count,seed,world=1,window=64):
    """Reorder fixed per-rank windows only; preserve every original patient-aware draw.

    Always generate the entire final window before truncation. A shorter resumed
    request therefore has exactly the same prefix as an uninterrupted request.
    """
    if world<1 or window<4 or window%4:raise ValueError('Invalid locality window')
    block=world*window;padded=((count+block-1)//block)*block
    original=sample_draws(records,padded,seed,world);result=[]
    for start in range(0,padded,block):
        streams=[]
        for rank in range(world):
            groups=OrderedDict()
            for visit,z in original[start+rank:start+block:world]:groups.setdefault(visit,[]).append(z)
            streams.append([(visit,z) for visit,zs in groups.items() for z in sorted(zs)])
        result.extend(streams[rank][i] for i in range(window) for rank in range(world))
    return result[:count]
