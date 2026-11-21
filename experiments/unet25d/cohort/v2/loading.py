"""Bounded existing-shard loading, aligned physical masks and lesion crops."""
import gzip,hashlib,io,json,tarfile
from pathlib import Path
from collections import OrderedDict
import numpy as np
import torch
from torch.nn import functional as F
from ..loading import ShardCache
from .geometry import crop,effective,crop_origin
from .sampling import lesion_weight

def component_exposure(ids,target,weights,selected):
    """Diagnostic counts only; target and component weighting remain unchanged."""
    positive=target.bool();chosen=(ids==selected)&positive
    count=int(chosen.sum());total=int(positive.sum())
    return dict(selected_component_foreground_pixels=count,incidental_foreground_pixels=total-count,selected_component_BCE_share=float(weights[chosen].sum()/weights[positive].sum()) if total else None,target_scope='all enhancing components in crop')

class Cache(ShardCache):
    def __init__(self,root,fetch,c,inventory=None):
        super().__init__(root,fetch,c['gpu_cache_bytes'],c['decoded_cache_bytes_per_worker'])
        self.c=c;self.inventory=inventory or {};self.component_arrays=OrderedDict();self.component_bytes=0
        self.mask_counts={'planes':0,'fallbacks':0,'gt_excluded':0};self.fallback_reasons={}
    def arrays(self,visit,z):
        shard=next(r for r in visit['shards'] if r['start']<=z<r['end']);key=shard['sha256'];path=self.get(shard)
        if key not in self.decoded:
            if shard['decoded_bytes']>self.decoded_limit:raise MemoryError('Decoded shard bound')
            while self.decoded and self.decoded_bytes+shard['decoded_bytes']>self.decoded_limit:
                _,old=self.decoded.popitem(last=False);self.decoded_bytes-=sum(v.nbytes for v in old)
            with tarfile.open(path) as a:arrays=tuple(np.load(io.BytesIO(a.extractfile(n).read()),allow_pickle=False) for n in ('images.npy','raw-labels.npy'))
            if sum(x.nbytes for x in arrays)!=shard['decoded_bytes']:raise ValueError('Shard decoded bytes differ')
            self.decoded[key]=arrays;self.decoded_bytes+=shard['decoded_bytes']
        self.decoded.move_to_end(key);images,labels=self.decoded[key]
        indices=np.clip([z-1,z,z+1],0,visit['metadata']['shape'][2]-1)-shard['halo_start']
        return images[indices].transpose(1,0,2,3).reshape(12,*images.shape[-2:]).copy(),labels[z-shard['start']]
    def components(self,exam):
        if exam not in self.component_arrays:
            r=self.inventory[exam]['labels'];path=self.root/(r['sha256']+'.npy.gz')
            if not path.exists():self.fetch(r['key'],r['sha256'],path)
            else:
                with path.open('rb') as f:
                    if hashlib.file_digest(f,'sha256').hexdigest()!=r['sha256']:raise ValueError('Cached auxiliary integrity')
            with gzip.open(path,'rb') as f:arr=np.load(f,allow_pickle=False)
            if arr.dtype!=np.uint16 or arr.nbytes>self.c['component_cache_bytes']:raise MemoryError('Component cache encoding/bound')
            while self.component_arrays and self.component_bytes+arr.nbytes>self.c['component_cache_bytes']:
                _,old=self.component_arrays.popitem(last=False);self.component_bytes-=old.nbytes
            self.component_arrays[exam]=arr;self.component_bytes+=arr.nbytes
        self.component_arrays.move_to_end(exam);return self.component_arrays[exam]
    def sample_v2(self,visit,row,training=True):
        m=visit['metadata'];z=row['z'];images,raw=self.arrays(visit,z)
        roi,valid,reason=effective(images,m,z,self.c,raw==3 if training else None)
        self.mask_counts['planes']+=1
        if reason:self.mask_counts['fallbacks']+=1;self.fallback_reasons[reason]=self.fallback_reasons.get(reason,0)+1
        if not training:self.mask_counts['validation_GT_excluded_voxels']=self.mask_counts.get('validation_GT_excluded_voxels',0)+int(((raw==3)&~roi).sum())
        origin=row['origin'];size=row['scale'];x=crop(images*roi[None],origin,size);y=crop((raw==3)[None],origin,size);mask=crop(roi[None],origin,size)
        if not np.any(crop(valid,origin,size)):raise ValueError('Crop outside acquired support')
        ids=None
        if training:
            ids=crop((self.components(m['exam_id'])[:,:,z] if (raw==3).any() else np.zeros(raw.shape,np.uint16))[None],origin,size)
            if not np.array_equal(ids>0,y):raise ValueError('Component ID/raw label alignment mismatch')
        x=torch.from_numpy(x.copy());y=torch.from_numpy(y.astype(np.float32));mask=torch.from_numpy(mask.astype(np.float32))
        ids=torch.from_numpy(ids.astype(np.float32)) if ids is not None else None
        if size!=512:
            x=F.interpolate(x[None],size=(512,512),mode='bilinear',align_corners=False)[0]
            y=F.interpolate(y[None],size=(512,512),mode='nearest-exact')[0];mask=F.interpolate(mask[None],size=(512,512),mode='nearest-exact')[0]
            if ids is not None:ids=F.interpolate(ids[None],size=(512,512),mode='nearest-exact')[0]
        x*=mask
        weights=torch.ones_like(y);exposure={}
        if training:
            if bool(y.any())!=bool(row['positive']):raise ValueError('Planned versus actual foreground mismatch')
            if row['kind']=='replay' and not bool((ids==row['component']).any()):raise ValueError('Sampled lesion missing from actual crop')
            properties={r['id']:r for r in self.inventory[m['exam_id']]['components']}
            for label in torch.unique(ids).int().tolist():
                if label==0:continue
                loc=ids==label;pixels=int(loc.sum());volume=properties[label]['volume_mm3'];weights[loc]=lesion_weight(volume,self.c)/pixels
                exposure[str(label)]={'volume_mm3':volume,'pixels':pixels}
            if bool(y.any()):weights[y.bool()]*=float(y.sum())/float(weights[y.bool()].sum())
        identity=dict(row,actual_positive=bool(y.any()),valid_pixels=int(mask.sum()),foreground_pixels=int(y.sum()),source_size=size,zoom_scale=512/size,mask_fallback=reason,lesions=exposure)
        if training and row['kind']=='replay':identity.update(component_exposure(ids,y,weights,row['component']))
        if x.shape!=(12,512,512) or y.shape!=(1,512,512) or not torch.isfinite(x).all():raise ValueError('Frozen input shape/finiteness')
        return x,y,mask,weights,identity
    def sample(self,visit,tile,z):
        # Native validation remains full-FOV and GT-independent. No support mask
        # clips final predictions; only the same image-only input suppression.
        row=dict(origin=list(tile),scale=512,z=z)
        x,y,_,_,_=self.sample_v2(visit,row,training=False);return x,y

class TrainingInput(torch.utils.data.Dataset):
    def __init__(self,catalogue,inventory,plan,lease,root,c,cursor=0,rank=0):
        from ..stream import SignedFetch
        if cursor%2:raise ValueError('DDP cursor alignment')
        self.visits={v['metadata']['exam_id']:v for v in catalogue['visits'] if v['partition']=='train'}
        self.draws=plan[cursor+rank::2];self.inventory=inventory;self.cache=Cache(Path(root)/str(rank),SignedFetch(lease),c,inventory)
    def __len__(self):return len(self.draws)
    def __getitem__(self,i):
        row=self.draws[i]
        if row['exam'] not in self.visits:raise ValueError('Nontraining draw')
        return self.cache.sample_v2(self.visits[row['exam']],row,training=True)

def collate(rows):
    if len(rows)!=1:raise ValueError('Frozen local batch one')
    x,y,m,w,identity=rows[0];return x[None],y[None],m[None],w[None],identity
