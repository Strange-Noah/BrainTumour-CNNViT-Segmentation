"""Bounded shard cache. fetch(key,sha,destination) is supplied by the existing data staging path."""
import io,tarfile,hashlib
from collections import OrderedDict
from pathlib import Path
import numpy as np
import torch
class ShardCache:
    def __init__(self,root,fetch,max_bytes=3*1024**3,max_decoded_bytes=512*1024**2):
        self.root=Path(root);self.root.mkdir(parents=True,exist_ok=True);self.fetch=fetch;self.limit=max_bytes;self.cache=OrderedDict();self.bytes=0;self.decoded=OrderedDict();self.decoded_bytes=0;self.decoded_limit=max_decoded_bytes
        self.verified=set()
        for path in sorted(self.root.glob('*.tar.gz'),key=lambda p:p.stat().st_mtime):self.cache[path.name[:-7]]=(path,path.stat().st_size);self.bytes+=path.stat().st_size
        while self.cache and self.bytes>self.limit:
            _,(path,size)=self.cache.popitem(last=False);path.unlink();self.bytes-=size
    def get(self,receipt):
        key=receipt['sha256']
        if key not in self.cache:
            if receipt['bytes']>self.limit:raise MemoryError('Shard exceeds cache budget')
            while self.cache and self.bytes+receipt['bytes']>self.limit:
                _,(path,size)=self.cache.popitem(last=False);path.unlink();self.bytes-=size
            path=self.root/(key+'.tar.gz');self.fetch(receipt['key'],key,path)
            with path.open('rb') as f:actual=hashlib.file_digest(f,'sha256').hexdigest()
            if actual!=key or path.stat().st_size!=receipt['bytes']:path.unlink();raise ValueError('Shard integrity mismatch')
            self.cache[key]=(path,receipt['bytes']);self.bytes+=receipt['bytes'];self.verified.add(key)
        path=self.cache[key][0]
        if key not in self.verified:
            with path.open('rb') as f:actual=hashlib.file_digest(f,'sha256').hexdigest()
            if actual!=key or path.stat().st_size!=receipt['bytes']:raise ValueError('Cached shard integrity mismatch')
            self.verified.add(key)
        self.cache.move_to_end(key);return path
    def sample(self,visit,tile,z):
        shard=next(r for r in visit['shards'] if r['start']<=z<r['end']);path=self.get(shard);x,y=tile
        key=shard['sha256']
        if key not in self.decoded:
            if shard['decoded_bytes']>self.decoded_limit:raise MemoryError('Decoded shard exceeds bound')
            while self.decoded and self.decoded_bytes+shard['decoded_bytes']>self.decoded_limit:
                _,old=self.decoded.popitem(last=False);self.decoded_bytes-=sum(v.nbytes for v in old)
            with tarfile.open(path) as archive:
                arrays=tuple(np.load(io.BytesIO(archive.extractfile(n).read()),allow_pickle=False) for n in ['images.npy','raw-labels.npy'])
            if sum(v.nbytes for v in arrays)!=shard['decoded_bytes']:raise ValueError('Decoded shard size mismatch')
            self.decoded[key]=arrays;self.decoded_bytes+=shard['decoded_bytes']
        self.decoded.move_to_end(key);images,labels=self.decoded[key]
        centers=np.clip([z-1,z,z+1],0,visit['metadata']['shape'][2]-1)-shard['halo_start']
        sample=images[centers,:,x:x+512,y:y+512].transpose(1,0,2,3).reshape(12,512,512).copy()
        target=(labels[z-shard['start'],x:x+512,y:y+512]==3).astype(np.float32)[None]
        return torch.from_numpy(sample),torch.from_numpy(target)
