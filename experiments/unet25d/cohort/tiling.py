import numpy as np
from functools import lru_cache

def starts(length,tile=512,stride=384):
    if length<tile:raise ValueError('Pad the physical visit grid before tiling')
    return sorted(set(range(0,length-tile+1,stride))|{length-tile})

def windows(shape):return [(x,y) for y in starts(shape[1]) for x in starts(shape[0])]

@lru_cache(maxsize=1)
def weights(tile=512):
    a=(np.arange(tile,dtype=np.float64)-(tile-1)/2)/(tile*.25)
    return np.maximum(np.exp(-.5*(a[:,None]**2+a[None,:]**2)),1e-6).astype(np.float32)

def merge_logits(tiles,shape):
    """One axial plane; keep memory bounded independently of visit depth."""
    total=np.zeros(shape[:2],np.float32);mass=np.zeros_like(total);w=weights()
    expected=windows(shape);seen=[]
    for (x,y),logits in tiles:
        if (x,y) not in expected or (x,y) in seen or logits.shape!=(512,512) or not np.isfinite(logits).all():raise ValueError('Invalid/incomplete tiled logits')
        total[x:x+512,y:y+512]+=logits*w;mass[x:x+512,y:y+512]+=w;seen.append((x,y))
    if set(seen)!=set(expected) or np.any(mass<=0):raise ValueError('Uncovered physical field of view')
    return total/mass
