"""Physical validity is independent of intensity and GT. GT only guards train fallback."""
import numpy as np
from scipy import ndimage

def fov(meta,z,origin=(0,0),size=None):
    shape=tuple(size or meta['shape'][:2]);x,y=np.indices(shape,dtype=np.float64);x+=origin[0];y+=origin[1]
    transform=np.linalg.inv(np.asarray(meta['native_affine']))@np.asarray(meta['canonical_affine'])
    good=np.ones(shape,bool)
    for a,n in enumerate(meta['native_shape']):
        q=transform[a,0]*x+transform[a,1]*y+transform[a,2]*z+transform[a,3]
        good&=(q>=-.5+1e-5)&(q<n-.5-1e-5)
    return good

def support(images,valid,meta,c):
    if images.shape[-2:]!=valid.shape or not np.isfinite(images).all():raise ValueError('Support alignment/finiteness')
    # Intensities inform anatomy only, never acquired-FOV validity. Zeros inside
    # anatomical holes remain valid after fill, even at normalized mean intensity.
    seed=np.any(np.abs(images)>1e-6,axis=0)&valid
    fraction=seed.sum()/max(1,valid.sum())
    if fraction<c['support_min_seed_fraction']:return valid.copy(),'uncertain_image_support'
    basis=np.asarray(meta['canonical_affine'])[:3,:2]
    minimum=float(np.linalg.svd(basis,compute_uv=False).min())
    if not np.isfinite(minimum) or minimum<=0:raise ValueError('Invalid physical spacing')
    # min singular value gives a conservative enclosing index radius even for
    # sheared/oblique affines; it cannot make the physical margin too narrow.
    r=max(1,int(np.ceil(c['support_closing_mm']/minimum)))
    if r>24:return valid.copy(),'support_physical_scale_uncertain'
    a,b=np.ogrid[-r:r+1,-r:r+1];disk=a*a+b*b<=r*r
    closed=ndimage.binary_closing(seed,structure=disk,border_value=0)
    filled=ndimage.binary_fill_holes(closed|seed)
    expanded=ndimage.distance_transform_edt(~filled)<=c['support_margin_mm']/minimum
    result=(expanded|seed)&valid
    if result.sum()<c['support_min_fraction']*valid.sum():return valid.copy(),'support_safety_fraction'
    return result,None

def effective(images,meta,z,c,training_target=None):
    valid=fov(meta,z);anatomy,reason=support(images,valid,meta,c);result=valid&anatomy
    if training_target is not None:
        target=np.asarray(training_target,bool)
        if np.any(target&~valid):raise ValueError('Training GT outside acquired geometry; reject, do not erase')
        if np.any(target&~result):result=valid.copy();reason='training_GT_preservation_fallback'
        if np.any(target&~result):raise AssertionError('Foreground preservation contract')
    return result,valid,reason

def crop(array,origin,size):
    """Identical padded index transform for MRI, IDs and masks; no label shift."""
    x,y=origin;out=np.zeros((*array.shape[:-2],size,size),array.dtype)
    a,b=max(0,x),max(0,y);u,v=min(array.shape[-2],x+size),min(array.shape[-1],y+size)
    if u>a and v>b:out[...,a-x:u-x,b-y:v-y]=array[...,a:u,b:v]
    return out

def crop_origin(component,z,size,jitter,shape,seed):
    rng=np.random.default_rng(seed);bounds=component['planes'][str(z)];lo=np.array(bounds['lo']);hi=np.array(bounds['hi'])
    while size<512 and np.any(hi-lo>size):size=192 if size==128 and np.max(hi-lo)<=192 else 256 if size<256 and np.max(hi-lo)<=256 else 512
    center=np.asarray(bounds['centroid'])+rng.uniform(-jitter*size,jitter*size,2)
    origin=np.floor(center-(size-1)/2).astype(int)
    # Keep the entire selected in-plane lesion if it fits. Boundary padding is
    # permitted; a >512 lesion is an explicitly recorded partial-context sample.
    for k in range(2):
        if hi[k]-lo[k]<=size:origin[k]=np.clip(origin[k],hi[k]-size,lo[k])
    return origin.tolist(),size,bool(np.any(hi-lo>size))
