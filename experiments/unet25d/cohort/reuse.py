"""Reuse compatible verified benchmark normalization; only recover missing raw labels."""
import hashlib,json
from pathlib import Path
import nibabel as nib
import numpy as np
from ..persist_cache import read,sha
from ..repair_validation import CACHE_PREFIX,CACHE_SHA
from research.geometry import mask_geometry

def previous_index(client):
    raw=read(client,CACHE_PREFIX+'/manifest.json')
    if sha(raw)!=CACHE_SHA:raise ValueError('Original prepared-cache identity mismatch')
    return {r['path']:r for r in json.loads(raw)['files']}

def reuse_visit(ds,row,folder,client,files):
    prefix='prepared/'+row['exam_id']+'/'
    if prefix+'visit.json' not in files:return None
    def materialize(name):
        item=files[prefix+name];path=folder/name;h=hashlib.sha256();size=0
        with path.open('wb') as out:
            for chunk in item['chunks']:
                block=read(client,CACHE_PREFIX+'/chunks/'+chunk['sha256'])
                if len(block)!=chunk['bytes'] or sha(block)!=chunk['sha256']:raise ValueError('Reused preparation integrity failure')
                out.write(block);h.update(block);size+=len(block)
        if h.hexdigest()!=item['sha256'] or size!=item['bytes']:raise ValueError('Reused logical-file mismatch')
        return path
    meta=json.loads(materialize('visit.json').read_text())
    if meta['oblique_resampled'] or np.max(nib.affines.obliquity(np.asarray(meta['native_affine'])))>1e-4:return None
    if meta['patient_id']!=row['patient_id']:raise ValueError('Reused preparation identity mismatch')
    materialize('images.npy')
    reference=nib.Nifti1Image(np.empty(meta['native_shape'],np.uint8),np.asarray(meta['native_affine']));reference.header.set_xyzt_units('mm')
    label=nib.load(ds.fetch(row['segmentation']));warnings=mask_geometry(label,reference,row['exam_id']);raw=np.asanyarray(label.dataobj)
    if not np.isfinite(raw).all() or not np.isin(raw,[0,1,2,3,4]).all():raise ValueError('Invalid original labels')
    raw=raw.astype(np.uint8);np.save(folder/'native-raw-labels.npy',raw);np.save(folder/'native-target.npy',(raw==3).astype(np.uint8))
    transform=nib.orientations.ornt_transform(nib.orientations.io_orientation(reference.affine),nib.orientations.axcodes2ornt(('R','A','S')))
    canonical=np.pad(nib.orientations.apply_orientation(raw,transform),meta['padding']);np.save(folder/'raw-labels.npy',canonical);np.save(folder/'target.npy',(canonical==3).astype(np.uint8))
    meta.update(source_oblique=False,grid_policy='native-T1c-with-affine',geometry_warnings=warnings,reused_normalization_manifest_sha256=CACHE_SHA)
    return meta
