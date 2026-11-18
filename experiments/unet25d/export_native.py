"""Shared prediction export contract; also usable on saved logits with CPU only."""
import csv,hashlib,os
from pathlib import Path
import nibabel as nib
import numpy as np
from .data import restore
from research.evaluate import export
ROOT=Path(__file__).resolve().parents[2]
SPLIT_SHA='57f5af55141e16ecb0b85fdda63cc355175d9b35271126784eb22766698b36f0'

def export_native(prediction,meta,destination,training_sha,checkpoint_ref,split_path=None,expected_split_sha=SPLIT_SHA):
    split_path=Path(split_path) if split_path else ROOT/'protocol/v1/patients.csv'
    digest=hashlib.sha256(split_path.read_bytes()).hexdigest()
    if digest!=expected_split_sha:raise ValueError('Frozen split checksum mismatch')
    with split_path.open() as f:partitions={r['patient_id']:r['partition'] for r in csv.DictReader(f)}
    if partitions.get(meta['patient_id'])!='validation':raise ValueError('Expected frozen validation patient')
    if list(prediction.shape)!=meta['shape']:raise ValueError('Saved canonical logits shape mismatch')
    for start in range(0,prediction.shape[-1],16):
        if not np.isfinite(prediction[:,:,start:start+16]).all():raise FloatingPointError('Nonfinite saved logits')
    native=restore(prediction,meta,order=1)>=0
    reference=nib.Nifti1Image(np.empty(meta['native_shape'],dtype=np.uint8),np.asarray(meta['native_affine']));reference.header.set_xyzt_units('mm')
    export(native,reference,destination,dict(patient_id=meta['patient_id'],exam_id=meta['exam_id'],task='enhancing',git_sha=training_sha,model_version='unet25d-fp32-integrated',split_sha256=digest,checkpoint_ref=checkpoint_ref,export_git_sha=os.environ.get('NOVO_SOURCE_SHA','unrecorded')))
    recovered=nib.load(str(Path(destination)/'seg.nii.gz'));array=np.asarray(recovered.dataobj)
    if recovered.shape!=tuple(meta['native_shape']) or not np.allclose(recovered.affine,reference.affine,atol=1e-4) or not np.isin(array,[0,1]).all() or not np.array_equal(array,native):raise ValueError('Exported NIfTI roundtrip mismatch')
    return native,reference
