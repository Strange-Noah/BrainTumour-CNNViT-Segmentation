"""Reuse canonical manifest and completed audit headers; never inspect MRI pixels."""
import csv,hashlib,itertools,json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import nibabel as nib
import numpy as np
from cloud.storage import connect,DATASET
from .prepare import CONFIG,ROOT,MANIFEST_SHA
from .tiling import windows

def planning_summary():
    c,b=connect();body=c.get_object(Bucket=b,Key=f'{DATASET}/manifests/{MANIFEST_SHA}/visits.jsonl')['Body'];raw=body.read();body.close()
    if hashlib.sha256(raw).hexdigest()!=MANIFEST_SHA:raise ValueError('Manifest identity changed')
    visits=[json.loads(line) for line in raw.splitlines()];config=json.loads(CONFIG.read_text());split=ROOT/'protocol/v1/patients.csv'
    if hashlib.sha256(split.read_bytes()).hexdigest()!=config['split_sha256']:raise ValueError('Frozen split changed')
    with split.open() as f:partitions={r['patient_id']:r['partition'] for r in csv.DictReader(f)}
    body=c.get_object(Bucket=b,Key=f'{DATASET}/reports/audit/full-v1/final/index.json')['Body'];index=json.loads(body.read());body.close()
    def headers(row):
        body=c.get_object(Bucket=b,Key=row['prefix']+'/files.jsonl')['Body'];raw=body.read();body.close()
        if hashlib.sha256(raw).hexdigest()!=row['files_sha256']:raise ValueError('Existing audit metadata identity changed')
        return [r for r in map(json.loads,raw.splitlines()) if r['key'].endswith('-t1c.nii.gz')]
    refs={r['exam_id']:r for group in ThreadPoolExecutor(4).map(headers,index) for r in group}
    summary={p:dict(patients=len([x for x in partitions.values() if x==p]),visits=0,axial_planes=0,tile_planes=0,oversized_visits=0,oblique_visits=0,uncompressed_image_label_bytes=0) for p in ['train','validation']}
    for visit in visits:
        partition=partitions[visit['patient_id']]
        if partition not in summary:continue
        header=refs[visit['exam_id']];aff=np.asarray(header['effective_affine']);shape=np.asarray(header['shape']);orientation=nib.orientations.io_orientation(aff);axes=np.argsort(orientation[:,0]).astype(int);spacing=nib.affines.voxel_sizes(aff)[axes]
        oblique=bool(np.max(np.abs(nib.affines.obliquity(aff)))>1e-4)
        grid=shape[axes].copy() # Native lattice is retained; no oblique canvas expansion.
        grid[:2]=np.maximum(grid[:2],512);s=summary[partition];s['visits']+=1;s['axial_planes']+=int(grid[2]);s['tile_planes']+=len(windows(grid))*int(grid[2]);s['oversized_visits']+=int(any(shape[:2]>512));s['oblique_visits']+=int(oblique);s['uncompressed_image_label_bytes']+=int(np.prod(grid))*17
    summary['test_pixels_accessed']=False;summary['basis']='completed ingestion manifest and completed audit headers only; no repeated audit or MRI transfer';summary['split_sha256']=config['split_sha256']
    print(json.dumps(summary,indent=2))
if __name__=='__main__':planning_summary()
