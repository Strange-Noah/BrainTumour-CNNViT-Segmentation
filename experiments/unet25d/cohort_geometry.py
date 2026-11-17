"""Existing audit metadata only: no medical pixel downloads or preprocessing."""
import hashlib,json
from concurrent.futures import ThreadPoolExecutor
from collections import Counter
import nibabel as nib
import numpy as np
from cloud.storage import connect,DATASET

def main():
    c,b=connect();index=json.loads(c.get_object(Bucket=b,Key=f'{DATASET}/reports/audit/full-v1/final/index.json')['Body'].read())
    def read(r):
        data=c.get_object(Bucket=b,Key=r['prefix']+'/files.jsonl')['Body'].read();assert hashlib.sha256(data).hexdigest()==r['files_sha256'];return [json.loads(l) for l in data.splitlines()]
    references=[f for rows in ThreadPoolExecutor(8).map(read,index) for f in rows if f['key'].endswith('-t1c.nii.gz')]
    assert len(references)==1065,'Reference filename selection must cover all exams'
    shapes=Counter();canonical=Counter();orientations=Counter();oblique=0
    for r in references:
        affine=np.asarray(r['effective_affine']);o=nib.orientations.io_orientation(affine);axes=np.argsort(o[:,0]).astype(int);shape=tuple(np.asarray(r['shape'])[axes]);canonical[str(tuple(map(int,shape[:2])))]+=1;shapes[str(tuple(r['shape'][:2]))]+=1;orientations[''.join(r['orientation'])]+=1;oblique+=int(np.max(np.abs(nib.affines.obliquity(affine)))>1e-4)
    print(json.dumps(dict(examinations=len(references),native_inplane=dict(shapes),ras_inplane=dict(canonical),orientations=dict(orientations),oblique_count=oblique,source='complete private full-v1 audit; no pixel downloads'),indent=2))
if __name__=='__main__':main()
