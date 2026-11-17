"""Inspect two previously cached train visits; refuses new imaging downloads."""
import hashlib,json,tempfile
from pathlib import Path
import numpy as np
from research.dataset import Dataset
from .data import prepare_visit,restore,SliceCache

def main():
    root=Path('/private/tmp/novo-smp-private')
    ds=Dataset(root/'manifest.jsonl','protocol/v1/patients.csv',root/'cache','train')
    def cached(relative):
        response=ds.client.head_object(Bucket=ds.bucket,Key='novo-nordisk/datasets/ucsd-bmets/v1/raw/'+relative);sha=response['Metadata']['sha256'];path=ds.cache/(sha+('.nii.gz' if relative.endswith('.gz') else '.nii'))
        if not path.exists():raise RuntimeError('Required representative not already cached; no imaging transfer allowed in this test')
        with path.open('rb') as stream:assert hashlib.file_digest(stream,'sha256').hexdigest()==sha
        return path
    ds.fetch=cached
    empty=set(json.loads(Path('/private/tmp/novo-comprehensive-audit/summary.json').read_text())['empty_mask_examinations'])
    negative=next(i for i,r in enumerate(ds.visits) if r['exam_id'] in empty)
    results=[]
    for index in [0,negative]:
        sample=ds.load(index)
        with tempfile.TemporaryDirectory() as name:
            meta=prepare_visit(sample,Path(name)/'visit');target=np.load(Path(name)/'visit/target.npy');assert np.array_equal(restore(target,meta),sample['target'])
            dataset=SliceCache(name,[('visit',0),('visit',meta['shape'][-1]-1)]);first,_,_=dataset[0];last,_,_=dataset[1];assert first.shape==(12,512,512) and np.array_equal(first[0],first[1]) and np.array_equal(last[1],last[2])
            results.append(dict(shape=meta['shape'],enhancing_voxels=int(sample['target'].sum()),positive_slices=len(meta['positive']),negative_slices=len(meta['negative']),restoration_exact=True,boundaries_correct=True,normalization=meta['normalization']))
            del dataset,first,last,target
        del sample
    print(json.dumps(dict(representatives=results,new_image_downloads=0,gpu_allocated=False),indent=2))
if __name__=='__main__':main()
