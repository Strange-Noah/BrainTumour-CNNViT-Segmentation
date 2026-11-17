"""Read existing audit metadata and inspect two train visits; never write raw data."""
import csv, hashlib, json, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import nibabel as nib
import numpy as np
from cloud.storage import connect, DATASET
from research.dataset import Dataset

def main():
    work=Path(sys.argv[1]);work.mkdir(parents=True,exist_ok=True);work.chmod(0o700)
    client,bucket=connect();index=json.loads(client.get_object(Bucket=bucket,Key=f'{DATASET}/reports/audit/full-v1/final/index.json')['Body'].read())
    def read(row):
        data=client.get_object(Bucket=bucket,Key=row['prefix']+'/files.jsonl')['Body'].read()
        assert hashlib.sha256(data).hexdigest()==row['files_sha256']
        return [json.loads(l) for l in data.splitlines()]
    files=[f for rows in ThreadPoolExecutor(8).map(read,index) for f in rows]
    train={r['patient_id'] for r in csv.DictReader(open('protocol/v1/patients.csv')) if r['partition']=='train'}
    masks=[f for f in files if 'label_voxels' in f and f['exam_id'].rsplit('_',1)[0] in train]
    enhancing=[r['label_voxels'].get('3.0',0) for r in masks]
    pos=next(r for r in masks if r['label_voxels'].get('3.0',0)>0); neg=next(r for r in masks if r.get('empty_mask'))
    manifest=work/'manifest.jsonl';client.download_file(bucket,f'{DATASET}/manifests/040ec6c523d5beae43a30b64fede2bee226e1a08e8c14b2cca8043b01b9c39f4/visits.jsonl',str(manifest))
    ds=Dataset(manifest,'protocol/v1/patients.csv',work/'cache','train');samples=[]
    for chosen in [pos,neg]:
        i=next(i for i,r in enumerate(ds.visits) if r['exam_id']==chosen['exam_id']);s=ds.load(i)
        target=s['target'];ref=s['reference'];positive=target.any(axis=(0,1));stats=[]
        for arr in s['images']:
            values=arr[arr!=0]; stats.append(dict(nonzero_fraction=float((arr!=0).mean()),percentiles=np.percentile(values,[0,0.5,1,50,99,99.5,100]).tolist(),mean=float(values.mean()),std=float(values.std())))
        samples.append(dict(kind='positive' if chosen is pos else 'negative',orientation=list(nib.aff2axcodes(ref.affine)),shape=list(ref.shape),spacing=list(map(float,ref.header.get_zooms())),slice_axis_code=nib.aff2axcodes(ref.affine)[2],adjacent_center_distance_mm=float(np.linalg.norm(ref.affine[:3,2])),positive_slices=int(positive.sum()),negative_slices=int((~positive).sum()),enhancing_voxels=int(target.sum()),modality_statistics=stats,float32_four_volume_bytes=int(s['images'].nbytes)))
    report=dict(train_visits=len(masks),train_native_axis2_slices=sum(r['shape'][2] for r in masks),train_empty_raw_masks=sum(bool(r.get('empty_mask')) for r in masks),train_no_enhancing_visits=sum(v==0 for v in enhancing),train_enhancing_voxel_percentiles=np.percentile(enhancing,[0,25,50,75,90,99,100]).tolist(),train_positive_enhancing_voxels_below_100=sum(0<v<100 for v in enhancing),train_positive_negative_slice_counts='not present in existing full audit; two-visit counts below only; no full-cohort preprocessing launched',representatives=samples)
    print(json.dumps(report,indent=2))
if __name__=='__main__':main()
