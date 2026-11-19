"""One-time resumable visit-to-R2 shards. CPU only; no infrastructure changes."""
import argparse,csv,hashlib,io,json,os,tarfile,tempfile,time,gzip,shutil
from pathlib import Path
import nibabel as nib
import numpy as np
from nibabel.processing import resample_from_to
from cloud.storage import connect,verify,put_once,DATASET
from research.dataset import Dataset
from research.geometry import require_mm,mask_geometry
from ..data import prepare_visit
from .tiling import windows
ROOT=Path(__file__).resolve().parents[3];CONFIG=Path(__file__).with_name('config.json');MANIFEST_SHA='040ec6c523d5beae43a30b64fede2bee226e1a08e8c14b2cca8043b01b9c39f4'

def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def policy_hash():
    config=json.loads(CONFIG.read_text());keys=['schema','split_sha256','modalities','offsets','target_labels','preserve_raw_labels','tile','tile_stride','gaussian_sigma_pixels','normalization','physical_grid','image_interpolation','label_interpolation','centers_per_shard','context_halo']
    return hashlib.sha256(json.dumps({k:config[k] for k in keys},sort_keys=True).encode()).hexdigest()
def read_json_record(client,bucket,key):
    response=client.get_object(Bucket=bucket,Key=key);raw=response['Body'].read();response['Body'].close()
    if hashlib.sha256(raw).hexdigest()!=response.get('Metadata',{}).get('sha256'):raise ValueError('Prepared metadata integrity mismatch')
    return json.loads(raw)
def keybase():return 'novo-nordisk/experiments/unet25d/full-cohort/'+policy_hash()
def array_member(archive,name,array):
    data=io.BytesIO();np.save(data,array,allow_pickle=False);info=tarfile.TarInfo(name);info.size=data.tell();data.seek(0);archive.addfile(info,data)

def prepare_dataset_visit(ds,index,folder):
    row=ds.visits[index];reference=nib.load(ds.fetch(row['modalities']['t1c']));require_mm(reference)
    label=nib.load(ds.fetch(row['segmentation']));warnings=mask_geometry(label,reference,row['exam_id'])
    raw=np.asanyarray(label.dataobj)
    if not np.isfinite(raw).all() or not np.isin(raw,[0,1,2,3,4]).all():raise ValueError('Invalid original label')
    raw=raw.astype(np.uint8)
    def images():
        for name in ('t1n','t1c','t2','t2f'):
            image=nib.load(ds.fetch(row['modalities'][name]));require_mm(image,name)
            yield image # Direct source affine -> final physical grid; no intermediate oblique interpolation.
    meta=prepare_visit(dict(reference=reference,images=images(),raw_labels=raw,target=raw==3,visit=row),folder,allow_tiles=True)
    meta['geometry_warnings']=warnings
    return meta

def shard_visit(folder,meta,client,bucket,prefix):
    images=np.load(folder/'images.npy',mmap_mode='r');labels=np.load(folder/'raw-labels.npy',mmap_mode='r');target=np.load(folder/'target.npy',mmap_mode='r');rows=[]
    meta['tiles']=[]
    for x,y in windows(meta['shape']):
        positive=np.flatnonzero(target[x:x+512,y:y+512,:].any(axis=(0,1))).tolist()
        meta['tiles'].append(dict(origin=[x,y],positive=positive,negative=sorted(set(range(meta['shape'][2]))-set(positive))))
    for start in range(0,meta['shape'][2],8):
        end=min(start+8,meta['shape'][2]);lo=max(0,start-1);hi=min(meta['shape'][2],end+1);path=folder/'shard.tar.gz'
        with tarfile.open(folder/'shard.tar','w') as archive:
            array_member(archive,'images.npy',np.ascontiguousarray(images[:,:,:,lo:hi].transpose(3,0,1,2)))
            array_member(archive,'raw-labels.npy',np.ascontiguousarray(labels[:,:,start:end].transpose(2,0,1)))
        with (folder/'shard.tar').open('rb') as src,path.open('wb') as raw_output:
            with gzip.GzipFile(filename='',fileobj=raw_output,mode='wb',mtime=0,compresslevel=1) as dest:shutil.copyfileobj(src,dest)
        (folder/'shard.tar').unlink()
        receipt=put_once(client,bucket,prefix+'/shards/'+digest(path)+'.tar.gz',path)
        rows.append(dict(**receipt,start=start,end=end,halo_start=lo,decoded_bytes=(hi-lo)*4*meta['shape'][0]*meta['shape'][1]*4+(end-start)*meta['shape'][0]*meta['shape'][1]));path.unlink()
    native=folder/'native-labels.npy.gz'
    with (folder/'native-raw-labels.npy').open('rb') as src,native.open('wb') as raw_output:
        with gzip.GzipFile(filename='',fileobj=raw_output,mode='wb',mtime=0,compresslevel=1) as dest:shutil.copyfileobj(src,dest)
    native_receipt=put_once(client,bucket,prefix+'/native-labels/'+digest(native)+'.npy.gz',native)
    return dict(metadata=meta,shards=rows,native_labels=native_receipt)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--partition',choices=['train','validation'],required=True);parser.add_argument('--start',type=int,default=0);parser.add_argument('--patients',type=int,default=1);parser.add_argument('--minutes',type=int,default=30);a=parser.parse_args()
    if a.start<0 or not 1<=a.patients<=210 or not 1<=a.minutes<=30:raise ValueError('Invalid bounded preparation batch')
    import torch
    if torch.cuda.is_available():raise RuntimeError('Preparation is CPU only')
    split=ROOT/'protocol/v1/patients.csv';config=json.loads(CONFIG.read_text())
    if digest(split)!=config['split_sha256']:raise ValueError('Frozen split changed')
    reader,raw_bucket=connect();writer,output_bucket=connect(output=True)
    if output_bucket!='novo-nordisk-pbl-outputs':raise ValueError('Private outputs required')
    deadline=time.monotonic()+min(a.minutes,30)*60;completed=skipped=0
    from .reuse import previous_index,reuse_visit
    previous=previous_index(writer)
    with tempfile.TemporaryDirectory(prefix='novo-private-prepare-') as d:
        root=Path(d);root.chmod(0o700);manifest=root/'visits.jsonl'
        with manifest.open('wb') as f:verify(reader,raw_bucket,f'{DATASET}/manifests/{MANIFEST_SHA}/visits.jsonl',MANIFEST_SHA,f)
        ds=Dataset(manifest,split,root/'raw',a.partition);patients=sorted({r['patient_id'] for r in ds.visits})[a.start:a.start+a.patients]
        for patient in patients:
            for index,row in enumerate(ds.visits):
                if row['patient_id']!=patient:continue
                if time.monotonic()>deadline:print(json.dumps(dict(completed_visits=completed,previously_prepared=skipped,stopped='bounded_cpu_time',gpu_allocated=False)));return
                prefix=keybase()+'/'+a.partition+'/'+row['exam_id'];key=prefix+'/receipt.json'
                try:
                    receipt=read_json_record(writer,output_bucket,key)
                except writer.exceptions.NoSuchKey:receipt=None
                if receipt:
                    if receipt['data_policy_sha256']!=policy_hash():raise ValueError('Incompatible existing preparation')
                    skipped+=1;continue
                folder=root/'visit';folder.mkdir();meta=reuse_visit(ds,row,folder,writer,previous)
                if meta is None:meta=prepare_dataset_visit(ds,index,folder)
                record=shard_visit(folder,meta,writer,output_bucket,prefix)
                record.update(data_policy_sha256=policy_hash(),config_sha256=digest(CONFIG),source_sha=os.environ['NOVO_SOURCE_SHA'],partition=a.partition,split_sha256=config['split_sha256'])
                path=root/'receipt.json';path.write_text(json.dumps(record));put_once(writer,output_bucket,key,path)
                import shutil
                shutil.rmtree(folder)
                for file in ds.cache.iterdir():file.unlink()
                completed+=1
    print(json.dumps(dict(completed_visits=completed,previously_prepared=skipped,gpu_allocated=False)))
if __name__=='__main__':main()
