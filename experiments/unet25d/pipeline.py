"""Restartable stages: CPU prepare, bounded GPU campaign, fresh-process recovery."""
import argparse, contextlib, hashlib, json, os, subprocess, sys, time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from .data import prepare_visit
from .monitor import run

ROOT=Path(__file__).resolve().parents[2]
CONFIG=ROOT/'experiments/unet25d/config.json'

def digest(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

@contextlib.contextmanager
def stage(work,name):
    record=work/f'{name}.stage.json';start=time.time()
    record.write_text(json.dumps(dict(stage=name,status='running',started=start)))
    try:
        yield
    except Exception as e:
        record.write_text(json.dumps(dict(stage=name,status='failed',error_type=type(e).__name__,seconds=time.time()-start)));raise
    else:record.write_text(json.dumps(dict(stage=name,status='complete',seconds=time.time()-start)))

def prepare(work,on_visit=None):
    import csv,numpy as np
    from research.dataset import Dataset
    from cloud.storage import connect,DATASET
    work.mkdir(parents=True,exist_ok=True);work.chmod(0o700)
    with stage(work,'environment'):
        import torch
        if torch.cuda.is_available():raise RuntimeError('Prepare in CPU session before requesting GPUs')
        check=subprocess.run([sys.executable,'-m','pip','check'],capture_output=True,text=True)
        (work/'dependency-check.txt').write_text(check.stdout+check.stderr)
        if check.returncode:(work/'dependency-warning.json').write_text(json.dumps(dict(pip_check_passed=False,note='Existing environment conflicts recorded; pinned model initialization/forward stage must still pass')))
    with stage(work,'storage_preflight'):
        import tempfile
        from cloud.storage import connect,put_once
        output,bucket=connect(output=True)
        if bucket!='novo-nordisk-pbl-outputs':raise ValueError('Separate output bucket required')
        fixture=work/'storage-fixture.json';fixture.write_text(json.dumps(dict(config_sha256=digest(CONFIG),purpose='benchmark-output-preflight')))
        receipt=put_once(output,bucket,'novo-nordisk/experiments/unet25d/preflight/'+digest(CONFIG)+'/fixture.json',fixture)
        (work/'storage-preflight.json').write_text(json.dumps(receipt))
    with stage(work,'manifest'):
        client,bucket=connect();manifest=work/'manifest.jsonl';client.download_file(bucket,f'{DATASET}/manifests/040ec6c523d5beae43a30b64fede2bee226e1a08e8c14b2cca8043b01b9c39f4/visits.jsonl',str(manifest))
        train=Dataset(manifest,ROOT/'protocol/v1/patients.csv',work/'raw-cache','train');validation=Dataset(manifest,ROOT/'protocol/v1/patients.csv',work/'raw-cache','validation')
    with stage(work,'cohort_geometry'):
        index=json.loads(client.get_object(Bucket=bucket,Key=f'{DATASET}/reports/audit/full-v1/final/index.json')['Body'].read())
        def get(row):
            b=client.get_object(Bucket=bucket,Key=row['prefix']+'/files.jsonl')['Body'].read();assert hashlib.sha256(b).hexdigest()==row['files_sha256'];return [json.loads(l) for l in b.splitlines()]
        headers=[f for records in ThreadPoolExecutor(8).map(get,index) for f in records]
        masks={f['exam_id']:f for f in headers if 'label_voxels' in f}
        from collections import Counter
        import nibabel as nib
        references=[f for f in headers if f['key'].endswith('-t1c.nii.gz')]
        shapes=Counter(str(f['shape']) for f in references)
        oblique_count=sum(bool(np.max(np.abs(nib.affines.obliquity(np.asarray(f['effective_affine']))))>1e-4) for f in references)
        (work/'cohort-geometry.json').write_text(json.dumps(dict(t1c_shape_counts=dict(shapes),oblique_t1c_count=oblique_count,note='Native shapes only; strict RAS/oblique check per prepared visit; do not assume canonical 512 from native shape'),indent=2))
    with stage(work,'initialization'):
        from experiments.smp_audit.model_probe import main as probe
        probe() # Verifies pinned encoder and CPU forward before allocating GPUs.
    with stage(work,'cache'):
        import nibabel as nib,itertools
        def eligible(r):
            f=masks[r['exam_id']];affine=np.asarray(f['effective_affine']);orientation=nib.orientations.io_orientation(affine);axes=np.argsort(orientation[:,0]).astype(int)
            if np.max(np.abs(nib.affines.obliquity(affine)))<=1e-4:return all(np.asarray(f['shape'])[axes][:2]<=512)
            corners=np.array(list(itertools.product(*[[-.5,n-.5] for n in f['shape']])));world=nib.affines.apply_affine(affine,corners);spacing=np.asarray(nib.affines.voxel_sizes(affine))[axes];return bool(np.all(np.ceil(np.ptp(world,axis=0)/spacing)[:2]<=512))
        pool=[r for r in train.visits if eligible(r)]
        (work/'benchmark-subset-policy.json').write_text(json.dumps(dict(purpose='bounded benchmark only; not a full-training exclusion policy',eligible_train_visits=len(pool),all_train_visits=len(train.visits))))
        selected=[];seen=set()
        for want_positive in [True,False]*4:
            row=next(r for r in pool if r['patient_id'] not in seen and ((masks[r['exam_id']]['label_voxels'].get('3.0',0)>0) if want_positive else masks[r['exam_id']].get('empty_mask',False)))
            selected.append(row);seen.add(row['patient_id'])
        train_records=[];conflicts=[]
        for partition,ds,rows in [('train',train,selected),('validation',validation,[next(r for r in validation.visits if eligible(r))])]:
            for row in rows:
                folder=work/'prepared'/row['exam_id'];complete=folder/'visit.json'
                if complete.exists(): meta=json.loads(complete.read_text())
                else:
                    sample=ds.load(next(i for i,r in enumerate(ds.visits) if r['exam_id']==row['exam_id']))
                    from .data import GeometryConflict
                    try:meta=prepare_visit(sample,folder)
                    except GeometryConflict as e:conflicts.append(dict(exam=row['exam_id'],reason=str(e)));continue
                    finally:del sample
                if on_visit:on_visit(folder)
                if partition=='train':train_records.append(meta)
                else:(work/'prepared/validation-index.json').write_text(json.dumps([meta]))
                if sum(p.stat().st_size for p in (work/'prepared').rglob('*') if p.is_file())>16*1024**3:raise RuntimeError('Prepared disk-cache cap 16 GiB exceeded')
                # Bound decoded acquisition cache to a single visit, then release it.
                for file in (work/'raw-cache').iterdir():file.unlink()
        (work/'geometry-conflicts.json').write_text(json.dumps(conflicts))
        if conflicts:raise ValueError('Geometry conflict: review explicit records; no silent exclusions')
        (work/'prepared/train-index.json').write_text(json.dumps(train_records))
    with stage(work,'cache_integrity'):
        receipts={str(p.relative_to(work/'prepared')):digest(p) for p in (work/'prepared').rglob('*') if p.is_file()}
        (work/'cache-receipts.json').write_text(json.dumps(receipts))
        (work/'config.json').write_bytes(CONFIG.read_bytes())

def campaign(work):
    if not (work/'cache_integrity.stage.json').exists():raise ValueError('CPU preparation required')
    if json.loads((work/'cache_integrity.stage.json').read_text())['status']!='complete':raise ValueError('CPU preparation incomplete')
    if (work/'config.json').read_bytes()!=CONFIG.read_bytes():raise ValueError('Config changed since preparation')
    for name,sha in json.loads((work/'cache-receipts.json').read_text()).items():
        if digest(work/'prepared'/name)!=sha:raise ValueError('Prepared cache changed')
    from cloud.storage import connect
    kaggle_artifacts=os.environ.get('NOVO_ARTIFACT_MODE')=='kaggle'
    if not kaggle_artifacts:
        output_client,output_bucket=connect(output=True)
        if output_bucket!='novo-nordisk-pbl-outputs':raise ValueError('Output preflight missing')
    import torch
    if torch.cuda.device_count()!=2 or not all('T4' in torch.cuda.get_device_name(i) for i in range(2)):raise RuntimeError('Require actual free dual T4 runtime')
    start=time.time();deadline=start+3300 # 55 minutes in-process; launcher adds 60 minute outer timeout.
    candidates=[('A-short',1,1,1,50,120),('B-short',2,1,1,50,120)]
    completed=[]
    with stage(work,'benchmark'):
        for name,ranks,batch,workers,steps,seconds in candidates:
            folder=work/name;command=[sys.executable,'-m','torch.distributed.run','--standalone',f'--nproc_per_node={ranks}','-m','experiments.unet25d.train','--cache',str(work/'prepared'),'--out',str(folder),'--steps',str(steps),'--seconds',str(seconds),'--batch',str(batch),'--workers',str(workers)]
            if run(command,folder,deadline)!=0:raise RuntimeError('Benchmark failed; inspect private diagnostics, no automatic retry')
            completed.append((name,ranks,batch,workers))
        # Sustain both baselines for ten measured steady-state minutes each.
        for name,ranks,batch,workers in [('A-sustained',1,1,1),('B-sustained',2,1,1)]:
            if time.time()+780>deadline:break
            folder=work/name;seconds=660 if 'sustained' in name else 120
            code=run([sys.executable,'-m','torch.distributed.run','--standalone',f'--nproc_per_node={ranks}','-m','experiments.unet25d.train','--cache',str(work/'prepared'),'--out',str(folder),'--steps','100000' if 'sustained' in name else '50','--seconds',str(seconds),'--batch',str(batch),'--workers',str(workers)],folder,deadline)
            if code:break # Preserve evidence; no silent resolution, worker or architecture change.
            completed.append((name,ranks,batch,workers))
    subprocess.run([sys.executable,'-m','experiments.unet25d.report',str(work)],check=True)
    measured=json.loads((work/'performance.json').read_text())
    stable=[r for r in measured if r['sustained_accepted'] and r['rss_plateau'] and r['late_rank0_throughput']>=.75*r['early_rank0_throughput']]
    if not stable:raise RuntimeError('No accepted sustained candidate; stop, retain diagnostics')
    best=max(stable,key=lambda r:r['samples_per_second']);ranks=best['world_size']
    for name,batch,workers in [('C-workers2',1,2),('D-batch2',2,1)]:
        if time.time()+300>deadline:break
        folder=work/name
        command=[sys.executable,'-m','torch.distributed.run','--standalone',f'--nproc_per_node={ranks}','-m','experiments.unet25d.train','--cache',str(work/'prepared'),'--out',str(folder),'--steps','50','--seconds','120','--batch',str(batch),'--workers',str(workers)]
        if run(command,folder,deadline):break
        completed.append((name,ranks,batch,workers))
    with stage(work,'validation_smoke'):
        subprocess.run([sys.executable,'-m','experiments.unet25d.validate',str(work/'prepared'),str(work/'B-short/checkpoint.pt'),str(work/'validation')],check=True,timeout=300)
    with stage(work,'resume_training'):
        command=[sys.executable,'-m','torch.distributed.run','--standalone','--nproc_per_node=2','-m','experiments.unet25d.train','--cache',str(work/'prepared'),'--out',str(work/'resume'),'--steps','2','--seconds','60','--resume',str(work/'B-short/checkpoint.pt')]
        if run(command,work/'resume',deadline):raise RuntimeError('Resume continuation failed')
        completed.append(('resume',2,1,1))
    subprocess.run([sys.executable,'-m','experiments.unet25d.report',str(work)],check=True)
    (work/'campaign.json').write_text(json.dumps(dict(started=start,seconds=time.time()-start,device_hours=2*(time.time()-start)/3600,quota_accounting='unconfirmed platform accounting; elapsed session and device-time reported separately',completed=completed)))
    if kaggle_artifacts:
        with stage(work,'local_checkpoint_recovery'):
            subprocess.run([sys.executable,'-m','experiments.unet25d.recover_local',str(work)],check=True)
        return # Private Kaggle outputs are persisted/readback-verified by Actions.
    with stage(work,'persistence'):
        from cloud.storage import connect,put_once
        client,bucket=connect(output=True)
        if bucket!='novo-nordisk-pbl-outputs':raise ValueError('Output bucket required')
        run_id=hashlib.sha256((digest(CONFIG)+str(start)).encode()).hexdigest()[:16];prefix=f'novo-nordisk/experiments/unet25d/{run_id}'
        receipts=[]
        paths=[]
        for folder in [work/name for name,*_ in completed]+[work/'validation']:
            paths.extend(p for p in folder.rglob('*') if p.is_file())
        paths.extend(p for p in work.iterdir() if p.is_file() and p.name not in ['manifest.jsonl','output-receipts.json','cache-receipts.json'])
        for p in paths:receipts.append(put_once(client,bucket,prefix+'/'+str(p.relative_to(work)),p))
        (work/'output-receipts.json').write_text(json.dumps(receipts))
    with stage(work,'independent_readback'):
        subprocess.run([sys.executable,'-m','experiments.unet25d.recover',str(work/'output-receipts.json'),str(work/'recovered')],check=True)

def persist_failure(work,error):
    # Best-effort diagnostics, independently of successful stage completion.
    from cloud.storage import connect,put_once
    (work/'failure.json').write_text(json.dumps(dict(error_type=type(error).__name__,time=time.time())))
    try:
        client,bucket=connect(output=True)
        if bucket!='novo-nordisk-pbl-outputs':return
        prefix='novo-nordisk/experiments/unet25d/failures/'+str(int(time.time()))
        paths=list(work.glob('*.stage.json'))+[work/'failure.json']
        for folder in work.iterdir():
            if folder.is_dir() and folder.name not in ['prepared','raw-cache','hf','recovered']:
                paths.extend(p for p in folder.iterdir() if p.is_file() and p.suffix in ['.json','.jsonl','.log','.pt'])
        receipts=[put_once(client,bucket,prefix+'/'+str(p.relative_to(work)),p) for p in paths]
        (work/'failure-output-receipts.json').write_text(json.dumps(receipts))
    except Exception as persistence_error:
        (work/'failure-persistence.json').write_text(json.dumps(dict(error_type=type(persistence_error).__name__,persisted=False)))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','campaign']);p.add_argument('work',type=Path);a=p.parse_args();os.chdir(ROOT)
    work=a.work.resolve();work.mkdir(parents=True,exist_ok=True)
    try:(prepare if a.mode=='prepare' else campaign)(work)
    except Exception as error:
        persist_failure(work,error)
        raise
