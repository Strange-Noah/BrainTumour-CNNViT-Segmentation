"""Model entry adapter within the existing Actions/Kaggle/R2 bridge; no new infrastructure."""
import json,os,time
from pathlib import Path
from .prepare import CONFIG

def render(sha,session,budget):
    return '''import os,sys,time,io,urllib.request,zipfile,subprocess,shutil
from pathlib import Path
os.environ['NOVO_SESSION_STARTED_AT']=str(time.time())
sha=__SHA_LITERAL__
root=Path('/tmp/novo-source');root.mkdir(exist_ok=True)
with urllib.request.urlopen('https://github.com/Strange-Noah/BrainTumour-CNNViT-Segmentation/archive/'+sha+'.zip') as r:z=zipfile.ZipFile(io.BytesIO(r.read()))
for entry in z.infolist():
 p=root/Path(*Path(entry.filename).parts[1:])
 if not p.resolve().is_relative_to(root.resolve()):raise ValueError('Invalid source archive')
 if entry.is_dir():p.mkdir(parents=True,exist_ok=True)
 else:p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(z.read(entry))
subprocess.run([sys.executable,'-m','pip','install','-q','-r',str(root/'experiments/unet25d/requirements.txt')],check=True)
inputs=list(Path('/kaggle/input').rglob('private-data-lease.json'))
if len(inputs)!=1:raise ValueError('Exactly one private cohort input required')
source=inputs[0].parent;out=Path('/kaggle/working/novo-unet25d-full');os.environ['NOVO_SOURCE_SHA']=sha
command=[sys.executable,'-m','torch.distributed.run','--standalone','--nproc_per_node=2','-m','experiments.unet25d.cohort.train_full','--input',str(source),'--output',str(out),'--approved-budget-seconds',__BUDGET_LITERAL__,'--session-seconds',__SESSION_LITERAL__]
if (source/'resume.pt').exists():command+=['--resume',str(source/'resume.pt')]
try:subprocess.run(command,cwd=root,check=True)
finally:shutil.rmtree(out/'cache',ignore_errors=True)
'''.replace('__SHA_LITERAL__',repr(sha)).replace('__BUDGET_LITERAL__',repr(str(budget))).replace('__SESSION_LITERAL__',repr(str(session)))

def submit_one(dataset,kernel):
    config=json.loads(CONFIG.read_text())
    if not config['formal_training_authorized'] or config['approved_budget_seconds']<=0:raise PermissionError('Formal training has no owner budget authorization')
    from cloud.kaggle_benchmark_bridge import WORK,cli,collect
    from cloud.storage import connect
    from .stage import latest_key,publish_from_existing_runtime
    c,b=connect(output=True);prior=0
    try:
        response=c.get_object(Bucket=b,Key=latest_key());state=json.loads(response['Body'].read())['state'];response['Body'].close();prior=state['consumed_seconds']
    except c.exceptions.NoSuchKey:pass
    remaining=int(config['approved_budget_seconds']-prior)
    if remaining<600:print(json.dumps(dict(formal_budget_exhausted=True,gpu_started=False)));return
    if 'state' in locals() and state['epoch']>=config['maximum_proposed_coverage_epochs'] and not state['pending_validation']:print(json.dumps(dict(research_training_complete=True,gpu_started=False)));return
    publish_from_existing_runtime(dataset)
    readiness=time.monotonic()+600
    while time.monotonic()<readiness:
        try:
            if 'READY' in cli('datasets','status',dataset).upper():break
        except RuntimeError:pass # Existing Kaggle publication processing can temporarily reject a status read.
        time.sleep(20)
    else:raise TimeoutError('Private input publication did not finish processing')
    session=min(3600,int(os.environ['GPU_SESSION_SECONDS']),remaining)
    folder=WORK/'full-kernel';folder.mkdir(exist_ok=True);sha=os.environ['NOVO_SOURCE_SHA'];(folder/'run.py').write_text(render(sha,session,config['approved_budget_seconds']))
    (folder/'kernel-metadata.json').write_text(json.dumps(dict(id=kernel,title='Novo Full Cohort Training',code_file='run.py',language='python',kernel_type='script',is_private=True,enable_gpu=True,enable_internet=True,dataset_sources=[dataset],competition_sources=[],kernel_sources=[],machine_shape='NvidiaTeslaT4')))
    cli('kernels','push','-p',str(folder),'-t',str(session),'--accelerator','NvidiaTeslaT4')
    deadline=time.monotonic()+4500
    while time.monotonic()<deadline:
        status=cli('kernels','status',kernel).upper()
        if any(x in status for x in ['COMPLETE','ERROR','CANCEL']):break
        time.sleep(45)
    else:raise TimeoutError('Bounded formal session did not reach terminal state')
    collect(kernel,dataset,'formal')
    if 'COMPLETE' not in status:raise RuntimeError('Formal training correctness failure; stop without speculative retry')
