"""Model entry adapter within the existing Actions/Kaggle/R2 bridge; no new infrastructure."""
import json,os,time
from pathlib import Path
from .settings import CONFIG

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
    from .unattended import launch
    return launch(dataset,kernel)
