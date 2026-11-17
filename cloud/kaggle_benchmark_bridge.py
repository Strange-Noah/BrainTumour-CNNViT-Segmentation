"""Actions-only R2 credentials; Kaggle receives private files and no project secrets."""
import hashlib,json,os,re,shutil,subprocess,sys,tempfile,time
from pathlib import Path
from cloud.storage import connect,put_once,verify
ROOT=Path.cwd();WORK=Path(os.environ.get('RUNNER_TEMP','/tmp'))/'novo-benchmark-bridge'
WORK.mkdir(exist_ok=True);WORK.chmod(0o700)
def cli(*args):
 p=subprocess.run(['kaggle',*args],capture_output=True,text=True,timeout=600)
 if p.returncode:raise RuntimeError('Kaggle CLI failed: '+args[0]+' '+args[1])
 return p.stdout

def publish_result(name,path):
 c,b=connect(output=True);return put_once(c,b,f'novo-nordisk/experiments/unet25d/bridge/{os.environ["GITHUB_RUN_ID"]}/{name}',path)

def restore_existing(work):
 from experiments.unet25d.persist_cache import BASE,sha,read
 c,b=connect(output=True)
 for page in c.get_paginator('list_objects_v2').paginate(Bucket=b,Prefix=BASE):
  for row in page.get('Contents',[]):
   if not row['Key'].endswith('/independent-verification.json'):continue
   prefix=row['Key'].rsplit('/',1)[0];receipt=json.loads(read(c,row['Key']));raw=read(c,prefix+'/manifest.json')
   if sha(raw)!=receipt['manifest_sha256']:raise ValueError('Persisted cache receipt mismatch')
   manifest=json.loads(raw)
   if manifest['config_sha256']!=hashlib.sha256((ROOT/'experiments/unet25d/config.json').read_bytes()).hexdigest():continue
   for file in manifest['files']:
    dest=(work/file['path']).resolve()
    if not dest.is_relative_to(work.resolve()):raise ValueError('Unsafe cache path')
    dest.parent.mkdir(parents=True,exist_ok=True);h=hashlib.sha256();size=0
    with dest.open('wb') as f:
     for chunk in file['chunks']:
      data=read(c,prefix+'/chunks/'+chunk['sha256'])
      if sha(data)!=chunk['sha256']:raise ValueError('Cache chunk mismatch')
      f.write(data);h.update(data);size+=len(data)
    if h.hexdigest()!=file['sha256'] or size!=file['bytes']:raise ValueError('Reconstructed cache mismatch')
   return True
 return False

def prepare(dataset):
 from experiments.unet25d.pipeline import prepare as build
 work=WORK/'cache';work.mkdir(exist_ok=True);os.environ['HF_HOME']=str(work/'hf')
 c,b=connect(output=True)
 stage_prefix='novo-nordisk/experiments/unet25d/bridge/prepared-visits-v1/'+hashlib.sha256((ROOT/'experiments/unet25d/config.json').read_bytes()).hexdigest()+'/'
 # Restore per-visit checkpoints, including partial prior build progress.
 for page in c.get_paginator('list_objects_v2').paginate(Bucket=b,Prefix=stage_prefix):
  for row in page.get('Contents',[]):
   if not row['Key'].endswith('/complete.json'):continue
   record=json.loads(c.get_object(Bucket=b,Key=row['Key'])['Body'].read())
   for item in record:
    path=(work/'prepared'/item['relative']).resolve()
    if not path.is_relative_to((work/'prepared').resolve()):raise ValueError('Invalid visit checkpoint')
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('wb') as stream:verify(c,b,item['key'],item['sha256'],stream)
 def save_visit(folder):
  records=[]
  for path in sorted(folder.iterdir()):
   if path.is_file():
    r=put_once(c,b,stage_prefix+folder.name+'/'+path.name,path);r['relative']=str(path.relative_to(work/'prepared'));records.append(r)
  receipt=WORK/'visit-complete.json';receipt.write_text(json.dumps(records,sort_keys=True))
  put_once(c,b,stage_prefix+folder.name+'/complete.json',receipt)
 restored=restore_existing(work)
 if not restored:
  # Single regeneration authorization: immutable claim stops repeated whole-cache builds.
  c,b=connect(output=True);claim=WORK/'build-claim.json';claim.write_text(json.dumps({'recovery_authorization':'v2','config_sha256':hashlib.sha256((ROOT/'experiments/unet25d/config.json').read_bytes()).hexdigest()},sort_keys=True))
  key='novo-nordisk/experiments/unet25d/bridge/bounded-cache-recovery-v2.json'
  from botocore.exceptions import ClientError
  try:c.head_object(Bucket=b,Key=key)
  except ClientError as error:
   if error.response['ResponseMetadata']['HTTPStatusCode']!=404:raise
  else:
   if not list((work/'prepared').glob('*/visit.json')):raise RuntimeError('Recovery already claimed without reusable visit checkpoints; no blind regeneration')
  put_once(c,b,key,claim)
  (work/'launch.json').write_text(json.dumps(dict(source_sha=os.environ['NOVO_SOURCE_SHA'],mode='actions-cpu-prepare')))
  build(work,on_visit=save_visit)
  from experiments.unet25d.persist_cache import upload
  upload(work) # Preserve the sole regeneration before attempting Kaggle publication.
 subprocess.run([sys.executable,'-m','experiments.unet25d.verify_mount',str(work)],check=True)
 data=WORK/'dataset';data.mkdir()
 for name in ['prepared','hf']:shutil.move(str(work/name),data/name)
 for name in ['config.json','cache-receipts.json','cache_integrity.stage.json','launch.json','manifest.jsonl']:
  if (work/name).exists():shutil.copy2(work/name,data/name)
 # No environment files, credentials or raw NIfTI cache are published.
 (data/'dataset-metadata.json').write_text(json.dumps(dict(id=dataset,title='Novo Unet25d Benchmark Cache',licenses=[{'name':'other'}])))
 cli('datasets','create','-p',str(data),'-r','zip','-t') # default private; never --public
 ready=WORK/'publication.json';ready.write_text(json.dumps(dict(dataset=dataset,config_sha256=hashlib.sha256((data/'config.json').read_bytes()).hexdigest(),restored_existing=restored,regenerated=not restored,source_sha=os.environ['NOVO_SOURCE_SHA'])))
 print(json.dumps(dict(publication=publish_result('publication.json',ready))))

def kernel_code(mode,sha):
 return '''import os,sys,subprocess,urllib.request,zipfile,io,json
from pathlib import Path
sha=__SOURCE_LITERAL__
root=Path('/tmp/novo-source');root.mkdir(exist_ok=True)
with urllib.request.urlopen('https://github.com/Strange-Noah/BrainTumour-CNNViT-Segmentation/archive/'+sha+'.zip',timeout=60) as r:z=zipfile.ZipFile(io.BytesIO(r.read()))
for item in z.infolist():
 dest=root/Path(*Path(item.filename).parts[1:])
 if not dest.resolve().is_relative_to(root.resolve()):raise RuntimeError('Invalid archive')
 if item.is_dir():dest.mkdir(parents=True,exist_ok=True)
 else:dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(z.read(item))
subprocess.run([sys.executable,'-m','pip','install','-q','-r',str(root/'experiments/unet25d/requirements.txt')],check=True)
candidates=list(Path('/kaggle/input').rglob('cache_integrity.stage.json'))
if len(candidates)!=1:raise RuntimeError('Prepared dataset must expose exactly one complete cache')
cache=candidates[0].parent
os.environ['NOVO_SOURCE_SHA']=sha
os.environ['NOVO_ARTIFACT_MODE']='kaggle'
if __MODE_LITERAL__=='validate':
 p=subprocess.run([sys.executable,'-m','experiments.unet25d.verify_mount',str(cache)],cwd=root,check=True,capture_output=True,text=True)
 Path('/kaggle/working/cache-validation.json').write_text(p.stdout)
else:
 try:subprocess.run([sys.executable,'-m','experiments.unet25d.kaggle_launcher','campaign','--cpu-artifacts',str(cache)],cwd=root,check=True)
 finally:
  link=Path('/kaggle/working/novo-unet25d/prepared')
  if link.is_symlink():link.unlink()
'''.replace('__SOURCE_LITERAL__',repr(sha)).replace('__MODE_LITERAL__',repr(mode))

def submit(dataset,kernel,mode):
 deadline=time.monotonic()+600
 while 'READY' not in cli('datasets','status',dataset).upper():
  if time.monotonic()>deadline:raise TimeoutError('Private dataset processing incomplete')
  time.sleep(20)
 sha=os.environ['NOVO_SOURCE_SHA'];folder=WORK/'kernel';folder.mkdir(exist_ok=True)
 ref=kernel if mode=='benchmark' else 'mathewzhou/novo-unet25d-cache-validation'
 title='Novo Unet25d Bounded Benchmark' if mode=='benchmark' else 'Novo Unet25d Cache Validation'
 (folder/'run.py').write_text(kernel_code(mode,sha));(folder/'kernel-metadata.json').write_text(json.dumps(dict(id=ref,title=title,code_file='run.py',language='python',kernel_type='script',is_private=True,enable_gpu=mode=='benchmark',enable_internet=True,dataset_sources=[dataset],competition_sources=[],kernel_sources=[],machine_shape='NvidiaTeslaT4' if mode=='benchmark' else None)))
 if mode=='benchmark':
  c,b=connect(output=True);prior=c.get_object(Bucket=b,Key='novo-nordisk/experiments/unet25d/bridge/cache-validation.json')
  record=json.loads(prior['Body'].read())
  if record['dataset']!=dataset or not record['verified']:raise ValueError('Independent CPU validation required')
 args=['kernels','push','-p',str(folder),'-t','3600' if mode=='benchmark' else '1200']
 if mode=='benchmark':args+=['--accelerator','NvidiaTeslaT4']
 print(json.dumps(dict(submitted=ref,mode=mode,source_sha=sha,quota_before=cli('quota'))));cli(*args)
 deadline=time.monotonic()+(4500 if mode=='benchmark' else 1800)
 while time.monotonic()<deadline:
  status=cli('kernels','status',ref).upper();print(json.dumps(dict(kernel=ref,complete='COMPLETE' in status,error='ERROR' in status)),flush=True)
  if any(s in status for s in ['COMPLETE','ERROR','CANCEL']):break
  time.sleep(45)
 else:raise TimeoutError('Kaggle polling deadline exceeded; inspect platform hard session cap')
 collect(ref,dataset,mode)
 if 'COMPLETE' not in status:raise RuntimeError('Kaggle run failed; private diagnostics persisted')

def collect(ref,dataset,mode):
 folder=WORK/'outputs';folder.mkdir(exist_ok=True)
 cli('kernels','output',ref,'-p',str(folder))
 c,b=connect(output=True);receipts=[];prefix=f'novo-nordisk/experiments/unet25d/bridge/{os.environ["GITHUB_RUN_ID"]}/outputs/'
 for path in folder.rglob('*'):
  if path.is_file():receipts.append(put_once(c,b,prefix+str(path.relative_to(folder)),path))
 # New client readback, with full SHA checks after transfer out of Kaggle.
 reader,read_bucket=connect(output=True)
 for r in receipts:verify(reader,read_bucket,r['key'],r['sha256'])
 if mode=='validate':
  result=json.loads(next(folder.rglob('cache-validation.json')).read_text())
  if not result.get('verified'):raise ValueError('Private input validation failed')
  result['dataset']=dataset;path=WORK/'validation.json';path.write_text(json.dumps(result))
  # Config-specific input selected once; a conflicting publication must be reviewed.
  put_once(c,b,'novo-nordisk/experiments/unet25d/bridge/cache-validation.json',path)
 report=WORK/'collection.json';report.write_text(json.dumps(dict(kernel=ref,dataset=dataset,mode=mode,receipts=receipts,independent_readback=True,quota_after=cli('quota'))))
 print(json.dumps(dict(collection=publish_result('collection.json',report),files=len(receipts))))

def main():
 if not os.environ.get('KAGGLE_API_TOKEN'):raise ValueError('Cloud Kaggle API token missing')
 mode=os.environ['MODE'];dataset=os.environ['DATASET'];kernel=os.environ['KERNEL']
 if not all(re.fullmatch('mathewzhou/[a-z0-9-]+',s) for s in [dataset,kernel]):raise ValueError('Expected owner-private resource names')
 if mode=='prepare':prepare(dataset)
 elif mode in ['validate','benchmark']:submit(dataset,kernel,mode)
 elif mode=='collect':collect(kernel,dataset,'benchmark')
 else:raise ValueError('Unsupported mode')
if __name__=='__main__':main()
