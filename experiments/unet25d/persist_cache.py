"""Resumable private cache transfer; no preprocessing and no GPU imports."""
import argparse,hashlib,json,os,subprocess,sys,uuid
from contextlib import closing
from pathlib import Path,PurePosixPath
import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

BUCKET='novo-nordisk-pbl-outputs'
BASE='novo-nordisk/experiments/unet25d/prepared-cache/'
CHUNK=32*1024*1024

def sha(data):return hashlib.sha256(data).hexdigest()
def file_sha(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def connect():
    return boto3.client('s3',endpoint_url='https://fec4311cb2caaffaf5e1742997da99b3.r2.cloudflarestorage.com',region_name='auto',aws_access_key_id=os.environ['R2_OUTPUT_ACCESS_KEY_ID'],aws_secret_access_key=os.environ['R2_OUTPUT_SECRET_ACCESS_KEY'],config=Config(connect_timeout=10,read_timeout=60,retries={'max_attempts':2,'mode':'standard'}))
def read(c,key):
    response=c.get_object(Bucket=BUCKET,Key=key)
    with closing(response['Body']) as body:return body.read()
def check(c,key,digest,size):
    response=c.get_object(Bucket=BUCKET,Key=key);h=hashlib.sha256();n=0
    with closing(response['Body']) as body:
        for block in body.iter_chunks(chunk_size=1024*1024):h.update(block);n+=len(block)
    if h.hexdigest()!=digest or n!=size:raise ValueError('Remote object integrity mismatch')
def put(c,key,data):
    digest=sha(data)
    try:
        head=c.head_object(Bucket=BUCKET,Key=key)
    except ClientError as e:
        if e.response['ResponseMetadata']['HTTPStatusCode']!=404:raise
        try:c.put_object(Bucket=BUCKET,Key=key,Body=data,ContentLength=len(data),Metadata={'sha256':digest},IfNoneMatch='*')
        except ClientError as conflict:
            if conflict.response['ResponseMetadata']['HTTPStatusCode']!=412:raise
    else:
        if head['ContentLength']!=len(data) or head.get('Metadata',{}).get('sha256')!=digest:raise ValueError('Existing object conflicts; refusing overwrite')
    check(c,key,digest,len(data))
    return dict(sha256=digest,bytes=len(data))
def inventory(root):
    required=['config.json','cache-receipts.json','cache_integrity.stage.json','launch.json','manifest.jsonl']
    files={name:root/name for name in required}
    config=json.loads(files['config.json'].read_text());receipt=json.loads(files['cache-receipts.json'].read_text())
    if json.loads(files['cache_integrity.stage.json'].read_text())['status']!='complete':raise ValueError('Cache integrity stage not complete')
    if config['full_training_authorized'] or config['shape']!=[512,512]:raise ValueError('Unexpected benchmark configuration')
    for name,digest in receipt.items():
        path=(root/'prepared'/name).resolve()
        if not path.is_relative_to((root/'prepared').resolve()):raise ValueError('Invalid receipt path')
        if file_sha(path)!=digest:raise ValueError('Existing prepared file differs from CPU receipt')
        files['prepared/'+name]=path
    snapshot='hf/hub/models--smp-hub--resnet34.imagenet/snapshots/'+config['checkpoint_revision']
    for name in ['config.json','model.safetensors']:files[snapshot+'/'+name]=root/snapshot/name
    if file_sha(files[snapshot+'/model.safetensors'])!=config['checkpoint_sha256']:raise ValueError('Pretrained checkpoint checksum mismatch')
    source=json.loads(files['launch.json'].read_text())['source_sha']
    if len(source)!=40 or any(x not in '0123456789abcdef' for x in source):raise ValueError('Missing original source SHA')
    if file_sha(files['manifest.jsonl'])!='040ec6c523d5beae43a30b64fede2bee226e1a08e8c14b2cca8043b01b9c39f4':raise ValueError('Dataset manifest checksum mismatch')
    return files,source

def verify(prefix,expected):
    if not prefix.startswith(BASE) or '..' in PurePosixPath(prefix).parts:raise ValueError('Invalid cache prefix')
    c=connect();raw=read(c,prefix+'/manifest.json')
    if sha(raw)!=expected:raise ValueError('Transfer manifest checksum mismatch')
    manifest=json.loads(raw);total=0
    for row in manifest['files']:
        h=hashlib.sha256();n=0
        for chunk in row['chunks']:
            data=read(c,prefix+'/chunks/'+chunk['sha256'])
            if sha(data)!=chunk['sha256'] or len(data)!=chunk['bytes']:raise ValueError('Independent chunk readback failed')
            h.update(data);n+=len(data)
        if h.hexdigest()!=row['sha256'] or n!=row['bytes']:raise ValueError('Independent logical file readback failed')
        total+=n
    receipt=dict(verified=True,verification='fresh-process GET of every chunk and whole-file SHA-256',files=len(manifest['files']),bytes=total,source_sha=manifest['source_sha'],config_sha256=manifest['config_sha256'],manifest_sha256=expected,cache_id=prefix.removeprefix(BASE),gpu_allocated=False)
    put(c,prefix+'/independent-verification.json',(json.dumps(receipt,sort_keys=True)+'\n').encode());print(json.dumps(receipt))

def upload(root):
    files,source=inventory(root)
    plan=dict(source_sha=source,files=[dict(path=name,sha256=file_sha(path),bytes=path.stat().st_size) for name,path in sorted(files.items())])
    plan_sha=sha(json.dumps(plan,sort_keys=True).encode());state=root/'.r2-cache-transfer.json'
    if state.exists():
        saved=json.loads(state.read_text())
        if saved['plan_sha256']!=plan_sha:raise ValueError('Cache changed since transfer began; preserve original state')
    else:
        saved=dict(cache_id=uuid.uuid4().hex,plan_sha256=plan_sha);temporary=state.with_suffix('.tmp');temporary.write_text(json.dumps(saved));temporary.replace(state)
    prefix=BASE+saved['cache_id'];c=connect();rows=[]
    for row in plan['files']:
        chunks=[];h=hashlib.sha256();n=0
        with files[row['path']].open('rb') as f:
            while data:=f.read(CHUNK):
                h.update(data);n+=len(data);chunks.append(put(c,prefix+'/chunks/'+sha(data),data))
        if h.hexdigest()!=row['sha256'] or n!=row['bytes']:raise ValueError('Source changed during upload')
        rows.append(dict(**row,chunks=chunks))
    manifest=dict(schema='novo-private-cache-chunks-v1',source_sha=source,config_sha256=file_sha(root/'config.json'),plan_sha256=plan_sha,files=rows)
    raw=(json.dumps(manifest,sort_keys=True)+'\n').encode();put(c,prefix+'/manifest.json',raw)
    (root/'r2-cache-upload.json').write_text(json.dumps(dict(prefix=prefix,manifest_sha256=sha(raw))))
    # Child reads remote objects only; never consults source cache or writer receipts.
    subprocess.run([sys.executable,str(Path(__file__).resolve()),'verify',prefix,sha(raw)],check=True,timeout=7200)

def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['upload','verify']);p.add_argument('location');p.add_argument('digest',nargs='?');a=p.parse_args()
    try:
        if a.mode=='upload':upload(Path(a.location).resolve())
        else:verify(a.location,a.digest)
    except Exception as error:
        print(json.dumps(dict(verified=False,error_type=type(error).__name__,gpu_allocated=False)),file=sys.stderr);sys.exit(1)
if __name__=='__main__':main()
