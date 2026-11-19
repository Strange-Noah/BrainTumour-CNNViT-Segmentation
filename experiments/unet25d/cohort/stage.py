"""Data adapter for the existing Actions-authenticated private Kaggle publisher.

No workflow or credential changes. This module must run only in the existing
publisher runtime which already has KAGGLE_API_TOKEN; never send it to a notebook.
"""
import hashlib,json,time,tempfile
from pathlib import Path
from cloud.storage import connect,verify

def make_private_input(catalogue_key,catalogue_sha,folder,session_seconds):
    if not 60<=session_seconds<=12*3600:raise ValueError('Invalid approved session duration')
    c,b=connect(output=True)
    if b!='novo-nordisk-pbl-outputs':raise ValueError('Private project output bucket required')
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True);folder.chmod(0o700)
    path=folder/'catalogue.json'
    with path.open('wb') as out:verify(c,b,catalogue_key,catalogue_sha,out)
    catalogue=json.loads(path.read_text())
    if not catalogue['summary']['complete'] or catalogue['summary']['test_included']:raise ValueError('Incomplete or test-containing input catalogue')
    lifetime=min(18*3600,session_seconds+6*3600);objects={}
    for v in catalogue['visits']:
        if v['partition'] not in ['train','validation']:raise ValueError('Test access forbidden')
        for obj in [*v['shards'],v['native_labels']]:
            objects[obj['key']]=dict(sha256=obj['sha256'],url=c.generate_presigned_url('get_object',Params={'Bucket':b,'Key':obj['key']},ExpiresIn=lifetime,HttpMethod='GET'))
    (folder/'private-data-lease.json').write_text(json.dumps(dict(expires_at=time.time()+lifetime,catalogue_sha256=catalogue_sha,objects=objects)))
    # The existing publisher uploads only this small PRIVATE input package.
    # Its bearer URLs grant GET on listed objects only and expire after the session.
    return dict(catalogue_sha256=catalogue_sha,private_input_files=2,expires_in_seconds=lifetime)

def restore_encoder(folder,client):
    # Reuse only the already verified 87 MB encoder dependencies, not the MRI cache.
    from ..persist_cache import read,sha
    from .settings import CACHE_PREFIX,CACHE_SHA
    raw=read(client,CACHE_PREFIX+'/manifest.json')
    if sha(raw)!=CACHE_SHA:raise ValueError('Encoder cache manifest identity mismatch')
    restored=[]
    for row in json.loads(raw)['files']:
        if not row['path'].startswith('hf/') or not row['path'].endswith(('/config.json','/model.safetensors')):continue
        path=Path(folder)/row['path'];path.parent.mkdir(parents=True,exist_ok=True);h=hashlib.sha256()
        with path.open('wb') as f:
            for chunk in row['chunks']:
                data=read(client,CACHE_PREFIX+'/chunks/'+chunk['sha256'])
                if sha(data)!=chunk['sha256'] or len(data)!=chunk['bytes']:raise ValueError('Encoder dependency integrity mismatch')
                h.update(data);f.write(data)
        if h.hexdigest()!=row['sha256']:raise ValueError('Encoder dependency checksum mismatch')
        restored.append((row['path'],h.hexdigest()))
    from .settings import CONFIG
    config=json.loads(CONFIG.read_text());weights=[h for path,h in restored if path.endswith('/model.safetensors') and config['encoder_checkpoint_revision'] in path]
    if weights!=[config['encoder_checkpoint_sha256']] or len(restored)!=2:raise ValueError('Pinned encoder dependency set is incomplete')

def wait_private_ready(dataset,cli,seconds=900):
    # A successful private upload may temporarily return an API error while the
    # new version is becoming visible. Poll only status, never resubmit data here.
    deadline=time.monotonic()+seconds
    while time.monotonic()<deadline:
        try:
            if 'READY' in cli('datasets','status',dataset).upper():return
        except RuntimeError:pass
        time.sleep(20)
    raise TimeoutError('Private input publication did not finish processing')

def wait_published_version(dataset,folder,cli,api,seconds=900):
    # Search/list results can lag a successful upload. Bind a version to the
    # exact small publication receipt, never to an eventually-consistent number.
    expected=(Path(folder)/'input-package.json').read_bytes();deadline=time.monotonic()+seconds
    with tempfile.TemporaryDirectory(prefix='novo-private-version-') as d:
        root=Path(d);root.chmod(0o700)
        while time.monotonic()<deadline:
            try:
                row=next(r for r in api.dataset_list(mine=True,search=dataset.split('/')[1]) if r.ref==dataset)
                if not row.is_private:raise ValueError('Input publication must remain private')
                version=int(row.current_version_number)
                cli('datasets','download','-d',dataset+'/'+str(version),'-f','input-package.json','-p',str(root),'--force','-q')
                receipt=root/'input-package.json'
                if receipt.exists() and receipt.read_bytes()==expected:return version
            except (RuntimeError,StopIteration):pass
            time.sleep(20)
    raise TimeoutError('New private input version has not exposed the exact publication receipt')

def publish_from_existing_runtime(dataset,session_seconds=12*3600,final_check=False):
    from cloud.kaggle_benchmark_bridge import WORK,cli
    from .settings import keybase
    if dataset!='mathewzhou/novo-unet25d-full-cohort-input':raise ValueError('Unexpected private input target')
    c,b=connect(output=True);response=c.get_object(Bucket=b,Key=keybase()+'/ready.json');raw=response['Body'].read();response['Body'].close()
    if hashlib.sha256(raw).hexdigest()!=response['Metadata']['sha256']:raise ValueError('Prepared catalogue pointer identity mismatch')
    ready=json.loads(raw)
    if not ready['complete']:raise ValueError('One-time cohort preparation is not complete')
    folder=WORK/'cohort-private-input';result=make_private_input(ready['manifest']['key'],ready['manifest']['sha256'],folder,session_seconds);restore_encoder(folder,c)
    from .settings import CONFIG
    (folder/'training-config.json').write_bytes(CONFIG.read_bytes())
    try:
        response=c.get_object(Bucket=b,Key=latest_key());pointer=json.loads(response['Body'].read());response['Body'].close()
        if pointer['state']['catalogue_sha256']!=ready['manifest']['sha256']:raise ValueError('Refuse to switch prepared inputs during an experiment')
        with (folder/'resume.pt').open('wb') as f:verify(c,b,pointer['checkpoint']['key'],pointer['checkpoint']['sha256'],f)
    except c.exceptions.NoSuchKey:pass
    (folder/'dataset-metadata.json').write_text(json.dumps(dict(id=dataset,title='Novo full-cohort private training input',licenses=[{'name':'other'}],description='Private training references. Original TCIA usage terms apply. Never redistribute data leases publicly.')))
    import uuid,os
    package=dict(publication_id=uuid.uuid4().hex,source_sha=os.environ['NOVO_SOURCE_SHA'],files={name:hashlib.sha256((folder/name).read_bytes()).hexdigest() for name in ['catalogue.json','private-data-lease.json','training-config.json']})
    (folder/'input-package.json').write_text(json.dumps(package,sort_keys=True))
    # metadata/status calls are control-plane only; no tokens/URLs are printed.
    import subprocess
    check=subprocess.run(['kaggle','datasets','metadata',dataset,'-p',str(WORK/'existing-cohort-metadata')],capture_output=True,text=True)
    if check.returncode==0:
        meta=json.loads((WORK/'existing-cohort-metadata/dataset-metadata.json').read_text())
        if not meta['info']['isPrivate']:raise ValueError('Refuse public dataset publication')
        cli('datasets','version','-p',str(folder),'-m','Refresh private data lease','-r','zip','-t')
    else:cli('datasets','create','-p',str(folder),'-r','zip','-t')
    import time,os,tempfile
    wait_private_ready(dataset,cli)
    from kaggle.api.kaggle_api_extended import KaggleApi
    api=KaggleApi();api.authenticate();version=wait_published_version(dataset,folder,cli,api)
    from .settings import read_json_record
    key=keybase()+'/training-input-readiness/'+science_hash()+'.json'
    try:prior=read_json_record(c,b,key)
    except c.exceptions.NoSuchKey:prior=None
    if final_check and prior is None:
        from .accept_input import check_published_input
        result=check_published_input(dataset,version,folder);result.update(source_sha=os.environ['NOVO_SOURCE_SHA'],encoder_sha256=json.loads(CONFIG.read_text())['encoder_checkpoint_sha256'])
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'input-readiness.json';path.write_text(json.dumps(result,sort_keys=True))
            from cloud.storage import put_once
            result['receipt']=put_once(c,b,key,path)
    elif prior:
        result=prior
        if not prior['verified'] or prior['catalogue_sha256']!=ready['manifest']['sha256']:raise ValueError('Prior input readiness differs')
    else:raise ValueError('Complete final CPU input check before formal launch')
    published=dict(private_input_published=dataset,version=version,catalogue_sha256=ready['manifest']['sha256'],gpu_allocated=False,input_readiness=result)
    print(json.dumps(published));return published

from .settings import science_hash

def latest_key():return 'novo-nordisk/experiments/unet25d/full-runs/'+science_hash()+'/latest.json'

def persist_training_state(folder,client,bucket,receipts):
    paths=list(Path(folder).rglob('training-state.json'))
    if not paths:return # Preserve failure diagnostics even when no new checkpoint was produced.
    if len(paths)!=1:raise ValueError('Ambiguous formal training state')
    state=json.loads(paths[0].read_text())
    if state['science_sha256']!=science_hash() or state['data_mode']!='full-cohort-v1':raise ValueError('Experiment state mismatch')
    checkpoint=next(r for r in receipts if r['key'].endswith('/novo-unet25d-full/checkpoint.pt'))
    pointer=dict(state=state,checkpoint=checkpoint)
    best=next((r for r in receipts if r['key'].endswith('/novo-unet25d-full/best.pt')),None)
    try:
        response=client.get_object(Bucket=bucket,Key=latest_key());old=json.loads(response['Body'].read());response['Body'].close()
        if old['state']['step']>state['step']:raise ValueError('Refuse experiment-state rollback')
        if not best:best=old.get('best')
    except client.exceptions.NoSuchKey:pass
    if best:pointer['best']=best
    raw=json.dumps(pointer,sort_keys=True).encode();client.put_object(Bucket=bucket,Key=latest_key(),Body=raw,Metadata={'sha256':hashlib.sha256(raw).hexdigest()})
