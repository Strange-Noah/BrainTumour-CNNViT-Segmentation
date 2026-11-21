"""Same private publication/recovery mechanism, isolated v2 objects and input ID."""
import hashlib,json,os,tempfile,uuid,subprocess
from pathlib import Path
from cloud.storage import connect,verify,put_once
from ..stage import restore_encoder,wait_private_ready,wait_published_version
from ..settings import read_json_record
from .settings import config,science_hash,prefix,latest_key,aux_key,DATASET

def publish(session_seconds):
    from cloud.kaggle_benchmark_bridge import WORK,cli
    from kaggle.api.kaggle_api_extended import KaggleApi
    client,b=connect(output=True);ready=read_json_record(client,b,aux_key())
    if ready['science_sha256']!=science_hash() or not ready['complete']:raise ValueError('v2 Gate0 input identity')
    folder=WORK/'v2-private-input';folder.mkdir(exist_ok=True);folder.chmod(0o700)
    receipts={'catalogue.json':ready['catalogue'],**ready['objects']}
    for name,r in receipts.items():
        with (folder/name).open('wb') as f:verify(client,b,r['key'],r['sha256'],f)
    gate=json.loads((folder/'GATE0_ACCEPTANCE.json').read_text())
    if not gate['complete'] or gate['tests']['failed'] or gate['tests']['errors']:raise ValueError('v2 CPU acceptance incomplete')
    cat=json.loads((folder/'catalogue.json').read_text());inventory=json.loads((folder/'inventory.json').read_text());objects={};lifetime=min(18*3600,session_seconds+6*3600)
    for visit in cat['visits']:
        if visit['partition'] not in ('train','validation'):raise ValueError('Sealed test input forbidden')
        for r in [*visit['shards'],visit['native_labels']]:objects[r['key']]=r
    for row in inventory.values():objects[row['labels']['key']]=row['labels']
    lease={'catalogue_sha256':ready['catalogue']['sha256'],'expires_at':__import__('time').time()+lifetime,'objects':{key:dict(sha256=r['sha256'],url=client.generate_presigned_url('get_object',Params={'Bucket':b,'Key':key},ExpiresIn=lifetime,HttpMethod='GET')) for key,r in objects.items()}}
    (folder/'private-data-lease.json').write_text(json.dumps(lease));restore_encoder(folder,client)
    output_base=prefix()+'/live/'+uuid.uuid4().hex
    slots=[]
    for i in range(80):
        key=output_base+'/'+str(i)+'.tar'
        def signed(k,method):return client.generate_presigned_url(method,Params=dict(Bucket=b,Key=k,**({'IfNoneMatch':'*'} if method=='put_object' else {})),ExpiresIn=lifetime)
        slots.append(dict(key=key,put=signed(key,'put_object'),get=signed(key,'get_object'),receipt_put=signed(key+'.json','put_object')))
    (folder/'private-output-lease.json').write_text(json.dumps(dict(expires_at=lease['expires_at'],slots=slots)))

    from .settings import CONFIG
    (folder/'training-config.json').write_bytes(CONFIG.read_bytes());(folder/'resume.pt').unlink(missing_ok=True)
    try:pointer=read_json_record(client,b,latest_key())
    except client.exceptions.NoSuchKey:pointer=None
    if pointer:
        if pointer['state']['science_sha256']!=science_hash():raise ValueError('Refuse foreign/v1 checkpoint')
        with (folder/'resume.pt').open('wb') as f:verify(client,b,pointer['checkpoint']['key'],pointer['checkpoint']['sha256'],f)
    from .cpu_gate_acceptance import run as real_cpu_acceptance
    real_cpu_acceptance(folder)
    (folder/'dataset-metadata.json').write_text(json.dumps(dict(id=DATASET,title='Novo U-Net25D v2 private input',licenses=[{'name':'other'}],description='Private training data references and auxiliary inventory. TCIA terms apply. Never redistribute.')))
    files=[p for p in folder.iterdir() if p.is_file() and p.name not in ('dataset-metadata.json','input-package.json')]
    package={'publication_id':uuid.uuid4().hex,'source_sha':os.environ['NOVO_SOURCE_SHA'],'science_sha256':science_hash(),'files':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in files}}
    (folder/'input-package.json').write_text(json.dumps(package,sort_keys=True))
    metadata=WORK/'v2-existing-metadata';probe=subprocess.run(['kaggle','datasets','metadata',DATASET,'-p',str(metadata)],capture_output=True,text=True)
    if probe.returncode==0:
        if not json.loads((metadata/'dataset-metadata.json').read_text())['info']['isPrivate']:raise ValueError('Refuse public input publication')
        cli('datasets','version','-p',str(folder),'-m','Refresh isolated v2 private lease','-r','zip','-t')
    else:cli('datasets','create','-p',str(folder),'-r','zip','-t')
    wait_private_ready(DATASET,cli);api=KaggleApi();api.authenticate();version=wait_published_version(DATASET,folder,cli,api)
    # Independent CPU readback of exact new receipt, config, plan and inventory.
    with tempfile.TemporaryDirectory(prefix='novo-private-v2-read-') as d:
        dest=Path(d);dest.chmod(0o700)
        for name in ('input-package.json','training-config.json','plan.json','inventory.json'):
            cli('datasets','download','-d',DATASET+'/'+str(version),'-f',name,'-p',str(dest),'-q')
            if not (dest/name).exists():
                import zipfile
                with zipfile.ZipFile(dest/(name+'.zip')) as z:(dest/name).write_bytes(z.read(name))
            if (dest/name).read_bytes()!=(folder/name).read_bytes():raise ValueError('Independent published v2 input differs')
    return dict(output_live_prefix=output_base,dataset=DATASET,version=version,catalogue_sha256=ready['catalogue']['sha256'],private_input_verified=True,science_sha256=science_hash())

def collect(kernel):
    from cloud.kaggle_benchmark_bridge import WORK,cli
    c,b=connect(output=True);folder=WORK/'v2-outputs';folder.mkdir(exist_ok=True);cli('kernels','output',kernel,'-p',str(folder));receipts=[]
    base=prefix()+'/collections/'+os.environ['GITHUB_RUN_ID']
    for p in sorted(folder.rglob('*')):
        if p.is_file():receipts.append(put_once(c,b,base+'/outputs/'+str(p.relative_to(folder)),p))
    reader,rb=connect(output=True)
    for r in receipts:verify(reader,rb,r['key'],r['sha256'])
    paths=list(folder.rglob('training-state.json'))
    if len(paths)>1:raise ValueError('Ambiguous v2 state')
    pointer=None
    if paths:
        state=json.loads(paths[0].read_text())
        if state['science_sha256']!=science_hash() or state['data_mode']!='foreground-lesion-multiscale-v2':raise ValueError('Wrong recovered science')
        checkpoint=next(r for r in receipts if r['key'].endswith('/novo-unet25d-v2/checkpoint.pt'))
        try:old=read_json_record(c,b,latest_key())
        except c.exceptions.NoSuchKey:old=None
        if old and old['state']['step']>state['step']:raise ValueError('No v2 rollback')
        pointer=dict(state=state,checkpoint=checkpoint)
        from .settings import write_json
        write_json(c,b,latest_key(),pointer)
    report=dict(kernel=kernel,receipts=receipts,independent_readback=True,science_sha256=science_hash())
    p=WORK/'v2-collection.json';p.write_text(json.dumps(report,sort_keys=True));receipt=put_once(c,b,base+'/collection.json',p)
    print(json.dumps(dict(event='v2_collection_complete',files=len(receipts),collection=receipt,checkpoint_recovered=pointer is not None)),flush=True)
    return pointer,receipt
