"""One active authorized training run; the existing CPU bridge recovers it on a narrow cron."""
import json,os,re,time
from pathlib import Path
from cloud.storage import connect
from .settings import ACTIVE_KEY,PROVIDER_SESSION_SECONDS,CONFIG,science_hash,read_json_record
DATASET='mathewzhou/novo-unet25d-full-cohort-input'
TERMINAL=('COMPLETE','ERROR','CANCEL')

def read(c,b,key):
    try:return read_json_record(c,b,key)
    except c.exceptions.NoSuchKey:return None

def write(c,b,record):
    import hashlib
    raw=json.dumps(record,sort_keys=True).encode();c.put_object(Bucket=b,Key=ACTIVE_KEY,Body=raw,Metadata={'sha256':hashlib.sha256(raw).hexdigest()})

def session_budget(total,reserved):
    remaining=int(total-reserved)
    return min(PROVIDER_SESSION_SECONDS,remaining) if remaining>=1200 else 0

def authorization():
    config=json.loads(CONFIG.read_text())
    allowed=config['formal_training_authorized'] or os.environ.get('FORMAL_TRAINING_AUTHORIZED')=='true'
    budget=int(os.environ.get('APPROVED_GPU_BUDGET_SECONDS') or config['approved_budget_seconds'])
    if not allowed or budget<1200:raise PermissionError('Owner total GPU budget authorization required before any GPU submission')
    return budget

def require_integrated_source(sha):
    import subprocess
    if not re.fullmatch('[0-9a-f]{40}',sha):raise ValueError('Invalid pinned source')
    subprocess.run(['git','merge-base','--is-ancestor',sha,'origin/main'],check=True,capture_output=True)

def start_session(c,b,record):
    from cloud.kaggle_benchmark_bridge import WORK,cli
    from .stage import publish_from_existing_runtime
    from .kaggle import render
    require_integrated_source(record['source_sha'])
    seconds=session_budget(record['approved_budget_seconds'],record['reserved_seconds'])
    if not seconds:record['status']='budget_complete';write(c,b,record);return
    if record.get('status')!='preparing':
        record['session_no']+=1;record['session_seconds']=seconds
        record['kernel']=record['kernel_base']+'-'+str(record['campaign_id'])+'-s'+str(record['session_no'])
    else:seconds=record['session_seconds']
    if len(record['kernel'].split('/')[1])>64:raise ValueError('Training kernel slug too long')
    record['status']='preparing';write(c,b,record)
    published=publish_from_existing_runtime(DATASET,seconds);record['catalogue_sha256']=published['catalogue_sha256'];record['input_version']=published['version']
    deadline=time.monotonic()+900
    while 'READY' not in cli('datasets','status',DATASET).upper():
        if time.monotonic()>deadline:raise TimeoutError('Private input still processing; no GPU submitted')
        time.sleep(20)
    folder=WORK/'full-kernel';folder.mkdir(exist_ok=True)
    (folder/'run.py').write_text(render(record['source_sha'],seconds,record['approved_budget_seconds']))
    (folder/'kernel-metadata.json').write_text(json.dumps(dict(id=record['kernel'],title='Novo Full Cohort Training',code_file='run.py',language='python',kernel_type='script',is_private=True,enable_gpu=True,enable_internet=True,dataset_sources=[DATASET+'/'+str(record['input_version'])],competition_sources=[],kernel_sources=[],machine_shape='NvidiaTeslaT4')))
    # Persist intent before submission. An interrupted/ambiguous push is never blindly retried.
    record['reserved_seconds']+=seconds;record['status']='submitting';record['submitted_at']=time.time();write(c,b,record)
    result=cli('kernels','push','-p',str(folder),'-t',str(seconds),'--accelerator','NvidiaTeslaT4')
    version=re.search(r'[Vv]ersion\s+(\d+)',result)
    if version is None:raise RuntimeError('Ambiguous GPU submission result; inspect the unique kernel status before any retry')
    record['version']=int(version.group(1));record['status']='running';write(c,b,record)
    print(json.dumps(dict(submitted=record['kernel'],source_sha=record['source_sha'],session_seconds=seconds,approved_budget_seconds=record['approved_budget_seconds'],reserved_seconds=record['reserved_seconds'],catalogue_sha256=record['catalogue_sha256'],input_version=record['input_version'],execution_url='https://www.kaggle.com/code/'+record['kernel'],recovery='existing half-hourly CPU bridge')),flush=True)

def observe(c,b,record):
    from cloud.kaggle_benchmark_bridge import cli
    wall_deadline=time.monotonic()+3600;first=None;last=None
    while time.monotonic()<wall_deadline:
        status=cli('kernels','status',record['kernel']).upper()
        if any(s in status for s in TERMINAL):return recover(c,b,record,status)
        logs=cli('kernels','logs',record['kernel'])
        for line in logs.splitlines():
            marker=line.find('{"event": "training_update"')
            if marker<0:continue
            try:event=json.loads(line[marker:])
            except json.JSONDecodeError:continue
            if event['step']>record.get('last_observed_step',-1):
                record['last_observed_step']=event['step'];last=event
                if first is None:first=time.monotonic()
                print(json.dumps(event),flush=True)
        if first is not None and time.monotonic()-first>=600:
            record['initial_health']=last;record['initial_observation_complete']=True;record['expected_session_end_unix']=last['session_started_at']+record['session_seconds'];write(c,b,record)
            print(json.dumps(dict(initial_observation_complete=True,kernel=record['kernel'],unattended=True,expected_session_end_unix=record['expected_session_end_unix'],health=last)),flush=True);return
        time.sleep(30)
    print(json.dumps(dict(initial_observation_pending=True,kernel=record['kernel'],unattended_recovery_enabled=True)),flush=True)

def launch(dataset,kernel):
    budget=authorization() # No GPU API call before explicit authorization.
    if os.environ.get('GITHUB_ACTOR')!='Strange-Noah':raise PermissionError('Only the project owner may authorize formal GPU execution')
    if dataset!=DATASET:raise ValueError('Frozen private cohort input required')
    c,b=connect(output=True);old=read(c,b,ACTIVE_KEY)
    if old and old['status'] not in ['budget_complete','training_complete']:raise RuntimeError('An active or blocked formal run already exists; recover it instead of duplicating training')
    record=dict(status='authorized',campaign_id=os.environ['GITHUB_RUN_ID'],kernel_base=kernel,dataset=dataset,source_sha=os.environ['NOVO_SOURCE_SHA'],science_sha256=science_hash(),approved_budget_seconds=budget,reserved_seconds=0,session_no=0,collections=[])
    if old and old['status']=='budget_complete':
        if old['science_sha256']!=science_hash() or budget<=old['reserved_seconds']:raise ValueError('Explicit increased cumulative budget is required to continue the same experiment')
        record=old;record['approved_budget_seconds']=budget
    elif old:raise ValueError('The maximum proposed training passes are already complete')
    start_session(c,b,record)
    if record['status']=='running':observe(c,b,record)

def recover(c=None,b=None,record=None,status=None):
    from cloud.kaggle_benchmark_bridge import cli,collect
    from .stage import latest_key
    if c is None:c,b=connect(output=True)
    if record is None:record=read(c,b,ACTIVE_KEY)
    if not record or record['status'] in ['training_complete','budget_complete','failed','blocked']:
        print(json.dumps(dict(recovery_idle=True,gpu_submitted=False)));return
    if record['science_sha256']!=science_hash():raise ValueError('Frozen training science changed; do not resume')
    if record['status']=='preparing':
        try:start_session(c,b,record)
        except Exception:
            record['preparation_errors']=record.get('preparation_errors',0)+1
            if record['preparation_errors']>=2:record['status']='blocked'
            write(c,b,record);raise
        return
    status=status or cli('kernels','status',record['kernel']).upper()
    if not any(s in status for s in TERMINAL):
        if time.time()-record['submitted_at']>18*3600:record['status']='blocked';write(c,b,record);raise TimeoutError('Provider session did not reach terminal status')
        print(json.dumps(dict(kernel=record['kernel'],running=True,gpu_submitted=False)));return
    try:
        evidence=collect(record['kernel'],DATASET,'formal');record['collections'].append(evidence);record['collection_errors']=0
    except Exception as error:
        record['collection_errors']=record.get('collection_errors',0)+1;record['last_error_type']=type(error).__name__
        if record['collection_errors']>=2:record['status']='blocked'
        write(c,b,record);raise
    if 'COMPLETE' not in status:record['status']='failed';write(c,b,record);raise RuntimeError('Formal correctness failure; private outputs recovered, no GPU retry')
    latest=read(c,b,latest_key())
    collection_prefix=evidence['collection']['key'].rsplit('/collection.json',1)[0]+'/outputs/'
    if not latest or latest['state']['catalogue_sha256']!=record['catalogue_sha256'] or not latest['checkpoint']['key'].startswith(collection_prefix):
        record['status']='blocked';write(c,b,record);raise ValueError('No valid full-cohort checkpoint from completed session')
    record['catalogue_sha256']=latest['state']['catalogue_sha256']
    if latest['state']['epoch']>=json.loads(CONFIG.read_text())['maximum_proposed_coverage_epochs'] and not latest['state']['pending_validation']:
        record['status']='training_complete';write(c,b,record);print(json.dumps(dict(training_complete=True,kernel=record['kernel'])));return
    start_session(c,b,record) # Existing resume checkpoint; minimum sessions, no discretionary approval.
