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
    if allowed and os.environ.get('FREE_RESOURCE_AUTHORIZED')=='true':
        quota=free_quota();budget=free_seconds(quota)
    if not allowed or budget<1200:raise PermissionError('Owner free GPU allowance or explicit total budget required')
    return budget

def free_quota():
    from cloud.kaggle_benchmark_bridge import cli
    raw=cli('quota','--format','json');start=raw.find('[\n')
    if start<0:start=raw.find('[{')
    if start<0:raise ValueError('Free GPU quota unreadable; no submission')
    rows=json.loads(raw[start:]);gpu=next(r for r in rows if r['resource']=='GPU')
    for key in ['remaining','used','total']:
        if not re.fullmatch(r'\d+(\.\d+)?h',gpu[key]):raise ValueError('Unrecognized GPU quota units')
    return dict(remaining_seconds=float(gpu['remaining'][:-1])*3600,used_seconds=float(gpu['used'][:-1])*3600,refresh_at=gpu['refreshAt'])

def free_seconds(quota,rate=None):
    config=json.loads(CONFIG.read_text())
    # Prior accepted dual-T4 campaign: .45 reported quota-hours for at least
    # 1305.61 measured compute seconds. 1.25 conservatively exceeds that rate.
    return max(0,int((quota['remaining_seconds']-config['quota_reserve_seconds'])/(rate or config['quota_safety_rate'])))

def require_integrated_source(sha):
    import subprocess
    if not re.fullmatch('[0-9a-f]{40}',sha):raise ValueError('Invalid pinned source')
    ref = os.environ.get('NOVO_APPROVED_WORKSTREAM_REF', 'origin/main')
    if ref not in ('origin/main', 'origin/model/unet25d-kaggle'):
        raise PermissionError('Unapproved workstream reference')
    if ref != 'origin/main' and sha != os.environ.get('NOVO_SOURCE_SHA'):
        raise PermissionError('Workstream source differs from approved execution pin')
    subprocess.run(['git','merge-base','--is-ancestor',sha,ref],check=True,capture_output=True)

def start_session(c,b,record):
    from cloud.kaggle_benchmark_bridge import WORK,cli
    from .stage import publish_from_existing_runtime
    from .kaggle import render
    require_integrated_source(record['source_sha'])
    seconds=session_budget(record['approved_budget_seconds'],record['reserved_seconds'])
    if record.get('free_resource_authorized'):
        quota=free_quota();record['quota_before_session']=quota
        if quota['refresh_at']!=record['initial_quota']['refresh_at']:
            record['status']='paused_resource_limit';record['pause_reason']='quota renewal requires fresh owner allowance';write(c,b,record);return
        seconds=min(seconds,free_seconds(quota,record['quota_safety_rate']))
    if seconds<1200:record['status']='paused_resource_limit';write(c,b,record);return
    if record.get('status')!='preparing':
        record['session_no']+=1;record['session_seconds']=seconds
        record['kernel']=record['kernel_base']+'-'+str(record['campaign_id'])+'-s'+str(record['session_no'])
    else:seconds=record['session_seconds']
    if len(record['kernel'].split('/')[1])>64:raise ValueError('Training kernel slug too long')
    record['status']='preparing';write(c,b,record)
    published=publish_from_existing_runtime(DATASET,seconds);record['catalogue_sha256']=published['catalogue_sha256'];record['input_version']=published['version']
    folder=WORK/'full-kernel';folder.mkdir(exist_ok=True)
    (folder/'run.py').write_text(render(record['source_sha'],seconds,record['approved_budget_seconds']))
    (folder/'kernel-metadata.json').write_text(json.dumps(dict(id=record['kernel'],title=record['kernel'].split('/')[1],code_file='run.py',language='python',kernel_type='script',is_private=True,enable_gpu=True,enable_internet=True,dataset_sources=[DATASET+'/'+str(record['input_version'])],competition_sources=[],kernel_sources=[],machine_shape='NvidiaTeslaT4')))
    # Persist intent before submission. An interrupted/ambiguous push is never blindly retried.
    record['reserved_seconds']+=seconds;record['status']='submitting';record['submitted_at']=time.time();write(c,b,record)
    result=cli('kernels','push','-p',str(folder),'-t',str(seconds),'--accelerator','NvidiaTeslaT4')
    version=re.search(r'[Vv]ersion\s+(\d+)',result)
    if version is None:raise RuntimeError('Ambiguous GPU submission result; inspect the unique kernel status before any retry')
    returned=re.search(r'https://www\.kaggle\.com/code/(mathewzhou/[a-z0-9-]+)',result)
    if returned:record['requested_kernel']=record['kernel'];record['kernel']=returned.group(1)
    record['version']=int(version.group(1));record['status']='running';write(c,b,record)
    print(json.dumps(dict(submitted=record['kernel'],source_sha=record['source_sha'],session_seconds=seconds,approved_budget_seconds=record['approved_budget_seconds'],reserved_seconds=record['reserved_seconds'],catalogue_sha256=record['catalogue_sha256'],input_version=record['input_version'],execution_url='https://www.kaggle.com/code/'+record['kernel'],recovery='existing half-hourly CPU bridge')),flush=True)

def observe(c,b,record):
    # One-shot Kaggle logs are persisted only; live sessions require SSE follow.
    import subprocess,threading,queue
    from cloud.kaggle_benchmark_bridge import cli
    proc=subprocess.Popen(['kaggle','kernels','logs',record['kernel'],'--follow'],stdout=subprocess.PIPE,stderr=subprocess.DEVNULL,text=True,bufsize=1)
    lines=queue.Queue(maxsize=10000)
    def reader():
        for line in proc.stdout:lines.put(line)
    threading.Thread(target=reader,daemon=True).start()
    wall_deadline=time.monotonic()+3600;first=None;last=None;last_status=0
    try:
        while time.monotonic()<wall_deadline:
            if time.monotonic()-last_status>=30:
                status=cli('kernels','status',record['kernel']).upper();last_status=time.monotonic()
                if any(s in status for s in TERMINAL):return recover(c,b,record,status)
            try:line=lines.get(timeout=10)
            except queue.Empty:
                if proc.poll() is not None:break
                continue
            event=parse_event(line)
            if event is None:continue
            if event['event']=='checkpoint_saved':record['last_checkpoint']=event;continue
            if event['step']<=record.get('last_observed_step',-1):continue
            record['last_observed_step']=event['step'];last=event
            if first is None:first=event['unix_time']
            print(json.dumps(event),flush=True)
            if event['unix_time']-first>=600:
                if record.get('free_resource_authorized'):
                    quota=free_quota();record['initial_observation_quota']=quota
                    before=record['quota_before_session'];elapsed=time.time()-record['submitted_at']
                    if quota['refresh_at']==before['refresh_at'] and elapsed>0:
                        rate=max(0,quota['used_seconds']-before['used_seconds'])/elapsed
                        record['quota_safety_rate']=max(record['quota_safety_rate'],rate*1.1)
                record['initial_health']=last;record['initial_observation_complete']=True;record['expected_session_end_unix']=last['session_started_at']+record['session_seconds'];write(c,b,record)
                print(json.dumps(dict(initial_observation_complete=True,kernel=record['kernel'],unattended=True,expected_session_end_unix=record['expected_session_end_unix'],health=last,last_checkpoint=record.get('last_checkpoint'),quota_after_observation=record.get('initial_observation_quota'),quota_safety_rate=record.get('quota_safety_rate'))),flush=True);return
        print(json.dumps(dict(initial_observation_pending=True,kernel=record['kernel'],unattended_recovery_enabled=True)),flush=True)
    finally:
        proc.terminate()
        try:proc.wait(timeout=5)
        except subprocess.TimeoutExpired:proc.kill();proc.wait(timeout=5)
        proc.stdout.close()

def parse_event(line):
    for kind in ['training_update','checkpoint_saved']:
        marker=line.find('{"event": "'+kind+'"')
        if marker>=0:
            try:return json.loads(line[marker:])
            except json.JSONDecodeError:return None
    return None

def launch(dataset,kernel):
    if os.environ.get('GITHUB_ACTOR')!='Strange-Noah':raise PermissionError('Only the project owner may authorize formal GPU execution')
    budget=authorization() # No GPU API call before actual owner authorization.
    if dataset!=DATASET:raise ValueError('Frozen private cohort input required')
    c,b=connect(output=True);old=read(c,b,ACTIVE_KEY)
    if old and old['status'] not in ['budget_complete','training_complete','failed']:raise RuntimeError('An active or blocked formal run already exists; recover it instead of duplicating training')
    record=dict(status='authorized',campaign_id=os.environ['GITHUB_RUN_ID'],kernel_base=kernel,dataset=dataset,source_sha=os.environ['NOVO_SOURCE_SHA'],science_sha256=science_hash(),approved_budget_seconds=budget,reserved_seconds=0,session_no=0,collections=[])
    if os.environ.get('FREE_RESOURCE_AUTHORIZED')=='true':record.update(free_resource_authorized=True,initial_quota=free_quota(),quota_safety_rate=json.loads(CONFIG.read_text())['quota_safety_rate'])
    if old and old['status']=='failed':
        from .recovery import LEGACY_SCIENCE,CHECKPOINT_SHA,CHECKPOINT_STEP
        from .settings import read_json_record
        if old['science_sha256']!=LEGACY_SCIENCE or old['source_sha']!='e818635a9e7df4d91792afaad7fc9ff3dd4b2a23':raise ValueError('Only the diagnosed campaign may resume')
        pointer=read_json_record(c,b,'novo-nordisk/experiments/unet25d/full-runs/'+LEGACY_SCIENCE+'/latest.json')
        if pointer['checkpoint']['sha256']!=CHECKPOINT_SHA or pointer['state']['step']!=CHECKPOINT_STEP:raise ValueError('Wrong recovered checkpoint')
        record=old;record.update(status='authorized',source_sha=os.environ['NOVO_SOURCE_SHA'],science_sha256=science_hash(),initial_observation_complete=False,last_observed_step=-1,recovery_authorization_run=os.environ['GITHUB_RUN_ID'],legacy_science_sha256=LEGACY_SCIENCE)
        # Preserve the original cumulative allowance and whole-session reservation;
        # neither the failed session nor its resource charge is reset/refunded.
    elif old and old['status']=='budget_complete':
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
    if not record or record['status'] in ['training_complete','budget_complete','paused_resource_limit','failed','blocked']:
        print(json.dumps(dict(recovery_idle=True,gpu_submitted=False)));return
    if record['science_sha256']!=science_hash():raise ValueError('Frozen training science changed; do not resume')
    if record['status']=='preparing':
        try:start_session(c,b,record)
        except Exception:
            record['preparation_errors']=record.get('preparation_errors',0)+1
            if record['preparation_errors']>=2:record['status']='blocked'
            write(c,b,record);raise
        return
    status=status or resolved_status(c,b,record)
    if not any(s in status for s in TERMINAL):
        if time.time()-record['submitted_at']>18*3600:record['status']='blocked';write(c,b,record);raise TimeoutError('Provider session did not reach terminal status')
        if not record.get('initial_observation_complete'):return observe(c,b,record)
        print(json.dumps(dict(kernel=record['kernel'],running=True,gpu_submitted=False)));return
    try:
        evidence=collect(record['kernel'],DATASET,'formal');record['collections'].append(evidence);record['collection_errors']=0
    except Exception as error:
        record['collection_errors']=record.get('collection_errors',0)+1;record['last_error_type']=type(error).__name__
        if record['collection_errors']>=2:record['status']='blocked'
        write(c,b,record);raise
    if 'COMPLETE' not in status:record['status']='failed';write(c,b,record);raise RuntimeError('Formal correctness failure; private outputs recovered, no GPU retry')
    if record.get('free_resource_authorized'):
        quota=free_quota();record['quota_after_session']=quota;before=record['quota_before_session']
        if quota['refresh_at']==before['refresh_at']:
            # A completed session provides a less noisy actual platform charge.
            rate=max(0,quota['used_seconds']-before['used_seconds'])/max(1,time.time()-record['submitted_at'])
            record['quota_safety_rate']=max(record['quota_safety_rate'],rate*1.1)
    latest=read(c,b,latest_key())
    collection_prefix=evidence['collection']['key'].rsplit('/collection.json',1)[0]+'/outputs/'
    if not latest or latest['state']['catalogue_sha256']!=record['catalogue_sha256'] or not latest['checkpoint']['key'].startswith(collection_prefix):
        record['status']='blocked';write(c,b,record);raise ValueError('No valid full-cohort checkpoint from completed session')
    record['catalogue_sha256']=latest['state']['catalogue_sha256']
    if latest['state'].get('scientific_stop') in ['TARGET_REACHED','PLATEAU_BELOW_TARGET'] and not latest['state']['pending_validation']:
        record['status']='training_complete';write(c,b,record);print(json.dumps(dict(training_complete=True,kernel=record['kernel'])));return
    start_session(c,b,record) # Existing resume checkpoint; minimum sessions, no discretionary approval.

def resolved_status(c,b,record):
    """Repair only the demonstrated first formal title/slug mismatch, never push."""
    from cloud.kaggle_benchmark_bridge import cli,WORK
    from .kaggle import render
    try:return cli('kernels','status',record['kernel']).upper()
    except RuntimeError:
        if record['source_sha']!='e818635a9e7df4d91792afaad7fc9ff3dd4b2a23' or record['session_no']!=1:raise
        canonical='mathewzhou/novo-full-cohort-training'
        folder=WORK/'canonical-recovery';folder.mkdir(exist_ok=True)
        cli('kernels','pull',canonical,'-p',str(folder),'-m')
        meta=json.loads((folder/'kernel-metadata.json').read_text());files=list(folder.glob('*.py'))
        expected=render(record['source_sha'],record['session_seconds'],record['approved_budget_seconds'])
        if meta.get('id')!=canonical or not meta.get('is_private') or len(files)!=1 or files[0].read_text().strip()!=expected.strip():raise ValueError('Canonical recovery code/identity mismatch')
        status=cli('kernels','status',canonical).upper()
        record['requested_kernel']=record['kernel'];record['kernel']=canonical;record['canonical_source_verified']=True;write(c,b,record)
        return status
