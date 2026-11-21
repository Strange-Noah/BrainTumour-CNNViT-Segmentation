"""Existing Actions → private Kaggle → R2 path; v2-only campaign state."""
import json,os,re,time,subprocess,threading,queue,math
from pathlib import Path
from cloud.storage import connect
from ..unattended import free_quota,free_seconds,require_integrated_source
from ..settings import read_json_record
from .settings import config,science_hash,prefix,active_key,aux_key,latest_key,write_json,DATASET,KERNEL

TERMINAL=('COMPLETE','ERROR','CANCEL')
# Exact already-reviewed CPU producer. Only exception bookkeeping/idempotence
# changed since this source; loss/geometry/inventory/schedule/config are identical.
ACCEPTED_CPU_PRODUCER='54b34d139e5161be9be26e9e433fdd70cb65b95c'
def read(c,b,key):
    if not key.startswith(prefix()+'/'):raise PermissionError('v2 state namespace required')
    try:return read_json_record(c,b,key)
    except c.exceptions.NoSuchKey:return None

def render(sha,seconds,budget):
    from cloud.repository import source_archive_prefix
    return f'''import os,sys,time,io,zipfile,urllib.request,subprocess,shutil
from pathlib import Path
os.environ['NOVO_SESSION_STARTED_AT']=str(time.time())
sha={sha!r}
root=Path('/tmp/novo-source');root.mkdir(exist_ok=True)
with urllib.request.urlopen({source_archive_prefix()!r}+sha+'.zip',timeout=120) as r:z=zipfile.ZipFile(io.BytesIO(r.read()))
for entry in z.infolist():
 p=root/Path(*Path(entry.filename).parts[1:])
 if not p.resolve().is_relative_to(root.resolve()):raise ValueError('Unsafe immutable source archive')
 if entry.is_dir():p.mkdir(parents=True,exist_ok=True)
 else:p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(z.read(entry))
subprocess.run([sys.executable,'-m','pip','install','-q','-r',str(root/'experiments/unet25d/requirements.txt')],check=True)
inputs=list(Path('/kaggle/input').rglob('private-data-lease.json'))
if len(inputs)!=1:raise ValueError('Exactly one private v2 input required')
source=inputs[0].parent;out=Path('/kaggle/working/novo-unet25d-v2');os.environ['NOVO_SOURCE_SHA']=sha
cmd=[sys.executable,'-m','torch.distributed.run','--standalone','--nproc_per_node=2','-m','experiments.unet25d.cohort.v2.train','--input',str(source),'--output',str(out),'--session-seconds',{str(seconds)!r},'--approved-budget-seconds',{str(budget)!r}]
if (source/'resume.pt').exists():cmd+=['--resume',str(source/'resume.pt')]
try:subprocess.run(cmd,cwd=root,check=True)
finally:
 for name in ('cache','fixed-cache','validation-cache','early-cache'):shutil.rmtree(out/name,ignore_errors=True)
'''

def start(c,b,record):
    from cloud.kaggle_benchmark_bridge import WORK,cli
    from .stage import publish
    require_integrated_source(record['source_sha'])
    if record['science_sha256']!=science_hash() or not record.get('auto_continue'):raise PermissionError('Wrong or disabled v2 continuation')
    quota=free_quota()
    if quota['refresh_at']!=record['initial_quota']['refresh_at']:
        record.update(status='paused_resource_limit',pause_reason='Future-week renewal is not authorized',auto_continue=False);write_json(c,b,active_key(),record);return
    remaining=record['approved_budget_seconds']-record['reserved_seconds'];seconds=min(43200,remaining,free_seconds(quota,record['quota_safety_rate']))
    if seconds<1200:record.update(status='paused_resource_limit',auto_continue=False,pause_reason='Current free allowance exhausted');write_json(c,b,active_key(),record);return
    record.update(status='preparing',session_seconds=seconds,quota_before_session=quota);write_json(c,b,active_key(),record)
    published=publish(seconds);record['live_prefix']=published['output_live_prefix'];record['session_no']+=1;record['kernel']=KERNEL+'-'+str(record['campaign_id'])+'-s'+str(record['session_no']);record['input_version']=published['version'];record['catalogue_sha256']=published['catalogue_sha256']
    if len(record['kernel'].split('/')[1])>64:raise ValueError('Provider slug length')
    folder=WORK/'v2-kernel';folder.mkdir(exist_ok=True);launcher=render(record['source_sha'],seconds,record['approved_budget_seconds']);(folder/'run.ipynb').write_text(json.dumps(dict(nbformat=4,nbformat_minor=5,metadata=dict(kernelspec=dict(display_name='Python 3',language='python',name='python3')),cells=[dict(cell_type='code',metadata={},execution_count=None,outputs=[],source=launcher.splitlines(keepends=True))])))
    (folder/'kernel-metadata.json').write_text(json.dumps(dict(id=record['kernel'],title=record['kernel'].split('/')[1],code_file='run.ipynb',language='python',kernel_type='notebook',is_private=True,enable_gpu=True,enable_internet=True,dataset_sources=[DATASET+'/'+str(published['version'])],competition_sources=[],kernel_sources=[],machine_shape='NvidiaTeslaT4')))
    record['reserved_seconds']+=seconds;record.update(status='submitting',submitted_at=time.time());write_json(c,b,active_key(),record)
    # Ambiguous submission is never retried automatically.
    result=cli('kernels','push','-p',str(folder),'-t',str(seconds),'--accelerator','NvidiaTeslaT4')
    version=re.search(r'[Vv]ersion\s+(\d+)',result);returned=re.search(r'https://www\.kaggle\.com/code/(mathewzhou/[a-z0-9-]+)',result)
    if version is None:raise RuntimeError('Ambiguous v2 submission; inspect exact unique provider job before any retry')
    if returned:record['requested_kernel']=record['kernel'];record['kernel']=returned[1]
    record.update(status='running',execution_state='SUBMITTED',gpu_allocation_confirmed=False,version=int(version[1]),initial_observation_complete=False);write_json(c,b,active_key(),record)
    print(json.dumps(dict(event='V2_JOB_SUBMITTED',source_sha=record['source_sha'],science_sha256=science_hash(),kernel=record['kernel'],url='https://www.kaggle.com/code/'+record['kernel'],session_seconds=seconds,cumulative_budget_seconds=record['approved_budget_seconds'],quota_before=quota,campaign_key=active_key())),flush=True)

def launch():
    if os.environ.get('GITHUB_ACTOR')!='Strange-Noah' or os.environ.get('FORMAL_TRAINING_AUTHORIZED')!='true' or os.environ.get('FREE_RESOURCE_AUTHORIZED')!='true':raise PermissionError('Actual owner and explicit free v2 training authorization required')
    if os.environ.get('DATASET')!=DATASET or os.environ.get('KERNEL')!=KERNEL:raise PermissionError('Registered isolated resources only')
    from cloud.kaggle_benchmark_bridge import cli
    for old_slug in ('mathewzhou/novo-unet25d-v2-44906471093-s1','mathewzhou/novo-unet25d-v2-44906471093-s2'):
        if 'CANCEL' not in cli('kernels','status',old_slug).upper():raise PermissionError('Old execution not confirmed terminated')
    require_integrated_source(os.environ['NOVO_SOURCE_SHA']);c,b=connect(output=True)
    if read(c,b,active_key()) is not None:raise RuntimeError('Existing v2 record; recover, never duplicate or reset allowance')
    ready=read(c,b,aux_key())
    if not ready or not ready['complete'] or ready['science_sha256']!=science_hash() or ready['source_sha'] not in (os.environ['NOVO_SOURCE_SHA'],ACCEPTED_CPU_PRODUCER):raise ValueError('Complete exact-source v2 CPU Gate0 first')
    # Metadata-only whole-cohort bounds: do not discover an unsupported visit
    # after consuming GPU allowance. No MRI/label pixel pass is repeated.
    catalogue=read_json_record(c,b,ready['catalogue']['key']);limits=config()
    for visit in catalogue['visits']:
        if any(r['bytes']>limits['gpu_cache_bytes'] or r['decoded_bytes']>limits['decoded_cache_bytes_per_worker'] for r in visit['shards']):raise MemoryError('Prepared shard exceeds v2 bounded cache; no GPU submitted')
        if visit['partition']=='train' and math.prod(visit['metadata']['shape'])*2>limits['component_cache_bytes']:raise MemoryError('Training component volume exceeds bounded cache; no GPU submitted')
    quota=free_quota();budget=free_seconds(quota,config()['quota_safety_rate'])
    old_key='novo-nordisk/experiments/unet25d/v2/1d6fa16a689916e4225d561b7f2217e61f0ec423cddb9894b5ac3f8636ee8110/active.json'
    old=read_json_record(c,b,old_key)
    if old['initial_quota']['refresh_at']!=quota['refresh_at']:raise PermissionError('Original weekly boundary remains binding')
    prior_used=max(0.,quota['used_seconds']-old['initial_quota']['used_seconds'])
    budget=min(budget,int(old['approved_budget_seconds']-prior_used))

    if budget<2400:raise RuntimeError('Insufficient currently available free GPU allowance; no submission')
    record=dict(prior_campaign_key=old_key,prior_quota_used_seconds=prior_used,original_budget_seconds=old['approved_budget_seconds'],status='authorized',campaign_id=os.environ['GITHUB_RUN_ID'],source_sha=os.environ['NOVO_SOURCE_SHA'],science_sha256=science_hash(),initial_quota=quota,approved_budget_seconds=budget,reserved_seconds=0,quota_safety_rate=config()['quota_safety_rate'],session_no=0,collections=[],auto_continue=True)
    write_json(c,b,active_key(),record);start(c,b,record)
    if record['status']=='running':observe(c,b,record)

def observe(c,b,record):
    from cloud.kaggle_benchmark_bridge import cli
    proc=subprocess.Popen(['kaggle','kernels','logs',record['kernel'],'--follow'],stdout=subprocess.PIPE,stderr=subprocess.DEVNULL,text=True,bufsize=1);lines=queue.Queue(maxsize=10000)
    def receive():
        for line in proc.stdout:lines.put(line)
    threading.Thread(target=receive,daemon=True).start();deadline=time.monotonic()+3600;first=None;last_status=0
    try:
        while time.monotonic()<deadline:
            if time.monotonic()-last_status>30:
                status=cli('kernels','status',record['kernel']).upper();last_status=time.monotonic()
                if any(k in status for k in TERMINAL):return recover(c,b,record,status)
            try:line=lines.get(timeout=10)
            except queue.Empty:
                if proc.poll() is not None:break
                continue
            marker=line.find('{"event":')
            if marker<0:continue
            try:event=json.loads(line[marker:])
            except json.JSONDecodeError:continue
            kind=event.get('event')
            if kind not in ('v2_gpu_allocated','v2_private_snapshot','training_update','checkpoint_saved','v2_learning_gate','v2_gate1_complete','v2_gate1_failed'):continue
            print(json.dumps(event),flush=True)
            if kind=='v2_gpu_allocated':record.update(execution_state='GPU_ALLOCATED',gpu_allocation_confirmed=True)
            if kind=='v2_learning_gate':record['execution_state']='GATE1'
            if kind=='v2_private_snapshot':
                from cloud.storage import verify
                if not event['key'].startswith(prefix()+'/live/'):raise ValueError('Foreign snapshot namespace')
                from .persistence import recover_snapshot
                durable=recover_snapshot(c,b,event)
                record['last_verified_snapshot']=event;record['checkpoint_r2_verified']=durable
                print(json.dumps(dict(event='v2_snapshot_independently_verified',kind=event['kind'],sha256=event['sha256'])),flush=True)
            if kind=='training_update':record['execution_state']='FORMAL_TRAINING'
            if kind in ('v2_learning_gate','v2_gate1_complete'):record['last_learning_gate']=event
            if kind=='checkpoint_saved':record['last_checkpoint']=event
            if kind!='training_update':continue
            if first is None:first=event['unix_time']
            if event['unix_time']-first>=600 and record.get('checkpoint_r2_verified'):
                record.update(execution_state='PHASE1_UNATTENDED_ACTIVE',initial_observation_complete=True,initial_health=event,quota_after_observation=free_quota());write_json(c,b,active_key(),record)
                print(json.dumps(dict(event='v2_ten_minute_observation_complete',kernel=record['kernel'],health=event,unattended_recovery='existing trusted main half-hourly collector',quota=record['quota_after_observation'])),flush=True);return
        record['observation_pending']=True;write_json(c,b,active_key(),record)
        print(json.dumps(dict(initial_observation_pending=True,kernel=record['kernel'])),flush=True)
    finally:
        proc.terminate()
        try:proc.wait(timeout=5)
        except subprocess.TimeoutExpired:proc.kill();proc.wait(timeout=5)
        proc.stdout.close()

def recover(c=None,b=None,record=None,status=None):
    from cloud.kaggle_benchmark_bridge import cli
    from .stage import collect
    if c is None:c,b=connect(output=True)
    record=record or read(c,b,active_key())
    if not record or record['status'] in ('AWAIT_PHASE2_DESIGN','failed','blocked','paused_resource_limit') or not record.get('auto_continue'):
        print(json.dumps(dict(v2_recovery_idle=True,gpu_submitted=False)));return
    if record['source_sha']!=os.environ['NOVO_SOURCE_SHA'] or record['science_sha256']!=science_hash():raise PermissionError('Pinned v2 science/source changed')
    if record['status'] in ('authorized','preparing'):
        # No reservation/submission intent exists in these states; CPU publication
        # may resume. A submitting state never gets an automatic second push.
        start(c,b,record)
        if record['status']=='running':observe(c,b,record)
        return
    if record.get('live_prefix'):
        from .persistence import recover_snapshot
        listing=c.list_objects_v2(Bucket=b,Prefix=record['live_prefix']+'/')
        for item in sorted(listing.get('Contents',[]),key=lambda x:x['LastModified']):
            if not item['Key'].endswith('.tar.json'):continue
            event=read_json_record(c,b,item['Key'])
            if event['key'] in record.get('verified_live_keys',[]):continue
            durable=recover_snapshot(c,b,event)
            record.setdefault('verified_live_keys',[]).append(event['key']);record['last_live_snapshot']=event;record['checkpoint_r2_verified']=durable or record.get('checkpoint_r2_verified',False)
        write_json(c,b,active_key(),record)
    status=status or cli('kernels','status',record['kernel']).upper()
    if not any(s in status for s in TERMINAL):
        if 'QUEUED' in status:record['execution_state']='QUEUED';write_json(c,b,active_key(),record)
        if time.time()-record['submitted_at']>18*3600:record.update(status='blocked',auto_continue=False);write_json(c,b,active_key(),record);raise TimeoutError('Provider did not terminate within hard session expectation')
        if not record.get('initial_observation_complete'):observe(c,b,record)
        else:print(json.dumps(dict(v2_running=True,kernel=record['kernel'],gpu_submitted=False)))
        return
    try:pointer,receipt=collect(record['kernel'])
    except Exception as error:
        record['collection_errors']=record.get('collection_errors',0)+1;record['last_error_type']=type(error).__name__
        if record['collection_errors']>=2:record.update(status='blocked',auto_continue=False)
        write_json(c,b,active_key(),record);raise
    record['collections'].append(receipt);record['quota_after_session']=free_quota()
    if pointer is None and 'COMPLETE' in status:
        candidate=read_json_record(c,b,latest_key()) if record.get('checkpoint_r2_verified') else None
        if candidate and candidate['checkpoint']['key'].startswith(record.get('live_prefix','INVALID')+'/'):pointer=candidate
    record['execution_state']='COLLECTED'
    if 'COMPLETE' not in status or pointer is None:record.update(status='failed',auto_continue=False);write_json(c,b,active_key(),record);raise RuntimeError('v2 correctness failure; outputs preserved, no speculative GPU retry')
    stop=pointer['state']['scientific_stop']
    if stop or not pointer['state']['learning_gate_passed']:
        record.update(status=stop or 'failed',auto_continue=False);write_json(c,b,active_key(),record)
        print(json.dumps(dict(event='v2_campaign_stopped',status=record['status'],final_state=pointer['state'],checkpoint=pointer['checkpoint'],phase2_launched=False)),flush=True);return
    start(c,b,record)
    if record['status']=='running' and not record.get('initial_observation_complete'):observe(c,b,record)

def main():
    mode=os.environ['MODE']
    if mode=='prepare':
        from .prepare import main as prepare
        from .stage import publish
        prepare();
        from .cpu_transport import run
        run();result=publish(43200);print(json.dumps(dict(event='v2_CPU_input_ready',**result)),flush=True)
    elif mode=='formal':launch()
    elif mode in ('recover','collect'):recover()
    else:raise ValueError('v2 exposes only preparation, formal, recovery; no extra benchmark')

if __name__=='__main__':main()
