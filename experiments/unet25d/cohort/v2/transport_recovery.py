"""One owner-authorized transport repair; retain science, budget and checkpoint."""
import hashlib,json,math,os,tempfile,time
from pathlib import Path
from . import control
from .settings import active_key,latest_key,aux_key,write_json,science_hash
from ..settings import read_json_record
from ..unattended import require_integrated_source
from .recover_preserve import preserve,GPU_SOURCE


def preflight(c,b,pointer):
    """CPU SHA-only read of a bounded resume window, never decoding or training."""
    from cloud.storage import verify
    ready=read_json_record(c,b,aux_key())
    with tempfile.TemporaryDirectory(prefix='novo-resume-read-') as folder:
        root=Path(folder)
        def load(receipt):
            path=root/receipt['sha256']
            with path.open('wb') as out:verify(c,b,receipt['key'],receipt['sha256'],out)
            return json.loads(path.read_text())
        catalogue=load(ready['catalogue']);inventory=load(ready['objects']['inventory.json']);plan=load(ready['objects']['plan.json'])
        visits={v['metadata']['exam_id']:v for v in catalogue['visits'] if v['partition']=='train'}
        cursor=pointer['state']['cursor'];objects={}
        # Include the durable resume prefix and the entire previous failure window.
        for row in plan[cursor:min(len(plan),max(cursor+216,96008))]:
            if row['exam'] not in visits:raise PermissionError('Nontraining preflight row')
            visit=visits[row['exam']];shard=next(s for s in visit['shards'] if s['start']<=row['z']<s['end'])
            for obj in (shard,inventory[row['exam']]['labels']):objects[obj['key']]=obj
        count=0;size=0
        for obj in objects.values():
            verify(c,b,obj['key'],obj['sha256']);count+=1;size+=obj['bytes']
        return dict(verified_objects=count,bytes_verified=size,resume_step=pointer['state']['step'],gpu_allocated=False)


def reconcile_reservation(record,quota):
    if quota['refresh_at']!=record['initial_quota']['refresh_at']:raise PermissionError('No future-week allowance')
    # CLI quota is rounded to .01 hour. Two rounding units conservatively bound
    # both endpoints; unsaved GPU work remains charged, never inferred from step.
    charged=math.ceil(max(0,quota['used_seconds']-record['initial_quota']['used_seconds'])+72)
    if charged>record['reserved_seconds']:raise ValueError('Actual use exceeds prior reservation')
    return charged


def repair_and_resume(c,b,record):
    if os.environ.get('GITHUB_ACTOR')!='Strange-Noah' or os.environ.get('FORMAL_TRAINING_AUTHORIZED')!='true' or os.environ.get('FREE_RESOURCE_AUTHORIZED')!='true':raise PermissionError('Explicit owner repair dispatch only')
    source=os.environ['NOVO_SOURCE_SHA'];require_integrated_source(source)
    if record['source_sha']!=GPU_SOURCE or record.get('transport_recovery_used') or record['science_sha256']!=science_hash():raise PermissionError('One exact failed campaign only')
    from cloud.kaggle_benchmark_bridge import cli
    status=cli('kernels','status',record['kernel']).upper()
    if 'ERROR' not in status:raise PermissionError('Old provider execution must be ERROR')
    preserve() # Independent archive readback, committed outputs, monotonic pointer.
    record=read_json_record(c,b,active_key());pointer=read_json_record(c,b,latest_key())
    state=pointer['state']
    if not state['learning_gate_passed'] or state['phase']!='PHASE1' or state['step']<46784 or state['cursor']!=2*state['step']:raise ValueError('Valid passed-Gate1 formal checkpoint required')
    # Verify complete optimizer and both-rank RNG before publishing a resume.
    import torch
    from cloud.storage import verify
    with tempfile.TemporaryDirectory(prefix='novo-resume-state-') as folder:
        path=Path(folder)/'checkpoint.pt'
        with path.open('wb') as out:verify(c,b,pointer['checkpoint']['key'],pointer['checkpoint']['sha256'],out)
        checkpoint=torch.load(path,map_location='cpu',weights_only=True)
        from .state import require_resume
        require_resume(checkpoint,state['catalogue_sha256'],state['inventory_sha256'],state['plan_sha256'])
        if checkpoint['step']!=state['step'] or len(checkpoint['rank_rng'])!=2 or {float(s['step']) for s in checkpoint['optimizer']['state'].values()}!={float(state['step'])}:raise ValueError('Checkpoint optimizer/RNG mismatch')
    proof=preflight(c,b,pointer);quota=control.free_quota();charged=reconcile_reservation(record,quota)
    record.setdefault('execution_history',[]).append(dict(kernel=record['kernel'],source_sha=record['source_sha'],status='ERROR',last_logged_step=47900,resumed_checkpoint=pointer['checkpoint'],state_step=state['step'],old_reservation=record['reserved_seconds'],charged_upper_seconds=charged,preflight=proof))
    record.update(source_sha=source,status='preparing',auto_continue=True,transport_recovery_used=True,reserved_seconds=charged,checkpoint_r2_verified=True)
    write_json(c,b,active_key(),record)
    print(json.dumps(dict(event='v2_transport_repair_verified',source_sha=source,science_sha256=science_hash(),checkpoint_sha256=pointer['checkpoint']['sha256'],resume_step=state['step'],charged_upper_seconds=charged,approved_budget_seconds=record['approved_budget_seconds'],**{k:v for k,v in proof.items() if k!='resume_step'})),flush=True)
    control.recover(c,b,record)


def main():
    require_integrated_source(os.environ['NOVO_SOURCE_SHA'])
    c,b=control.connect(output=True);record=control.read(c,b,active_key())
    if os.environ['MODE']=='collect' and record and record['source_sha']==GPU_SOURCE:
        preserve();return
    if record and record['source_sha']==GPU_SOURCE and not record.get('transport_recovery_used'):
        if os.environ['MODE']=='recover' and os.environ.get('FORMAL_TRAINING_AUTHORIZED')=='true':repair_and_resume(c,b,record);return
        print(json.dumps(dict(gpu_submitted=False,repair_dispatch_pending=True)));return
    control.main()

if __name__=='__main__':main()
