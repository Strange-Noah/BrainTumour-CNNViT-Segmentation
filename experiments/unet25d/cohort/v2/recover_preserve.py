"""Preserve-only CPU repair for immutable signed GPU receipt format. No GPU start."""
import json,os,re
from . import control
from .settings import prefix,active_key,latest_key,science_hash
from ..unattended import require_integrated_source
from ..settings import read_json_record

GPU_SOURCE='072b55ee1869d23c3cd4b928833e767c9aee4211'

from .receipt_io import read_receipt,LatestReceiptClient

def preserve():
    require_integrated_source(os.environ['NOVO_SOURCE_SHA'])
    c,b=control.connect(output=True);record=read_json_record(c,b,active_key())
    if record['source_sha']!=GPU_SOURCE or record['science_sha256']!=science_hash():raise PermissionError('Existing exact campaign only')
    from cloud.kaggle_benchmark_bridge import cli
    status=cli('kernels','status',record['kernel']).upper()
    if not any(s in status for s in ('ERROR','CANCEL')):raise PermissionError('Preserve-only repair requires terminated failed provider job')
    live=record['live_prefix']
    # Mark stopped before recovery. This operation has no submission path.
    record.update(auto_continue=False,status='recovering_failed')
    control.write_json(c,b,active_key(),record)
    from .persistence import recover_snapshot
    listing=LatestReceiptClient(c,live).list_objects_v2(Bucket=b,Prefix=live+'/')
    for item in listing.get('Contents',[]):
        event=read_receipt(c,b,item['Key'],live)
        if event['kind']!='formal_checkpoint':raise ValueError('Latest failed-run receipt is not a formal checkpoint')
        recover_snapshot(c,b,event)
        record['last_verified_snapshot']=event;record['checkpoint_r2_verified']=True
    from .stage import collect
    pointer,receipt=collect(record['kernel'])
    record.setdefault('collections',[]).append(receipt)
    record.update(auto_continue=False,status='failed',execution_state='COLLECTED',quota_after_session=control.free_quota())
    control.write_json(c,b,active_key(),record)
    final=read_json_record(c,b,active_key());pointer=read_json_record(c,b,latest_key())
    if final['auto_continue'] or final['status']!='failed':raise ValueError('Failed campaign must remain stopped')
    print(json.dumps(dict(event='v2_failed_campaign_preserved',provider_status='ERROR',status=final['status'],auto_continue=False,gpu_submitted=False,checkpoint=pointer['checkpoint'],state={k:pointer['state'][k] for k in ('step','cursor','mandatory_unique','replay_consumed','phase','learning_gate_passed')})),flush=True)

if __name__=='__main__':preserve()
