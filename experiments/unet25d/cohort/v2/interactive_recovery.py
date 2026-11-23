"""Resume the interrupted s3 interactive execution from its verified R2 snapshot."""
import json
import os
from pathlib import Path
import tempfile

from cloud.storage import verify
from ..settings import read_json_record
from ..unattended import require_integrated_source
from . import control
from .persistence import recover_snapshot
from .receipt_io import read_receipt
from .settings import active_key, latest_key, science_hash, write_json
from .state import require_resume
from .transport_recovery import preflight, reconcile_reservation

FAILED_SOURCE = 'ebf72f80a784e440532c80c717b880b312fe798c'
FAILED_KERNEL = 'mathewzhou/novo-unet25d-v2-44950328254-s3'
SNAPSHOT_SHA = '6e8ba88d651773471a9e8e0edf77d51dd6481771f8405c2f9135ca5b0b52864d'


def resume(client, bucket, record):
    from cloud.kaggle_benchmark_bridge import cli
    import torch

    if (os.environ.get('GITHUB_ACTOR'), os.environ.get('FORMAL_TRAINING_AUTHORIZED'),
            os.environ.get('FREE_RESOURCE_AUTHORIZED')) != ('Strange-Noah', 'true', 'true'):
        raise PermissionError('Explicit owner recovery dispatch required')
    if record['kernel'] != FAILED_KERNEL or record.get('interactive_recovery_used'):
        raise PermissionError('Only one recovery of the adjudicated failed execution')
    from kaggle.api.kaggle_api_extended import KaggleApi
    from kagglesdk.kernels.types.kernels_api_service import ApiGetAcceleratorQuotaStatisticsRequest
    from .receipt_io import LatestReceiptClient

    def check_inactive():
        if 'KernelWorkerStatus.CANCEL_ACKNOWLEDGED' not in cli('kernels', 'status', FAILED_KERNEL):
            raise PermissionError('The saved batch execution must be cancelled')
        api = KaggleApi()
        api.authenticate()
        with api.build_kaggle_client() as kaggle:
            quota = kaggle.kernels.kernels_api_client.get_accelerator_quota_statistics(
                ApiGetAcceleratorQuotaStatisticsRequest()).gpu_quota
        if quota.time_reserved.total_seconds() != 0 or quota.time_used.total_seconds() != 55598.912:
            raise PermissionError('GPU accounting changed since the stopped interactive execution')
    check_inactive()
    if record['science_sha256'] != science_hash():
        raise ValueError('Campaign science changed')
    live = record['live_prefix']
    receipt_key = live + '/20.tar.json'
    listing = LatestReceiptClient(client, live)
    newest = listing.list_objects_v2(Bucket=bucket, Prefix=live + '/').get('Contents', [])
    if [item['Key'] for item in newest] != [receipt_key]:
        raise ValueError('The independently verified snapshot is no longer the newest receipt')
    receipt = read_receipt(client, bucket, receipt_key, live)
    if receipt['sha256'] != SNAPSHOT_SHA or receipt['kind'] != 'formal_checkpoint':
        raise ValueError('Unexpected terminal snapshot')
    recover_snapshot(client, bucket, receipt)
    pointer = read_json_record(client, bucket, latest_key())
    state = pointer['state']
    if pointer['checkpoint']['sha256'] != 'ed447984974ba15cfce0790c30ae8f1f5f788a2a5dba2e92fb902b3a5474f140':
        raise ValueError('Checkpoint differs from independent verification')
    if state['step'] != 94475 or state['phase'] != 'PHASE1' or not state['learning_gate_passed']:
        raise ValueError('Unexpected checkpoint cursor or science state')
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / 'checkpoint.pt'
        with path.open('wb') as dest:
            verify(client, bucket, pointer['checkpoint']['key'], pointer['checkpoint']['sha256'], dest)
        checkpoint = torch.load(path, map_location='cpu', weights_only=True)
        require_resume(checkpoint, state['catalogue_sha256'], state['inventory_sha256'], state['plan_sha256'])
        if (checkpoint['step'] != state['step'] or len(checkpoint['rank_rng']) != 2
                or {float(s['step']) for s in checkpoint['optimizer']['state'].values()} != {float(state['step'])}):
            raise ValueError('Invalid optimizer step or rank RNG state')
        del checkpoint
    proof = preflight(client, bucket, pointer, last_logged_step=94475)
    check_inactive()
    newest = listing.list_objects_v2(Bucket=bucket, Prefix=live + '/').get('Contents', [])
    if [item['Key'] for item in newest] != [receipt_key]:
        raise ValueError('New snapshot appeared during recovery')
    charged = reconcile_reservation(record, control.free_quota())
    record.setdefault('execution_history', []).append({
        'kernel': FAILED_KERNEL, 'source_sha': FAILED_SOURCE, 'status': 'interactive_inactive_inferred',
        'last_logged_step': 94475, 'resumed_checkpoint': pointer['checkpoint'],
        'state_step': state['step'], 'charged_upper_seconds': charged, 'preflight': proof,
    })
    record.update(source_sha=os.environ['NOVO_SOURCE_SHA'], auto_continue=True, status='preparing',
                  reserved_seconds=charged, interactive_recovery_used=True, checkpoint_r2_verified=False)
    write_json(client, bucket, active_key(), record)
    print(json.dumps(dict(event='v2_interactive_recovery_verified', checkpoint=pointer['checkpoint'],
                          charged_upper_seconds=charged, **proof)), flush=True)
    control.recover(client, bucket, record)


def main():
    require_integrated_source(os.environ['NOVO_SOURCE_SHA'])
    client, bucket = control.connect(output=True)
    record = control.read(client, bucket, active_key())
    if record and record['source_sha'] == FAILED_SOURCE:
        if os.environ['MODE'] == 'recover' and os.environ.get('FORMAL_TRAINING_AUTHORIZED') == 'true':
            resume(client, bucket, record)
        else:
            print(json.dumps(dict(recovery_dispatch_pending=True, gpu_submitted=False)))
        return
    control.main()


if __name__ == '__main__':
    main()
