"""Resume the exact failed s2 once, after independent state and data checks."""
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

FAILED_SOURCE = '9c13b991de9231277270dd115b2fa05248cac18d'
FAILED_KERNEL = 'mathewzhou/novo-unet25d-v2-44950328254-s2'
SNAPSHOT_SHA = 'e3a8c71b51b3d57fb11a9e194168d57b28bd9a70c5eadecf010a9383d686ac06'


def resume(client, bucket, record):
    from cloud.kaggle_benchmark_bridge import cli
    import torch

    if (os.environ.get('GITHUB_ACTOR'), os.environ.get('FORMAL_TRAINING_AUTHORIZED'),
            os.environ.get('FREE_RESOURCE_AUTHORIZED')) != ('Strange-Noah', 'true', 'true'):
        raise PermissionError('Explicit owner recovery dispatch required')
    if record['kernel'] != FAILED_KERNEL or record.get('checksum_recovery_used'):
        raise PermissionError('Only one recovery of the adjudicated failed execution')
    if 'KernelWorkerStatus.ERROR' not in cli('kernels', 'status', FAILED_KERNEL):
        raise PermissionError('Provider termination must be confirmed')
    if record['science_sha256'] != science_hash():
        raise ValueError('Campaign science changed')
    live = record['live_prefix']
    receipt = read_receipt(client, bucket, live + '/27.tar.json', live)
    if receipt['sha256'] != SNAPSHOT_SHA or receipt['kind'] != 'formal_checkpoint':
        raise ValueError('Unexpected terminal snapshot')
    recover_snapshot(client, bucket, receipt)
    pointer = read_json_record(client, bucket, latest_key())
    state = pointer['state']
    if state['step'] != 74387 or state['phase'] != 'PHASE1' or not state['learning_gate_passed']:
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
    proof = preflight(client, bucket, pointer, last_logged_step=75000)
    charged = reconcile_reservation(record, control.free_quota())
    record.setdefault('execution_history', []).append({
        'kernel': FAILED_KERNEL, 'source_sha': FAILED_SOURCE, 'status': 'ERROR',
        'last_logged_step': 75000, 'resumed_checkpoint': pointer['checkpoint'],
        'state_step': state['step'], 'charged_upper_seconds': charged, 'preflight': proof,
    })
    record.update(source_sha=os.environ['NOVO_SOURCE_SHA'], auto_continue=True, status='preparing',
                  reserved_seconds=charged, checksum_recovery_used=True, checkpoint_r2_verified=False)
    write_json(client, bucket, active_key(), record)
    print(json.dumps(dict(event='v2_checksum_recovery_verified', checkpoint=pointer['checkpoint'],
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
