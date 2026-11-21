"""One owner-authorized queue replacement; scientific runtime remains immutable."""
import hashlib,json,os,subprocess,tempfile,time,zipfile
from pathlib import Path

SCIENCE_SOURCE='3ddc324e114d0a0cd740234b5ba26ec750238d2a'
ORIGINAL='mathewzhou/novo-unet25d-v2-44906471093-s1'
INPUT_VERSION=3
EXPECTED_SCIENCE='1d6fa16a689916e4225d561b7f2217e61f0ec423cddb9894b5ac3f8636ee8110'


def released_reservation(record,status,quota,files):
    if 'CANCEL' not in status.upper():raise PermissionError('Old provider session is not cancelled')
    if files:raise PermissionError('Unexpected old outputs: recover before replacement')
    if quota['used_seconds']!=0:raise PermissionError('Nonzero GPU usage: manual reservation accounting required')
    if record['kernel']!=ORIGINAL or record['session_no']!=1 or 'queue_replacement' in record:
        raise PermissionError('Exactly one replacement of the original session only')
    if record['source_sha']!=SCIENCE_SOURCE or record['science_sha256']!=EXPECTED_SCIENCE:
        raise PermissionError('Original scientific identity required')
    if record['initial_quota']['refresh_at']!=quota['refresh_at']:raise PermissionError('No future-week renewal')
    if record['reserved_seconds']!=43200 or record['session_seconds']!=43200:
        raise PermissionError('Unexpected reservation; never reset cumulative budget')
    return record['reserved_seconds']-record['session_seconds']


def verified_input(c,b):
    from cloud.kaggle_benchmark_bridge import cli
    from .settings import DATASET,aux_key,read_json_record
    ready=read_json_record(c,b,aux_key())
    if not ready['complete'] or ready['science_sha256']!=EXPECTED_SCIENCE:raise ValueError('Original Gate0 required')
    with tempfile.TemporaryDirectory(prefix='novo-v2-replacement-private-') as d:
        root=Path(d);root.chmod(0o700)
        for name in ('input-package.json','private-data-lease.json','training-config.json','plan.json','inventory.json'):
            cli('datasets','download','-d',DATASET+'/'+str(INPUT_VERSION),'-f',name,'-p',d,'-q')
            p=root/name
            if not p.exists():
                with zipfile.ZipFile(root/(name+'.zip')) as z:p.write_bytes(z.read(name))
            p.chmod(0o600)
        package=json.loads((root/'input-package.json').read_text())
        if package['source_sha']!=SCIENCE_SOURCE or package['science_sha256']!=EXPECTED_SCIENCE:
            raise ValueError('Fresh package scientific source mismatch')
        for name in ('private-data-lease.json','training-config.json','plan.json','inventory.json'):
            if hashlib.sha256((root/name).read_bytes()).hexdigest()!=package['files'][name]:raise ValueError('Private input checksum')
        for name in ('plan.json','inventory.json'):
            if package['files'][name]!=ready['objects'][name]['sha256']:raise ValueError('Accepted plan/inventory changed')
        cat=ready['catalogue']['sha256'];lease=json.loads((root/'private-data-lease.json').read_text())
        if package['files']['catalogue.json']!=cat or lease['catalogue_sha256']!=cat:raise ValueError('Frozen catalogue changed')
        config=json.loads((root/'training-config.json').read_text())
        if hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()!=EXPECTED_SCIENCE:raise ValueError('Scientific config changed')
        if lease['expires_at']-time.time()<45000:raise ValueError('Lease insufficient for twelve-hour session plus margin')
        return dict(dataset=DATASET,version=INPUT_VERSION,catalogue_sha256=cat,private_input_verified=True,science_sha256=EXPECTED_SCIENCE,lease_expires_at=lease['expires_at'])


def replace():
    from cloud.kaggle_benchmark_bridge import cli
    from kaggle.api.kaggle_api_extended import KaggleApi
    import importlib
    from cloud.storage import connect
    from . import control,stage
    from .settings import active_key,science_hash,write_json
    if os.environ.get('GITHUB_ACTOR')!='Strange-Noah' or os.environ.get('FORMAL_TRAINING_AUTHORIZED')!='true' or os.environ.get('FREE_RESOURCE_AUTHORIZED')!='true':
        raise PermissionError('Explicit owner replacement/free authorization required')
    c,b=connect(output=True);record=control.read(c,b,active_key())
    if not record or science_hash()!=EXPECTED_SCIENCE:raise PermissionError('Existing v2 campaign required')
    # Preserve all original model/data/source identities and budget. Input was
    # refreshed exactly once in CPU run 44929363445; never publish again here.
    status=cli('kernels','status',ORIGINAL).upper()
    a=KaggleApi();a.authenticate();types=importlib.import_module('kagglesdk.kernels.types.kernels_api_service')
    with a.build_kaggle_client() as api:
        q=types.ApiListKernelSessionOutputRequest();q.user_name='mathewzhou';q.kernel_slug=ORIGINAL.split('/')[1];q.page_size=100
        outputs=api.kernels.kernels_api_client.list_kernel_session_output(q)
        if outputs.next_page_token:raise PermissionError('Unexpected paginated original outputs')
    quota=control.free_quota();new_reserved=released_reservation(record,status,quota,outputs.files)
    publication=verified_input(c,b)
    # Independent fresh provider confirmation immediately before state transition.
    final_status=cli('kernels','status',ORIGINAL).upper()
    released_reservation(record,final_status,quota,outputs.files)
    record['queue_replacement']=dict(original_kernel=ORIGINAL,original_version=record['version'],terminal_status='CANCEL_ACKNOWLEDGED',verified_at=time.time(),released_unused_seconds=43200,input_version=INPUT_VERSION,lease_expires_at=publication['lease_expires_at'],operator_run=os.environ['GITHUB_RUN_ID'])
    record.update(reserved_seconds=new_reserved,status='authorized',auto_continue=True,initial_observation_complete=False)
    write_json(c,b,active_key(),record)
    old_publish=stage.publish
    def existing_publication(seconds):
        if seconds>43200 or publication['lease_expires_at']-time.time()<seconds+1800:raise ValueError('Fresh lease no longer sufficient')
        return publication
    stage.publish=existing_publication
    try:control.start(c,b,record)
    finally:stage.publish=old_publish
    if record['kernel']==ORIGINAL:raise RuntimeError('Distinct replacement identity required')
    # Provider readback: current only version, exact approved launcher, private,
    # T4 and expected attached dataset. Publication readback above pins v3 bytes;
    # provider metadata may normalize the displayed dataset reference.
    with a.build_kaggle_client() as api:
        q=types.ApiGetKernelRequest();q.user_name='mathewzhou';q.kernel_slug=record['kernel'].split('/')[1]
        got=api.kernels.kernels_api_client.get_kernel(q);meta=got.metadata
        expected=control.render(SCIENCE_SOURCE,record['session_seconds'],record['approved_budget_seconds'])
        if got.blob.source!=expected or not meta.is_private or not meta.enable_gpu or meta.machine_shape!='NvidiaTeslaT4' or meta.current_version_number!=record['version']:
            record.update(status='blocked',auto_continue=False);write_json(c,b,active_key(),record);raise ValueError('Replacement provider metadata mismatch; never submit retry')
        if not any(s.startswith(publication['dataset']) for s in meta.dataset_data_sources):raise ValueError('Expected private input absent')
    record['queue_replacement']['new_submission_verified']=True;write_json(c,b,active_key(),record)
    print(json.dumps(dict(event='v2_queue_replacement_complete',old_kernel=ORIGINAL,old_terminal_verified=True,kernel=record['kernel'],input_version=INPUT_VERSION,input_version_submitted_explicitly=True,provider_input_version_readback='normalized dataset reference; exact mounted version checked by package identity at startup',lease_expires_at=publication['lease_expires_at'],source_sha=SCIENCE_SOURCE,science_sha256=EXPECTED_SCIENCE,approved_budget_seconds=record['approved_budget_seconds'],reserved_seconds=record['reserved_seconds'],gpu_submissions=1)),flush=True)
    control.observe(c,b,record)


def main():
    from ..unattended import require_integrated_source
    require_integrated_source(os.environ['NOVO_SOURCE_SHA'])
    # Operations wrapper is separately reviewed; GPU launcher and all scientific
    # source remain at the owner's immutable SHA, including future recovery.
    os.environ['NOVO_SOURCE_SHA']=SCIENCE_SOURCE
    if os.environ['MODE']=='formal':replace()
    else:
        from . import control
        c,b=control.connect(output=True);record=control.read(c,b,control.active_key())
        if record and record['kernel']==ORIGINAL:
            print(json.dumps(dict(v2_replacement_waiting=True,gpu_submitted=False)));return
        control.main()

if __name__=='__main__':main()
