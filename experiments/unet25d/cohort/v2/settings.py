"""v2 science identity; reuse immutable prepared inputs, never v1 training state."""
import hashlib,json
from pathlib import Path
from ..settings import keybase as prepared_keybase, policy_hash, read_json_record
ROOT=Path(__file__).resolve().parents[4]
CONFIG=Path(__file__).resolve().parents[2]/'runs/v2/config.json'
DATASET='mathewzhou/novo-unet25d-v2-input'
KERNEL='mathewzhou/novo-unet25d-v2'
SCHEMA='foreground-lesion-multiscale-v2'
def config():return json.loads(CONFIG.read_text())
def sha(raw):return hashlib.sha256(raw).hexdigest()
def science_hash():return sha(json.dumps(config(),sort_keys=True).encode())
def prefix():return 'novo-nordisk/experiments/unet25d/v2/'+science_hash()
def active_key():return prefix()+'/active.json'
def latest_key():return prefix()+'/latest.json'
def aux_key():return prefix()+'/aux/ready.json'
def write_json(c,b,key,value):
    if not key.startswith(prefix()+'/'):raise PermissionError('v2 output namespace required')
    raw=json.dumps(value,sort_keys=True).encode();c.put_object(Bucket=b,Key=key,Body=raw,Metadata={'sha256':sha(raw)})
def validate_catalogue(cat):
    import csv
    from ..settings import ROOT as project_root,digest
    path=project_root/'protocol/v1/patients.csv'
    if digest(path)!=config()['split_sha256']:raise ValueError('Frozen split changed')
    with path.open() as f:parts={r['patient_id']:r['partition'] for r in csv.DictReader(f)}
    if cat['summary']['patients']!={'train':210,'validation':45} or cat['summary']['test_included'] or len(cat['visits'])!=905:raise ValueError('Frozen prepared cohort required')
    seen=set()
    for v in cat['visits']:
        m=v['metadata'];e=m['exam_id']
        if e in seen or v['partition'] not in ('train','validation') or parts[m['patient_id']]!=v['partition']:raise ValueError('Duplicate or nontraining partition identity')
        seen.add(e)
        if m['oblique_resampled'] or m['grid_policy']!='native-T1c-with-affine':raise ValueError('Expected preserved native physical lattice')
    for partition,patients,visits in (('train',210,746),('validation',45,159)):
        rows=[v for v in cat['visits'] if v['partition']==partition]
        if len(rows)!=visits or len({v['metadata']['patient_id'] for v in rows})!=patients:raise ValueError('Actual cohort coverage mismatch')
