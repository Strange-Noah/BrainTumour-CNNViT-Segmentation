"""Frozen identities and operational limits; safe in the CPU control plane."""
import hashlib,json
from pathlib import Path
PROVIDER_SESSION_SECONDS=12*3600
FINALIZATION_MARGIN_SECONDS=600
ROOT=Path(__file__).resolve().parents[3];CONFIG=Path(__file__).with_name('config.json');MANIFEST_SHA='040ec6c523d5beae43a30b64fede2bee226e1a08e8c14b2cca8043b01b9c39f4'

def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def policy_hash():
    config=json.loads(CONFIG.read_text());keys=['schema','split_sha256','modalities','offsets','target_labels','preserve_raw_labels','tile','tile_stride','gaussian_sigma_pixels','normalization','physical_grid','image_interpolation','label_interpolation','centers_per_shard','context_halo']
    return hashlib.sha256(json.dumps({k:config[k] for k in keys},sort_keys=True).encode()).hexdigest()
def read_json_record(client,bucket,key):
    response=client.get_object(Bucket=bucket,Key=key);raw=response['Body'].read();response['Body'].close()
    if hashlib.sha256(raw).hexdigest()!=response.get('Metadata',{}).get('sha256'):raise ValueError('Prepared metadata integrity mismatch')
    return json.loads(raw)
def keybase():return 'novo-nordisk/experiments/unet25d/full-cohort/'+policy_hash()

def science_config():
    c=json.loads(CONFIG.read_text());return {k:v for k,v in c.items() if k not in ['formal_training_authorized','approved_budget_seconds']}
def science_hash():return hashlib.sha256(json.dumps(science_config(),sort_keys=True).encode()).hexdigest()
CACHE_PREFIX='novo-nordisk/experiments/unet25d/prepared-cache/fee663bf826c48dfb7bac6609813410d'
CACHE_SHA='37176945ceff56d7d8411e74a40e676a66a168f39780aea58d7659793034547e'
ACTIVE_KEY='novo-nordisk/experiments/unet25d/full-training-active.json'
