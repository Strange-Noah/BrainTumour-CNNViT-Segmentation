"""CPU-only mount and streaming-integrity check; never reads R2 credentials."""
import argparse, hashlib, json
from pathlib import Path

def sha(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()

def main():
    parser=argparse.ArgumentParser();parser.add_argument('root',type=Path);a=parser.parse_args();root=a.root.resolve()
    stage=json.loads((root/'cache_integrity.stage.json').read_text())
    if stage.get('status')!='complete':raise ValueError('CPU integrity stage incomplete')
    if (root/'config.json').read_bytes()!=Path(__file__).with_name('config.json').read_bytes():raise ValueError('Prepared configuration differs from the frozen project configuration')
    config=json.loads((root/'config.json').read_text());receipts=json.loads((root/'cache-receipts.json').read_text())
    if config['shape']!=[512,512] or config['positive_labels']!=[3] or config['full_training_authorized']:raise ValueError('Unexpected configuration')
    size=0
    for name,expected in receipts.items():
        path=(root/'prepared'/name).resolve()
        if not path.is_relative_to((root/'prepared').resolve()):raise ValueError('Invalid receipt path')
        if sha(path)!=expected:raise ValueError('Prepared object SHA-256 mismatch')
        size+=path.stat().st_size
    train=json.loads((root/'prepared/train-index.json').read_text());validation=json.loads((root/'prepared/validation-index.json').read_text())
    if len(train)!=config['train_cache_visits'] or len(validation)!=config['validation_smoke_visits']:raise ValueError('Incomplete visit index')
    weight=root/'hf/hub/models--smp-hub--resnet34.imagenet/snapshots'/config['checkpoint_revision']/'model.safetensors'
    if sha(weight)!=config['checkpoint_sha256']:raise ValueError('Offline pretrained weight cache missing or invalid')
    print(json.dumps(dict(verified=True,prepared_files=len(receipts),prepared_bytes=size,train_visits=len(train),validation_visits=len(validation),config_sha256=sha(root/'config.json'),receipt_sha256=sha(root/'cache-receipts.json'),offline_weight_verified=True,gpu_allocated=False)))
if __name__=='__main__':main()
