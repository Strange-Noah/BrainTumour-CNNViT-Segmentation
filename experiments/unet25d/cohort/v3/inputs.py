"""Bind accepted V2 data objects to a new, private V3 runtime package."""
import argparse
import hashlib
import json
import re
from pathlib import Path

from .settings import CONFIG, config, science_hash
from ..v2.settings import config as v2_config

FROZEN_FILES = {
    'catalogue.json': '04b59724c5cc6ac654f85d51ad19e8a8a0766944d8211ecfd03a3495a418de1e',
    'inventory.json': '17664e661adb06523d0d938a7409c654df25989c50c21b3c244d5d14ffea2722',
    'plan.json': '4029aee0a723fce9794ee2ce5ee05e61f7c357656029bb55f8d0ec02d87d8860',
    'SAMPLING_AUDIT.json': '6660521f3b802628e7b67fd221861cab4b3c68e8f6852303eb9b161d1f17892e',
}
PLAN_SHA = '359433d8d4e5c8bdfa1008b2db26dd85945ff9683e8638fc4d290e3430d9e2a3'
REQUIRED = {*FROZEN_FILES, 'training-config.json', 'private-data-lease.json',
            'private-output-lease.json'}


def file_sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def encoder_snapshot(root, names):
    c = config()
    snapshots = [Path(name) for name in names
                 if name.startswith('hf/') and name.endswith('/model.safetensors')
                 and Path(name).parent.name == c['encoder_checkpoint_revision']]
    if len(snapshots) != 1:
        raise ValueError('Exactly one frozen encoder snapshot required')
    weights = snapshots[0]
    metadata = weights.with_name('config.json')
    if (names[weights.as_posix()] != c['encoder_checkpoint_sha256']
            or metadata.as_posix() not in names):
        raise ValueError('Frozen encoder dependencies are incomplete')
    return root / weights.parent


def verify_manifest(package, root):
    c = config()
    if c.pop('execution_schedule', None) != 'gate1-then-phase1-training-only':
        raise ValueError('V3 campaign requires the frozen training-only schedule')
    old = v2_config()
    for key in ('schema', 'bce_reduction', 'bce_positive_coefficient'):
        c.pop(key, None)
        old.pop(key, None)
    if c != old:
        raise ValueError('V3 must preserve the accepted V2 non-BCE configuration')
    files = package['files']
    if (not REQUIRED <= files.keys()
            or any(files.get(k) != v for k, v in FROZEN_FILES.items())
            or files['training-config.json'] != file_sha(CONFIG)
            or ('resume.pt' in files) != (root / 'resume.pt').is_file()):
        raise ValueError('V3 private package identity mismatch')
    encoder_snapshot(root, files)
    inventory = {p.relative_to(root).as_posix() for p in root.rglob('*')
                 if p.is_file() and p != root / 'input-package.json'}
    if inventory != files.keys():
        raise ValueError('Unverified file in V3 private input')
    for name in files:
        path = root / name
        if not path.resolve().is_relative_to(root.resolve()):
            raise ValueError('V3 private package path escapes input')


def write_manifest(root, source_sha):
    """Finalize a staged private directory. No publishing or remote writes."""
    root = Path(root)
    if not re.fullmatch(r'[0-9a-f]{40}', source_sha):
        raise ValueError('Expected reviewed V3 source SHA')
    manifest = root / 'input-package.json'
    if manifest.exists():
        raise FileExistsError('Use a new V3 staging directory')
    files = {p.relative_to(root).as_posix(): file_sha(p)
             for p in root.rglob('*') if p.is_file()}
    package = dict(source_sha=source_sha, science_sha256=science_hash(), files=files)
    verify_manifest(package, root)
    from .persistence import verify_lease
    verify_lease(root)
    manifest.write_text(json.dumps(package, sort_keys=True))
    return package

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('private_folder', type=Path)
    parser.add_argument('--source-sha', required=True)
    args = parser.parse_args()
    package = write_manifest(args.private_folder, args.source_sha)
    print(json.dumps({k: package[k] for k in ('source_sha', 'science_sha256')}))
