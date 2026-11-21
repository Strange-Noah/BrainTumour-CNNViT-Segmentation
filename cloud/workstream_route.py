"""Trusted-main routing, evaluated before personal code or secrets."""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
from cloud.resolve_workstream import resolve


def route(name, source, actor, ref, event, mode='', dataset='', kernel='', cpu=False):
    if ref != 'refs/heads/main':
        raise PermissionError('Credential-bearing dispatch must originate on main')
    registry = json.loads(Path('cloud/workstreams.json').read_text())
    record = registry[name]
    if event == 'schedule' and not record['auto_recover']:
        return {'enabled': 'false', 'reason': 'automatic recovery disabled'}
    if event not in ('schedule', 'workflow_dispatch'):
        raise PermissionError('Unsupported dispatch event')
    if actor != record['authorized_actor']:
        raise PermissionError('Unauthorized dispatch actor')
    if cpu:
        if not re.fullmatch('[0-9a-f]{40}', source):
            raise PermissionError('Expected immutable source SHA')
        shared = subprocess.run(['git', 'merge-base', '--is-ancestor', source, 'origin/main'], check=False).returncode == 0
        if shared:
            return {'enabled': 'true', 'source_sha': source, 'approved_ref': 'origin/main'}
    elif mode == 'formal' and not record['formal_enabled']:
        raise PermissionError('Formal training disabled for archived campaign')
    elif mode == 'recover' and not record['auto_recover']:
        return {'enabled': 'false', 'reason': 'automatic recovery disabled'}
    source = source or record['source_sha']
    record = resolve(name, source, actor)
    for field, value in (('dataset', dataset), ('kernel', kernel)):
        if value and value != record[field]:
            raise PermissionError('Unregistered resource name')
    return {'enabled': 'true', 'source_sha': source, 'approved_ref': 'origin/'+record['branch'], 'dataset': record['dataset'], 'kernel': record['kernel'], 'entrypoint': record['entrypoint']}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--cpu', action='store_true')
    args = p.parse_args()
    result = route(os.environ.get('WORKSTREAM', 'unet25d'), os.environ.get('SOURCE_SHA', ''), os.environ['DISPATCH_ACTOR'], os.environ['DISPATCH_REF'], os.environ['DISPATCH_EVENT'], os.environ.get('MODE', ''), os.environ.get('DATASET', ''), os.environ.get('KERNEL', ''), args.cpu)
    with open(os.environ['GITHUB_OUTPUT'], 'a') as f:
        for key,value in result.items():
            f.write(f'{key}={value}\n')
    print(json.dumps(result, sort_keys=True))


if __name__ == '__main__':
    main()
