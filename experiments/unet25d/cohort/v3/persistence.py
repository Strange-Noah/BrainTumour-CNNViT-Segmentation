"""Reuse V2 signed archive transport only for V3 object identities."""
import json
import time
from pathlib import Path
from urllib.parse import unquote, urlsplit

from ..v2.persistence import snapshot as shared_snapshot
from .settings import prefix


def verify_lease(root, min_expiry=None):
    lease = json.loads((Path(root) / 'private-output-lease.json').read_text())
    if lease['expires_at'] < (time.time() if min_expiry is None else min_expiry):
        raise ValueError('V3 private output lease expires too soon')
    slots = lease['slots']
    if not slots or len({s['key'] for s in slots}) != len(slots):
        raise ValueError('Unique V3 output slots required')
    for slot in slots:
        key = slot['key']
        if not key.startswith(prefix() + '/live/') or not key.endswith('.tar'):
            raise ValueError('V3 output namespace required')
        for name, suffix in [('put', ''), ('get', ''), ('receipt_put', '.json')]:
            url = urlsplit(slot[name])
            expect = '/novo-nordisk-pbl-outputs/' + key + suffix
            host = url.hostname or ''
            path = unquote(url.path)
            vhost = host.startswith('novo-nordisk-pbl-outputs.')
            accept = path == expect or (vhost and path == '/' + key + suffix)
            if url.scheme != 'https' or not host.endswith('.r2.cloudflarestorage.com') or not accept:
                raise ValueError('Signed V3 URL does not identify the declared output')


def snapshot(output, input_root, kind):
    verify_lease(input_root)
    return shared_snapshot(output, input_root, kind)
