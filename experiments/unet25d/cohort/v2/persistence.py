"""Short-lived, output-prefix-only signed persistence; no permanent GPU credentials."""
import hashlib,io,json,tarfile,time,urllib.request,uuid
from pathlib import Path

def snapshot(output,input_root,kind):
    lease=json.loads((Path(input_root)/'private-output-lease.json').read_text())
    if time.time()>lease['expires_at']-300:raise RuntimeError('Output lease insufficient; preserve local checkpoint and stop')
    used=Path(output)/'persistence-cursor.json';cursor=json.loads(used.read_text())['next'] if used.exists() else 0
    if cursor>=len(lease['slots']):raise RuntimeError('Output slots exhausted')
    tmp=used.with_suffix('.tmp');tmp.write_text(json.dumps({'next':cursor+1}));tmp.replace(used)
    slot=lease['slots'][cursor];buffer=io.BytesIO()
    with tarfile.open(fileobj=buffer,mode='w') as archive:
        for p in sorted(Path(output).rglob('*')):
            if p.is_file() and p.suffix in ('.pt','.json','.jsonl') and not (kind=='formal_checkpoint' and p.name=='learning-diagnostic.pt') and not any(x in p.relative_to(output).parts for x in ('cache','fixed-cache','validation-cache','early-cache')):archive.add(p,arcname=str(p.relative_to(output)),recursive=False)
    data=buffer.getvalue();sha=hashlib.sha256(data).hexdigest()
    with urllib.request.urlopen(urllib.request.Request(slot['put'],data=data,headers={'If-None-Match':'*'},method='PUT'),timeout=180) as r:
        if r.status not in (200,201):raise RuntimeError('Private upload failed')
    h=hashlib.sha256()
    with urllib.request.urlopen(slot['get'],timeout=180) as r:
        while b:=r.read(1024**2):h.update(b)
    if h.hexdigest()!=sha:raise ValueError('Private checkpoint readback mismatch')
    receipt=dict(event='v2_private_snapshot',kind=kind,key=slot['key'],sha256=sha,bytes=len(data),slot=cursor)
    payload=json.dumps(receipt).encode()
    with urllib.request.urlopen(urllib.request.Request(slot['receipt_put'],data=payload,headers={'If-None-Match':'*'},method='PUT'),timeout=60):pass
    used.write_text(json.dumps({'next':cursor+1}))
    print(json.dumps(receipt),flush=True);return receipt

def recover_snapshot(client,bucket,event):
    """Independent CPU readback and monotonic resume pointer, including live runs."""
    import tempfile,shutil
    from cloud.storage import verify,put_once
    from ..settings import read_json_record
    from .settings import prefix,latest_key,write_json,science_hash
    if not event['key'].startswith(prefix()+'/live/'):raise PermissionError('Foreign live archive')
    with tempfile.TemporaryDirectory(prefix='novo-live-readback-') as folder:
        root=Path(folder);archive=root/'snapshot.tar'
        with archive.open('wb') as out:verify(client,bucket,event['key'],event['sha256'],out)
        if event['kind']!='formal_checkpoint':return False
        with tarfile.open(archive) as source:
            state=json.load(source.extractfile('training-state.json'))
            if state['science_sha256']!=science_hash() or state['cursor']!=2*state['step']:raise ValueError('Live checkpoint identity/cursor')
            try:old=read_json_record(client,bucket,latest_key())
            except client.exceptions.NoSuchKey:old=None
            if old and old['state']['step']>state['step']:raise ValueError('Live checkpoint rollback')
            path=root/'checkpoint.pt'
            with source.extractfile('checkpoint.pt') as src,path.open('wb') as dst:shutil.copyfileobj(src,dst)
        receipt=put_once(client,bucket,event['key']+'/checkpoint.pt',path)
        write_json(client,bucket,latest_key(),dict(state=state,checkpoint=receipt,live_archive_sha256=event['sha256']))
        return True
