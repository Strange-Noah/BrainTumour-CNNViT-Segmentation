"""Build the normal training input catalogue from complete preparation receipts."""
import csv,hashlib,json,os,tempfile
from pathlib import Path
from cloud.storage import connect,put_once,DATASET
from .settings import ROOT,CONFIG,MANIFEST_SHA,policy_hash,keybase,read_json_record

def main():
    read,b=connect();write,out=connect(output=True)
    if out!='novo-nordisk-pbl-outputs':raise ValueError('Private output bucket required')
    try:
        ready=read_json_record(write,out,keybase()+'/ready.json')
        if ready.get('data_policy_sha256')!=policy_hash():raise ValueError('Existing prepared policy mismatch')
        ready['already_complete']=True;print(json.dumps(ready));return
    except write.exceptions.NoSuchKey:pass
    body=read.get_object(Bucket=b,Key=f'{DATASET}/manifests/{MANIFEST_SHA}/visits.jsonl')['Body'];raw=body.read();body.close()
    if hashlib.sha256(raw).hexdigest()!=MANIFEST_SHA:raise ValueError('Manifest identity changed')
    split=ROOT/'protocol/v1/patients.csv';config=json.loads(CONFIG.read_text())
    if hashlib.sha256(split.read_bytes()).hexdigest()!=config['split_sha256']:raise ValueError('Frozen split changed')
    with split.open() as f:parts={r['patient_id']:r['partition'] for r in csv.DictReader(f)}
    records=[];expected=[];missing=0
    for row in map(json.loads,raw.splitlines()):
        partition=parts[row['patient_id']]
        if partition=='test':continue
        expected.append(row)
        try:
            r=read_json_record(write,out,keybase()+'/'+partition+'/'+row['exam_id']+'/receipt.json')
        except write.exceptions.NoSuchKey:missing+=1;continue
        if r['data_policy_sha256']!=policy_hash() or r['partition']!=partition or r['metadata']['patient_id']!=row['patient_id']:raise ValueError('Prepared visit identity mismatch')
        if r['shards'][0]['start']!=0 or r['shards'][-1]['end']!=r['metadata']['shape'][2] or any(a['end']!=b['start'] for a,b in zip(r['shards'],r['shards'][1:])):raise ValueError('Missing prepared slice range')
        records.append(r)
    summary=dict(data_policy_sha256=policy_hash(),complete=missing==0,missing_visits=missing,prepared_visits=len(records),patients={p:len({r['metadata']['patient_id'] for r in records if r['partition']==p}) for p in ['train','validation']},compressed_shard_bytes=sum(s['bytes'] for r in records for s in r['shards']),test_included=False,reused_normalized_visits=sum('reused_normalization_manifest_sha256' in r['metadata'] for r in records))
    if missing:print(json.dumps(summary));return
    if summary['patients']!={'train':210,'validation':45}:raise ValueError('Incomplete frozen patient coverage')
    from .accept_input import coverage
    summary.update(coverage(records,expected));summary['compressed_prepared_bytes']=summary['compressed_shard_bytes']+summary['compressed_native_label_bytes']
    with tempfile.TemporaryDirectory() as d:
        path=Path(d)/'manifest.json';path.write_text(json.dumps(dict(config_sha256=hashlib.sha256(CONFIG.read_bytes()).hexdigest(),data_policy_sha256=policy_hash(),split_sha256=config['split_sha256'],visits=records,summary=summary)))
        summary['manifest']=put_once(write,out,keybase()+'/catalogues/'+hashlib.sha256(path.read_bytes()).hexdigest()+'.json',path)
        ready=Path(d)/'ready.json';ready.write_text(json.dumps(summary,sort_keys=True));put_once(write,out,keybase()+'/ready.json',ready)
    print(json.dumps(summary))
if __name__=='__main__':main()
