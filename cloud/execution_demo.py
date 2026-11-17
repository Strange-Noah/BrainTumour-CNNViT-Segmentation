"""Autonomous cloud acceptance: private manifest read, computation, output recovery."""
import argparse,hashlib,json,os,subprocess,sys,tempfile
from pathlib import Path
from cloud.storage import connect,DATASET,PREFIX,put_once,verify
MANIFEST='040ec6c523d5beae43a30b64fede2bee226e1a08e8c14b2cca8043b01b9c39f4'
def main():
    p=argparse.ArgumentParser();p.add_argument('--verify',nargs=2,metavar=('KEY','SHA'));a=p.parse_args()
    if a.verify:
        client,bucket=connect(output=True);key,digest=a.verify
        verify(client,bucket,key,digest)
        print(json.dumps(dict(independent_readback=True,sha256=digest)));return
    client,bucket=connect();response=client.get_object(Bucket=bucket,Key=f'{DATASET}/manifests/{MANIFEST}/visits.jsonl')
    with response['Body'] as body:data=body.read()
    if hashlib.sha256(data).hexdigest()!=MANIFEST:raise ValueError('Manifest digest mismatch')
    visits=[json.loads(line) for line in data.splitlines()];patients={v['patient_id'] for v in visits}
    if len(visits)!=1065 or len(patients)!=300:raise ValueError('Unexpected frozen dataset counts')
    report=dict(source_sha=os.environ['NOVO_SOURCE_SHA'],workflow_sha=os.environ['GITHUB_SHA'],run_id=os.environ['GITHUB_RUN_ID'],input_sha256=MANIFEST,input_bytes=len(data),patients=len(patients),visits=len(visits),computation='parse verified canonical manifest and count unique patients/visits',environment='GitHub Actions ubuntu-24.04 standard CPU',gpu_allocated=False,manual_notebook_execution=False)
    output,output_bucket=connect(output=True)
    if output_bucket!='novo-nordisk-pbl-outputs' or output_bucket==bucket:raise ValueError('Separate output bucket required')
    key=f'{PREFIX}/execution/acceptance/{report["run_id"]}/result.json'
    with tempfile.TemporaryDirectory() as name:
        path=Path(name)/'result.json';path.write_text(json.dumps(report,sort_keys=True)+'\n');receipt=put_once(output,output_bucket,key,path)
    subprocess.run([sys.executable,'-m','cloud.execution_demo','--verify',key,receipt['sha256']],check=True)
    print(json.dumps(dict(accepted=True,**report,output_receipt=receipt)))
if __name__=='__main__':main()
