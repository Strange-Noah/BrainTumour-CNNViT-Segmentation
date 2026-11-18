"""Recover verified checkpoints and export already computed logits; never rerun inference."""
import argparse,hashlib,json,os,re,subprocess,sys,tempfile
from pathlib import Path
import numpy as np
from cloud.storage import connect,verify,put_once
from .persist_cache import read,sha
from .export_native import export_native
CACHE_PREFIX='novo-nordisk/experiments/unet25d/prepared-cache/fee663bf826c48dfb7bac6609813410d'
CACHE_SHA='37176945ceff56d7d8411e74a40e676a66a168f39780aea58d7659793034547e'

def main():
    parser=argparse.ArgumentParser();parser.add_argument('gpu_run');parser.add_argument('collection_sha');parser.add_argument('training_sha');args=parser.parse_args()
    if not re.fullmatch('[0-9]+',args.gpu_run) or not all(re.fullmatch('[0-9a-f]{'+str(n)+'}',v) for n,v in [(64,args.collection_sha),(40,args.training_sha)]):raise ValueError('Invalid immutable execution references')
    c,b=connect(output=True)
    if b!='novo-nordisk-pbl-outputs':raise ValueError('Private output bucket required')
    base='novo-nordisk/experiments/unet25d/bridge/'+args.gpu_run
    raw=read(c,base+'/collection.json')
    if sha(raw)!=args.collection_sha:raise ValueError('GPU collection receipt checksum mismatch')
    collection=json.loads(raw);receipts=collection['receipts']
    if not collection['independent_readback']:raise ValueError('GPU output readback incomplete')
    manifest_raw=read(c,CACHE_PREFIX+'/manifest.json')
    if sha(manifest_raw)!=CACHE_SHA:raise ValueError('Original cache manifest changed')
    row=next(r for r in json.loads(manifest_raw)['files'] if r['path']=='prepared/validation-index.json')
    if row['bytes']>4*1024**2:raise ValueError('Unexpected validation metadata size')
    index=b''
    for chunk in row['chunks']:
        data=read(c,CACHE_PREFIX+'/chunks/'+chunk['sha256'])
        if len(data)!=chunk['bytes'] or sha(data)!=chunk['sha256']:raise ValueError('Validation metadata chunk mismatch')
        index+=data
    if len(index)!=row['bytes'] or sha(index)!=row['sha256']:raise ValueError('Validation metadata checksum mismatch')
    with tempfile.TemporaryDirectory(prefix='novo-private-cpu-recovery-') as name:
        root=Path(name);root.chmod(0o700);model_root=root/'models';outputs=[]
        def recover(receipt,destination):
            destination.parent.mkdir(parents=True,exist_ok=True)
            with destination.open('wb') as f:verify(c,b,receipt['key'],receipt['sha256'],f)
            if destination.stat().st_size!=receipt['bytes']:raise ValueError('Recovered size mismatch')
            return destination
        def select(suffix):
            matches=[r for r in receipts if r['key'].endswith(suffix)]
            if len(matches)!=1:raise ValueError('Missing or ambiguous verified GPU output')
            return matches[0]
        for stage in ['A-sustained','B-sustained','resume']:
            row=select('/'+stage+'/checkpoint.pt');recover(row,model_root/stage/'checkpoint.pt')
        resume=json.loads(recover(select('/resume/resume-verification.json'),root/'resume.json').read_text())
        if not resume['verified'] or resume['successful_updates']!=4:raise ValueError('GPU optimizer continuation incomplete')
        # Fresh CPU process restores original model/optimizer state; no training.
        subprocess.run([sys.executable,'-m','experiments.unet25d.recover_local',str(model_root)],check=True,timeout=300)
        recovery=json.loads((model_root/'checkpoint-recovery.json').read_text())
        if recovery['checkpoints']!=3:raise ValueError('Not all healthy checkpoints recovered')
        import torch
        state=torch.load(model_root/'resume/checkpoint.pt',map_location='cpu',weights_only=True)
        if state['git_sha']!=args.training_sha:raise ValueError('Unexpected training source')
        del state
        for i,meta in enumerate(json.loads(index)):
            if not re.fullmatch('[A-Za-z0-9_-]+',meta['exam_id']):raise ValueError('Unsafe visit reference')
            suffix='/validation/'+meta['exam_id']+'/'
            # Metrics exist only after all GPU inference slices and native evaluation completed.
            metrics=json.loads(recover(select(suffix+'metrics.json'),root/f'metrics-{i}.json').read_text())
            if not 0<=metrics['dice']<=1:raise ValueError('Invalid completed validation metric')
            path=recover(select(suffix+'canonical-prediction.npy'),root/f'logits-{i}.npy')
            prediction=np.load(path,mmap_mode='r');dest=root/'exports'/meta['exam_id']
            export_native(prediction,meta,dest,args.training_sha,select('/resume/checkpoint.pt')['key'])
            del prediction
            output_base=base+'/cpu-recovery/'+os.environ['GITHUB_RUN_ID']+'/exports/'+meta['exam_id']
            for file in dest.iterdir():outputs.append(put_once(c,b,output_base+'/'+file.name,file))
        reader,rb=connect(output=True)
        for row in outputs:verify(reader,rb,row['key'],row['sha256'])
        report=dict(verified=True,gpu_allocated=False,new_inference=False,new_training=False,gpu_run=args.gpu_run,training_source_sha=args.training_sha,repair_source_sha=os.environ['NOVO_SOURCE_SHA'],collection_sha256=args.collection_sha,checkpoints_recovered=recovery['checkpoints'],resume_verified=True,validation_exports=len(json.loads(index)),native_roundtrip_verified=True,independent_readback=True,outputs=outputs)
        path=root/'report.json';path.write_text(json.dumps(report,indent=2)+'\n');receipt=put_once(c,b,base+'/cpu-recovery/'+os.environ['GITHUB_RUN_ID']+'/report.json',path)
        print(json.dumps(dict(verified=True,gpu_allocated=False,new_inference=False,checkpoints_recovered=3,validation_exports=report['validation_exports'],report=receipt)))
if __name__=='__main__':
    try:main()
    except Exception as error:
        print(json.dumps(dict(verified=False,gpu_allocated=False,error_type=type(error).__name__)),flush=True);sys.exit(1)
