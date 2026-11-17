"""Fresh-process output GET/hash and safe full training-state restore."""
import json,sys
from pathlib import Path
import torch
from cloud.storage import connect, verify
from .train import Model

def main():
    records=json.loads(Path(sys.argv[1]).read_text());folder=Path(sys.argv[2]);folder.mkdir(parents=True,exist_ok=True);client,bucket=connect(output=True);checkpoints=[]
    for i,r in enumerate(records):
        if not r['key'].startswith('novo-nordisk/experiments/unet25d/'):raise ValueError('Invalid prefix')
        path=folder/f'{i}-{Path(r["key"]).name}'
        with path.open('wb') as stream:verify(client,bucket,r['key'],r['sha256'],stream)
        if path.stat().st_size!=r['bytes']:raise ValueError('Size mismatch')
        if Path(r['key']).name=='checkpoint.pt':checkpoints.append(path)
    for path in checkpoints:
        state=torch.load(path,map_location='cpu',weights_only=True)
        model=Model(encoder_name='resnet34',encoder_weights=None,in_channels=12,classes=1,activation=None)
        model.load_state_dict(state['model'],strict=True)
        optimizer=torch.optim.AdamW([{'params':model.encoder.parameters()},{'params':list(model.decoder.parameters())+list(model.segmentation_head.parameters())}]);optimizer.load_state_dict(state['optimizer'])
        assert state['step']>0 and state['sampler'] and state['rng_cpu'] is not None
        del model,optimizer,state
    print(json.dumps(dict(verified=True,artifacts=len(records),checkpoints=len(checkpoints),training_continuation='not exercised by this CPU restore')))
if __name__=='__main__':main()
