"""Fresh CPU process restores full private Kaggle training checkpoints."""
import json,sys
from pathlib import Path
import torch
from .train import Model
root=Path(sys.argv[1]);count=0;skipped=0
for path in root.glob('*/checkpoint.pt'):
 state=torch.load(path,map_location='cpu',weights_only=True)
 if state['status']!='success':
  skipped+=1;del state;continue
 model=Model(encoder_name='resnet34',encoder_weights=None,in_channels=12,classes=1,activation=None);model.load_state_dict(state['model'],strict=True)
 optimizer=torch.optim.AdamW([{'params':model.encoder.parameters()},{'params':list(model.decoder.parameters())+list(model.segmentation_head.parameters())}]);optimizer.load_state_dict(state['optimizer'])
 assert state['step']>0 and state['rank_rng'] and state['sampler']
 count+=1;del optimizer,model,state
if not count:raise ValueError('No checkpoint recovered')
(root/'checkpoint-recovery.json').write_text(json.dumps(dict(verified=True,checkpoints=count,diagnostic_checkpoints_skipped=skipped,method='fresh-process strict model and optimizer restoration')))
