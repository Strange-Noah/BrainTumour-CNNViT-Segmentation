"""CPU-only inspection of a verified existing private checkpoint; no GPU submission."""
import json,tempfile
from pathlib import Path

def inspect_state(state):
 import torch
 def describe(name,t):
  t=t.detach().cpu();finite=torch.isfinite(t)
  return dict(name=name,shape=list(t.shape),finite=bool(finite.all()),maximum_abs=float(t[finite].abs().max()) if bool(finite.any()) and t.numel() else 0.)
 model=[describe(k,v) for k,v in state['model'].items() if v.numel()]
 optimizer=[describe(str(k)+'.'+n,v) for k,s in state['optimizer']['state'].items() for n,v in s.items() if torch.is_tensor(v) and v.numel()]
 return dict(step=state['step'],phase=state['phase'],phase1_unique=state['phase1_unique'],consumed_seconds=state['consumed_seconds'],source_sha=state['source_sha'],model_largest=sorted(model,key=lambda x:x['maximum_abs'],reverse=True)[:30],optimizer_largest=sorted(optimizer,key=lambda x:x['maximum_abs'],reverse=True)[:20],nonfinite=[r for r in model+optimizer if not r['finite']],model_tensors=len(model),optimizer_tensors=len(optimizer),cuda_allocated=False)

def main():
 import torch
 from cloud.storage import connect,verify
 from .settings import read_json_record
 from .stage import latest_key
 if torch.cuda.is_available():raise RuntimeError('Diagnostic must run on CPU')
 torch.set_num_threads(2);c,b=connect(output=True)
 from .recovery import LEGACY_SCIENCE,CHECKPOINT_SHA
 pointer=read_json_record(c,b,'novo-nordisk/experiments/unet25d/full-runs/'+LEGACY_SCIENCE+'/latest.json')
 if pointer['checkpoint']['sha256']!=CHECKPOINT_SHA:raise ValueError('Wrong recovery checkpoint')
 with tempfile.TemporaryDirectory(prefix='private-checkpoint-diagnostic-') as d:
  p=Path(d)/'checkpoint.pt'
  with p.open('wb') as f:verify(c,b,pointer['checkpoint']['key'],pointer['checkpoint']['sha256'],f)
  state=torch.load(p,map_location='cpu',weights_only=True);report=inspect_state(state);report['checkpoint_sha256']=pointer['checkpoint']['sha256'];print(json.dumps(report),flush=True)
  replay(state,c,b,Path(d))
def replay(state,c,b,root):
 import torch,hashlib
 from cloud.storage import verify
 from .settings import keybase,read_json_record,science_config
 from .sampling import exhaustive_plan
 from .loading import ShardCache
 from .train_full import CohortModel
 from ..train import Model,loss_fn
 from .recovery import reset_statistics,require_migration,CHECKPOINT_SHA
 require_migration(state,science_config(),CHECKPOINT_SHA)
 ready=read_json_record(c,b,keybase()+'/ready.json');path=root/'catalogue.json'
 with path.open('wb') as f:verify(c,b,ready['manifest']['key'],ready['manifest']['sha256'],f)
 catalogue=json.loads(path.read_text());visits=[v for v in catalogue['visits'] if v['partition']=='train'];by_id={v['metadata']['exam_id']:v for v in visits};plan,sha=exhaustive_plan(visits)
 def fetch(key,expected,dest):
  with Path(dest).open('wb') as f:verify(c,b,key,expected,f)
 cache=ShardCache(root/'cache',fetch)
 results=[]
 for position in [148668,148669,154400]:
  e,t,z,_=plan[position];x,y=cache.sample(by_id[e],t,z);row=dict(position=position,input_max_abs=float(x.abs().max()))
  for policy,cls in [('legacy',Model),('recovered',CohortModel)]:
   model=cls(encoder_name='resnet34',encoder_weights=None,in_channels=12,classes=1,activation=None).float().train();model.load_state_dict(state['model'])
   if policy=='recovered':reset_statistics(model)
   optimizer=torch.optim.AdamW([{'params':model.encoder.parameters(),'lr':1e-4},{'params':list(model.decoder.parameters())+list(model.segmentation_head.parameters()),'lr':3e-4}],weight_decay=1e-4);optimizer.load_state_dict(state['optimizer'])
   # Use independent optimizer tensors for every replay; never mutate the saved state.
   import copy
   optimizer.load_state_dict(copy.deepcopy(state['optimizer']))
   optimizer.zero_grad(set_to_none=True);out=model(x[None]);loss=loss_fn(out,y[None]);loss.backward()
   gradients=all(bool(torch.isfinite(p.grad).all()) for p in model.parameters() if p.grad is not None)
   if policy=='recovered' and (not bool(torch.isfinite(loss)) or not gradients):raise FloatingPointError('CPU recovery forward/backward failed')
   if policy=='recovered':optimizer.step()
   buffers=[v for k,v in model.state_dict().items() if k.endswith('running_var')]
   finite=all(bool(torch.isfinite(v).all()) for v in model.state_dict().values()) and all(bool(torch.isfinite(v).all()) for s in optimizer.state.values() for v in s.values() if torch.is_tensor(v))
   row[policy]=dict(loss=float(loss.detach()),gradients_finite=gradients,state_finite=finite,largest_running_variance=max(float(v.abs().max()) for v in buffers))
   if policy=='recovered' and not finite:raise FloatingPointError('CPU recovered optimizer/state failed')
   del optimizer,model,out,loss
  results.append(row);print(json.dumps(dict(event='cpu_recovery_replay',**row)),flush=True)
 print(json.dumps(dict(event='cpu_recovery_complete',verified=True,real_training_positions=len(results),plan_sha256=sha,cursor_advanced=False,gpu_allocated=False)),flush=True)

if __name__=='__main__':main()
