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
 torch.set_num_threads(2);c,b=connect(output=True);pointer=read_json_record(c,b,latest_key())
 with tempfile.TemporaryDirectory(prefix='private-checkpoint-diagnostic-') as d:
  p=Path(d)/'checkpoint.pt'
  with p.open('wb') as f:verify(c,b,pointer['checkpoint']['key'],pointer['checkpoint']['sha256'],f)
  state=torch.load(p,map_location='cpu',weights_only=True);report=inspect_state(state);report['checkpoint_sha256']=pointer['checkpoint']['sha256'];print(json.dumps(report),flush=True)
if __name__=='__main__':main()
