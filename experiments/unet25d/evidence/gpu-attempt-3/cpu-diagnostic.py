import io,json,os,subprocess,sys,urllib.request,zipfile
from pathlib import Path
sha='5f0eee17ba10927fc267903e2a6ba1cd23499fc6';root=Path('/tmp/project');root.mkdir()
with urllib.request.urlopen('https://github.com/Strange-Noah/BrainTumour-CNNViT-Segmentation/archive/'+sha+'.zip',timeout=60) as r:z=zipfile.ZipFile(io.BytesIO(r.read()))
for entry in z.infolist():
 p=root/Path(*Path(entry.filename).parts[1:])
 if entry.is_dir():p.mkdir(parents=True,exist_ok=True)
 else:p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(z.read(entry))
subprocess.run([sys.executable,'-m','pip','install','-q','-r',str(root/'experiments/unet25d/requirements.txt')],check=True)
sys.path.insert(0,str(root))
import torch
from experiments.unet25d.data import sample_draws,SliceCache
from experiments.unet25d.train import Model,loss_fn
torch.set_num_threads(2)
cache=next(p.parent for p in Path('/kaggle/input').rglob('cache_integrity.stage.json') if (p.parent/'prepared/train-index.json').exists())
checkpoint=next(Path('/kaggle/input').rglob('checkpoint.pt'))
import hashlib
with checkpoint.open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()=='dafc256caccd3cdeba54210ccb6bfe33166cde4f57e56daf4a3c6b61a340f0b1'
s=torch.load(checkpoint,map_location='cpu',weights_only=True);records=json.loads((cache/'prepared/train-index.json').read_text());draw=sample_draws(records,836,42)[-1];x,y,_=SliceCache(cache/'prepared',[draw])[0];results={}
for precision in ['current_fp16_with_island','encoder_and_first_decoder_fp32']:
 model=Model(encoder_name='resnet34',encoder_weights=None,in_channels=12,classes=1,activation=None);model.load_state_dict(s['model']);model.train()
 for m in model.decoder.modules():
  if isinstance(m,torch.nn.BatchNorm2d):m.reset_running_stats()
 if precision=='encoder_and_first_decoder_fp32':
  original_encoder=model.encoder.forward
  def stable_encoder(value,original_encoder=original_encoder):
   with torch.autocast('cpu',enabled=False):return original_encoder(value.float())
  model.encoder.forward=stable_encoder
 stages=[]
 def capture(name):
  def hook(module,args,out):
   stages.append(dict(module=name,finite=bool(torch.isfinite(out).all()),max_abs=float(out.detach().abs().max()),dtype=str(out.dtype)))
  return hook
 for name,m in model.named_modules():
  if isinstance(m,(torch.nn.Conv2d,torch.nn.BatchNorm2d)) or type(m).__name__=='BasicBlock':m.register_forward_hook(capture(name))
 with torch.autocast('cpu',dtype=torch.float16,enabled=precision!='full_float32'):
  output=model(x[None]);loss=loss_fn(output,y[None])
 gradient_finite=None
 if torch.isfinite(loss):
  optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4);scaler=torch.amp.GradScaler('cpu',init_scale=128);scaler.scale(loss).backward();scaler.unscale_(optimizer)
  gradient_finite=all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in model.parameters())
  del optimizer,scaler
 results[precision]=dict(gradient_finite=gradient_finite,loss_finite=bool(torch.isfinite(loss)),loss=float(loss) if torch.isfinite(loss) else None,first_nonfinite=next((r for r in stages if not r['finite']),None),max_finite_activation=max(r['max_abs'] for r in stages if r['finite']),stages=stages)
 Path('/kaggle/working/precision-partial.json').write_text(json.dumps(results,indent=2))
 del model
r=dict(source_sha=sha,draw_position=835,input_finite=bool(torch.isfinite(x).all()),input_max_abs=float(x.abs().max()),target_voxels=int(y.sum()),results=results,gpu_allocated=False,optimizer_updates=0)
Path('/kaggle/working/forward-diagnostic.json').write_text(json.dumps(r,indent=2));print(json.dumps({k:v for k,v in r.items() if k!='results'}));print({k:{n:v for n,v in value.items() if n!='stages'} for k,value in results.items()})
