import unittest
import torch
from .train import amp_step
class AMPRecovery(unittest.TestCase):
 def test_real_scaler_skips_then_recovers(self):
  p=torch.nn.Parameter(torch.tensor(1.));opt=torch.optim.SGD([p],lr=.1);scaler=torch.amp.GradScaler('cpu',init_scale=128)
  scaler.scale(p).backward();p.grad.fill_(float('inf'));scaler.unscale_(opt)
  skips,updated=amp_step(scaler,opt,True,False,0)
  self.assertFalse(updated);self.assertEqual(p.item(),1.);self.assertEqual(scaler.get_scale(),64.)
  scaler.scale(p).backward();scaler.unscale_(opt);skips,updated=amp_step(scaler,opt,True,True,skips)
  self.assertTrue(updated);self.assertAlmostEqual(p.item(),.9,places=5);self.assertEqual(skips,1)
 def test_fatal_cases_remain_fatal(self):
  with self.assertRaises(FloatingPointError):amp_step(None,None,False,False,0)
  with self.assertRaises(FloatingPointError):amp_step(None,None,True,False,3)

class OptimizerCounter(unittest.TestCase):
 def test_adamw_counter_matches_valid_updates_only(self):
  p=torch.nn.Parameter(torch.tensor(1.));optimizer=torch.optim.AdamW([p],lr=.01);scaler=torch.amp.GradScaler('cpu',init_scale=128);counter=0;skips=0
  for finite in [True,False,True]:
   optimizer.zero_grad(set_to_none=True);scaler.scale(p).backward()
   if not finite:p.grad.fill_(float('inf'))
   scaler.unscale_(optimizer);skips,updated=amp_step(scaler,optimizer,True,finite,skips);counter+=int(updated)
   self.assertEqual(float(optimizer.state[p]['step']),counter)
  self.assertEqual(counter,2)

class PrecisionIsland(unittest.TestCase):
 def test_state_contract_and_autocast_boundary(self):
  import segmentation_models_pytorch as smp
  from .train import Model
  torch.set_num_threads(2)
  kwargs=dict(encoder_name='resnet34',encoder_weights=None,in_channels=12,classes=1,activation=None)
  base=smp.Unet(**kwargs);model=Model(**kwargs).train();model.load_state_dict(base.state_dict(),strict=True)
  self.assertEqual([(k,tuple(v.shape)) for k,v in base.state_dict().items()],[(k,tuple(v.shape)) for k,v in model.state_dict().items()])
  self.assertEqual(sum(p.numel() for p in base.parameters()),sum(p.numel() for p in model.parameters()))
  seen={}
  def hook(name):
   def record(module,args,out):seen[name]=out.dtype
   return record
  model.encoder.conv1.register_forward_hook(hook('encoder_conv'))
  model.encoder.layer4[2].register_forward_hook(hook('residual'))
  model.decoder.blocks[0].conv1.register_forward_hook(hook('island'))
  model.decoder.blocks[0].conv2.register_forward_hook(hook('next'))
  with torch.autocast('cpu',dtype=torch.float16):out=model(torch.randn(1,12,64,64))
  self.assertTrue(torch.isfinite(out).all());self.assertEqual(seen,{'encoder_conv':torch.float32,'residual':torch.float32,'island':torch.float32,'next':torch.float16})

if __name__=='__main__':unittest.main()
