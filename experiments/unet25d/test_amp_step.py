"""Legacy filename; the authorized baseline now uses only full FP32."""
import unittest
import torch
from .train import fp32_step,Model,RUNTIME_PRECISION,optimizer_matches
class FP32Baseline(unittest.TestCase):
 def test_finite_updates_and_nonfinite_guards(self):
  p=torch.nn.Parameter(torch.tensor(1.));optimizer=torch.optim.AdamW([p],lr=.01)
  for step in range(1,4):
   optimizer.zero_grad();p.square().backward();fp32_step(optimizer,True,bool(torch.isfinite(p.grad)));self.assertEqual(float(optimizer.state[p]['step']),step)
  before=p.detach().clone()
  for loss,grad in [(False,True),(True,False)]:
   with self.assertRaises(FloatingPointError):fp32_step(optimizer,loss,grad)
   self.assertTrue(torch.equal(before,p))
 def test_optimizer_restore_checks_moments_not_only_step(self):
  import copy
  p=torch.nn.Parameter(torch.tensor(1.));opt=torch.optim.AdamW([p]);p.square().backward();opt.step();saved=copy.deepcopy(opt.state_dict())
  q=torch.nn.Parameter(torch.tensor(1.));restored=torch.optim.AdamW([q]);restored.load_state_dict(copy.deepcopy(saved))
  self.assertTrue(optimizer_matches(restored,saved));restored.state[q]['exp_avg'].add_(1)
  self.assertFalse(optimizer_matches(restored,saved))
 def test_fp32_network_and_state_contract(self):
  import segmentation_models_pytorch as smp
  torch.set_num_threads(2);kwargs=dict(encoder_name='resnet34',encoder_weights=None,in_channels=12,classes=1,activation=None)
  base=smp.Unet(**kwargs);model=Model(**kwargs).train();model.load_state_dict(base.state_dict(),strict=True)
  self.assertEqual([(k,tuple(v.shape)) for k,v in base.state_dict().items()],[(k,tuple(v.shape)) for k,v in model.state_dict().items()])
  seen=[]
  for m in model.modules():
   if isinstance(m,(torch.nn.Conv2d,torch.nn.BatchNorm2d)):m.register_forward_hook(lambda m,args,out:seen.append(out.dtype))
  out=model(torch.randn(1,12,64,64));out.square().mean().backward()
  self.assertEqual(set(seen),{torch.float32});self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()));self.assertFalse(RUNTIME_PRECISION['autocast'])
if __name__=='__main__':unittest.main()
