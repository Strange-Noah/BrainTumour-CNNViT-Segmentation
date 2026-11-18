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
if __name__=='__main__':unittest.main()
