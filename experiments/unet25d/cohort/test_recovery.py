import copy,unittest,torch
from .settings import science_config,policy_hash
from .recovery import legacy_config,require_migration,reset_statistics,CHECKPOINT_SHA
from .train_full import CohortModel
class Recovery(unittest.TestCase):
 def test_exact_migration_preserves_identity_and_rejects_wrong_checkpoint(self):
  c=science_config();old=dict(config=legacy_config(c),step=74334,phase1_unique=148668,source_sha='e818635a9e7df4d91792afaad7fc9ff3dd4b2a23')
  require_migration(old,c,CHECKPOINT_SHA)
  for bad in [dict(old,step=74335),dict(old,phase1_unique=148666),dict(old,config=dict(old['config'],encoder_lr=.1))]:
   with self.assertRaises(ValueError):require_migration(bad,c,CHECKPOINT_SHA)
  with self.assertRaises(ValueError):require_migration(old,c,'wrong')
  self.assertEqual(policy_hash(),'3bbb16ca8e4659d70c3dc3e5503150e2f10fa7be9da0d7291865484719d7318a')
 def test_train_eval_and_statistics_reset_preserve_learned_tensors(self):
  torch.set_num_threads(2);model=CohortModel(encoder_name='resnet34',encoder_weights=None,in_channels=12,classes=1,activation=None)
  model.train();self.assertTrue(all(m.training for m in model.encoder.modules() if isinstance(m,torch.nn.BatchNorm2d)))
  model.eval();self.assertFalse(any(m.training for m in model.modules() if isinstance(m,torch.nn.BatchNorm2d)))
  model.train();self.assertTrue(all(m.training for m in model.encoder.modules() if isinstance(m,torch.nn.BatchNorm2d)))
  original={k:v.clone() for k,v in model.named_parameters()};rng=torch.get_rng_state().clone()
  for m in model.modules():
   if isinstance(m,torch.nn.BatchNorm2d):m.running_var.fill_(3e34)
  reset_statistics(model);self.assertTrue(torch.equal(rng,torch.get_rng_state()))
  self.assertTrue(all(torch.equal(original[k],v) for k,v in model.named_parameters()))
  x=torch.randn(1,12,64,64);y=model(x);y.square().mean().backward()
  self.assertTrue(torch.isfinite(y).all());self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
  self.assertLess(max(float(m.running_var.max()) for m in model.modules() if isinstance(m,torch.nn.BatchNorm2d)),100)
if __name__=='__main__':unittest.main()
