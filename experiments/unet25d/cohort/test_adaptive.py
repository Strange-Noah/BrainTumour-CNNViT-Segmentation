"""Focused CPU contracts. Optional private catalogue check never emits identifiers."""
import copy,json,os,tempfile,unittest
from collections import Counter
from pathlib import Path
import numpy as np
import torch
from .sampling import exhaustive_plan,adaptive_plan,resume_rank,plan_digest
from .adaptive import Difficulty,initial_control,feedback,commit_update,validation_completed
from .metrics import control_metrics,lesion_strata
from .mining import scores,lesion_properties
from .settings import policy_hash,CONFIG

def visits(n=10):
 return [dict(partition='train',metadata=dict(patient_id=f'p{i}',exam_id=f'e{i}',shape=[512,512,6],tiles=[dict(origin=[0,0],positive=[1,2],negative=[0,3,4,5])])) for i in range(n)]
def metric(d=.2,s=.3,f=4,loss=.7):
 return dict(complete=True,positive_patient_dice=d,patient_lesion_sensitivity=s,patient_fp_per_scan=f,small_lesion_sensitivity={'lt100_mm3':s},validation_loss=loss)
class Adaptive(unittest.TestCase):
 def test_exact_ddp_resume_and_prefetch(self):
  plan,sha=exhaustive_plan(visits());self.assertEqual(len(plan),60)
  a=resume_rank(plan,0,0);b=resume_rank(plan,0,1)
  ids=lambda ds:{(x,tuple(t),z) for x,t,z,_ in ds}
  self.assertFalse(ids(a)&ids(b));self.assertEqual(ids(a)|ids(b),ids(plan))
  cursor=8;prefetched=plan[cursor:cursor+8]
  self.assertEqual(resume_rank(plan,cursor,0),a[4:]);self.assertEqual(len(ids(plan[:cursor])),cursor)
  self.assertEqual(exhaustive_plan(visits())[1],sha)
  # Cursor is committed only after both ranks complete one actual optimizer update.
  model=torch.nn.Linear(1,1);opt=torch.optim.AdamW(model.parameters());consumed=0
  state=dict(global_cursor=0,step=0,phase=1,phase1_unique=0,phase1_updates=0,phase2_consumed=0,category_consumed={})
  for _ in range(4):opt.zero_grad();model(torch.ones(2,1)).sum().backward();opt.step();commit_update(state,plan);consumed+=2
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'c.pt';torch.save(dict(cursor=consumed,phase=1,unique=consumed,plan_sha=sha,optimizer=opt.state_dict()),p)
   r=torch.load(p,weights_only=True);self.assertEqual(r['cursor'],cursor);self.assertEqual(r['plan_sha'],sha)
  self.assertEqual(consumed,len(ids(plan[:cursor])));self.assertEqual(len(prefetched),8);self.assertEqual(state['phase1_unique'],consumed)
  while state['global_cursor']<len(plan):commit_update(state,plan)
  state.update(pending_validation=True,validation_progress=[],phase2_cycle=0,active_plan=plan)
  saved=json.loads(json.dumps(state));validation_completed(saved);self.assertEqual(saved['phase'],2);self.assertEqual(saved['phase1_unique'],60);self.assertEqual(saved['global_cursor'],0)
  saved['active_plan']=plan;commit_update(saved,plan);self.assertEqual(saved['phase2_consumed'],2);self.assertEqual(saved['phase1_unique'],60)
 def test_phase2_accounting_caps_resume_and_train_only(self):
  vs=visits();pool=Difficulty(vs);c=initial_control()
  plan,sha,report=adaptive_plan(vs,0,c['ratios'],pool,0,total=1000)
  self.assertEqual(Counter(d[3] for d in plan),dict(baseline=400,hard_positive=250,hard_negative=200,ordinary_negative=150))
  self.assertLessEqual(report['max_patient'],report['patient_cap']);self.assertLessEqual(report['max_visit'],report['visit_cap'])
  self.assertEqual(plan_digest(json.loads(json.dumps(plan))),sha)
  self.assertEqual(adaptive_plan(vs,0,c['ratios'],pool,0,total=1000)[1],sha)
  with self.assertRaises(ValueError):Difficulty([dict(vs[0],partition='validation')])
  with self.assertRaises(ValueError):pool.update([dict(exam='foreign',step=1,rank=0,z=0,tile=[0,0],kind='positive',score=1)],1)
 def test_bounded_decayed_mining_and_peak_negatives(self):
  pool=Difficulty(visits());rows=[dict(exam='e0',tile=[0,0],z=i,kind='negative',score=float(i),step=1,rank=0) for i in range(100)]
  pool.update(rows,1);self.assertEqual(len(pool.records),32);pool.update([],200002);self.assertEqual(pool.records,{})
  y=torch.zeros(1,1,16,16);logits=torch.full_like(y,-20);logits[0,0,0,0]=20
  result=scores(logits,y,dict(exam='e0',tile=[0,0],z=0,min_volume_mm3=0),1,0)
  self.assertGreater(result[0]['score'],.5);self.assertEqual(len(result),1)
 def test_positive_dice_excludes_mutual_empty_and_size_is_3d(self):
  affine=np.eye(4);truth=np.zeros((8,8,8),bool);truth[1:3,1:3,1:3]=True
  bins=lesion_strata(truth,truth,affine);self.assertEqual(bins['lt100_mm3'],dict(truth=1,matched=1))
  empty={s:dict(truth=0,matched=0) for s in bins}
  rows=[dict(patient_id='a',dice=.2,truth_lesions=1,lesion_sensitivity=1,false_positives=2,lesion_strata=bins,loss_sum=2,loss_planes=2),dict(patient_id='a',dice=1.,truth_lesions=0,lesion_sensitivity=None,false_positives=0,lesion_strata=empty,loss_sum=2,loss_planes=2),dict(patient_id='b',dice=1.,truth_lesions=0,lesion_sensitivity=None,false_positives=0,lesion_strata=empty,loss_sum=2,loss_planes=2)]
  r=control_metrics(rows,True);self.assertEqual(r['positive_patient_dice'],.2);self.assertEqual(r['positive_dice_defined_patients'],1);self.assertEqual(r['patient_fp_per_scan'],.5)
  self.assertFalse(control_metrics(rows,False)['complete'])
 def test_controller_phase_transition_target_plateau_and_restore(self):
  c=initial_control()
  with self.assertRaises(ValueError):feedback(c,metric(),phase2=False,exhaustive_complete=False)
  with self.assertRaises(ValueError):feedback(c,dict(metric(),complete=False),phase2=True,exhaustive_complete=True)
  c=feedback(c,metric(),phase2=False,exhaustive_complete=True)
  for _ in range(3):c=feedback(json.loads(json.dumps(c)),metric(),phase2=True,exhaustive_complete=True)
  self.assertEqual(c['scientific_stop'],'PLATEAU_BELOW_TARGET');self.assertGreaterEqual(c['ratios']['baseline'],.3)
  c=feedback(initial_control(),metric(.7,.8,2),phase2=False,exhaustive_complete=True);self.assertIsNone(c['scientific_stop'])
  c=feedback(c,metric(.7,.8,2),phase2=True,exhaustive_complete=True);self.assertEqual(c['scientific_stop'],'TARGET_REACHED')
  c=feedback(initial_control(),metric(.3,.5,1),phase2=False,exhaustive_complete=True);self.assertAlmostEqual(c['ratios']['hard_positive'],.35)
  reverted=feedback(c,metric(.2,.4,3,.6),phase2=True,exhaustive_complete=True);self.assertEqual(reverted['ratios'],initial_control()['ratios'])
 def test_prepared_policy_and_science_baseline_unchanged(self):
  self.assertEqual(policy_hash(),'3bbb16ca8e4659d70c3dc3e5503150e2f10fa7be9da0d7291865484719d7318a')
  c=json.loads(CONFIG.read_text());self.assertEqual(c['loss'],'0.5 BCE + 0.5 soft Dice');self.assertEqual(c['precision'],'float32');self.assertFalse(c['augmentation']);self.assertEqual(c['global_batch'],2)
 def test_3d_gt_properties_not_2d_size(self):
  v=visits(1)[0];m=v['metadata'];m.update(shape=[512,512,6],native_shape=[8,8,6],native_affine=np.diag([2,2,2,1]).tolist(),orientation_transform=[[0,1],[1,1],[2,1]],padding=[[252,252],[252,252],[0,0]])
  m['tiles'][0]['positive']=[1,2];v['native_labels']=dict(key='synthetic',sha256='synthetic')
  import gzip
  def fetch(k,s,p):
   raw=np.zeros((8,8,6),np.uint8);raw[2:4,2:4,1:3]=3
   with gzip.open(p,'wb') as f:np.save(f,raw)
  with tempfile.TemporaryDirectory() as d:
   props=lesion_properties(v,fetch,d);self.assertAlmostEqual(props[str((0,0,1))],64);self.assertAlmostEqual(props[str((0,0,2))],64)
 def test_optional_frozen_inventory(self):
  path=os.environ.get('NOVO_PRIVATE_CATALOGUE')
  if not path:self.skipTest('metadata-only private catalogue not supplied')
  catalogue=json.loads(Path(path).read_text());vs=[v for v in catalogue['visits'] if v['partition']=='train'];plan,sha=exhaustive_plan(vs)
  self.assertEqual(len(plan),220384);self.assertEqual(len(vs),746);self.assertEqual(len(plan)//2,110192)
  for rank in [0,1]:self.assertEqual(len(resume_rank(plan,0,rank)),110192)
  self.assertEqual(len({e for e,t,z,c in plan}),746)
  print(json.dumps(dict(exhaustive_positions=len(plan),optimizer_updates=len(plan)//2,plan_sha256=sha,train_visits=len(vs))))
 def test_free_quota_guard_and_collection_compatibility(self):
  from unittest.mock import patch
  from . import unattended as u
  from .stage import persist_training_state
  from .settings import science_hash
  from unittest.mock import Mock
  with patch('cloud.kaggle_benchmark_bridge.cli',return_value='warning\n[{"resource":"GPU","remaining":"29.22h","used":"0.78h","total":"30.00h","refreshAt":"fixed"}]'):
   q=u.free_quota();self.assertEqual(u.free_seconds(q),83433);self.assertLess(u.free_seconds(q,2),u.free_seconds(q))
  self.assertEqual(u.free_seconds(dict(remaining_seconds=899)),0)
  class Missing(Exception):pass
  client=Mock();client.exceptions.NoSuchKey=Missing;client.get_object.side_effect=Missing
  state=dict(science_sha256=science_hash(),data_mode='adaptive-two-phase-v1',step=4,phase=1,phase1_unique=8,scientific_stop=None)
  with tempfile.TemporaryDirectory() as d:
   (Path(d)/'training-state.json').write_text(json.dumps(state))
   receipts=[dict(key='private/novo-unet25d-full/'+name+'.pt',sha256='synthetic') for name in ['checkpoint','best-positive-dice','best-tradeoff']]
   persist_training_state(d,client,'private',receipts)
  saved=json.loads(client.put_object.call_args.kwargs['Body']);self.assertEqual(saved['state']['phase1_unique'],8);self.assertIn('best-tradeoff',saved)
if __name__=='__main__':unittest.main()
