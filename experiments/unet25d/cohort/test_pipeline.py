import unittest
from collections import Counter
import numpy as np
from .tiling import windows,merge_logits,weights
from .sampling import epoch_plan,resume_rank
class FullInput(unittest.TestCase):
 def test_640_four_windows_and_constant_logit_roundtrip(self):
  shape=(640,640);self.assertEqual(windows(shape),[(0,0),(128,0),(0,128),(128,128)])
  self.assertTrue((weights()>0).all());out=merge_logits([(o,np.full((512,512),-2,np.float32)) for o in windows(shape)],shape);self.assertTrue(np.allclose(out,-2));self.assertFalse((out>=0).any())
 def test_merge_before_threshold_and_all_pixels_covered(self):
  shape=(640,640);tiles=[(o,np.full((512,512),1 if i==0 else -10,np.float32)) for i,o in enumerate(windows(shape))];merged=merge_logits(tiles,shape);self.assertLess(merged[320,320],0)
  with self.assertRaises(ValueError):merge_logits(tiles[:-1],shape)
 def test_native_640_preparation_preserves_labels_and_full_shape(self):
  import tempfile,json
  from pathlib import Path
  import nibabel as nib
  from ..data import prepare_visit
  with tempfile.TemporaryDirectory() as d:
   image=np.ones((640,640,2),np.float32);image[100:200]*=2;raw=np.zeros(image.shape,np.uint8);raw[120:125,120:125]=3;raw[200:205,200:205]=2
   ref=nib.Nifti1Image(image,np.eye(4));ref.header.set_xyzt_units('mm')
   meta=prepare_visit(dict(reference=ref,images=(ref for _ in range(4)),raw_labels=raw,target=raw==3,visit=dict(exam_id='synthetic',patient_id='synthetic')),Path(d),allow_tiles=True)
   self.assertEqual(meta['shape'],[640,640,2]);self.assertTrue(np.array_equal(np.load(Path(d)/'raw-labels.npy'),raw));self.assertTrue(np.array_equal(np.load(Path(d)/'native-raw-labels.npy'),raw))
   import hashlib,shutil
   from unittest.mock import patch
   from .prepare import shard_visit
   from .loading import ShardCache
   store=Path(d)/'store';store.mkdir();objects={}
   def put(client,bucket,key,path):
    sha=hashlib.sha256(path.read_bytes()).hexdigest();dest=store/sha;shutil.copyfile(path,dest);objects[key]=dest;return dict(key=key,sha256=sha,bytes=dest.stat().st_size)
   with patch('experiments.unet25d.cohort.prepare.put_once',side_effect=put):record=shard_visit(Path(d),meta,None,'private','prefix')
   def fetch(key,sha,dest):shutil.copyfile(objects[key],dest)
   cache=ShardCache(Path(d)/'cache',fetch);x,y=cache.sample(record,[0,0],0)
   self.assertEqual(x.shape,(12,512,512));self.assertEqual(y.sum(),25);self.assertTrue(np.array_equal(x[0],x[1]));self.assertLessEqual(cache.decoded_bytes,cache.decoded_limit)

 def test_oblique_native_spacing_full_field_and_label_consistency(self):
  import tempfile
  from pathlib import Path
  import nibabel as nib
  from ..data import prepare_visit,restore
  angle=np.deg2rad(2);affine=np.eye(4);affine[:2,:2]=[[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]]
  source=np.ones((640,640,2),np.float32);source[100:200]*=2;raw=np.zeros(source.shape,np.uint8);raw[120:125,120:125]=3
  reference=nib.Nifti1Image(source,affine);reference.header.set_xyzt_units('mm')
  with tempfile.TemporaryDirectory() as d:
   meta=prepare_visit(dict(reference=reference,images=(reference for _ in range(4)),raw_labels=raw,target=raw==3,visit=dict(exam_id='synthetic',patient_id='synthetic')),Path(d),allow_tiles=True)
   self.assertTrue(meta['source_oblique']);self.assertFalse(meta['oblique_resampled']);self.assertEqual(meta['shape'][0],640)
   self.assertTrue(np.array_equal(np.load(Path(d)/'target.npy'),np.load(Path(d)/'raw-labels.npy')==3))
   grid=np.indices(meta['shape'],dtype=np.float32);a=np.asarray(meta['canonical_affine']);ramp=a[0,0]*grid[0]+a[0,1]*grid[1]+a[0,2]*grid[2]+a[0,3];native=restore(ramp,meta,order=1);coords=np.indices(source.shape,dtype=np.float32);expected=affine[0,0]*coords[0]+affine[0,1]*coords[1]
   self.assertTrue(np.allclose(native,expected,atol=1e-3))
 def test_formal_budget_is_not_authorized_and_rendered_entry_is_valid(self):
  from .kaggle import render,submit_one
  code=render('a'*40,3600,28800);compile(code,'private-kaggle-entry','exec')
  self.assertIn("NOVO_SOURCE_SHA",code);self.assertIn("NOVO_SESSION_STARTED_AT",code)
  with self.assertRaises(PermissionError):submit_one('mathewzhou/novo-unet25d-full-cohort-input','mathewzhou/formal-training')
 def test_private_reader_is_spawn_serializable(self):
  import pickle
  from .stream import SignedFetch
  fetch=pickle.loads(pickle.dumps(SignedFetch(dict(expires_at=0,objects={}))))
  with self.assertRaises(RuntimeError):fetch('object','digest',None)
 def test_validation_summary_returns_on_nonzero_rank_and_restores_training(self):
  import tempfile,torch
  from unittest.mock import patch
  from .validation import validate_patients
  visits=[dict(partition='validation',metadata=dict(patient_id=f'p{i}',exam_id=f'v{i}')) for i in range(45)]
  results=[dict(patient_id=f'p{i}',exam_id=f'v{i}',dice=.5,hd95_mm=1.,lesion_sensitivity=.5,false_positives=0,absolute_volume_error_mm3=0,truth_lesions=2,lesion_strata={s:dict(truth=1,matched=0) for s in ['lt100_mm3','100_to_lt1000_mm3','ge1000_mm3']},loss_sum=1,loss_planes=2) for i in range(45)]
  for r in results:r['threshold_grid']={str(t):dict(dice=.5,truth_lesions=2,matched_lesions=1,lesion_sensitivity=.5,false_positives=0) for t in [.3,.7]}
  model=torch.nn.Linear(1,1).train()
  with tempfile.TemporaryDirectory() as d,patch('torch.distributed.is_initialized',return_value=True),patch('torch.distributed.get_world_size',return_value=2),patch('torch.distributed.get_rank',return_value=1),patch('torch.distributed.all_gather_object',side_effect=lambda out,local:out.__setitem__(slice(None),[[],[]])):
   result=validate_patients(model,dict(visits=visits),None,None,d,'source','checkpoint',5,previous_results=results)
  self.assertTrue(result['complete']);self.assertEqual(result['validation_patients'],45);self.assertTrue(model.training)
 def test_complete_patient_visit_tile_coverage_and_resume(self):
  visits=[dict(partition='train',metadata=dict(patient_id=f'p{i}',exam_id=f'v{i}',tiles=[dict(origin=[0,0],positive=[1] if i<2 else [],negative=[0,2])])) for i in range(4)]
  plan,sha=epoch_plan(visits,0);again,sha2=epoch_plan(visits,0);self.assertEqual(sha,sha2);self.assertEqual(plan,again);self.assertEqual({d[0] for d in plan},{f'v{i}' for i in range(4)});self.assertEqual(Counter(d[3] for d in plan),{True:128,False:128})
  for rank in [0,1]:self.assertEqual(resume_rank(plan,14,rank),plan[14+rank::2])
  visits[0]['partition']='test'
  with self.assertRaises(ValueError):epoch_plan(visits,0)
if __name__=='__main__':unittest.main()
