"""CPU-only control/data contracts; never submit an actual notebook."""
import copy,json,os,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from . import unattended as u
from .settings import science_hash,CONFIG,policy_hash
class Unattended(unittest.TestCase):
 def record(self):return dict(status='running',science_sha256=science_hash(),approved_budget_seconds=86400,reserved_seconds=43200,session_no=1,kernel='mathewzhou/synthetic',catalogue_sha256='sha',submitted_at=0,collections=[])
 def test_provider_budget_cap_and_no_unauthorized_launch(self):
  self.assertEqual(u.session_budget(86400,0),43200);self.assertEqual(u.session_budget(28800,0),28800);self.assertEqual(u.session_budget(28800,28800),0)
  with patch.dict(os.environ,{'FORMAL_TRAINING_AUTHORIZED':'false','APPROVED_GPU_BUDGET_SECONDS':'0'}),patch.object(u,'connect') as c:
   with self.assertRaises(PermissionError):u.launch(u.DATASET,'mathewzhou/synthetic')
   c.assert_not_called()
 def test_no_active_record_cannot_submit_gpu(self):
  with patch.object(u,'connect',return_value=(None,None)),patch.object(u,'read',return_value=None),patch('cloud.kaggle_benchmark_bridge.cli') as cli,patch.object(u,'start_session') as start:u.recover()
  cli.assert_not_called();start.assert_not_called()
 def test_terminal_recovery_is_idempotent(self):
  for status in ['training_complete','budget_complete','failed','blocked']:
   with patch.object(u,'read',return_value=dict(status=status)),patch('cloud.kaggle_benchmark_bridge.collect') as collect:u.recover(object(),'private')
   collect.assert_not_called()
 def test_completed_session_collects_before_resuming(self):
  record=self.record();latest=dict(state=dict(catalogue_sha256='sha',epoch=5,pending_validation=False),checkpoint=dict(key='private/outputs/checkpoint.pt'));events=[]
  with patch('cloud.kaggle_benchmark_bridge.collect',side_effect=lambda *a:events.append('collect') or dict(collection=dict(key='private/collection.json'))),patch.object(u,'read',return_value=latest),patch.object(u,'start_session',side_effect=lambda *a:events.append('resume')):
   u.recover(object(),'private',record,'COMPLETE')
  self.assertEqual(events,['collect','resume'])
 def test_final_validation_and_error_stop_further_gpu(self):
  for status,epoch,pending,terminal in [('COMPLETE',30,False,'training_complete'),('ERROR',5,False,'failed')]:
   record=self.record();latest=dict(state=dict(catalogue_sha256='sha',epoch=epoch,pending_validation=pending),checkpoint=dict(key='private/outputs/checkpoint.pt'))
   with patch('cloud.kaggle_benchmark_bridge.collect',return_value=dict(collection=dict(key='private/collection.json'))),patch.object(u,'read',return_value=latest),patch.object(u,'write'),patch.object(u,'start_session') as start:
    if status=='ERROR':
     with self.assertRaises(RuntimeError):u.recover(object(),'private',record,status)
    else:u.recover(object(),'private',record,status)
   self.assertEqual(record['status'],terminal);start.assert_not_called()
 def test_two_collection_failures_block_instead_of_blind_retries(self):
  record=self.record()
  with patch('cloud.kaggle_benchmark_bridge.collect',side_effect=OSError),patch.object(u,'write'),patch.object(u,'start_session') as start:
   for _ in range(2):
    with self.assertRaises(OSError):u.recover(object(),'private',record,'COMPLETE')
  self.assertEqual(record['status'],'blocked');start.assert_not_called()
 def test_long_session_render_and_frozen_science(self):
  from .kaggle import render
  code=render('a'*40,43200,86400);compile(code,'long-formal-session','exec');self.assertIn('43200',code)
  c=json.loads(CONFIG.read_text());self.assertFalse(c['formal_training_authorized']);self.assertEqual(c['approved_budget_seconds'],0);self.assertEqual(c['precision'],'float32')
  self.assertEqual(c['validation_every_coverage_epochs'],5);self.assertEqual(c['checkpoint_every_minutes'],10)
 def test_collection_control_plane_needs_no_torch_import(self):
  import subprocess,sys
  result=subprocess.run([sys.executable,'-c',"import sys; from experiments.unet25d.cohort.stage import latest_key; from experiments.unet25d.cohort.unattended import recover; assert 'torch' not in sys.modules; print(latest_key())"],capture_output=True,text=True)
  self.assertEqual(result.returncode,0,result.stderr)
 def test_full_session_uses_pinned_private_version_and_reserves_budget_once(self):
  record=dict(status='authorized',approved_budget_seconds=86400,reserved_seconds=0,session_no=0,kernel_base='mathewzhou/novo-full',campaign_id='123',source_sha='a'*40)
  def cli(*args):return 'READY' if args[:2]==('datasets','status') else 'Kernel version 1 successfully pushed'
  with tempfile.TemporaryDirectory() as d,patch('cloud.kaggle_benchmark_bridge.WORK',Path(d)),patch('cloud.kaggle_benchmark_bridge.cli',side_effect=cli) as call,patch('experiments.unet25d.cohort.stage.publish_from_existing_runtime',return_value=dict(catalogue_sha256='sha',version=3)),patch.object(u,'write'),patch.object(u,'require_integrated_source'):
   u.start_session(None,None,record)
   meta=json.loads((Path(d)/'full-kernel/kernel-metadata.json').read_text())
  self.assertEqual(record['reserved_seconds'],43200);self.assertEqual(record['session_seconds'],43200);self.assertEqual(meta['dataset_sources'],[u.DATASET+'/3']);self.assertTrue(meta['is_private']);self.assertEqual(record['status'],'running')
  self.assertIn(unittest.mock.call('kernels','push','-p',unittest.mock.ANY,'-t','43200','--accelerator','NvidiaTeslaT4'),call.call_args_list)
 def test_receipt_catalogue_completeness_and_geometry(self):
  import hashlib,numpy as np
  from .accept_input import coverage
  from .settings import keybase
  records=[];expected=[]
  for i in range(905):
   partition='train' if i<746 else 'validation';exam='synthetic-'+str(i);patient='synthetic-'+partition+'-'+str(i%(210 if partition=='train' else 45));wh=640 if i<4 else 512;affine=np.eye(4)
   if i<51:
    angle=.01;affine[:2,:2]=[[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]]
   from .tiling import windows
   meta=dict(exam_id=exam,patient_id=patient,shape=[wh,wh,2],native_shape=[wh,wh,2],native_affine=affine.tolist(),canonical_affine=affine.tolist(),orientation_transform=[[0,1],[1,1],[2,1]],padding=[[0,0]]*3,normalization=[{}]*4,grid_policy='native-T1c-with-affine',oblique_resampled=False,source_oblique=i<51,tiles=[dict(origin=list(o),positive=[1],negative=[0]) for o in windows([wh,wh,2])])
   def obj(name):
    sha=hashlib.sha256((exam+name).encode()).hexdigest();return dict(key=keybase()+'/'+sha+name,sha256=sha,bytes=123)
   shard=dict(**obj('.tar.gz'),start=0,end=2,halo_start=0,decoded_bytes=2*4*wh*wh*4+2*wh*wh)
   records.append(dict(partition=partition,metadata=meta,shards=[shard],native_labels=obj('.npy.gz')));expected.append(dict(exam_id=exam,patient_id=patient))
  result=coverage(records,expected);self.assertEqual(result['shard_count'],905);self.assertEqual(result['oblique_visits'],51)
  records[0]['metadata']['tiles'].pop()
  with self.assertRaises(ValueError):coverage(records,expected)
  records[0]=records[1]
  with self.assertRaises(ValueError):coverage(records,expected)
 def test_new_private_publication_visibility_is_polled_without_reupload(self):
  from .stage import wait_private_ready
  with patch('experiments.unet25d.cohort.stage.time.monotonic',side_effect=[0,1,2,3]),patch('experiments.unet25d.cohort.stage.time.sleep') as sleep:
   cli=unittest.mock.Mock(side_effect=[RuntimeError('temporary API visibility'), 'processing','ready']);wait_private_ready('private/input',cli)
  self.assertEqual(cli.call_args_list,[unittest.mock.call('datasets','status','private/input')]*3);self.assertEqual(sleep.call_count,2)
  with patch('experiments.unet25d.cohort.stage.time.monotonic',side_effect=[0,1,901]),patch('experiments.unet25d.cohort.stage.time.sleep'):
   with self.assertRaises(TimeoutError):wait_private_ready('private/input',unittest.mock.Mock(return_value='processing'))
if __name__=='__main__':unittest.main()
