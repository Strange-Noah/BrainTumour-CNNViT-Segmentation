"""CPU correctness checks for geometry, gradients, identities and full-plan policy."""
import copy,json,tempfile,unittest
from pathlib import Path
import numpy as np
import torch
from .settings import config
from .geometry import fov,effective,crop,crop_origin
from .loss import balanced_loss
from .inventory import build
from .sampling import build_plan,digest,commit_update

def fixture(n=16):
    affine=np.eye(4);m=dict(exam_id='synthetic',patient_id='synthetic',shape=[n,n,3],native_shape=[n,n,3],native_affine=affine.tolist(),canonical_affine=affine.tolist(),orientation_transform=[[0,1],[1,1],[2,1]],padding=[[0,0]]*3,tiles=[])
    return m

class Correctness(unittest.TestCase):
    def test_fov_zeros_and_padding(self):
        m=fixture();m['native_shape']=[12,12,3];m['padding']=[[2,2],[2,2],[0,0]];m['canonical_affine'][0][3]=-2;m['canonical_affine'][1][3]=-2
        valid=fov(m,1);self.assertEqual(int(valid.sum()),144)
        roi,_,reason=effective(np.zeros((12,16,16)),m,1,config());self.assertTrue(np.array_equal(roi,valid));self.assertIsNotNone(reason)
    def test_oblique_affine_fov(self):
        m=fixture();a=np.eye(4);a[:2,:2]=[[.8,-.6],[.6,.8]];m['native_affine']=a.tolist();m['canonical_affine']=a.tolist();self.assertEqual(int(fov(m,1).sum()),256)
    def test_GT_safety_fallback(self):
        m=fixture(64);x=np.zeros((12,64,64));x[:,20:40,20:40]=1;y=np.zeros((64,64),bool);y[1,1]=1
        c=config();c['support_margin_mm']=1;c['support_min_fraction']=.01
        roi,valid,reason=effective(x,m,1,c,y);self.assertTrue(roi[1,1]);self.assertEqual(reason,'training_GT_preservation_fallback')
    def test_loss_positive_and_negative_gradients(self):
        l=torch.zeros((1,1,8,8),requires_grad=True);y=torch.zeros_like(l);y[:,:,3,3]=1;m=torch.zeros_like(y);m[:,:,1:7,1:7]=1
        loss,parts=balanced_loss(l,y,m,return_parts=True);loss.backward()
        self.assertTrue(torch.all(l.grad[~m.bool()]==0));self.assertLess(float(l.grad[0,0,3,3]),0);self.assertGreater(float(l.grad[0,0,2,2]),0)
        self.assertTrue(torch.allclose(loss,.5*parts['bce']+.5*parts['soft_dice']))
    def test_tiny_not_diluted(self):
        grads=[]
        for n in (16,512):
            l=torch.zeros((1,1,n,n),requires_grad=True);y=torch.zeros_like(l);y[:,:,0,0]=1;balanced_loss(l,y,torch.ones_like(y)).backward();grads.append(abs(float(l.grad[0,0,0,0])))
        self.assertGreater(grads[1],.12);self.assertLess(abs(grads[0]-grads[1]),.03)
    def test_empty_valid_learning(self):
        l=torch.zeros((1,1,16,16),requires_grad=True);y=torch.zeros_like(l);loss=balanced_loss(l,y,torch.ones_like(y));loss.backward();self.assertTrue(torch.all(l.grad>0));self.assertTrue(torch.isfinite(loss))
    def test_invalid_supervision_rejected(self):
        with self.assertRaises(ValueError):balanced_loss(torch.zeros(1,1,2,2),torch.ones(1,1,2,2),torch.zeros(1,1,2,2))
    def test_crop_alignment_all_scales(self):
        image=np.zeros((12,640,640),np.float32);image[:,0,0]=7;label=np.zeros((1,640,640),np.float32);label[:,0,0]=1
        for size in (128,192,256,512):
            x=crop(image,[-size//2,-size//2],size);y=crop(label,[-size//2,-size//2],size);self.assertEqual(x.shape,(12,size,size));self.assertEqual(float(x[0][y[0].astype(bool)][0]),7)
            from torch.nn.functional import interpolate
            out=interpolate(torch.from_numpy(y)[None],(512,512),mode='nearest');self.assertTrue(bool(out.any()));self.assertEqual(tuple(out.shape),(1,1,512,512))
    def test_component_volume_and_identity(self):
        m=fixture();m['tiles']=[dict(origin=[0,0],positive=[0,1],negative=[2])];raw=np.zeros((16,16,3),np.uint8);raw[0,0,0]=3;raw[10:12,10:12,1]=3;raw[4,4,2]=2
        ids,rows=build(raw,m);self.assertEqual(len(rows),2);self.assertEqual(sorted(r['volume_mm3'] for r in rows),[1.,4.]);self.assertEqual(ids[4,4,2],0)
        origin,size,partial=crop_origin(rows[0],0,128,.125,m['shape'],42);self.assertFalse(partial);self.assertEqual(int(crop(ids[:,:,0],origin,size).max()),rows[0]['id'])
    def test_large_lesion_escalation(self):
        r={'planes':{'0':dict(lo=[0,0],hi=[200,200],centroid=[100,100])}};_,size,_=crop_origin(r,0,128,.125,[512,512,1],42);self.assertEqual(size,256)
    def test_committed_coverage_not_prefetch(self):
        s=dict(cursor=0,step=0,mandatory_unique=0,replay_consumed=0,positive_samples=0,negative_samples=0);pair=[dict(kind='mandatory',positive=False),dict(kind='replay',positive=True)];untouched=copy.deepcopy(s);self.assertEqual(s,untouched);commit_update(s,pair);self.assertEqual(s['mandatory_unique'],1);self.assertEqual(s['cursor'],2)
    def test_complete_interleaving_synthetic(self):
        c=config();c.update(mandatory_positions=1200,replay_patient_cap=128,replay_visit_cap=128,replay_lesion_cap=128)
        visits=[];inventory={}
        for i in range(12):
            e=str(i);m=dict(exam_id=e,patient_id=e,shape=[512,512,100],positive=list(range(20)),tiles=[dict(origin=[0,0],positive=list(range(20)),negative=list(range(20,100)))])
            visits.append(dict(metadata=m,partition='train'));inventory[e]=dict(patient=e,components=[dict(id=1,volume_mm3=10,slices=list(range(20)))])
        a,sha,audit=build_plan(visits,inventory,c);b,sha2,_=build_plan(visits,inventory,c);self.assertEqual(sha,sha2);self.assertEqual(audit['mandatory_unique'],1200);self.assertLessEqual(audit['negative_only_streak']['max'],8)
        cursor=102;self.assertEqual(a[cursor:cursor+32],[r for pair in zip(a[cursor::2],a[cursor+1::2]) for r in pair][:32]);self.assertEqual(digest(a),digest(b))
    def test_zero_intensity_inside_anatomy(self):
        m=fixture(64);x=np.ones((12,64,64));x[:,30:34,30:34]=0
        roi,_,_=effective(x,m,1,config());self.assertTrue(roi[31,31])
    def test_no_validation_training_inventory(self):
        from .sampling import mandatory
        with self.assertRaises(ValueError):mandatory([dict(metadata={},partition='validation')])
    def test_resume_identity_and_prefix(self):
        from .state import initial,require_resume
        from ...train import RUNTIME_PRECISION
        state=initial('cat','inv','plan');state['runtime_precision']=RUNTIME_PRECISION
        self.assertTrue(require_resume(state,'cat','inv','plan'))
        with self.assertRaises(ValueError):require_resume(state,'other','inv','plan')
        state['normalization_migration']={'anything':True}
        with self.assertRaises(ValueError):require_resume(state,'cat','inv','plan')
    def test_weighted_components_bounded(self):
        l=torch.zeros((1,1,16,16),requires_grad=True);y=torch.zeros_like(l);y[:,:,0,0]=1;y[:,:,8:12,8:12]=1
        w=torch.ones_like(y);w[:,:,0,0]=4;w[:,:,8:12,8:12]=1/16
        total,parts=balanced_loss(l,y,torch.ones_like(y),w,True);parts['bce'].backward()
        tiny=abs(float(l.grad[0,0,0,0]));large=float(l.grad[0,0,8:12,8:12].abs().sum())
        self.assertLess(tiny/large,4.5);self.assertGreater(tiny/large,3.)
    def test_render_has_v2_only_entrypoint(self):
        from .control import render
        source=render('a'*40,43200,60000)
        self.assertIn('cohort.v2.train',source);self.assertNotIn('normalization_migration',source)
        self.assertIn('novo-unet25d-v2',source)
        self.assertNotIn('KAGGLE_API_TOKEN',source);self.assertNotIn('R2_OUTPUT_SECRET',source)
        compile(source,'rendered-v2','exec')
    def test_v1_incompatible(self):
        from .state import require_resume
        with self.assertRaises(ValueError):require_resume({'data_mode':'adaptive-two-phase-v1'},'cat','inv','plan')

def run_tests():
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(Correctness);result=unittest.TextTestRunner(verbosity=1).run(suite)
    if not result.wasSuccessful():raise ValueError('v2 CPU correctness failed')
    return dict(run=result.testsRun,failed=len(result.failures),errors=len(result.errors),skipped=len(result.skipped))

if __name__=='__main__':run_tests()
