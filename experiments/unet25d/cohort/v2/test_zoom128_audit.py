"""Bounded CPU regressions for zoom geometry and unambiguous diagnostics."""
import unittest
import numpy as np
import torch
from scipy import ndimage
from torch.nn import functional as F
from .geometry import crop
from .loading import Cache,component_exposure
from .loss import balanced_loss
from .settings import config

class SyntheticCache(Cache):
    def __init__(self):
        self.c=config();self.mask_counts={'planes':0,'fallbacks':0};self.fallback_reasons={}
        self.ids=np.zeros((512,512,1),np.uint16)
        self.ids[200:202,200:211,0]=1;self.ids[220:227,210:222,0]=2;self.ids[250,200:301,0]=3
        self.inventory={'synthetic':{'components':[{'id':1,'volume_mm3':10.},{'id':2,'volume_mm3':500.},{'id':3,'volume_mm3':800.}]}}
    def arrays(self,visit,z):return np.ones((12,512,512),np.float32),np.where(self.ids[:,:,0]>0,3,0).astype(np.uint8)
    def components(self,exam):return self.ids

class ZoomAuditTests(unittest.TestCase):
    def test_mixed_lesion_crop_keeps_all_gt_and_reports_selected_subset(self):
        cache=SyntheticCache();m=dict(exam_id='synthetic',shape=[512,512,1],native_shape=[512,512,1],native_affine=np.eye(4).tolist(),canonical_affine=np.eye(4).tolist())
        row=dict(z=0,origin=[192,192],scale=128,positive=True,kind='replay',component=1)
        x,y,mask,w,identity=cache.sample_v2({'metadata':m},row)
        self.assertEqual(tuple(x.shape),(12,512,512));self.assertEqual(int(y.sum()),207*16)
        self.assertEqual(identity['selected_component_foreground_pixels'],22*16)
        self.assertEqual(identity['incidental_foreground_pixels'],185*16)
        self.assertEqual(identity['target_scope'],'all enhancing components in crop')
        selected_weight=10**.25
        self.assertAlmostEqual(identity['selected_component_BCE_share'],selected_weight/(selected_weight+2),places=6)
        np.testing.assert_array_equal(y.numpy()[0],np.repeat(np.repeat(cache.ids[192:320,192:320,0]>0,4,0),4,1))

    def test_nearest_exact_integer_zoom_preserves_each_component(self):
        a=torch.tensor([[[[0.,1.],[2.,3.]]]])
        b=F.interpolate(a,scale_factor=4,mode='nearest-exact')
        np.testing.assert_array_equal(b.numpy(),np.repeat(np.repeat(a.numpy(),4,2),4,3))
        for label in (1,2,3):self.assertEqual(int((b==label).sum()),16)

    def test_bilinear_pixel_centers_match_independent_reference(self):
        rng=np.random.default_rng(7);a=rng.normal(size=(128,128)).astype(np.float32)
        u=np.clip((np.arange(512)+.5)/4-.5,0,127);i,j=np.meshgrid(u,u,indexing='ij')
        ref=ndimage.map_coordinates(a,[i,j],order=1,mode='nearest',prefilter=False)
        got=F.interpolate(torch.from_numpy(a)[None,None],(512,512),mode='bilinear',align_corners=False)[0,0].numpy()
        np.testing.assert_allclose(got,ref,atol=1e-6,rtol=1e-6)

    def test_negative_origin_padding_does_not_shift_pixels(self):
        a=np.arange(12,dtype=np.int32).reshape(3,4);got=crop(a,[-1,-2],6)
        expected=np.zeros((6,6),np.int32);expected[1:4,2:6]=a
        np.testing.assert_array_equal(got,expected)

    def test_loss_positive_negative_invalid_gradients_and_weight_invariance(self):
        y=torch.zeros(1,1,8,8);y[:,:,2:4,2:4]=1;m=torch.ones_like(y);m[:,:,:,0]=0
        z=torch.zeros_like(y,requires_grad=True);w=torch.ones_like(y)
        loss=balanced_loss(z,y,m,w);loss.backward()
        self.assertTrue(bool((z.grad[y.bool()]<0).all()));self.assertTrue(bool((z.grad[(m.bool()&~y.bool())]>0).all()));self.assertTrue(bool((z.grad[~m.bool()]==0).all()))
        self.assertAlmostEqual(float(loss.detach()),float(balanced_loss(z.detach(),y,m,w*3)),places=6)

    def test_empty_dice_is_expected_sum_probability_expression(self):
        z=torch.zeros(1,1,8,8,requires_grad=True);y=torch.zeros_like(z);m=torch.ones_like(z)
        loss,parts=balanced_loss(z,y,m,return_parts=True)
        self.assertAlmostEqual(float(parts['soft_dice'].detach()),1-1/(64*.5+1),places=6)
        loss.backward();self.assertTrue(bool((z.grad>0).all()))

if __name__=='__main__':unittest.main()
