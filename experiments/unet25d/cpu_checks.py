"""Targeted geometry/cache/loss/BN checks; not GPU benchmark evidence."""
import json,tempfile
from pathlib import Path
import nibabel as nib
import numpy as np
import torch
from .data import prepare_visit,restore,SliceCache,sample_draws
from .train import Model,loss_fn

def main():
    torch.set_num_threads(2);torch.manual_seed(42)
    with tempfile.TemporaryDirectory() as name:
        root=Path(name);arr=np.zeros((4,512,512,3),np.float32);arr[:,1:8,1:8,:]=np.arange(4*7*7*3).reshape(4,7,7,3)+1;target=np.zeros((512,512,3),np.uint8);target[2,3,1]=1
        affine=np.diag([-1.,1.,1.,1.]);ref=nib.Nifti1Image(arr[0],affine);ref.header.set_xyzt_units('mm');meta=prepare_visit(dict(reference=ref,images=arr,target=target,visit=dict(exam_id='synthetic',patient_id='synthetic')),root/'synthetic')
        raw=np.load(root/'synthetic/target.npy');assert np.array_equal(restore(raw,meta),target)
        ds=SliceCache(root,[('synthetic',0),('synthetic',2)]);x,y,info=ds[0];assert x.shape==(12,512,512) and y.shape==(1,512,512);assert torch.equal(x[0],x[1]);assert bool((x[:,20,20]==0).all())
        ds[1];assert ds.hits==1 and ds.misses==1
        negative=dict(meta,exam_id='negative',patient_id='negative',positive=[],negative=[0,1,2]);draws=sample_draws([meta,negative],40,42,2);assert draws==sample_draws([meta,negative],40,42,2)
        # Both rank streams contain positives/negatives, not one class per rank.
        for rank in (0,1):assert any(z==1 and v=='synthetic' for v,z in draws[rank::2]) and any(v=='negative' for v,z in draws[rank::2])
        small=np.ones((4,256,256,3),dtype=np.float32);small[:,0,0,:]=0
        small_target=np.zeros((256,256,3),dtype=np.uint8)
        small_ref=nib.Nifti1Image(small[0],np.eye(4));small_ref.header.set_xyzt_units('mm')
        padded=prepare_visit(dict(reference=small_ref,images=small,target=small_target,visit=dict(exam_id='padded',patient_id='padded')),root/'padded')
        assert padded['shape']==[512,512,3] and padded['padding'][:2]==[[128,128],[128,128]]
        assert np.array_equal(restore(np.load(root/'padded/target.npy'),padded),small_target)
        from .data import GeometryConflict
        bad_ref=nib.Nifti1Image(np.zeros((640,640,2),dtype=np.uint8),np.eye(4))
        try:prepare_visit(dict(reference=bad_ref,images=np.zeros((4,640,640,2),dtype=np.float32),target=np.zeros((640,640,2),dtype=np.uint8),visit=dict(exam_id='bad',patient_id='bad')),root/'bad')
        except GeometryConflict:pass
        else:raise AssertionError('640 FOV conflict must fail explicitly')
        angle=np.deg2rad(2.);rot=np.eye(4);rot[:2,:2]=[[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]]
        oblique_images=np.ones((4,32,32,3),dtype=np.float32);oblique_target=np.zeros((32,32,3),dtype=np.uint8);oblique_ref=nib.Nifti1Image(oblique_images[0],rot)
        oblique=prepare_visit(dict(reference=oblique_ref,images=oblique_images,target=oblique_target,visit=dict(exam_id='oblique',patient_id='oblique')),root/'oblique')
        grid=np.indices(tuple(oblique['shape']),dtype=np.float32);grid_affine=np.asarray(oblique['canonical_affine']);ramp=grid_affine[0,0]*grid[0]+grid_affine[0,3]
        restored=restore(ramp,oblique,order=1);native_grid=np.indices((32,32,3),dtype=np.float32);expected=rot[0,0]*native_grid[0]+rot[0,1]*native_grid[1]
        assert oblique['oblique_resampled'] and np.allclose(restored,expected,atol=1e-4)
    model=Model(encoder_name='resnet34',encoder_weights='imagenet',in_channels=12,classes=1,activation=None).train();before=[m.running_mean.clone() for m in model.encoder.modules() if isinstance(m,torch.nn.BatchNorm2d)];x=torch.randn(1,12,64,64);y=torch.zeros(1,1,64,64);loss=loss_fn(model(x),y);loss.backward()
    assert torch.isfinite(loss) and all(p.requires_grad and p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    model.train();after=[m.running_mean for m in model.encoder.modules() if isinstance(m,torch.nn.BatchNorm2d)];assert all(torch.equal(a,b) for a,b in zip(before,after));assert all(m.training for m in model.decoder.modules() if isinstance(m,torch.nn.BatchNorm2d))
    print(json.dumps(dict(geometry_roundtrip=True,padding256_roundtrip=True,conflict640_rejected=True,oblique_physical_ramp_roundtrip=True,replicated_boundary=True,zero_background=True,deterministic_patient_sampling=True,both_rank_classes=True,full_network_cpu_gradient=True,encoder_bn_running_stats_frozen=True,decoder_bn_train=True,cpu_empty_target_loss=float(loss.detach()),gpu_benchmark=False)))
if __name__=='__main__':main()
