"""Private bounded CPU forensics; no training, inference, R2 writes or GPU submission."""
import json,hashlib,gzip,os,math,urllib.request,ssl
from botocore.httpsession import get_cert_path
from pathlib import Path
from collections import Counter
import numpy as np
import torch
from torch.nn import functional as F
from scipy import ndimage
import nibabel as nib
from experiments.unet25d.cohort.v2.train import fixed_rows
from experiments.unet25d.cohort.v2.loading import Cache
from experiments.unet25d.cohort.v2.geometry import effective,crop,crop_origin
from experiments.unet25d.cohort.v2.loss import balanced_loss
from experiments.unet25d.cohort.v2.sampling import digest,lesion_weight
from experiments.unet25d.cohort.v2.settings import config,science_hash


def main():
    torch.set_num_threads(2)
    import argparse
    parser=argparse.ArgumentParser(description='CPU-only reconstruction of frozen v2 Gate1 128 sample; no model forward or GPU')
    parser.add_argument('--private-dir',type=Path,required=True,help='Private directory with exact verified input v3 metadata and GET lease')
    args=parser.parse_args();p=args.private_dir;p.mkdir(mode=0o700,parents=True,exist_ok=True);p.chmod(0o700);c=config()
    if p.resolve().is_relative_to(Path(__file__).resolve().parents[4]):raise ValueError('Private clinical audit files must remain outside the repository')
    assert not torch.cuda.is_available()
    package=json.loads((p/'input-package.json').read_text())
    assert package['source_sha']=='3ddc324e114d0a0cd740234b5ba26ec750238d2a' and package['science_sha256']==science_hash()
    for name in ('catalogue.json','inventory.json','plan.json','private-data-lease.json','training-config.json','GATE0_ACCEPTANCE.json','SAMPLING_AUDIT.json'):
        assert hashlib.sha256((p/name).read_bytes()).hexdigest()==package['files'][name]

    cat=json.loads((p/'catalogue.json').read_text());inv=json.loads((p/'inventory.json').read_text());plan=json.loads((p/'plan.json').read_text());lease=json.loads((p/'private-data-lease.json').read_text());rows=fixed_rows(plan,inv,c);row=rows[1];visit=next(v for v in cat['visits'] if v['metadata']['exam_id']==row['exam']);assert visit['partition']=='train';m=visit['metadata'];comp=next(x for x in inv[row['exam']]['components'] if x['id']==row['component']);z=row['z'];selected=row['component']
    assert digest(plan)==json.loads((p/'SAMPLING_AUDIT.json').read_text())['plan_SHA']
    def fetch(key,sha,destination):
     r=lease['objects'][key];assert r['sha256']==sha
     path=Path(destination);path.parent.mkdir(parents=True,exist_ok=True)
     h=hashlib.sha256()
     with urllib.request.urlopen(r['url'],context=ssl.create_default_context(cafile=get_cert_path(True)),timeout=120) as response:
      with path.open('wb') as f:
       while b:=response.read(1024**2):h.update(b);f.write(b)
     assert h.hexdigest()==sha;path.chmod(0o600)
    cache=Cache(p/'cache',fetch,c,inv)
    x,y,mask,w,identity=cache.sample_v2(visit,row);images,raw=cache.arrays(visit,z);ids=cache.components(row['exam'])[:,:,z];roi,valid,reason=effective(images,m,z,c,raw==3)
    origin=row['origin'];size=128
    recreated_origin,recreated_size,_=crop_origin(comp,z,row.get('requested_scale',128),c['crop_jitter_fraction'],m['shape'],row['seed']);assert recreated_origin==origin and recreated_size==128
    # Independent integer crop reference via coordinate arrays, not production crop.
    ii,jj=np.indices((size,size));aa=ii+origin[0];bb=jj+origin[1];inside=(aa>=0)&(aa<raw.shape[0])&(bb>=0)&(bb<raw.shape[1])
    def independent(a):
     out=np.zeros((*a.shape[:-2],size,size),a.dtype);out[...,inside]=a[...,aa[inside],bb[inside]];return out
    native_ids=independent(ids);native_y=independent(raw==3);native_mask=independent(roi);native_x=independent(images*roi[None]);expanded_ids=np.repeat(np.repeat(native_ids,4,0),4,1);expanded_y=np.repeat(np.repeat(native_y,4,0),4,1);expanded_mask=np.repeat(np.repeat(native_mask,4,0),4,1)
    np.testing.assert_array_equal(y.numpy()[0],expanded_y);np.testing.assert_array_equal(mask.numpy()[0],expanded_mask)
    # Independent bilinear pixel-center reference, clamped at source edge.
    u=np.clip((np.arange(512)+.5)/4-.5,0,127);uu,vv=np.meshgrid(u,u,indexing='ij');bilinear=np.stack([ndimage.map_coordinates(a,[uu,vv],order=1,mode='nearest',prefilter=False) for a in native_x]);before=F.interpolate(torch.from_numpy(native_x.copy())[None],(512,512),mode='bilinear',align_corners=False)[0].numpy();err=float(np.max(np.abs(bilinear-before)))
    np.testing.assert_allclose(x.numpy(),bilinear*expanded_mask[None],rtol=2e-6,atol=2e-6)
    # Reconstruct native annotation orientation and geometric FOV independently.
    refpath=p/'native-labels.npy.gz'
    if not refpath.exists():fetch(visit['native_labels']['key'],visit['native_labels']['sha256'],refpath)
    with gzip.open(refpath,'rb') as f:native_labels=np.load(f,allow_pickle=False)
    canonical=np.pad(nib.orientations.apply_orientation(native_labels,np.asarray(m['orientation_transform'])),m['padding']);np.testing.assert_array_equal(canonical[:,:,z],raw)
    pts=np.column_stack([*np.nonzero(raw==3),np.full(int((raw==3).sum()),z),np.ones(int((raw==3).sum()))]);native_pts=(np.linalg.inv(m['native_affine'])@np.asarray(m['canonical_affine'])@pts.T).T[:,:3]
    fov_ok=bool(((native_pts>=-.5+1e-5)&(native_pts<np.asarray(m['native_shape'])-.5-1e-5)).all())
    # Check production channel order directly against the decoded shard arrays.
    shard=next(r for r in visit['shards'] if r['start']<=z<r['end']);decoded=cache.decoded[shard['sha256']][0];zi=np.clip([z-1,z,z+1],0,m['shape'][2]-1)-shard['halo_start'];channels_ok=all(np.array_equal(images[3*a+b],decoded[zi[b],a]) for a in range(4) for b in range(3))
    labels=[int(v) for v in np.unique(native_ids) if v];counts={v:int((native_ids==v).sum()) for v in labels};upcounts={v:int((expanded_ids==v).sum()) for v in labels};assert all(upcounts[v]==16*counts[v] for v in labels);assert counts[selected]==comp['planes'][str(z)]['voxels']
    # Independent local 26-connected reference from raw GT around selected 3D bbox.
    lo=np.maximum(0,np.array([b[0] for b in comp['bbox']])-1);hi=np.minimum(np.asarray(canonical.shape),np.array([b[1] for b in comp['bbox']])+1);box=tuple(slice(a,b) for a,b in zip(lo,hi))
    reference_ids,_=ndimage.label(canonical[box]==3,structure=np.ones((3,3,3),bool));stored=cache.components(row['exam'])[box]==selected;seed=tuple(np.argwhere(stored)[0]);reference_component=reference_ids==reference_ids[seed];np.testing.assert_array_equal(reference_component,stored)
    reference_voxels=int(reference_component.sum());reference_volume=reference_voxels*abs(np.linalg.det(np.asarray(m['canonical_affine'])[:3,:3]));assert reference_voxels==comp['voxels'] and np.isclose(reference_volume,comp['volume_mm3'])
    properties={r['id']:r for r in inv[row['exam']]['components']}
    lesions=[]
    for label in labels:
     loc=expanded_ids==label;weights=w.numpy()[0][loc];lesions.append(dict(alias='selected' if label==selected else 'incidental_'+str(len(lesions)),selected=label==selected,native_pixels=counts[label],resized_pixels=upcounts[label],volume_bin='<100' if properties[label]['volume_mm3']<100 else '100-<1000' if properties[label]['volume_mm3']<1000 else '>=1000',positive_weight_min=float(weights.min()),positive_weight_max=float(weights.max()),positive_bce_weight_share=float(weights.sum()/w.numpy()[0][expanded_y].sum())))
    
    def stats(a,valid=None):
     a=np.asarray(a);v=a.ravel();ans=dict(finite=bool(np.isfinite(v).all()),mean=float(v.mean(dtype=np.float64)),std=float(v.std(dtype=np.float64)),quantiles=[float(t) for t in np.quantile(v,[0,.01,.05,.5,.95,.99,1])],zero_fraction=float((v==0).mean()))
     if valid is not None:ans.update(valid_occupancy=float(np.asarray(valid).mean()),valid_mean=float(a[valid].mean(dtype=np.float64)),valid_std=float(a[valid].std(dtype=np.float64)))
     return ans
    statistics=[]
    selected_native=native_ids==selected
    selected_enlarged=expanded_ids==selected
    ring_native=(ndimage.distance_transform_edt(~selected_native)<=5)&native_mask&~native_y
    ring_enlarged=np.repeat(np.repeat(ring_native,4,0),4,1)
    lesion_intensities=[dict(channel=k,modality=c['modalities'][k//3],offset=c['offsets'][k%3],native_selected=stats(native_x[k][selected_native]),resized_selected=stats(x.numpy()[k][selected_enlarged]),native_nearby_background=stats(native_x[k][ring_native]),resized_nearby_background=stats(x.numpy()[k][ring_enlarged])) for k in range(12)]
    for k in range(12):statistics.append(dict(channel=k,modality=c['modalities'][k//3],slice_offset=c['offsets'][k%3],prepared=stats(images[k],roi),crop=stats(native_x[k],native_mask),resized_before_final_mask=stats(before[k],expanded_mask),final=stats(x.numpy()[k],expanded_mask)))
    
    def loss_audit(yy,mm,ww,probability=.5):
     yy=yy[None];mm=mm[None];ww=ww[None];logit=math.log(probability/(1-probability));t=torch.full_like(yy,logit,requires_grad=True);loss,parts=balanced_loss(t,yy,mm,ww,True);loss.backward();grad=t.grad.detach().numpy()[0,0];truth=yy.numpy()[0,0].astype(bool);valid=mm.numpy()[0,0].astype(bool);weights=ww.numpy()[0,0].astype(np.float64);n=int(truth.sum());b=int((valid&~truth).sum());a=np.full(truth.shape,logit,dtype=np.float64);p0=1/(1+np.exp(-a));I=float(p0[truth].sum());D=float(p0[valid].sum()+truth.sum()+1);num=2*I+1
     pe=float(np.sum(np.logaddexp(0,-a[truth])*weights[truth])/weights[truth].sum()) if n else 0;ne=float(np.logaddexp(0,a[valid&~truth]).mean()) if b else 0
     reference=.5*(.5*pe+.5*ne if n and b else pe if n else ne)+.5*(1-num/D)
     analytic=np.zeros_like(a)
     if n:analytic[truth]+=.25*(p0[truth]-1)*weights[truth]/weights[truth].sum() if b else .5*(p0[truth]-1)*weights[truth]/weights[truth].sum()
     if b:analytic[valid&~truth]+=(.25 if n else .5)*p0[valid&~truth]/b
     analytic[valid]+=.5*(num-2*truth[valid]*D)/D**2*p0[valid]*(1-p0[valid])
     groups={'foreground':truth,'near_background':valid&~truth&(ndimage.distance_transform_edt(~truth)<=8),'far_background':valid&~truth&(ndimage.distance_transform_edt(~truth)>32),'invalid':~valid}
     if truth.shape==expanded_ids.shape and np.array_equal(truth,expanded_y):groups['selected_foreground']=expanded_ids==selected;groups['incidental_foreground']=truth&(expanded_ids!=selected)
     breakdown={}
     for name,loc in groups.items():
      vals=grad[loc];breakdown[name]=dict(pixels=int(loc.sum()),gradient_mean=float(vals.mean(dtype=np.float64)) if vals.size else None,gradient_min=float(vals.min()) if vals.size else None,gradient_max=float(vals.max()) if vals.size else None,gradient_l1=float(np.abs(vals).sum(dtype=np.float64)))
     # Double precision independent central difference of individual representative logits.
     finite=[]
     for name,loc in groups.items():
      positions=np.argwhere(loc)
      if not len(positions):continue
      i,j=positions[len(positions)//2];epsilon=.001
      def calc(delta):
       aa=a.copy();aa[i,j]+=delta;pp=1/(1+np.exp(-aa));positive=float(np.sum(np.logaddexp(0,-aa[truth])*weights[truth])/weights[truth].sum()) if n else 0;negative=float(np.logaddexp(0,aa[valid&~truth]).mean()) if b else 0
       return .5*(.5*positive+.5*negative if n and b else positive if n else negative)+.5*(1-(2*pp[truth].sum()+1)/(pp[valid].sum()+n+1))
      fd=(calc(epsilon)-calc(-epsilon))/(2*epsilon);finite.append(dict(group=name,independent_finite_difference=float(fd),production_autograd=float(grad[i,j]),absolute_error=float(abs(fd-grad[i,j]))))
     assert np.max(np.abs(grad-analytic))<1e-7
     return dict(probability=probability,positive_pixels=n,negative_pixels=b,loss=float(loss.detach()),independent_loss=reference,parts={k:float(v.detach()) for k,v in parts.items()},weighted_positive_contribution=.25*pe if n and b else .5*pe,negative_contribution=(.25 if n else .5)*ne,soft_dice_numerator=num,soft_dice_denominator=D,gradient_breakdown=breakdown,finite_difference=finite,max_analytic_gradient_error=float(np.max(np.abs(grad-analytic))))
    
    # Controlled same lesion and same physical crop center. No model inference.
    representations=[];center=np.asarray(origin)+(127/2)
    for size2 in (512,256,128):
     orig2=np.floor(center-(size2-1)/2).astype(int).tolist();row2=dict(row,origin=orig2,scale=size2);xx,yy,mm,ww,ident=cache.sample_v2(visit,row2)
     cc=crop(ids,orig2,size2);selected_native=int((cc==selected).sum());assert selected_native==counts[selected]
     entry=dict(scale=size2,selected_native_pixels=selected_native,total_native_pixels=int((cc>0).sum()),selected_resized_pixels=selected_native*(512//size2)**2,total_resized_pixels=int(yy.sum()),valid_pixels=int(mm.sum()),all_channel_std=[float(xx[k].std(unbiased=False)) for k in range(12)],loss=loss_audit(yy,mm,ww))
     representations.append(entry)
    canvas_x=torch.zeros_like(x);canvas_y=torch.zeros_like(y);canvas_m=torch.zeros_like(mask);canvas_w=torch.ones_like(w)
    canvas_x[:,192:320,192:320]=torch.from_numpy(native_x);canvas_y[:,192:320,192:320]=torch.from_numpy(native_y);canvas_m[:,192:320,192:320]=torch.from_numpy(native_mask)
    # Same component weighting formula in unzoomed crop.
    for label in labels:canvas_w[0,192:320,192:320][torch.from_numpy(native_ids==label)]=lesion_weight(properties[label]['volume_mm3'],c)/counts[label]
    canvas_w[canvas_y.bool()]*=float(canvas_y.sum())/float(canvas_w[canvas_y.bool()].sum())
    representations.append(dict(scale='128_embedded_no_zoom',selected_resized_pixels=counts[selected],total_resized_pixels=int(canvas_y.sum()),valid_pixels=int(canvas_m.sum()),loss=loss_audit(canvas_y,canvas_m,canvas_w)))
    # Weighted BCE is invariant to a common positive-weight scale in actual crop.
    test_logits=torch.zeros_like(y[None]);test_logits[y[None].bool()]=torch.linspace(-2,2,int(y.sum()))
    base=balanced_loss(test_logits,y[None],mask[None],w[None]);scaled=balanced_loss(test_logits,y[None],mask[None],w[None]*3)
    assert abs(float(base-scaled))<1e-6
    single=[]
    for volume in (1.,10000.):
     a=lesion_weight(volume,c)/int(y.sum());weights=torch.ones_like(y);weights[y.bool()]=a;weights[y.bool()]*=float(y.sum())/float(weights[y.bool()].sum());single.append(float(balanced_loss(test_logits,y[None],mask[None],weights[None])))
    empty=[]
    for probability in (.5,.0269,.001,.00001):empty.append(loss_audit(torch.zeros_like(y),mask,torch.ones_like(w),probability))
    exposures={str(rank):Counter(((step//2)*2+rank+step%2)%4 for step in range(400)) for rank in (0,1)}
    private=dict(row=row,component=comp,metadata=m,crop_center=center.tolist(),native_spacing=nib.affines.voxel_sizes(np.asarray(m['native_affine'])).tolist(),crop_physical_vectors=(np.asarray(m['canonical_affine'])[:3,:2]*128).tolist(),same_lesion_FOV_edge_lengths_mm={str(size): (np.linalg.norm(np.asarray(m['canonical_affine'])[:3,:2],axis=0)*size).tolist() for size in (128,256,512)},crop_jitter_pixels=(np.asarray(origin)+(127/2)-np.asarray(comp['planes'][str(z)]['centroid'])).tolist(),intersects_image_boundary=not bool(inside.all()),excluded_selected_pixels=int(comp['planes'][str(z)]['voxels']-counts[selected]),native_z_coordinates=native_pts[:3].tolist())
    (p/'exact_geometry_private.json').write_text(json.dumps(private,indent=2));(p/'exact_geometry_private.json').chmod(0o600)
    result=dict(source_sha='3ddc324e114d0a0cd740234b5ba26ec750238d2a',science_sha256=science_hash(),plan_digest=digest(plan),shape=list(x.shape),gate0_128_same_as_gate1=row==next(r for r in plan if r['kind']=='replay' and r['scale']==128),total_native_positive_pixels=int(native_y.sum()),total_resized_positive_pixels=int(y.sum()),selected_native_positive_pixels=counts[selected],selected_resized_positive_pixels=upcounts[selected],selected_fraction=upcounts[selected]/int(y.sum()),lesions_in_crop=len(labels),components=lesions,selected_component_fully_preserved=True,frozen_crop_origin_recomputed=True,crop_geometric_fov_pixels=int(crop(valid,origin,128).sum()),crop_intersects_padded_canvas_boundary=not bool(inside.all()),independent_3d_component_identity_verified=True,independent_component_volume_verified=True,gt_native_orientation_alignment=True,all_foreground_in_geometric_fov=fov_ok,channel_order_verified=channels_ok,max_independent_bilinear_error=err,mask_valid_pixels=int(mask.sum()),GT_excluded_by_mask=int((y.bool()&~mask.bool()).sum()),nonfinite_input=int((~torch.isfinite(x)).sum()),statistics=statistics,selected_lesion_intensities=lesion_intensities,loss_at_zero=loss_audit(y,mask,w),loss_at_uniform_0269=loss_audit(y,mask,w,.0269),same_lesion_scales=representations,weight_cancellation=dict(actual_common_weight_scale_difference=float(base-scaled),single_component_losses=single),negative_dice_curve=empty,ddp_fixed_exposure={r:dict(v) for r,v in exposures.items()},private_geometry_record='exact_geometry_private.json',normalization_recomputed=False,model_forward_executed=False,gpu_allocated=False)
    (p/'audit_sanitized.json').write_text(json.dumps(result,indent=2));(p/'audit_sanitized.json').chmod(0o600)
    print(json.dumps({k:result[k] for k in ['shape','total_native_positive_pixels','total_resized_positive_pixels','selected_native_positive_pixels','selected_resized_positive_pixels','selected_fraction','lesions_in_crop','selected_component_fully_preserved','max_independent_bilinear_error','mask_valid_pixels','GT_excluded_by_mask','channel_order_verified']}),flush=True)

if __name__=='__main__':main()
