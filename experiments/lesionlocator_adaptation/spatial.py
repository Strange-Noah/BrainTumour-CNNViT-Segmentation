"""CPU-only probes on official LesionLocator grids and fixed patch sizes."""
from copy import deepcopy
import math

import nibabel as nib
import numpy as np
from scipy import ndimage
import torch
from torch.nn import functional as F
from lesionlocator.inference.export_prediction import convert_predicted_logits_to_segmentation_with_correct_shape
from lesionlocator.modules.tracknet import TrackNet
from lesionlocator.utilities.plans_handling.plans_handler import PlansManager

from experiments.lesionlocator_benchmark.baseline import overlap_counts, propagate
from experiments.lesionlocator_benchmark.geometry import native_array, native_labels, _instance_report

REG=(175,175,175)
PATCH=(96,256,256)
CANDIDATES={
    "official": (3.,.841796875,.841796875),
    "depth_2": (2.,.841796875,.841796875),
    "depth_1_5": (1.5,.841796875,.841796875),
    "depth_1": (1.,.841796875,.841796875),
    "paper_geometry": (1.,.8,.8),
}


class GeometryOnly:
    input_shape=REG
    unet_patch_size=torch.tensor(PATCH)


def config_for(base,name,spacing,new_norm=None):
    plans=deepcopy(base)
    if new_norm is not None:
        plans["foreground_intensity_properties_per_channel"]["0"].update(new_norm)
    plans["configurations"][name]["spacing"]=list(spacing)
    manager=PlansManager(plans)
    return manager,manager.get_configuration(name)


def work_to_registration(coord,shape):
    return (np.asarray(coord)+.5)*np.asarray(REG)/np.asarray(shape)-.5


def registration_to_work(coord,shape):
    return (np.asarray(coord)+.5)*np.asarray(shape)/np.asarray(REG)-.5


def register_mask(mask):
    x=torch.from_numpy(np.ascontiguousarray(mask,dtype=np.float32))[None,None]
    return F.interpolate(x,size=REG,mode="nearest-exact")[0,0].numpy().astype(mask.dtype)


def boxes(mask,spacing):
    voxels=int(mask.sum())
    if not voxels:
        return {"voxels":0}
    lim=ndimage.find_objects(mask.astype(np.uint8))[0]
    spans=np.array([s.stop-s.start for s in lim])
    return dict(voxels=voxels,bbox=spans.tolist(),
                bbox_mm=(spans*np.asarray(spacing)).tolist(),
                aspect=(spans/spans.max()).tolist(),
                components=ndimage.label(mask,structure=np.ones((3,)*3,dtype=bool))[1])


def roi(mask,anatomy,target,spacing):
    if not mask.any():
        return {"available":False}
    p=torch.from_numpy(mask.astype(np.uint8))[None,None]
    coords=[int(i) for i in TrackNet.get_patch_around_mask(GeometryOnly(),p,True)[0]]
    bounds=[slice(coords[i],coords[i+1]) for i in range(0,6,2)]
    cut=mask[tuple(bounds)]
    real=np.array(cut.shape)
    pad=np.asarray(PATCH)-real
    if np.any(pad<0):
        raise ValueError("Unexpected upstream ROI crop")
    fg=int(cut.sum())
    coverage=float(target[tuple(bounds)].sum()/target.sum()) if target is not None and target.any() else None
    border=None
    position=None
    if fg:
        box=ndimage.find_objects(cut.astype(np.uint8))[0]
        border=[min(b.start,n-b.stop) for b,n in zip(box,real)]
        center=np.argwhere(cut).mean(axis=0)+pad//2
        position=(center/np.asarray(PATCH)).tolist()
    return dict(available=True,coordinates=coords,voxels=fg,retained=fg/int(mask.sum()),
                **{k:v for k,v in boxes(cut,spacing).items() if k!="voxels"},
                lesion_fraction=fg/math.prod(PATCH),pad_fraction=1-math.prod(real)/math.prod(PATCH),
                actual_shape=real.tolist(),pad_pairs=np.stack((pad//2,pad-pad//2),axis=1).tolist(),
                physical_field_of_view_mm=(real*np.asarray(spacing)).tolist(),
                nominal_field_of_view_mm=(np.asarray(PATCH)*spacing).tolist(),
                anatomy_fraction=int(anatomy[tuple(bounds)].sum())/math.prod(PATCH),
                position=position,min_border_distance_vox=border,target_coverage=coverage)


def preprocess(pre,manager,config,data,visit,prompt=None):
    x,m,props=pre.run_case([str(visit["image"])],None if prompt is None else [str(prompt)],
                           manager,config,data)
    return x,np.maximum(m,0),props


def evaluate_pair(base,name,data,spacing,history,target,root,normalization=None,metrics=True):
    m,c=config_for(base,name,spacing,normalization)
    pre=c.preprocessor_class(verbose=False)
    x0,h,p0=preprocess(pre,m,c,data,history,history["instances"])
    x1,_,p1=preprocess(pre,m,c,data,target)
    _,t,q1=preprocess(pre,m,c,data,target,target["instances"])
    fields=("bbox_used_for_cropping","shape_before_cropping","shape_after_cropping_and_before_resampling","spacing")
    if any(not np.array_equal(p1[k],q1[k]) for k in fields):
        raise ValueError("GT diagnostic and unlabelled target transforms disagree")
    tensors=[torch.from_numpy(x.astype(np.float16))[None] for x in (x0,x1)]
    thresh=int(np.floor(60./spacing[0]+1e-8))
    _,_,w0,w1=TrackNet.optional_z_translation(GeometryOnly(),*tensors,difference_threshold=thresh)
    _,_,old0,old1=(None,None,w0,w1) if thresh==20 else TrackNet.optional_z_translation(
        GeometryOnly(),*tensors,difference_threshold=20)
    h0=h[0,w0[0]:w0[1]]
    t1=t[0,w1[0]:w1[1]]
    reg=register_mask(h0)
    restored=F.interpolate(torch.from_numpy(reg.astype(np.float32))[None,None],
                           size=h0.shape,mode="nearest-exact")[0,0].numpy().astype(h.dtype)
    nominal=np.asarray(spacing)*h0.shape
    target_nominal=np.asarray(spacing)*t1.shape
    # Derive crop bounds from the unnormalized source using the same reader and
    # transpose. Preserve the nonzero mask as geometry, not a CT/MRI threshold.
    rr=m.image_reader_writer_class()
    anat=[]
    for visit,props,dims in ((history,p0,h.shape[1:]),(target,p1,t.shape[1:])):
        raw,_=rr.read_images([str(visit["image"])])
        mask=(raw[0]>-500) if visit.get("modality")=="ct" else (raw[0]!=0)
        mask=mask.transpose(m.transpose_forward)
        selection=tuple(slice(a,b) for a,b in props["bbox_used_for_cropping"])
        a=torch.from_numpy(mask[selection].astype(np.float32))[None,None]
        anat.append(F.interpolate(a,size=dims,mode="nearest-exact")[0,0].numpy()>0)
    areg=register_mask(anat[0][w0[0]:w0[1]].astype(np.uint8))
    result=dict(spacing=list(spacing),history_work_shape=list(h.shape[1:]),target_work_shape=list(t.shape[1:]),
                history_175_nominal_fov_mm=nominal.tolist(),history_175_nominal_spacing_mm=(nominal/175).tolist(),
                target_175_nominal_fov_mm=target_nominal.tolist(),target_175_nominal_spacing_mm=(target_nominal/175).tolist(),
                anatomy_fraction_175=float(areg.mean()),
                windows=[w0,w1],fixed_voxel_windows=[old0,old1],z_threshold_vox=thresh,
                z_input_dtype="float16",
                history_geometry=dict(affine=history["ref"].affine.tolist(),shape=history["ref"].shape,
                    transpose=m.transpose_forward,spacing=p0["spacing"],bbox=p0["bbox_used_for_cropping"],
                    original_crop_shape=p0["shape_after_cropping_and_before_resampling"]),
                target_geometry=dict(affine=target["ref"].affine.tolist(),shape=target["ref"].shape,
                    transpose=m.transpose_forward,spacing=p1["spacing"],bbox=p1["bbox_used_for_cropping"],
                    original_crop_shape=p1["shape_after_cropping_and_before_resampling"]))
    l0_native=propagate(history["labels"],history["ref"].affine,target["ref"])
    l0_file=root/"l0.nii.gz"
    nib.save(nib.Nifti1Image(l0_native.astype(np.int32),target["ref"].affine),l0_file)
    _,l0,_=preprocess(pre,m,c,data,target,l0_file)
    hnative,tnative=history["labels"],target["labels"]
    by_overlap=overlap_counts(l0_native,tnative)
    reverse=overlap_counts(tnative,l0_native)
    history_rows=[]
    for identity in np.unique(hnative):
        if identity==0: continue
        matches=by_overlap.get(identity,{})
        proxy=next(iter(matches)) if len(matches)==1 else None
        if proxy is not None and len(reverse[proxy])!=1:
            proxy=None
        wanted=(t[0]==proxy) if proxy is not None else None
        native=boxes(hnative==identity,nib.affines.voxel_sizes(history["ref"].affine))
        native["volume_mm3"]=native["voxels"]*abs(np.linalg.det(history["ref"].affine[:3,:3]))
        row=dict(id=int(identity),native=native,work=boxes(h[0]==identity,spacing),
                 registration=boxes(reg==identity,nominal/175),
                 history_roi=roi(h[0]==identity,anat[0],None,spacing),
                 direct_l0_roi=roi(l0[0]==identity,anat[1],wanted,spacing),
                 target_oracle_roi=roi(wanted,anat[1],wanted,spacing) if wanted is not None else None)
        history_rows.append(row)
    result["history"]=history_rows
    result["target_oracle"]=[dict(id=int(i),roi=roi(t[0]==i,anat[1],t[0]==i,spacing))
                             for i in np.unique(tnative) if i]
    if metrics:
        retained=np.zeros_like(h)
        recovered=np.zeros_like(h)
        retained[:,w0[0]:w0[1]]=h0
        recovered[:,w0[0]:w0[1]]=restored
        for tag,work in (("G1",h),("G2",retained),("G4",recovered)):
            back=native_array(native_labels(work,p0,m,c),m,p0,root/(tag+".nii.gz"),history["ref"])
            for entry in history_rows:
                i=entry["id"]
                scores=_instance_report(i,hnative==i,back==i,history["ref"].affine)
                scores["relative_volume_error"]=abs(scores["volume_change_mm3"])/entry["native"]["volume_mm3"]
                entry[tag]=scores
        foreground=(t[0]>0).astype(np.float32)
        label_manager=m.get_label_manager(data)
        for entry in result["target_oracle"]:
            i=entry["id"]
            fg=(t[0]==i).astype(foreground.dtype)
            exported=convert_predicted_logits_to_segmentation_with_correct_shape(
                np.stack((1-fg,fg)),m,c,label_manager,p1)
            back=native_array(exported,m,p1,root/"G5.nii.gz",target["ref"])
            score=_instance_report(i,tnative==i,back>0,target["ref"].affine)
            vol=abs(np.linalg.det(target["ref"].affine[:3,:3]))*score["reference_voxels"]
            score["native_volume_mm3"]=vol
            score["relative_volume_error"]=abs(score["volume_change_mm3"])/vol
            entry["G5"]=score
    return result
