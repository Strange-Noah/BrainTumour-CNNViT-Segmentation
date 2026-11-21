"""Full validation-patient evaluation after meaningful training, never diagnostic-wide evaluation."""
import gzip,io,json,tempfile
from pathlib import Path
import nibabel as nib
import numpy as np
import psutil
import torch
from ...data import restore
from research.evaluate import evaluate,export,patient_aggregates
from ..tiling import windows,merge_logits
from ..metrics import lesion_strata,control_metrics

def validate_patients(model,catalogue,cache,fetch,output,checkpoint_source,checkpoint_ref,completed_coverage_epochs,deadline=float('inf'),previous_results=None):
    if completed_coverage_epochs<1:raise ValueError('Full validation is scheduled after meaningful full-cohort training, not a diagnostic checkpoint')
    visits=[v for v in catalogue['visits'] if v['partition']=='validation']
    if len({v['metadata']['patient_id'] for v in visits})!=45:raise ValueError('Full frozen validation cohort required')
    import time,torch.distributed as dist
    world=dist.get_world_size() if dist.is_initialized() else 1;rank=dist.get_rank() if dist.is_initialized() else 0
    raw=model.module if hasattr(model,'module') else model
    if world>1:
        for buffer in raw.buffers():dist.broadcast(buffer,src=0)
    expected={v['metadata']['exam_id'] for v in visits}
    completed=list(previous_results or []);done={r['exam_id'] for r in completed}
    if not done<=expected or len(done)!=len(completed):raise ValueError('Invalid saved validation progress')
    visits=[v for v in visits if v['metadata']['exam_id'] not in done]
    bins=[[] for _ in range(world)];costs=[0]*world
    for v in sorted(visits,key=lambda v:len(windows(v['metadata']['shape']))*v['metadata']['shape'][2],reverse=True):
        owner=int(np.argmin(costs));bins[owner].append(v);costs[owner]+=len(windows(v['metadata']['shape']))*v['metadata']['shape'][2]
    output=Path(output);output.mkdir(parents=True,exist_ok=True);results=[];was_training=model.training;model.eval();device=next(raw.parameters()).device
    with torch.inference_mode():
        for round_index in range(max(map(len,bins),default=0)):
            memory=psutil.virtual_memory()
            stop=torch.tensor([int(time.time()>deadline-90),int(memory.available/memory.total<.15)],device=device)
            if world>1:dist.all_reduce(stop,op=dist.ReduceOp.MAX)
            if bool(stop[1]):raise MemoryError('Actual validation RAM pressure')
            if bool(stop[0]):break
            visit=bins[rank][round_index] if round_index<len(bins[rank]) else None
            if visit is None:
                if world>1:dist.barrier()
                continue
            m=visit['metadata'];folder=output/m['exam_id'];folder.mkdir();path=folder/'canonical-logits.npy';prediction=np.lib.format.open_memmap(path,mode='w+',dtype='float32',shape=tuple(m['shape']))
            finished_visit=True
            for z in range(m['shape'][2]):
                if time.time()>deadline:
                    finished_visit=False;break
                tiles=[]
                for origin in windows(m['shape']):
                    x,_=cache.sample(visit,origin,z);logits=raw(x[None].to(device))
                    if not torch.isfinite(logits).all():raise FloatingPointError('Nonfinite full-validation logits')
                    tiles.append((origin,logits[0,0].float().cpu().numpy()))
                prediction[:,:,z]=merge_logits(tiles,m['shape'])
            if not finished_visit:
                import shutil
                del prediction;shutil.rmtree(folder)
                if world>1:dist.barrier()
                continue
            prediction.flush();native_logits=restore(prediction,m,order=1);native=native_logits>=0
            native_path=folder/'native-labels.npy.gz';fetch(visit['native_labels']['key'],visit['native_labels']['sha256'],native_path)
            with gzip.open(native_path,'rb') as f:truth=np.load(f,allow_pickle=False)==3
            canonical_truth=np.pad(nib.orientations.apply_orientation(truth,np.asarray(m['orientation_transform'])),m['padding'])
            loss_sum=0.
            for z in range(m['shape'][2]):
                logits=np.asarray(prediction[:,:,z],dtype=np.float64);y=canonical_truth[:,:,z]
                probability=1/(1+np.exp(-np.clip(logits,-700,700)))
                bce=np.mean(np.logaddexp(0,logits)-y*logits)
                dice_loss=1-(2*(probability*y).sum()+1)/(probability.sum()+y.sum()+1)
                loss_sum+=float(.5*bce+.5*dice_loss)
            reference=nib.Nifti1Image(np.empty(m['native_shape'],np.uint8),np.asarray(m['native_affine']));reference.header.set_xyzt_units('mm')
            metrics=evaluate(native,truth,reference.affine)
            probability=1/(1+np.exp(-np.clip(native_logits,-80,80)))
            def distribution(values):
                counts,edges=np.histogram(values,bins=np.linspace(0,1,21))
                return dict(count=int(values.size),mean=float(values.mean()) if values.size else None,max=float(values.max()) if values.size else None,histogram=counts.tolist(),edges=edges.tolist())
            diagnostics=dict(positive_probability=distribution(probability[truth]),negative_probability=distribution(probability[~truth]),empty_prediction=not bool(native.any()),predicted_positive_mm3=float(native.sum()*abs(np.linalg.det(reference.affine[:3,:3]))))
            del probability
            grid={}
            for threshold in [.3,.7]:
                mask=native_logits>=np.log(threshold/(1-threshold))
                # Reuse the same logits; no additional model inference.
                from research.evaluate import components
                p,candidates=components(mask,reference.affine);t,lesions=components(truth,reference.affine)
                from scipy.optimize import linear_sum_assignment
                overlap=np.zeros((len(lesions),len(candidates)),bool);intersection=truth & mask
                if intersection.any():
                    pairs=np.unique(np.stack([t[intersection],p[intersection]],axis=1),axis=0);overlap[pairs[:,0]-1,pairs[:,1]-1]=True
                matches=0
                if overlap.size:
                    rows,cols=linear_sum_assignment(-overlap.astype(int));matches=int(overlap[rows,cols].sum())
                grid[str(threshold)]=dict(lesion_strata=lesion_strata(mask,truth,reference.affine),empty_prediction=not bool(mask.any()),predicted_positive_mm3=float(mask.sum()*abs(np.linalg.det(reference.affine[:3,:3]))),dice=float(2*intersection.sum()/(mask.sum()+truth.sum())) if mask.sum()+truth.sum() else 1.,truth_lesions=len(lesions),matched_lesions=matches,lesion_sensitivity=matches/len(lesions) if lesions else None,false_positives=len(candidates)-matches)
            metrics.update(**diagnostics,lesion_strata=lesion_strata(native,truth,reference.affine),loss_sum=loss_sum,loss_planes=m['shape'][2],threshold_grid=grid)
            results.append(dict(patient_id=m['patient_id'],exam_id=m['exam_id'],**metrics))
            export(native,reference,folder/'prediction',dict(patient_id=m['patient_id'],exam_id=m['exam_id'],task='enhancing',git_sha=checkpoint_source,split_sha256=catalogue['split_sha256'],model_version='unet25d-full-cohort',checkpoint_ref=checkpoint_ref))
            (folder/'metrics.json').write_text(json.dumps(metrics));del prediction,native,native_logits,truth,canonical_truth;path.unlink();native_path.unlink()
            if world>1:dist.barrier()
    if world>1:
        gathered=[None]*world;dist.all_gather_object(gathered,results);results=[r for group in gathered for r in group]
    results=completed+results
    complete={r['exam_id'] for r in results}==expected
    patients=patient_aggregates(results);summary=dict(complete=complete,validation_patients=len(patients),validation_visits=len(results),mean_patient_dice=float(np.mean([r['dice'] for r in patients])) if complete else None,patients=patients,visits=results,test_evaluated=False)
    if len({r['exam_id'] for r in results})!=len(results):raise ValueError('Duplicate validation exams')
    summary['control']=control_metrics(results,complete)
    if complete:
        summary['secondary_threshold_grid']={str(t):control_metrics([dict(r,**r['threshold_grid'][str(t)]) for r in results],True) for t in [.3,.7]}
        for t,grid in summary['secondary_threshold_grid'].items():
            grid['threshold_probability']=float(t)
            # Size strata above belong to the primary matching; do not mislabel
            # them as measurements at a secondary threshold.
            # v2 records actual per-threshold matching strata above.
            grid['pooled_matched']=sum(r['threshold_grid'][t]['matched_lesions'] for r in results)
            grid['pooled_total']=sum(r['truth_lesions'] for r in results)
            grid['empty_prediction_exams']=sum(r['threshold_grid'][t]['empty_prediction'] for r in results)
    summary['pooled_matched']=sum(r['matched_lesions'] for r in results)
    summary['pooled_total']=sum(r['truth_lesions'] for r in results)
    summary['empty_prediction_exams']=sum(r['empty_prediction'] for r in results)
    summary['empty_prediction_fraction']=summary['empty_prediction_exams']/len(results) if results else None
    mask_counts=cache.mask_counts
    if world>1:
        audits=[None]*world;dist.all_gather_object(audits,mask_counts)
        mask_counts={key:sum(a.get(key,0) for a in audits) for key in set().union(*(a.keys() for a in audits))}
    summary['mask_audit']=mask_counts
    if rank==0:(output/'validation.json').write_text(json.dumps(summary))
    model.train(was_training)
    return summary
