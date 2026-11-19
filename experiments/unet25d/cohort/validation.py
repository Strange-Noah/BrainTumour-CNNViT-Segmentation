"""Full validation-patient evaluation after meaningful training, never diagnostic-wide evaluation."""
import gzip,io,json,tempfile
from pathlib import Path
import nibabel as nib
import numpy as np
import psutil
import torch
from ..data import restore
from research.evaluate import evaluate,export,patient_aggregates
from .tiling import windows,merge_logits

def validate_patients(model,catalogue,cache,fetch,output,checkpoint_source,checkpoint_ref,completed_coverage_epochs,deadline=float('inf'),previous_results=None):
    if completed_coverage_epochs<1:raise ValueError('Full validation is scheduled after meaningful full-cohort training, not a diagnostic checkpoint')
    visits=[v for v in catalogue['visits'] if v['partition']=='validation']
    if len({v['metadata']['patient_id'] for v in visits})!=45:raise ValueError('Full frozen validation cohort required')
    import time,torch.distributed as dist
    world=dist.get_world_size() if dist.is_initialized() else 1;rank=dist.get_rank() if dist.is_initialized() else 0
    raw=model.module if hasattr(model,'module') else model
    if world>1:
        for buffer in raw.buffers():dist.broadcast(buffer,src=0)
    completed=list(previous_results or []);done={r['exam_id'] for r in completed};visits=[v for v in visits if v['metadata']['exam_id'] not in done]
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
            prediction.flush();native=restore(prediction,m,order=1)>=0
            native_path=folder/'native-labels.npy.gz';fetch(visit['native_labels']['key'],visit['native_labels']['sha256'],native_path)
            with gzip.open(native_path,'rb') as f:truth=np.load(f,allow_pickle=False)==3
            reference=nib.Nifti1Image(np.empty(m['native_shape'],np.uint8),np.asarray(m['native_affine']));reference.header.set_xyzt_units('mm')
            metrics=evaluate(native,truth,reference.affine);results.append(dict(patient_id=m['patient_id'],exam_id=m['exam_id'],**metrics))
            export(native,reference,folder/'prediction',dict(patient_id=m['patient_id'],exam_id=m['exam_id'],task='enhancing',git_sha=checkpoint_source,split_sha256=catalogue['split_sha256'],model_version='unet25d-full-cohort',checkpoint_ref=checkpoint_ref))
            (folder/'metrics.json').write_text(json.dumps(metrics));del prediction,native,truth;path.unlink();native_path.unlink()
            if world>1:dist.barrier()
    if world>1:
        gathered=[None]*world;dist.all_gather_object(gathered,results);results=[r for group in gathered for r in group]
    results=completed+results
    complete=len(results)==len([v for v in catalogue['visits'] if v['partition']=='validation'])
    patients=patient_aggregates(results);summary=dict(complete=complete,validation_patients=len(patients),validation_visits=len(results),mean_patient_dice=float(np.mean([r['dice'] for r in patients])) if complete else None,patients=patients,visits=results,test_evaluated=False)
    if rank==0:(output/'validation.json').write_text(json.dumps(summary))
    model.train(was_training)
    return summary
