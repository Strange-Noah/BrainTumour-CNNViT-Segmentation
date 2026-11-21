"""One v2 GPU campaign: real positive learning, independent Phase 1, native validation."""
import argparse,gzip,hashlib,json,os,random,time,traceback
from pathlib import Path
from collections import Counter
import numpy as np
import torch
import torch.distributed as dist
import psutil
import segmentation_models_pytorch as smp
from ...train import fp32_step,RUNTIME_PRECISION
from ..stream import SignedFetch
from .settings import config,science_hash,SCHEMA,validate_catalogue
from .loading import TrainingInput,Cache,collate
from .loss import balanced_loss
from .sampling import digest,commit_update
from .state import initial,require_resume

class Model(smp.Unet):
    def __init__(self,c):
        super().__init__(encoder_name='resnet34',encoder_weights='imagenet',in_channels=12,classes=1,activation=None)
        for m in self.modules():
            if isinstance(m,torch.nn.modules.batchnorm._BatchNorm):m.momentum=c['bn_momentum']

def make(c,local):
    torch.manual_seed(c['seed']);random.seed(c['seed']);np.random.seed(c['seed'])
    model=Model(c).float().cuda(local).train()
    optimizer=torch.optim.AdamW([dict(params=model.encoder.parameters(),lr=c['encoder_lr']),dict(params=list(model.decoder.parameters())+list(model.segmentation_head.parameters()),lr=c['decoder_lr'])],weight_decay=c['weight_decay'])
    return model,optimizer

def health(model,optimizer,c,local):
    tensors=list(model.state_dict().values())+[v for s in optimizer.state.values() for v in s.values() if torch.is_tensor(v)]
    finite=all(bool(torch.isfinite(v).all()) for v in tensors)
    variances=[float(m.running_var.max()) for m in model.modules() if isinstance(m,torch.nn.BatchNorm2d)]
    means=[float(m.running_mean.abs().max()) for m in model.modules() if isinstance(m,torch.nn.BatchNorm2d)]
    good=torch.tensor(int(finite and max(variances,default=0)<c['bn_variance_limit']),device=local);dist.all_reduce(good,op=dist.ReduceOp.MIN)
    if not bool(good):raise FloatingPointError('Nonfinite model/optimizer or unsafe BN variance; retain prior checkpoint')
    memory=psutil.virtual_memory();pressure=torch.tensor(int(memory.available/memory.total<.15),device=local);dist.all_reduce(pressure,op=dist.ReduceOp.MAX)
    if bool(pressure):raise MemoryError('Actual host RAM pressure')
    return dict(BN_running_variance_max=max(variances,default=0),BN_running_mean_abs_max=max(means,default=0),host_ram_available_fraction=memory.available/memory.total)

def fixed_rows(plan,inventory,c):
    rows=[]
    for scale in (512,128):
        candidates=[r for r in plan if r['kind']=='replay' and r['scale']==scale]
        selected=None
        for r in candidates:
            component=next(x for x in inventory[r['exam']]['components'] if x['id']==r['component'])
            if scale==512 and component['planes'][str(r['z'])]['voxels']<c['capacity_fullscale_min_component_plane_pixels']:continue
            if scale==128 and component['volume_mm3']>=c['capacity_small_lesion_max_mm3']:continue
            selected=r;break
        if selected is None:raise ValueError('Missing representative full-scale/tiny-lesion capacity examples')
        rows.append(selected)
    for kind in ('negative_enhancing','negative_empty'):rows.append(next(r for r in plan if r['negative_category']==kind))
    return rows

def measure(raw,examples,device,c):
    was=raw.training;raw.eval();results=[]
    with torch.no_grad():
        for x,y,m,w,identity in examples:
            x,y,m,w=[v[None].to(device) for v in (x,y,m,w)];logits=raw(x);loss,parts=balanced_loss(logits,y,m,w,True);p=logits.sigmoid();positive=y.bool();negative=m.bool()&~positive;pred=(p>=.5)&m.bool();intersection=(pred&positive).sum()
            results.append(dict(target_scope='all enhancing components in crop',selected_component_foreground_pixels=identity.get('selected_component_foreground_pixels'),incidental_foreground_pixels=identity.get('incidental_foreground_pixels'),selected_component_BCE_share=identity.get('selected_component_BCE_share'),positive=bool(positive.any()),scale=identity['scale'],loss=float(loss),positive_bce=float(parts['positive_bce']),negative_bce=float(parts['negative_bce']),positive_probability=float(p[positive].mean()) if positive.any() else None,negative_probability=float(p[negative].mean()) if negative.any() else None,foreground_dice=float(2*intersection/(pred.sum()+positive.sum()).clamp_min(1)) if positive.any() else None,predicted_positive_pixels=int(pred.sum()),target_pixels=int(positive.sum()),valid_pixels=int(m.sum()),negative_predicted_fraction=float(pred.sum()/m.sum()) if not positive.any() else None))
    raw.train(was)
    pos=[r for r in results if r['positive']];neg=[r for r in results if not r['positive']]
    passed=all(r['positive_probability']>=c['learning_gate_positive_probability'] and r['foreground_dice']>=c['learning_gate_dice'] and r['predicted_positive_pixels']>0 for r in pos) and all(r['negative_probability']<=c['learning_gate_negative_probability_max'] and r['negative_predicted_fraction']<=c['learning_gate_negative_predicted_fraction_max'] for r in neg)
    return dict(passed=passed,examples=results,positive_learning=any(r['positive_probability']>=.5 and r['foreground_dice']>=.01 for r in pos) and all(r['negative_predicted_fraction']<.5 for r in neg),not_just_total_loss=True)

def learning_gate(model,optimizer,examples,c,local,rank,output,deadline):
    first=measure(model.module,examples,local,c);started=time.time();step=0;latest=first
    for step in range(c['learning_gate_max_steps']):
        # Paired positive/negative updates expose both ranks to both classes.
        sample=examples[((step//2)*2+rank+step%2)%len(examples)];x,y,m,w,_=sample;x,y,m,w=[v[None].to(local) for v in (x,y,m,w)]
        optimizer.zero_grad(set_to_none=True);logits=model(x);loss=balanced_loss(logits,y,m,w);loss.backward();flags=torch.tensor(int(bool(torch.isfinite(loss)) and all(bool(torch.isfinite(p.grad).all()) for p in model.parameters() if p.grad is not None)),device=local);dist.all_reduce(flags,op=dist.ReduceOp.MIN);fp32_step(optimizer,bool(flags),bool(flags))
        if (step+1)%50==0:
            status=health(model.module,optimizer,c,local);latest=measure(model.module,examples,local,c)
            passed=torch.tensor(int(latest['passed']),device=local);dist.all_reduce(passed,op=dist.ReduceOp.MIN)
            if rank==0:print(json.dumps(dict(event='v2_learning_gate',diagnostic_step=step+1,metrics=latest,health=status)),flush=True)
            if bool(passed):break
        stop=torch.tensor(int(time.time()-started>=c['learning_gate_seconds'] or time.time()>=deadline),device=local);dist.all_reduce(stop,op=dist.ReduceOp.MAX)
        if bool(stop):break
    # Validation uses rank-0 running buffers, exactly as formal native validation.
    for buffer in model.module.buffers():dist.broadcast(buffer,0)
    latest=measure(model.module,examples,local,c);accepted=torch.tensor(int(latest['passed']),device=local);dist.all_reduce(accepted,op=dist.ReduceOp.MIN)
    result=dict(initial=first,final=latest,passed=bool(accepted),diagnostic_updates=step+1,elapsed_seconds=time.time()-started,formal_updates_carried_over=0,source_sha=os.environ['NOVO_SOURCE_SHA'],science_sha256=science_hash(),framework=dict(torch=str(torch.__version__),cuda=torch.version.cuda,smp=str(smp.__version__)))
    if rank==0:
        (output/'GPU_LEARNING_GATE.json').write_text(json.dumps(result));torch.save(dict(diagnostic_only=True,science_sha256=science_hash(),model=model.module.state_dict(),optimizer=optimizer.state_dict()),output/'learning-diagnostic.pt')
    if rank==0:print(json.dumps(dict(event='v2_gate1_complete',result=result)),flush=True)
    dist.barrier()
    return result

def mine(logits,y,m,identity,step):
    with torch.no_grad():
        prob=logits.detach().sigmoid();neg=prob[m.bool()&~y.bool()];top=float(neg.topk(min(256,neg.numel())).values.mean()) if neg.numel() else 0.;peak=float(neg.max()) if neg.numel() else 0.
        return dict(exam=identity['exam'],patient=identity['patient'],z=identity['z'],origin=identity['origin'],scale=identity['scale'],selected_component=identity.get('component'),actual_positive=identity['actual_positive'],hard_negative_eligible=not identity['actual_positive'],negative_category=identity['negative_category'],source_size=identity['source_size'],zoom_scale=identity['zoom_scale'],mask_fallback=identity['mask_fallback'],peak_false_positive=peak,topk_false_positive=top,score=.5*peak+.5*top,step=step,training_only=True)

def early_validation(raw,catalogue,cache,fetch,output,deadline):
    """Frozen first two validation exams; same full-FOV inference, no GT localization."""
    from ..tiling import windows,merge_logits
    from ...data import restore
    from research.evaluate import evaluate
    import nibabel as nib
    rows=[];was=raw.training;raw.eval();device=next(raw.parameters()).device
    with torch.inference_mode():
        for visit in sorted([v for v in catalogue['visits'] if v['partition']=='validation'],key=lambda v:v['metadata']['exam_id'])[:2]:
            if time.time()>deadline-120:break
            m=visit['metadata'];arr=np.empty(m['shape'],np.float32)
            for z in range(m['shape'][2]):
                if time.time()>deadline:raw.train(was);return dict(complete=False,visits=rows)
                tiles=[]
                for origin in windows(m['shape']):
                    x,_=cache.sample(visit,origin,z);logits=raw(x[None].to(device))
                    if not torch.isfinite(logits).all():raise FloatingPointError('Early validation logits')
                    tiles.append((origin,logits[0,0].cpu().numpy()))
                arr[:,:,z]=merge_logits(tiles,m['shape'])
            pred=restore(arr,m,order=1)>=0;path=output/'early-native-labels.npy.gz';fetch(visit['native_labels']['key'],visit['native_labels']['sha256'],path)
            with gzip.open(path,'rb') as f:truth=np.load(f,allow_pickle=False)==3
            rows.append(dict(exam=m['exam_id'],**evaluate(pred,truth,np.asarray(m['native_affine']))));path.unlink()
    raw.train(was);return dict(complete=len(rows)==2,visits=rows,diagnostic_only=True,model_selection=False)

def main():
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--session-seconds',type=int,required=True);p.add_argument('--approved-budget-seconds',type=int,required=True);p.add_argument('--resume',type=Path);a=p.parse_args()
    c=config();rank=int(os.environ['RANK']);local=int(os.environ['LOCAL_RANK']);world=int(os.environ['WORLD_SIZE'])
    if world!=2 or torch.cuda.device_count()!=2 or not all('T4' in torch.cuda.get_device_name(i) for i in (0,1)) or not 1200<=a.session_seconds<=43200 or a.approved_budget_seconds<a.session_seconds:raise ValueError('Approved free dual-T4 budget/configuration required')
    torch.cuda.set_device(local);dist.init_process_group('nccl');torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    started=float(os.environ['NOVO_SESSION_STARTED_AT']);deadline=started+a.session_seconds-600;output=a.output;output.mkdir(parents=True,exist_ok=True)
    os.environ['HF_HOME']=str(a.input/'hf');os.environ['HF_HUB_OFFLINE']='1'
    package=json.loads((a.input/'input-package.json').read_text())
    if package['science_sha256']!=science_hash() or package['source_sha']!=os.environ['NOVO_SOURCE_SHA']:raise ValueError('Pinned private input/source identity')
    for name,sha in package['files'].items():
        with (a.input/name).open('rb') as f:
            if hashlib.file_digest(f,'sha256').hexdigest()!=sha:raise ValueError('Private input integrity')
    catalogue=json.loads((a.input/'catalogue.json').read_text());validate_catalogue(catalogue);inventory=json.loads((a.input/'inventory.json').read_text());plan=json.loads((a.input/'plan.json').read_text());plan_sha=digest(plan);cat_sha=package['files']['catalogue.json'];inv_sha=package['files']['inventory.json'];lease=json.loads((a.input/'private-data-lease.json').read_text())
    if lease['catalogue_sha256']!=cat_sha:raise ValueError('Catalogue/lease mismatch')
    state=initial(cat_sha,inv_sha,plan_sha);model,optimizer=make(c,local)
    if a.resume:
        old=torch.load(a.resume,map_location='cpu',weights_only=True);require_resume(old,cat_sha,inv_sha,plan_sha);model.load_state_dict(old.pop('model'),strict=True);optimizer.load_state_dict(old.pop('optimizer'));rng=old.pop('rank_rng')[rank];torch.set_rng_state(rng['cpu']);torch.cuda.set_rng_state(rng['cuda'],local);random.setstate(rng['python']);state=old
        if {float(s['step']) for s in optimizer.state.values()}!={float(state['step'])}:raise ValueError('Optimizer resume step mismatch')
    consumed=plan[:state['cursor']]
    if sum(r['kind']=='mandatory' for r in consumed)!=state['mandatory_unique'] or sum(r['positive'] for r in consumed)!=state['positive_samples']:raise ValueError('Checkpoint prefix coverage mismatch')
    if plan_sha!=json.loads((a.input/'SAMPLING_AUDIT.json').read_text())['plan_SHA']:raise ValueError('Frozen accepted schedule identity mismatch')
    model=torch.nn.parallel.DistributedDataParallel(model,device_ids=[local],broadcast_buffers=True)
    fetch=SignedFetch(lease);fixed_cache=Cache(output/'fixed-cache'/str(rank),fetch,c,inventory);train_visits={v['metadata']['exam_id']:v for v in catalogue['visits'] if v['partition']=='train'}
    examples=[fixed_cache.sample_v2(train_visits[r['exam']],r) for r in fixed_rows(plan,inventory,c)]
    if not state['learning_gate_passed']:
        gate=learning_gate(model,optimizer,examples,c,local,rank,output,deadline)
        if not gate['passed']:
            if rank==0:print(json.dumps(dict(event='v2_gate1_failed',metrics=gate)),flush=True)
            dist.destroy_process_group();raise RuntimeError('Real positive-learning failed; no formal Phase1 and no speculative GPU retry')
        del model,optimizer;torch.cuda.empty_cache();model,optimizer=make(c,local);model=torch.nn.parallel.DistributedDataParallel(model,device_ids=[local],broadcast_buffers=True);state['learning_gate_passed']=True;state['learning_gate']=gate
    del fixed_cache
    prior=float(state['consumed_seconds']);deadline=min(deadline,started+a.approved_budget_seconds-prior-600);last_save=time.time();telemetry_start=time.monotonic();samples=0;wait=0.;previous=time.monotonic();pending=[];local_exposure=Counter();last_health={}
    def save():
        nonlocal pending,local_exposure
        health(model.module,optimizer,c,local)
        gathered=[None,None];dist.all_gather_object(gathered,pending);pending=[]
        measured=[None,None];dist.all_gather_object(measured,dict(local_exposure));local_exposure=Counter()
        for counts in measured:
            for key,value in counts.items():state['exposures'][key]=state['exposures'].get(key,0)+value
        for group in gathered:
            for row in group:
                key=row['exam']+'|'+row['negative_category'] if row['negative_category'] else row['exam']+'|positive_background'
                records=state['difficulty'].setdefault(key,[]);records.append(row);records.sort(key=lambda r:r['score'],reverse=True);del records[32:]
        rng=dict(cpu=torch.get_rng_state(),cuda=torch.cuda.get_rng_state(local),python=random.getstate());all_rng=[None,None];dist.all_gather_object(all_rng,rng)
        if optimizer.state and {float(s['step']) for s in optimizer.state.values()}!={float(state['step'])}:raise ValueError('Committed optimizer/checkpoint mismatch')
        if rank==0:
            record=dict(state,model=model.module.state_dict(),optimizer=optimizer.state_dict(),rank_rng=all_rng,runtime_precision=RUNTIME_PRECISION,source_sha=os.environ['NOVO_SOURCE_SHA'],config=c,framework=dict(torch=str(torch.__version__),cuda=torch.version.cuda,smp=str(smp.__version__),devices=[torch.cuda.get_device_name(i) for i in range(2)]),consumed_seconds=prior+time.time()-started)
            tmp=output/'checkpoint.tmp';torch.save(record,tmp);tmp.replace(output/'checkpoint.pt')
            public_keys=['data_mode','science_sha256','catalogue_sha256','inventory_sha256','plan_sha256','step','cursor','mandatory_unique','replay_consumed','positive_samples','negative_samples','phase','pending_validation','scientific_stop','learning_gate_passed','early_diagnostic_complete']
            meta={k:record[k] for k in public_keys};meta['consumed_seconds']=record['consumed_seconds'];meta['framework']=record['framework'];(output/'training-state.json').write_text(json.dumps(meta));(output/'exposures.json').write_text(json.dumps(state['exposures']));(output/'difficulty.json').write_text(json.dumps(state['difficulty']))
            print(json.dumps(dict(event='checkpoint_saved',step=state['step'],mandatory_unique=state['mandatory_unique'],replay_consumed=state['replay_consumed'],checkpoint_bytes=(output/'checkpoint.pt').stat().st_size)),flush=True)
        dist.barrier()
    try:
        while not state['scientific_stop']:
            expired=torch.tensor(int(time.time()>=deadline),device=local);dist.all_reduce(expired,op=dist.ReduceOp.MAX)
            if bool(expired):break
            if state['pending_validation']:
                from .validation import validate_patients
                result=validate_patients(model,catalogue,Cache(output/'validation-cache'/str(rank),fetch,c),fetch,output/'validation',os.environ['NOVO_SOURCE_SHA'],'checkpoint.pt',1,deadline,state['validation_progress'])
                state['validation_progress']=result['visits']
                if result['complete']:
                    state['pending_validation']=False;state['phase']='AWAIT_PHASE2_DESIGN';state['scientific_stop']='AWAIT_PHASE2_DESIGN'
                    if rank==0:(output/'PHASE1_RESULT.json').write_text(json.dumps(dict(complete=True,science_sha256=science_hash(),mandatory_unique=state['mandatory_unique'],replay_consumed=state['replay_consumed'],control=result['control'],secondary_threshold_grid=result['secondary_threshold_grid'],pooled_matched=result['pooled_matched'],pooled_total=result['pooled_total'],empty_prediction_exams=result['empty_prediction_exams'],status='AWAIT_PHASE2_DESIGN',test_evaluated=False)))
                save();break
            if state['mandatory_unique']>=int(c['early_mandatory_fraction']*c['mandatory_positions']) and not state['early_diagnostic_complete']:
                for buffer in model.module.buffers():dist.broadcast(buffer,0)
                check=measure(model.module,examples,local,c);passed=torch.tensor(int(check['positive_learning']),device=local);dist.all_reduce(passed,op=dist.ReduceOp.MIN)
                if not bool(passed):state['scientific_stop']='EARLY_POSITIVE_LEARNING_FAILURE';save();break
                postpone=torch.tensor(int(time.time()>deadline-300),device=local);dist.all_reduce(postpone,op=dist.ReduceOp.MAX)
                if bool(postpone):save();break
                for buffer in model.module.buffers():dist.broadcast(buffer,0)
                early=None
                if rank==0:
                    early=early_validation(model.module,catalogue,Cache(output/'early-cache',fetch,c),fetch,output,deadline);(output/'early-diagnostic.json').write_text(json.dumps(dict(training=check,validation=early)))
                payload=[early];dist.broadcast_object_list(payload,0);state['early_diagnostic_complete']=payload[0]['complete'];dist.barrier();save()
                if not state['early_diagnostic_complete']:break
            dataset=TrainingInput(catalogue,inventory,plan,lease,output/'cache',c,state['cursor'],rank)
            loader=torch.utils.data.DataLoader(dataset,batch_size=1,num_workers=c['workers_per_rank'],persistent_workers=True,prefetch_factor=c['prefetch_batches'],pin_memory=True,multiprocessing_context='spawn',collate_fn=collate,generator=torch.Generator().manual_seed(c['seed']))
            for x,y,m,w,identity in loader:
                wait+=time.monotonic()-previous;x,y,m,w=[v.cuda(local,non_blocking=True) for v in (x,y,m,w)];optimizer.zero_grad(set_to_none=True);logits=model(x);loss,parts=balanced_loss(logits,y,m,w,True);loss.backward();flags=torch.tensor(int(bool(torch.isfinite(loss)) and all(bool(torch.isfinite(p.grad).all()) for p in model.parameters() if p.grad is not None)),device=local);dist.all_reduce(flags,op=dist.ReduceOp.MIN);fp32_step(optimizer,bool(flags),bool(flags));pair=plan[state['cursor']:state['cursor']+2];commit_update(state,pair);samples+=2
                # Only completed optimizer updates count; workers/prefetch never
                # advance coverage. Per-example foreground and lesion pixels are
                # private measured exposure, distinct from planned categories.
                pending.append(mine(logits,y,m,identity,state['step']))
                for row in pair:
                    e=state['exposures'];e['patient:'+row['patient']]=e.get('patient:'+row['patient'],0)+1;e['visit:'+row['exam']]=e.get('visit:'+row['exam'],0)+1;e['scale:'+str(row['scale'])]=e.get('scale:'+str(row['scale']),0)+1
                    if row['kind']=='replay':key='selected_lesion:'+row['exam']+':'+str(row['component']);e[key]=e.get(key,0)+1
                local_exposure['rank'+str(rank)+'_foreground_pixels']+=identity['foreground_pixels'];local_exposure['rank'+str(rank)+'_valid_pixels']+=identity['valid_pixels']
                local_exposure['mask_fallbacks']+=int(identity['mask_fallback'] is not None)
                if identity['mask_fallback'] is not None:
                    local_exposure['mask_exception:'+identity['exam']+':'+str(identity['z'])+':'+identity['mask_fallback']]+=1
                for label,lesion in identity['lesions'].items():
                    local_exposure['incident_lesion:'+identity['exam']+':'+label]+=1
                    size='lt100' if lesion['volume_mm3']<100 else '100to1000' if lesion['volume_mm3']<1000 else 'ge1000'
                    local_exposure['lesion_size_exposure:'+size]+=1
                if state['step']%100==0 or state['step']==1:
                    last_health=health(model.module,optimizer,c,local);elapsed=time.monotonic()-telemetry_start
                    if rank==0:print(json.dumps(dict(event='training_update',step=state['step'],unix_time=time.time(),session_started_at=started,phase=state['phase'],mandatory_unique=state['mandatory_unique'],mandatory_total=c['mandatory_positions'],replay_consumed=state['replay_consumed'],samples_per_second=samples/elapsed,loader_wait_fraction=wait/elapsed,positive_bce=float(parts['positive_bce']),negative_bce=float(parts['negative_bce']),soft_dice=float(parts['soft_dice']),loss=float(loss),health=last_health,gpu_memory_allocated_bytes=torch.cuda.memory_allocated(local))),flush=True)
                    telemetry_start=time.monotonic();samples=0;wait=0.
                stop=torch.tensor([int(time.time()>=deadline),int(time.time()-last_save>=c['checkpoint_seconds'])],device=local);dist.all_reduce(stop,op=dist.ReduceOp.MAX)
                early_due=state['mandatory_unique']>=int(c['early_mandatory_fraction']*c['mandatory_positions']) and not state['early_diagnostic_complete']
                if bool(stop[1]):save();last_save=time.time()
                if bool(stop[0]) or early_due:break
                previous=time.monotonic()
            del loader
            if state['cursor']==len(plan):
                if state['mandatory_unique']!=c['mandatory_positions']:raise ValueError('Incomplete mandatory coverage')
                state['pending_validation']=True;state['phase']='PHASE1_VALIDATION'
            save()
        save()
    except BaseException:
        traceback.print_exc();os._exit(1)
    finally:dist.destroy_process_group()

if __name__=='__main__':main()
