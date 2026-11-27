"""V3 GPU entry point: real positive learning, independent Phase 1, native validation."""
import argparse,hashlib,json,os,random,re,time,traceback
from pathlib import Path
from unittest.mock import patch
from collections import Counter
import numpy as np
import torch
import torch.distributed as dist
import psutil
import segmentation_models_pytorch as smp
from ...train import fp32_step,RUNTIME_PRECISION
from ..v2.resilient_fetch import SignedFetch
from .settings import config,science_hash,validate_catalogue
from ..v2.loading import TrainingInput,Cache,collate
from .loss import masked_loss
from ..v2.sampling import digest,commit_update
from .state import complete_phase1,initial,load_checkpoint,plain_state

class Model(smp.Unet):
    def __init__(self,c):
        super().__init__(encoder_name='resnet34',encoder_weights='imagenet',in_channels=12,classes=1,activation=None)
        for m in self.modules():
            if isinstance(m,torch.nn.modules.batchnorm._BatchNorm):m.momentum=c['bn_momentum']

def pretrained_model(c, snapshot):
    # SMP 0.5.0 loads these two files before its native 3-to-12 expansion.
    # Bind the reviewed offline files so its internet fallback is never needed.
    settings = smp.encoders.encoders['resnet34']['pretrained_settings']['imagenet']
    if settings['revision'] != c['encoder_checkpoint_revision']:
        raise ValueError('Unexpected pretrained encoder revision')

    def offline(repo_id, filename, revision):
        if (repo_id != settings['repo_id'] or revision != settings['revision']
                or filename not in ('config.json', 'model.safetensors')):
            raise ValueError('Unexpected encoder dependency')
        return str(snapshot / filename)

    def reject_fallback(*args, **kwargs):
        raise ValueError('Pinned offline encoder could not be loaded')

    with patch.object(smp.encoders, 'hf_hub_download', offline), patch.object(
            smp.encoders, 'load_url', reject_fallback):
        return Model(c)


def make(c,local,snapshot):
    torch.manual_seed(c['seed']);random.seed(c['seed']);np.random.seed(c['seed'])
    model=pretrained_model(c,snapshot).float().cuda(local).train()
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
    devices=None
    if dist.get_rank()==0:
        import subprocess
        try:
            result=subprocess.run(['nvidia-smi','--query-gpu=index,utilization.gpu,memory.used,memory.total','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=3)
            if result.returncode==0:devices=[[int(v.strip()) for v in line.split(',')] for line in result.stdout.strip().splitlines()]
        except (OSError,ValueError,subprocess.TimeoutExpired):pass
    return dict(gpu_utilization_memory=devices,BN_running_variance_max=max(variances,default=0),BN_running_mean_abs_max=max(means,default=0),host_ram_available_fraction=memory.available/memory.total)

def fixed_rows(plan,inventory,c):
    rows=[]
    for scale in (512,128):
        candidates=[r for r in plan if r['kind']=='replay' and r['scale']==scale]
        eligible=[]
        for r in candidates:
            comp=next(x for x in inventory[r['exam']]['components'] if x['id']==r['component'])
            if scale==512 and comp['planes'][str(r['z'])]['voxels']<c['capacity_fullscale_min_component_plane_pixels']:continue
            if scale==128 and comp['volume_mm3']>=c['capacity_small_lesion_max_mm3']:continue
            eligible.append(r)
            if scale==512:break
        if not eligible:raise ValueError('Missing real capacity example')
        chosen=eligible[0]
        if scale==128:
            stress=dict(chosen,gate_role='stress_composite')
            for r in eligible:
                lo=np.asarray(r['origin']);hi=lo+128
                others=[x for x in inventory[r['exam']]['components'] if x['id']!=r['component'] and str(r['z']) in x['planes']]
                if not any(np.all(np.asarray(x['planes'][str(r['z'])]['lo'])<hi) and np.all(np.asarray(x['planes'][str(r['z'])]['hi'])>lo) for x in others):chosen=r;break
        rows.append(dict(chosen,gate_role='native_capacity' if scale==512 else 'tiny_capacity'))
    for kind in ('negative_enhancing','negative_empty'):rows.append(dict(next(r for r in plan if r['negative_category']==kind),gate_role=kind))
    rows.append(stress)
    return rows

def measure(raw,examples,device,c):
    was=raw.training;raw.eval();results=[]
    with torch.no_grad():
        for x,y,m,w,identity in examples:
            x,y,m,w=[v[None].to(device) for v in (x,y,m,w)];logits=raw(x);loss,parts=masked_loss(logits,y,m,w,True);p=logits.sigmoid();positive=y.bool();negative=m.bool()&~positive;pred=(p>=.5)&m.bool();intersection=(pred&positive).sum()
            component_metrics=[]
            for component,cm in identity.get('gate_component_masks',{}).items():
                cm=cm[None].to(device).bool();tp=(pred&cm).sum();other=positive&~cm;component_prediction=pred&~other
                component_metrics.append(dict(selected=component==identity.get('component'),target_pixels=int(cm.sum()),probability=float(p[cm].mean()),recall=float(tp/cm.sum()),dice=float(2*tp/(component_prediction.sum()+cm.sum()).clamp_min(1)),bce_coefficient=float(w[cm].sum()/m.sum())))
            results.append(dict(gate_role=identity.get('gate_role'),components=component_metrics,target_scope='all enhancing components in crop',selected_component_foreground_pixels=identity.get('selected_component_foreground_pixels'),incidental_foreground_pixels=identity.get('incidental_foreground_pixels'),selected_component_BCE_coefficient=next((r['bce_coefficient'] for r in component_metrics if r['selected']),None),positive=bool(positive.any()),scale=identity['scale'],loss=float(loss),bce=float(parts['bce']),positive_bce=float(parts['positive_bce']),negative_bce=float(parts['negative_bce']),positive_probability=float(p[positive].mean()) if positive.any() else None,negative_probability=float(p[negative].mean()) if negative.any() else None,foreground_dice=float(2*intersection/(pred.sum()+positive.sum()).clamp_min(1)) if positive.any() else None,predicted_positive_pixels=int(pred.sum()),target_pixels=int(positive.sum()),valid_pixels=int(m.sum()),negative_predicted_fraction=float(pred.sum()/m.sum()) if not positive.any() else None))
    raw.train(was)
    pos=[r for r in results if r['positive'] and r['gate_role']!='stress_composite'];neg=[r for r in results if not r['positive']]
    for r in pos:
        if r['gate_role']=='tiny_capacity':
            selected=next(x for x in r['components'] if x['selected']);r['capacity_probability']=selected['probability'];r['capacity_dice']=selected['dice'];r['capacity_recall']=selected['recall']
        else:r['capacity_probability']=r['positive_probability'];r['capacity_dice']=r['foreground_dice'];r['capacity_recall']=1.
    passed=all(r['capacity_probability']>=c['learning_gate_positive_probability'] and r['capacity_dice']>=c['learning_gate_dice'] and r['capacity_recall']>=c['learning_gate_selected_recall_min'] and r['predicted_positive_pixels']>0 for r in pos) and all(r['negative_probability']<=c['learning_gate_negative_probability_max'] and r['negative_predicted_fraction']<=c['learning_gate_negative_predicted_fraction_max'] for r in neg)
    return dict(passed=passed,examples=results,positive_learning=any(r['positive_probability']>=.5 and r['foreground_dice']>=.01 for r in pos) and all(r['negative_predicted_fraction']<.5 for r in neg),not_just_total_loss=True)

def learning_gate(model,optimizer,examples,c,local,rank,output,deadline):
    started=time.time();first=measure(model.module,examples,local,c);latest=first;rank_history=[]
    gate_deadline=min(started+c['learning_gate_seconds'],deadline)
    completed=0
    for step in range(c['learning_gate_max_steps']):
        stop=torch.tensor(int(time.time()>=gate_deadline),device=local);dist.all_reduce(stop,op=dist.ReduceOp.MAX)
        if bool(stop):break
        # Paired positive/negative updates expose both ranks to both classes.
        sample=examples[((step//2)*2+rank+step%2)%4];x,y,m,w,_=sample;x,y,m,w=[v[None].to(local) for v in (x,y,m,w)]
        optimizer.zero_grad(set_to_none=True);logits=model(x);loss=masked_loss(logits,y,m,w);loss.backward();flags=torch.tensor(int(bool(torch.isfinite(loss)) and all(bool(torch.isfinite(p.grad).all()) for p in model.parameters() if p.grad is not None)),device=local);dist.all_reduce(flags,op=dist.ReduceOp.MIN);fp32_step(optimizer,bool(flags),bool(flags))
        completed=step+1
        if (step+1)%c['learning_gate_measure_every']==0:
            status=health(model.module,optimizer,c,local)
            for buffer in model.module.buffers():dist.broadcast(buffer,0)
            latest=measure(model.module,examples,local,c)
            passed=torch.tensor(int(latest['passed']),device=local);dist.all_reduce(passed,op=dist.ReduceOp.MIN)
            rank_metrics=[None,None];dist.all_gather_object(rank_metrics,dict(rank=rank,metrics=latest,health=status));rank_history.append(dict(step=step+1,ranks=rank_metrics))
            if rank==0:print(json.dumps(dict(event='v3_learning_gate',diagnostic_step=step+1,metrics=latest,health=status,ranks=rank_metrics)),flush=True)
            if bool(passed):break
    # Preserve the frozen rank-0 BN policy and check the terminal Gate state.
    health(model.module,optimizer,c,local)
    for buffer in model.module.buffers():dist.broadcast(buffer,0)
    latest=measure(model.module,examples,local,c);accepted=torch.tensor(int(latest['passed']),device=local);dist.all_reduce(accepted,op=dist.ReduceOp.MIN)
    result=dict(initial=first,final=latest,rank_history=rank_history,bn_evaluation_policy='rank0 buffers broadcast before every capacity measurement on both ranks',passed=bool(accepted) and completed>0,diagnostic_updates=completed,elapsed_seconds=time.time()-started,formal_updates_carried_over=0,source_sha=os.environ['NOVO_SOURCE_SHA'],science_sha256=science_hash(),framework=dict(torch=str(torch.__version__),cuda=torch.version.cuda,smp=str(smp.__version__)))
    rank_rng=[None,None];dist.all_gather_object(rank_rng,dict(cpu=torch.get_rng_state(),cuda=torch.cuda.get_rng_state(local),python=random.getstate()))
    if rank==0:
        (output/'GPU_LEARNING_GATE.json').write_text(json.dumps(result));torch.save(dict(diagnostic_only=True,source_sha=os.environ['NOVO_SOURCE_SHA'],config=c,rank_rng=rank_rng,gate_result=result,science_sha256=science_hash(),model=model.module.state_dict(),optimizer=optimizer.state_dict()),output/'learning-diagnostic.pt')
    if rank==0:print(json.dumps(dict(event='v3_gate1_complete',result=result)),flush=True)
    dist.barrier()
    return result

def mine(logits,y,m,identity,step):
    with torch.no_grad():
        prob=logits.detach().sigmoid();neg=prob[m.bool()&~y.bool()];top=float(neg.topk(min(256,neg.numel())).values.mean()) if neg.numel() else 0.;peak=float(neg.max()) if neg.numel() else 0.
        return dict(exam=identity['exam'],patient=identity['patient'],z=identity['z'],origin=identity['origin'],scale=identity['scale'],selected_component=identity.get('component'),actual_positive=identity['actual_positive'],hard_negative_eligible=not identity['actual_positive'],negative_category=identity['negative_category'],source_size=identity['source_size'],zoom_scale=identity['zoom_scale'],mask_fallback=identity['mask_fallback'],peak_false_positive=peak,topk_false_positive=top,score=.5*peak+.5*top,step=step,training_only=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--session-seconds',type=int,required=True);p.add_argument('--approved-budget-seconds',type=int,required=True);p.add_argument('--resume',type=Path);a=p.parse_args()
    if os.environ.get('NOVO_V3_GPU_AUTHORIZED')!='true':
        raise PermissionError('Explicit V3 budget authorization required')
    source_sha=os.environ['NOVO_SOURCE_SHA']
    if not re.fullmatch(r'[0-9a-f]{40}', source_sha):
        raise ValueError('Pinned V3 source SHA required')
    c=config();rank=int(os.environ['RANK']);local=int(os.environ['LOCAL_RANK']);world=int(os.environ['WORLD_SIZE'])
    if world!=2 or torch.cuda.device_count()!=2 or not all('T4' in torch.cuda.get_device_name(i) for i in (0,1)) or not 1200<=a.session_seconds<=43200 or a.approved_budget_seconds<a.session_seconds:raise ValueError('Approved free dual-T4 budget/configuration required')
    torch.cuda.set_device(local);dist.init_process_group('nccl');torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    started=float(os.environ['NOVO_SESSION_STARTED_AT']);deadline=started+a.session_seconds-600;output=a.output
    package=json.loads((a.input/'input-package.json').read_text())
    if package['science_sha256']!=science_hash() or package['source_sha']!=os.environ['NOVO_SOURCE_SHA']:raise ValueError('Pinned private input/source identity')
    from .inputs import PLAN_SHA, encoder_snapshot, verify_manifest
    verify_manifest(package, a.input)
    encoder_dir=encoder_snapshot(a.input, package['files'])
    if (a.output.resolve().is_relative_to(a.input.resolve())
            or a.output.exists() and any(a.output.iterdir())):
        raise ValueError('V3 requires a fresh local output directory')
    if (a.input/'resume.pt').exists() != (a.resume is not None):
        raise ValueError('Packaged resume must be used explicitly')
    if a.resume is not None and a.resume.resolve() != (a.input/'resume.pt').resolve():
        raise ValueError('Resume only the checkpoint pinned by this V3 input')
    for name,sha in package['files'].items():
        with (a.input/name).open('rb') as f:
            if hashlib.file_digest(f,'sha256').hexdigest()!=sha:raise ValueError('Private input integrity')
    catalogue=json.loads((a.input/'catalogue.json').read_text());validate_catalogue(catalogue);inventory=json.loads((a.input/'inventory.json').read_text());plan=json.loads((a.input/'plan.json').read_text());plan_sha=digest(plan);cat_sha=package['files']['catalogue.json'];inv_sha=package['files']['inventory.json'];lease=json.loads((a.input/'private-data-lease.json').read_text())
    if lease['expires_at']-time.time()<a.session_seconds+300:raise ValueError('Input lease cannot cover session plus margin')
    if lease['catalogue_sha256']!=cat_sha:raise ValueError('Catalogue/lease mismatch')
    from .persistence import verify_lease
    verify_lease(a.input, started+a.session_seconds)
    if plan_sha!=PLAN_SHA or plan_sha!=json.loads((a.input/'SAMPLING_AUDIT.json').read_text())['plan_SHA']:
        raise ValueError('Frozen accepted schedule identity mismatch')
    dist.barrier()
    output.mkdir(parents=True,exist_ok=True)
    if rank==0:
        startup=dict(event='v3_gpu_allocated',source_sha=os.environ['NOVO_SOURCE_SHA'],science_sha256=science_hash(),world_size=world,devices=[torch.cuda.get_device_name(i) for i in range(2)],catalogue_sha256=cat_sha,plan_sha256=plan_sha)
        (output/'startup.json').write_text(json.dumps(startup));print(json.dumps(startup),flush=True)
        from .persistence import snapshot
        snapshot(output,a.input,'startup_transport_probe')
    dist.barrier()
    state=initial(cat_sha,inv_sha,plan_sha);model,optimizer=make(c,local,encoder_dir)
    if a.resume:
        old=load_checkpoint(a.resume,cat_sha,inv_sha,plan_sha,os.environ['NOVO_SOURCE_SHA']);model.load_state_dict(old.pop('model'),strict=True);optimizer.load_state_dict(old.pop('optimizer'));rng=old.pop('rank_rng')[rank];torch.set_rng_state(rng['cpu']);torch.cuda.set_rng_state(rng['cuda'],local);random.setstate(rng['python']);state=old
        if {float(s['step']) for s in optimizer.state.values()}!={float(state['step'])}:raise ValueError('Optimizer resume step mismatch')
    if state['scientific_stop'] is not None or state['pending_validation'] or state['validation_progress']:
        raise ValueError('This V3 campaign resumes only unfinished training')
    consumed=plan[:state['cursor']]
    if sum(r['kind']=='mandatory' for r in consumed)!=state['mandatory_unique'] or sum(r['positive'] for r in consumed)!=state['positive_samples']:raise ValueError('Checkpoint prefix coverage mismatch')
    model=torch.nn.parallel.DistributedDataParallel(model,device_ids=[local],broadcast_buffers=True)
    fetch=SignedFetch(lease);fixed_cache=Cache(output/'fixed-cache'/str(rank),fetch,c,inventory);train_visits={v['metadata']['exam_id']:v for v in catalogue['visits'] if v['partition']=='train'}
    examples=[fixed_cache.sample_v2(train_visits[r['exam']],r) for r in fixed_rows(plan,inventory,c)]
    if not state['learning_gate_passed']:
        gate=learning_gate(model,optimizer,examples,c,local,rank,output,deadline)
        if rank==0:
            from .persistence import snapshot
            snapshot(output,a.input,'gate1')
        dist.barrier()
        if not gate['passed']:
            if rank==0:print(json.dumps(dict(event='v3_gate1_failed',metrics=gate)),flush=True)
            dist.destroy_process_group();raise RuntimeError('Real positive-learning failed; no formal Phase1 and no speculative GPU retry')
        del model,optimizer;torch.cuda.empty_cache();model,optimizer=make(c,local,encoder_dir);model=torch.nn.parallel.DistributedDataParallel(model,device_ids=[local],broadcast_buffers=True);state['learning_gate_passed']=True;state['learning_gate']=gate
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
            record=dict(plain_state(state),model=model.module.state_dict(),optimizer=optimizer.state_dict(),rank_rng=all_rng,runtime_precision=RUNTIME_PRECISION,source_sha=os.environ['NOVO_SOURCE_SHA'],config=c,framework=dict(torch=str(torch.__version__),cuda=torch.version.cuda,smp=str(smp.__version__),devices=[torch.cuda.get_device_name(i) for i in range(2)]),consumed_seconds=prior+time.time()-started)
            tmp=output/'checkpoint.tmp';torch.save(record,tmp);tmp.replace(output/'checkpoint.pt')
            public_keys=['data_mode','science_sha256','catalogue_sha256','inventory_sha256','plan_sha256','step','cursor','mandatory_unique','replay_consumed','positive_samples','negative_samples','phase','pending_validation','scientific_stop','learning_gate_passed','early_diagnostic_complete']
            meta={k:record[k] for k in public_keys};meta['consumed_seconds']=record['consumed_seconds'];meta['framework']=record['framework'];(output/'training-state.json').write_text(json.dumps(meta));(output/'exposures.json').write_text(json.dumps(state['exposures']));(output/'difficulty.json').write_text(json.dumps(state['difficulty']))
            from .persistence import snapshot
            snapshot(output,a.input,'formal_checkpoint')
            print(json.dumps(dict(event='checkpoint_saved',step=state['step'],mandatory_unique=state['mandatory_unique'],replay_consumed=state['replay_consumed'],checkpoint_bytes=(output/'checkpoint.pt').stat().st_size)),flush=True)
        dist.barrier()
    try:
        while not state['scientific_stop']:
            expired=torch.tensor(int(time.time()>=deadline),device=local);dist.all_reduce(expired,op=dist.ReduceOp.MAX)
            if bool(expired):break
            dataset=TrainingInput(catalogue,inventory,plan,lease,output/'cache',c,state['cursor'],rank)
            loader=torch.utils.data.DataLoader(dataset,batch_size=1,num_workers=c['workers_per_rank'],persistent_workers=True,prefetch_factor=c['prefetch_batches'],pin_memory=True,multiprocessing_context='spawn',collate_fn=collate,generator=torch.Generator().manual_seed(c['seed']))
            for x,y,m,w,identity in loader:
                wait+=time.monotonic()-previous;x,y,m,w=[v.cuda(local,non_blocking=True) for v in (x,y,m,w)];optimizer.zero_grad(set_to_none=True);logits=model(x);loss,parts=masked_loss(logits,y,m,w,True);loss.backward();flags=torch.tensor(int(bool(torch.isfinite(loss)) and all(bool(torch.isfinite(p.grad).all()) for p in model.parameters() if p.grad is not None)),device=local);dist.all_reduce(flags,op=dist.ReduceOp.MIN);fp32_step(optimizer,bool(flags),bool(flags));pair=plan[state['cursor']:state['cursor']+2];commit_update(state,pair);samples+=2
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
                    if rank==0:
                        telemetry=dict(event='training_update',step=state['step'],unix_time=time.time(),session_started_at=started,phase=state['phase'],mandatory_unique=state['mandatory_unique'],mandatory_total=c['mandatory_positions'],replay_consumed=state['replay_consumed'],samples_per_second=samples/elapsed,loader_wait_fraction=wait/elapsed,bce=float(parts['bce']),positive_bce=float(parts['positive_bce']),negative_bce=float(parts['negative_bce']),soft_dice=float(parts['soft_dice']),loss=float(loss),health=last_health,gpu_memory_allocated_bytes=torch.cuda.memory_allocated(local))
                        with (output/'training-health.jsonl').open('a') as log:log.write(json.dumps(telemetry)+'\n')
                        print(json.dumps(telemetry),flush=True)
                    telemetry_start=time.monotonic();samples=0;wait=0.
                stop=torch.tensor([int(time.time()>=deadline),int(time.time()-last_save>=c['checkpoint_seconds'])],device=local);dist.all_reduce(stop,op=dist.ReduceOp.MAX)
                if bool(stop[1]) and state['cursor']<len(plan):save();last_save=time.time()
                if bool(stop[0]):break
                previous=time.monotonic()
            del loader
            if state['cursor']==len(plan):
                result=complete_phase1(state,c,len(plan))
                if rank==0:
                    (output/'PHASE1_RESULT.json').write_text(json.dumps(result))
                break
        save()
        if rank==0 and state['scientific_stop']=='PHASE1_TRAINING_COMPLETE':
            print(json.dumps(dict(event='v3_phase1_training_complete',**result)),flush=True)
    except BaseException:
        traceback.print_exc();os._exit(1)
    finally:dist.destroy_process_group()

if __name__=='__main__':main()
