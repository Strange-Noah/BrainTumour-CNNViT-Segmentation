"""Formal research entrypoint. Never invoke before the owner's explicit GPU budget authorization."""
import argparse,hashlib,json,os,random,time
from pathlib import Path
import numpy as np
import psutil
import torch
import torch.distributed as dist
from ..train import Model,loss_fn,fp32_step,RUNTIME_PRECISION
from .settings import CONFIG,PROVIDER_SESSION_SECONDS,FINALIZATION_MARGIN_SECONDS
from .stream import TrainingInput,SignedFetch
from .loading import ShardCache

def gpu_telemetry():
    import subprocess
    try:
        lines=subprocess.check_output(['nvidia-smi','--query-gpu=utilization.gpu,memory.used,memory.total','--format=csv,noheader,nounits'],text=True,timeout=5).splitlines()
        return [dict(utilization_percent=float(v[0]),memory_used_mib=float(v[1]),memory_total_mib=float(v[2])) for v in [line.split(',') for line in lines]]
    except Exception:return None

from .validation import validate_patients

def main():
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--approved-budget-seconds',type=int,required=True);p.add_argument('--session-seconds',type=int,default=PROVIDER_SESSION_SECONDS);p.add_argument('--resume',type=Path);a=p.parse_args()
    if not 600<=a.session_seconds<=PROVIDER_SESSION_SECONDS or a.approved_budget_seconds<a.session_seconds:raise ValueError('Explicit approved research budget required')
    rank=int(os.environ['RANK']);world=int(os.environ['WORLD_SIZE']);local=int(os.environ['LOCAL_RANK'])
    if world!=2 or torch.cuda.device_count()!=2 or not all('T4' in torch.cuda.get_device_name(i) for i in range(2)):raise RuntimeError('Frozen dual-T4 configuration required')
    torch.cuda.set_device(local);dist.init_process_group('nccl');torch.set_num_threads(2);torch.manual_seed(42);random.seed(42)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.deterministic=True
    config=json.loads(CONFIG.read_text());science={k:v for k,v in config.items() if k not in ['formal_training_authorized','approved_budget_seconds']};lease=json.loads((a.input/'private-data-lease.json').read_text());raw=(a.input/'catalogue.json').read_bytes();catalogue_sha=hashlib.sha256(raw).hexdigest()
    if catalogue_sha!=lease['catalogue_sha256']:raise ValueError('Input catalogue checksum mismatch')
    catalogue=json.loads(raw)
    if catalogue['summary']['patients']!={'train':210,'validation':45} or catalogue['summary']['test_included']:raise ValueError('Frozen research cohort mismatch')
    os.environ['HF_HOME']=str(a.input/'hf');os.environ['HF_HUB_OFFLINE']='1'
    model=Model(encoder_name='resnet34',encoder_weights='imagenet',in_channels=12,classes=1,activation=None).float().cuda(local).train()
    optimizer=torch.optim.AdamW([{'params':model.encoder.parameters(),'lr':config['encoder_lr']},{'params':list(model.decoder.parameters())+list(model.segmentation_head.parameters()),'lr':config['decoder_lr']}],weight_decay=config['weight_decay'])
    state=dict(data_mode='full-cohort-v1',catalogue_sha256=catalogue_sha,epoch=0,global_cursor=0,step=0,consumed_seconds=0.,pending_validation=False,validation_progress=[],best_dice=None,training_source_sha=os.environ['NOVO_SOURCE_SHA'])
    if a.resume:
        old=torch.load(a.resume,map_location='cpu',weights_only=True)
        if old['data_mode']!='full-cohort-v1' or old['catalogue_sha256']!=catalogue_sha or old['runtime_precision']!=RUNTIME_PRECISION or old['config']!=science:raise ValueError('Incompatible research checkpoint; diagnostic checkpoints are forbidden')
        model.load_state_dict(old.pop('model'),strict=True);optimizer.load_state_dict(old.pop('optimizer'));rng=old.pop('rank_rng')[rank];torch.set_rng_state(rng['cpu']);torch.cuda.set_rng_state(rng['cuda'],local);random.setstate(rng['python']);state=old
    model=torch.nn.parallel.DistributedDataParallel(model,device_ids=[local]);a.output.mkdir(parents=True,exist_ok=True)
    started=float(os.environ.get('NOVO_SESSION_STARTED_AT',str(time.time())));prior=float(state['consumed_seconds']);deadline=min(started+a.session_seconds-FINALIZATION_MARGIN_SECONDS,started+a.approved_budget_seconds-prior-FINALIZATION_MARGIN_SECONDS);last_save=time.time()
    def save():
        rng=dict(cpu=torch.get_rng_state(),cuda=torch.cuda.get_rng_state(local),python=random.getstate());all_rng=[None]*world;dist.all_gather_object(all_rng,rng)
        tensors=list(model.module.state_dict().values())+[v for s in optimizer.state.values() for v in s.values() if torch.is_tensor(v)]
        valid=torch.tensor(int(all(bool(torch.isfinite(v).all()) for v in tensors)),device=local);dist.all_reduce(valid,op=dist.ReduceOp.MIN)
        if not bool(valid):raise FloatingPointError('Invalid research checkpoint')
        if optimizer.state and {float(s['step']) for s in optimizer.state.values()}!={float(state['step'])}:raise ValueError('Research optimizer step mismatch')
        if rank==0:
            record=dict(state,status='success',sampler=dict(seed=42,world_size=world,epoch=state['epoch'],global_cursor=state['global_cursor'],plan_sha256=state.get('plan_sha256')),git_sha=state['training_source_sha'],model=model.module.state_dict(),optimizer=optimizer.state_dict(),rank_rng=all_rng,config=science,runtime_precision=RUNTIME_PRECISION,consumed_seconds=prior+time.time()-started,source_sha=os.environ['NOVO_SOURCE_SHA'],approved_budget_seconds=a.approved_budget_seconds,framework=dict(torch=str(torch.__version__),cuda=torch.version.cuda,smp='0.5.0'))
            tmp=a.output/'checkpoint.tmp';torch.save(record,tmp);tmp.replace(a.output/'checkpoint.pt')
            meta={k:record[k] for k in ['data_mode','catalogue_sha256','epoch','global_cursor','step','consumed_seconds','pending_validation','best_dice']}
            meta['science_sha256']=hashlib.sha256(json.dumps(science,sort_keys=True).encode()).hexdigest();(a.output/'training-state.json').write_text(json.dumps(meta))
        dist.barrier()
    try:
        while state['epoch']<config['maximum_proposed_coverage_epochs'] or state['pending_validation']:
            finished=torch.tensor(int(time.time()>=deadline),device=local);dist.all_reduce(finished,op=dist.ReduceOp.MAX)
            if bool(finished):break
            if state['pending_validation']:
                result=validate_patients(model,catalogue,ShardCache(a.output/'cache'/str(rank),SignedFetch(lease),config['gpu_cache_bytes'],config['decoded_cache_bytes_per_worker']),SignedFetch(lease),a.output/'validation'/f"epoch-{state['epoch']:04d}-step-{state['step']:08d}",state['training_source_sha'],'checkpoint.pt',state['epoch'],deadline,state['validation_progress'])
                state['validation_progress']=json.loads(json.dumps(result['visits']))
                if not result['complete']:break
                score=result['mean_patient_dice'];improved=state['best_dice'] is None or score>state['best_dice'];state['best_dice']=score if improved else state['best_dice'];state['pending_validation']=False;state['validation_progress']=[];save()
                if rank==0 and improved:
                    import shutil
                    shutil.copyfile(a.output/'checkpoint.pt',a.output/'best.pt')
                dist.barrier()
                if state['epoch']>=config['maximum_proposed_coverage_epochs']:break
            dataset=TrainingInput(catalogue,lease,a.output/'cache',state['epoch'],state['global_cursor'],rank,world)
            if 'plan_sha256' in state and state['global_cursor'] and state['plan_sha256']!=dataset.plan_sha:raise ValueError('Deterministic sampler mismatch')
            state['plan_sha256']=dataset.plan_sha
            loader=torch.utils.data.DataLoader(dataset,batch_size=1,num_workers=1,persistent_workers=True,prefetch_factor=2,pin_memory=True,multiprocessing_context='spawn',generator=torch.Generator().manual_seed(42))
            previous=time.monotonic();window_start=previous;wait=0.;window_samples=0
            for x,y in loader:
                wait+=time.monotonic()-previous
                x=x.cuda(local,non_blocking=True);y=y.cuda(local,non_blocking=True);optimizer.zero_grad(set_to_none=True);loss=loss_fn(model(x),y);loss.backward();flags=torch.stack([torch.isfinite(loss),torch.stack([torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None]).all()]).int();dist.all_reduce(flags,op=dist.ReduceOp.MIN);fp32_step(optimizer,bool(flags[0]),bool(flags[1]));state['training_source_sha']=os.environ['NOVO_SOURCE_SHA'];state['step']+=1;state['global_cursor']+=world;window_samples+=world
                if state['step']%100==0 or state['step']==1:
                    memory=psutil.virtual_memory()
                    pressure=torch.tensor(int(memory.available/memory.total<.15),device=local);dist.all_reduce(pressure,op=dist.ReduceOp.MAX)
                    if bool(pressure):raise MemoryError('Actual host RAM pressure')
                    elapsed=time.monotonic()-window_start
                    event=dict(event='training_update',unix_time=time.time(),session_started_at=started,step=state['step'],epoch=state['epoch'],global_cursor=state['global_cursor'],samples_per_second=window_samples/elapsed,loader_wait_fraction=wait/elapsed,loss=float(loss.detach()),loss_and_gradients_finite=True,host_ram_available_fraction=memory.available/memory.total,gpu_memory_allocated_bytes=torch.cuda.memory_allocated(local),gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated(local))
                    with (a.output/f'rank-{rank}.jsonl').open('a') as log:log.write(json.dumps(event)+'\n')
                    if rank==0:
                        event['gpus']=gpu_telemetry();event['checkpoint_interval_seconds']=600;event['validation_every_coverage_passes']=5;print(json.dumps(event),flush=True)
                    window_start=time.monotonic();wait=0.;window_samples=0
                stop=torch.tensor([int(time.time()>deadline),int(time.time()-last_save>=600)],device=local);dist.all_reduce(stop,op=dist.ReduceOp.MAX)
                if bool(stop[1]):save();last_save=time.time()
                if bool(stop[0]):break
                previous=time.monotonic()
            del loader
            if state['global_cursor']>=len(dataset.plan):state['epoch']+=1;state['global_cursor']=0;state['pending_validation']=state['epoch']%config['validation_every_coverage_epochs']==0
            save()
        save()
    except BaseException:
        import sys,traceback
        traceback.print_exc();sys.stderr.flush();os._exit(1) # Let torchrun stop peers instead of hanging in NCCL cleanup.
    finally:dist.destroy_process_group()
if __name__=='__main__':main()
