"""Full-FP32 bounded real-data training; launch with torchrun, never an epoch loop."""
import argparse, datetime, hashlib, json, os, random, signal, time
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist
import segmentation_models_pytorch as smp
from .data import SliceCache, locality_draws

RUNTIME_PRECISION={'dtype':'float32','autocast':False,'grad_scaler':False,'tf32':False}
LOADER_POLICY={'sampling':'patient-aware-exact-window-reorder-v1','window':64,'prefetch_factor':2,'persistent_workers':True,'max_mapped_bytes_per_worker':3*1024**3}

class Model(smp.Unet):
    def train(self,mode=True):
        super().train(mode)
        for m in self.encoder.modules():
            if isinstance(m,torch.nn.modules.batchnorm._BatchNorm):m.eval()
        return self

def loss_fn(logits,y):
    p=logits.float().sigmoid();dims=(1,2,3)
    dice=1-(2*(p*y).sum(dims)+1)/(p.sum(dims)+y.sum(dims)+1)
    return .5*torch.nn.functional.binary_cross_entropy_with_logits(logits.float(),y)+.5*dice.mean()

def fp32_step(optimizer,loss_finite,gradients_finite):
    if not loss_finite:raise FloatingPointError('Nonfinite FP32 loss')
    if not gradients_finite:raise FloatingPointError('Nonfinite FP32 gradients')
    optimizer.step()

def optimizer_matches(optimizer,saved):
    current=optimizer.state_dict()
    if current['param_groups']!=saved['param_groups'] or current['state'].keys()!=saved['state'].keys():return False
    for key,values in saved['state'].items():
        for name,value in values.items():
            actual=current['state'][key][name]
            if torch.is_tensor(value):
                if not torch.equal(actual.cpu(),value.cpu()):return False
            elif actual!=value:return False
    return True

def draws_hash(draws):return hashlib.sha256(json.dumps(draws).encode()).hexdigest()

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--cache',required=True);parser.add_argument('--out',required=True);parser.add_argument('--seconds',type=int,default=660);parser.add_argument('--steps',type=int,default=100000);parser.add_argument('--batch',type=int,default=1);parser.add_argument('--workers',type=int,default=1);parser.add_argument('--capacity',type=int,default=2);parser.add_argument('--resume');a=parser.parse_args()
    if not 1<=a.seconds<=900 or not 1<=a.steps<=100000 or a.workers not in (0,1,2) or a.batch not in (1,2) or a.capacity not in (1,2):raise ValueError('Benchmark bounds exceeded')
    rank=int(os.environ.get('RANK',0));world=int(os.environ.get('WORLD_SIZE',1));local=int(os.environ.get('LOCAL_RANK',0));torch.cuda.set_device(local)
    if world>1:dist.init_process_group('nccl',timeout=datetime.timedelta(seconds=120))
    def interrupted(*_):raise TimeoutError('Benchmark terminated by monitor')
    signal.signal(signal.SIGTERM,interrupted)
    torch.set_num_threads(2);torch.manual_seed(42);random.seed(42);np.random.seed(42)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    preparation_config=Path('experiments/unet25d/config.json').read_bytes()
    config=dict(json.loads(preparation_config),precision='cuda-float32',grad_scaler_init_scale=None,prefetch_factor=2,mmap_volumes_per_worker=a.capacity)
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    model=Model(encoder_name='resnet34',encoder_weights='imagenet',in_channels=12,classes=1,activation=None).float().cuda(local).train()
    optimizer=torch.optim.AdamW([{'params':model.encoder.parameters(),'lr':1e-4},{'params':list(model.decoder.parameters())+list(model.segmentation_head.parameters()),'lr':3e-4}],weight_decay=1e-4)
    runtime_loader=dict(**LOADER_POLICY,workers=a.workers,capacity=a.capacity)
    records=json.loads((Path(a.cache)/'train-index.json').read_text());draw_offset=0;offset=0;resume_verified=False
    if a.resume:
        state=torch.load(a.resume,map_location='cpu',weights_only=True)
        if state['status'] not in ('success','running') or state['config']!=config or state.get('runtime_precision')!=RUNTIME_PRECISION or state.get('runtime_loader')!=runtime_loader:raise ValueError('Resume status/configuration/precision/loader mismatch')
        if state['sampler']['world_size']!=world or state['local_batch']!=a.batch:raise ValueError('Exact resume requires same rank/batch allocation')
        model.load_state_dict(state['model'],strict=True);optimizer.load_state_dict(state['optimizer']);offset=state['step'];draw_offset=state['draw_position']
        if not all(torch.equal(v.cpu(),state['model'][k]) for k,v in model.state_dict().items()):raise ValueError('Model restoration mismatch')
        if {float(v['step']) for v in optimizer.state.values()}!={float(offset)} or not optimizer_matches(optimizer,state['optimizer']):raise ValueError('Optimizer restoration mismatch')
        rank_rng=state['rank_rng'][rank];torch.set_rng_state(rank_rng['cpu']);torch.cuda.set_rng_state(rank_rng['cuda'],local);random.setstate(rank_rng['python'])
        resume_verified=True
    if world>1:model=torch.nn.parallel.DistributedDataParallel(model,device_ids=[local],broadcast_buffers=True)
    raw=model.module if world>1 else model
    baseline={n:m.running_mean.clone() for n,m in raw.encoder.named_modules() if isinstance(m,torch.nn.BatchNorm2d)}
    global_draws=locality_draws(records,world*a.batch*(a.steps+draw_offset+32),42,world,LOADER_POLICY['window'])
    if a.resume and draws_hash(global_draws[world*a.batch*draw_offset:world*a.batch*(draw_offset+32)])!=state['sampler']['next_draws_sha256']:raise ValueError('Resumed sampler sequence mismatch')
    draws=global_draws[world*a.batch*draw_offset:world*a.batch*(draw_offset+a.steps)];dataset=SliceCache(a.cache,draws[rank::world],a.capacity,LOADER_POLICY['max_mapped_bytes_per_worker'])
    options=dict(batch_size=a.batch,num_workers=a.workers,pin_memory=True,drop_last=True,generator=torch.Generator().manual_seed(42))
    if a.workers:options.update(prefetch_factor=2,persistent_workers=True,multiprocessing_context='spawn')
    loader=torch.utils.data.DataLoader(dataset,**options);stream=iter(loader);start=time.monotonic();previous=start;step=offset;draw_position=draw_offset
    (out/'runtime.json').write_text(json.dumps(dict(precision=RUNTIME_PRECISION,loader=runtime_loader,rank=rank,world=world,batch=a.batch))) if rank==0 else None
    log=(out/f'rank-{rank}.jsonl').open('w');status='success'
    def save_state(status):
        rank_rng=dict(step=step,cpu=torch.get_rng_state(),cuda=torch.cuda.get_rng_state(local),python=random.getstate())
        torch.save(rank_rng,out/f'rng-{rank}.pt')
        if world>1 and status in ['success','running']:dist.barrier()
        optimizer_steps={float(v['step']) for v in optimizer.state.values() if 'step' in v}
        if optimizer_steps and optimizer_steps!={float(step)}:raise ValueError('Optimizer/checkpoint step counter mismatch')
        if status in ['success','running']:
            tensors=list(raw.state_dict().values())+[v for p in optimizer.state.values() for v in p.values() if torch.is_tensor(v)]
            valid=torch.tensor(int(all(bool(torch.isfinite(v).all()) for v in tensors)),device=local)
            if world>1:dist.all_reduce(valid,op=dist.ReduceOp.MIN)
            if not bool(valid):raise FloatingPointError('Invalid checkpoint tensor')
        if rank==0:
            all_rng=[torch.load(out/f'rng-{r}.pt',map_location='cpu',weights_only=True) for r in range(world)]
            next_draws=locality_draws(records,world*a.batch*(draw_position+32),42,world,LOADER_POLICY['window'])[world*a.batch*draw_position:]
            state=dict(model=raw.state_dict(),optimizer=optimizer.state_dict(),scheduler=None,scaler=None,config=config,preparation_config_sha256=hashlib.sha256(preparation_config).hexdigest(),runtime_precision=RUNTIME_PRECISION,runtime_loader=runtime_loader,step=step,epoch=0,draw_position=draw_position,local_batch=a.batch,git_sha=os.environ.get('NOVO_SOURCE_SHA','unrecorded'),sampler=dict(seed=42,consumed_global=draw_position*world*a.batch,world_size=world,next_draws_sha256=draws_hash(next_draws)),rank_rng=all_rng,status=status)
            filename='checkpoint.pt' if status in ('success','running') else 'diagnostic-checkpoint.pt'
            temporary=out/(filename+'.tmp');torch.save(state,temporary);temporary.replace(out/filename)
        if world>1 and status in ['success','running']:dist.barrier()
    last_checkpoint=time.monotonic()
    try:
        for i in range(a.steps):
            before=time.monotonic();x,y,cache=next(stream);assert tuple(x.shape[1:])==(12,512,512) and tuple(y.shape[1:])==(1,512,512);ready=time.monotonic();torch.cuda.synchronize();t=time.monotonic();x=x.cuda(local,non_blocking=True);y=y.cuda(local,non_blocking=True);torch.cuda.synchronize();h2d=time.monotonic()-t
            optimizer.zero_grad(set_to_none=True);t=time.monotonic();logits=model(x);loss=loss_fn(logits,y)
            torch.cuda.synchronize();forward=time.monotonic()-t;t=time.monotonic();loss.backward()
            flags=torch.stack([torch.isfinite(loss),torch.stack([torch.isfinite(p.grad).all() for p in raw.parameters() if p.grad is not None]).all()]).int()
            if world>1:dist.all_reduce(flags,op=dist.ReduceOp.MIN)
            torch.cuda.synchronize();backward=time.monotonic()-t;t=time.monotonic();fp32_step(optimizer,bool(flags[0]),bool(flags[1]));step+=1
            torch.cuda.synchronize();optim=time.monotonic()-t;draw_position=draw_offset+i+1;now=time.monotonic()
            row=dict(step=step,rank=rank,wall=time.time(),elapsed=now-start,phase='warmup' if i<10 else 'steady',samples=a.batch,optimizer_updated=True,precision='float32',iteration_seconds=now-previous,data_wait_seconds=ready-before,h2d_seconds=h2d,forward_seconds=forward,backward_including_sync_seconds=backward,optimizer_seconds=optim,loss=float(loss.detach()),finite=True,vram_allocated=torch.cuda.max_memory_allocated(),vram_reserved=torch.cuda.max_memory_reserved(),cache={k:v.tolist() for k,v in cache.items()})
            log.write(json.dumps(row)+'\n');log.flush();previous=now
            stop=torch.tensor([int(now-start>=a.seconds),int(rank==0 and now-last_checkpoint>=120)],device=local)
            if world>1:dist.all_reduce(stop,op=dist.ReduceOp.MAX)
            if bool(stop[1]):save_state('running');last_checkpoint=time.monotonic()
            if bool(stop[0]):break
        assert all(torch.equal(m.running_mean,baseline[n]) for n,m in raw.encoder.named_modules() if isinstance(m,torch.nn.BatchNorm2d))
    except Exception as e:status=type(e).__name__;raise
    finally:
        log.close();save_state(status);del stream,loader;dataset.close()
        if world>1:dist.destroy_process_group()
    if rank==0 and a.resume:
        if step-offset!=a.steps:raise ValueError('Resume did not finish requested optimizer updates')
        (out/'resume-verification.json').write_text(json.dumps(dict(verified=resume_verified,start_step=offset,final_step=step,successful_updates=step-offset,start_draw=draw_offset,final_draw=draw_position,model_and_optimizer_restored=True,rank_rng_restored=True,next_sampler_draws_verified=True,precision=RUNTIME_PRECISION)))
if __name__=='__main__':main()
