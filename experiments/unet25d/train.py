"""Bounded real-data CUDA benchmark; launch with torchrun, never an epoch loop."""
import argparse, datetime, hashlib, json, os, random, signal, time
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist
import segmentation_models_pytorch as smp
from .data import SliceCache, sample_draws

class Model(smp.Unet):
    def train(self,mode=True):
        super().train(mode)
        for m in self.encoder.modules():
            if isinstance(m,torch.nn.modules.batchnorm._BatchNorm):m.eval()
        return self

def loss_fn(logits,y):
    p=logits.float().sigmoid(); dims=(1,2,3)
    # Empty target: Dice=(smooth)/(sum predictions+smooth), penalizes false positives.
    dice=1-(2*(p*y).sum(dims)+1)/(p.sum(dims)+y.sum(dims)+1)
    return .5*torch.nn.functional.binary_cross_entropy_with_logits(logits.float(),y)+.5*dice.mean()

def amp_step(scaler,optimizer,loss_finite,gradients_finite,skips):
    """Bounded standard loss-scale recovery; never update weights with bad gradients."""
    if not loss_finite:raise FloatingPointError('Nonfinite loss')
    if gradients_finite:
        scaler.step(optimizer);scaler.update();return skips,True
    if skips>=3:raise FloatingPointError('FP16 gradient overflow retry cap exceeded')
    scaler.update(new_scale=scaler.get_scale()/2);optimizer.zero_grad(set_to_none=True)
    return skips+1,False

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--cache',required=True);parser.add_argument('--out',required=True);parser.add_argument('--seconds',type=int,default=120);parser.add_argument('--steps',type=int,default=50);parser.add_argument('--batch',type=int,default=1);parser.add_argument('--workers',type=int,default=1);parser.add_argument('--capacity',type=int,default=1);parser.add_argument('--resume');a=parser.parse_args()
    if not 1<=a.seconds<=900 or not 1<=a.steps<=100000 or a.workers not in (0,1,2,4) or a.batch not in (1,2):raise ValueError('Benchmark bounds exceeded')
    rank=int(os.environ.get('RANK',0));world=int(os.environ.get('WORLD_SIZE',1));local=int(os.environ.get('LOCAL_RANK',0));torch.cuda.set_device(local)
    if world>1:dist.init_process_group('nccl',timeout=datetime.timedelta(seconds=120))
    def interrupted(*_): raise TimeoutError('Benchmark terminated by monitor')
    signal.signal(signal.SIGTERM,interrupted)
    torch.set_num_threads(2);torch.manual_seed(42);random.seed(42);np.random.seed(42)
    config=json.loads(Path('experiments/unet25d/config.json').read_text());out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    model=Model(encoder_name='resnet34',encoder_weights='imagenet',in_channels=12,classes=1,activation=None).cuda(local).train()
    optimizer=torch.optim.AdamW([{'params':model.encoder.parameters(),'lr':1e-4},{'params':list(model.decoder.parameters())+list(model.segmentation_head.parameters()),'lr':3e-4}],weight_decay=1e-4)
    draw_offset=0;overflow_skips=0
    scaler=torch.amp.GradScaler('cuda',init_scale=config['grad_scaler_init_scale']);offset=0
    if a.resume:
        state=torch.load(a.resume,map_location='cpu',weights_only=True)
        if state['config']!=config:raise ValueError('Resume config mismatch')
        if state['sampler']['world_size']!=world or state['local_batch']!=a.batch:raise ValueError('Exact resume requires same rank/batch allocation')
        model.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer']);scaler.load_state_dict(state['scaler']);offset=state['step'];draw_offset=state.get('draw_position',offset);overflow_skips=state.get('amp_overflow_skips',0);torch.set_rng_state(state['rng_cpu']);random.setstate(state['rng_python'])
    if a.resume:
        rank_rng=state['rank_rng'][rank];torch.set_rng_state(rank_rng['cpu']);torch.cuda.set_rng_state(rank_rng['cuda'],local);random.setstate(rank_rng['python'])
    if world>1:model=torch.nn.parallel.DistributedDataParallel(model,device_ids=[local],broadcast_buffers=True)
    raw=model.module if world>1 else model
    baseline={n:m.running_mean.clone() for n,m in raw.encoder.named_modules() if isinstance(m,torch.nn.BatchNorm2d)}
    records=json.loads((Path(a.cache)/'train-index.json').read_text());draws=sample_draws(records,world*a.batch*(a.steps+draw_offset),42,world)[world*a.batch*draw_offset:]
    # One deterministic global draw sequence, disjoint positions for ranks.
    local_draws=draws[rank::world];dataset=SliceCache(a.cache,local_draws,a.capacity)
    options=dict(batch_size=a.batch,num_workers=a.workers,pin_memory=True,drop_last=True)
    if a.workers:options.update(prefetch_factor=1,persistent_workers=False,multiprocessing_context='spawn')
    loader=torch.utils.data.DataLoader(dataset,**options);stream=iter(loader);start=time.monotonic();previous=start;step=offset;draw_position=draw_offset
    log=(out/f'rank-{rank}.jsonl').open('w'); status='success'
    def save_state(status):
        rank_rng=dict(step=step,cpu=torch.get_rng_state(),cuda=torch.cuda.get_rng_state(local),python=random.getstate())
        torch.save(rank_rng,out/f'rng-{rank}.pt')
        if world>1 and status in ['success','running']:dist.barrier()
        if rank==0:
            all_rng=[torch.load(out/f'rng-{r}.pt',map_location='cpu',weights_only=True) for r in range(world)]
            state=dict(model=raw.state_dict(),optimizer=optimizer.state_dict(),scheduler=None,scaler=scaler.state_dict(),config=config,step=step,epoch=0,draw_position=draw_position,amp_overflow_skips=overflow_skips,local_batch=a.batch,git_sha=os.environ.get('NOVO_SOURCE_SHA','unrecorded'),sampler=dict(seed=42,consumed_global=draw_position*world*a.batch,world_size=world,draws_sha256=hashlib.sha256(json.dumps(draws).encode()).hexdigest()),rng_cpu=torch.get_rng_state(),rng_cuda=torch.cuda.get_rng_state(local),rng_python=random.getstate(),numpy_seed=42,rank_rng=all_rng,status=status)
            temporary=out/'checkpoint.tmp';torch.save(state,temporary);temporary.replace(out/'checkpoint.pt')
            if os.environ.get('R2_OUTPUT_ACCESS_KEY_ID'):
                from cloud.storage import connect,put_once
                client,bucket=connect(output=True)
                if bucket!='novo-nordisk-pbl-outputs':raise ValueError('Checkpoint persistence requires output bucket')
                run_id=os.environ.get('NOVO_RUN_ID')
                if not run_id:raise ValueError('Stable campaign run ID required')
                key=f'novo-nordisk/experiments/unet25d/{run_id}/{out.name}/periodic-step-{step:09d}.pt'
                upload_started=time.monotonic();receipt=put_once(client,bucket,key,out/'checkpoint.pt');receipt['upload_and_readback_seconds']=time.monotonic()-upload_started
                with (out/'periodic-receipts.jsonl').open('a') as receipts:receipts.write(json.dumps(receipt)+'\n')

    last_checkpoint=time.monotonic()
    try:
        for i in range(a.steps):
            before=time.monotonic();x,y,cache=next(stream);assert tuple(x.shape[1:])==(12,512,512) and tuple(y.shape[1:])==(1,512,512);ready=time.monotonic();torch.cuda.synchronize();t=time.monotonic();x=x.cuda(local,non_blocking=True);y=y.cuda(local,non_blocking=True);torch.cuda.synchronize();h2d=time.monotonic()-t
            optimizer.zero_grad(set_to_none=True);t=time.monotonic()
            with torch.autocast('cuda',dtype=torch.float16):logits=model(x);loss=loss_fn(logits,y)
            torch.cuda.synchronize();forward=time.monotonic()-t;t=time.monotonic();scaler.scale(loss).backward();scaler.unscale_(optimizer)
            finite=torch.isfinite(loss)&torch.stack([torch.isfinite(p.grad).all() for p in raw.parameters() if p.grad is not None]).all()
            if world>1:dist.all_reduce(finite,op=dist.ReduceOp.MIN)
            torch.cuda.synchronize();backward=time.monotonic()-t;t=time.monotonic()
            overflow_skips,updated=amp_step(scaler,optimizer,bool(torch.isfinite(loss)),bool(finite),overflow_skips)
            torch.cuda.synchronize();optim=time.monotonic()-t;draw_position=draw_offset+i+1;now=time.monotonic()
            row=dict(step=step,rank=rank,wall=time.time(),elapsed=now-start,phase='warmup' if i<10 else 'steady',samples=a.batch if updated else 0,optimizer_updated=updated,amp_scale=scaler.get_scale(),amp_overflow_skips=overflow_skips,iteration_seconds=now-previous,data_wait_seconds=ready-before,h2d_seconds=h2d,forward_seconds=forward,backward_including_sync_seconds=backward,optimizer_seconds=optim,loss=float(loss.detach()),finite=updated,vram_allocated=torch.cuda.max_memory_allocated(),vram_reserved=torch.cuda.max_memory_reserved(),cache={k:v.tolist() for k,v in cache.items()})
            log.write(json.dumps(row)+'\n');log.flush();previous=now
            stop=torch.tensor([int(now-start>=a.seconds),int(rank==0 and now-last_checkpoint>=120)],device=local)
            if world>1:dist.all_reduce(stop,op=dist.ReduceOp.MAX)
            if bool(stop[1]):
                save_state('running');last_checkpoint=time.monotonic()
            if bool(stop[0]):break
        assert all(torch.equal(m.running_mean,baseline[n]) for n,m in raw.encoder.named_modules() if isinstance(m,torch.nn.BatchNorm2d))
    except Exception as e:
        status=type(e).__name__;raise
    finally:
        log.close()
        save_state(status)
        del stream,loader,dataset
        if world>1:dist.destroy_process_group()
if __name__=='__main__':main()
