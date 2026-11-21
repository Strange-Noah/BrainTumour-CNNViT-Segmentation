"""Same BCE+Dice family, intentionally redesigned internal foreground supervision."""
import torch
from torch.nn import functional as F

def balanced_loss(logits,target,valid,component_weight=None,return_parts=False):
    if logits.shape!=target.shape or target.shape!=valid.shape:raise ValueError('Masked loss shapes')
    logits=logits.float();target=target.float();valid=valid.bool()
    if not torch.isfinite(logits).all() or not torch.isfinite(target).all():raise FloatingPointError('Nonfinite loss input')
    if torch.any(target.bool()&~valid):raise ValueError('GT excluded by loss mask')
    if not torch.all((target==0)|(target==1)):raise ValueError('Binary label-3 target required')
    w=torch.ones_like(target) if component_weight is None else component_weight.float()
    if torch.any((w<=0)&target.bool()) or not torch.isfinite(w).all():raise ValueError('Invalid component weights')
    dims=(1,2,3);pos=valid&target.bool();neg=valid&~target.bool();pc=pos.sum(dims);nc=neg.sum(dims)
    if not torch.all(valid.sum(dims)>0):raise ValueError('No acquired valid supervision')
    element=F.binary_cross_entropy_with_logits(logits,target,reduction='none')
    positive=(element*w*pos).sum(dims)/(w*pos).sum(dims).clamp_min(1)
    negative=(element*neg).sum(dims)/nc.clamp_min(1)
    bce=torch.where(pc>0,.5*positive+.5*negative,negative)
    # Positive-only ROI is unusual, but must not halve its learning signal.
    bce=torch.where((pc>0)&(nc==0),positive,bce)
    p=logits.sigmoid();m=valid.float()
    dice=1-(2*(p*target*m).sum(dims)+1)/((p*m).sum(dims)+(target*m).sum(dims)+1)
    total=.5*bce.mean()+.5*dice.mean()
    parts=dict(bce=bce.mean(),soft_dice=dice.mean(),positive_bce=positive.mean(),negative_bce=negative.mean())
    return (total,parts) if return_parts else total
