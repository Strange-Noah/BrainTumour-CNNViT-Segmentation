"""Class-unweighted masked BCE with positive lesion redistribution."""
import torch
from torch.nn import functional as F


def masked_loss(logits, target, valid, component_weight=None, return_parts=False):
    if logits.shape != target.shape or target.shape != valid.shape:
        raise ValueError('Masked loss shapes')
    logits = logits.float()
    target = target.float()
    valid = valid.bool()
    if not torch.isfinite(logits).all() or not torch.isfinite(target).all():
        raise FloatingPointError('Nonfinite loss input')
    if not torch.all((target == 0) | (target == 1)):
        raise ValueError('Binary label-3 target required')
    if torch.any(target.bool() & ~valid):
        raise ValueError('GT excluded by loss mask')
    dims = (1, 2, 3)
    valid_pixels = valid.sum(dims)
    if torch.any(valid_pixels == 0):
        raise ValueError('No acquired valid supervision')

    element = F.binary_cross_entropy_with_logits(logits, target, reduction='none')
    pos = valid & target.bool()
    neg = valid & ~target.bool()
    positive_pixels = pos.sum(dims)
    if component_weight is None:
        positive_weights = pos.float()
    else:
        if component_weight.shape != target.shape:
            raise ValueError('Component weight shape mismatch')
        # V2's loader normalizes the total foreground weight to its pixel count.
        # Ignore component weights on background and padding.
        positive_weights = torch.where(pos, component_weight.float(), 0.)
        if not torch.all(torch.isfinite(positive_weights) & ((positive_weights > 0) | ~pos)):
            raise ValueError('Invalid positive component weights')
        if not torch.allclose(positive_weights.sum(dims), positive_pixels.float(), rtol=1e-5, atol=1e-6):
            raise ValueError('Positive component weights must average one per sample')
    positive_loss = (element * positive_weights).sum(dims)
    negative_loss = (element * neg).sum(dims)
    bce = ((positive_loss + negative_loss) / valid_pixels).mean()
    p = logits.sigmoid()
    m = valid.float()
    dice = 1 - (2 * (p * target * m).sum(dims) + 1) / (
        (p * m).sum(dims) + (target * m).sum(dims) + 1
    )
    total = .5 * bce + .5 * dice.mean()
    if not return_parts:
        return total
    return total, dict(
        bce=bce, soft_dice=dice.mean(),
        positive_bce=(positive_loss / positive_pixels.clamp_min(1)).mean(),
        negative_bce=(negative_loss / neg.sum(dims).clamp_min(1)).mean(),
    )
