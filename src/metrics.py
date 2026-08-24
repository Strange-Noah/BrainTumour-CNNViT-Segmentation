from __future__ import annotations

import torch


def dice_score(logits: torch.Tensor, target: torch.Tensor, class_index: int, eps: float = 1e-6) -> torch.Tensor:
    pred = torch.argmax(logits, dim=1) == class_index
    truth = target == class_index
    intersection = torch.sum(pred & truth)
    denominator = torch.sum(pred) + torch.sum(truth)
    return (2.0 * intersection.float() + eps) / (denominator.float() + eps)


def whole_tumour_dice(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    pred = torch.argmax(logits, dim=1) > 0
    truth = target > 0
    intersection = torch.sum(pred & truth)
    denominator = torch.sum(pred) + torch.sum(truth)
    return (2.0 * intersection.float() + eps) / (denominator.float() + eps)

