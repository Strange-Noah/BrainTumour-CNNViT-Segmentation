from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class DiceCrossEntropyLoss(nn.Module):
    def __init__(self, dice_weight: float = 1.0, ce_weight: float = 1.0, eps: float = 1e-6):
        super().__init__()
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        self.eps = eps

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, target.long())
        probs = torch.softmax(logits, dim=1)
        target_1h = F.one_hot(target.long(), num_classes=logits.shape[1]).permute(0, 3, 1, 2)
        dims = (0, 2, 3)
        intersection = torch.sum(probs * target_1h, dims)
        denominator = torch.sum(probs + target_1h, dims)
        dice = (2.0 * intersection + self.eps) / (denominator + self.eps)
        dice_loss = 1.0 - dice.mean()
        return self.ce_weight * ce + self.dice_weight * dice_loss


class MulticlassForegroundDiceCELoss(nn.Module):
    def __init__(
        self,
        ce_weight: float = 0.5,
        dice_weight: float = 0.5,
        foreground_classes: tuple[int, ...] = (1, 2, 3),
        eps: float = 1e-6,
    ):
        super().__init__()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.foreground_classes = foreground_classes
        self.eps = eps

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> dict[str, torch.Tensor]:
        if logits.ndim != 4:
            raise ValueError(f"Expected logits with shape (B, C, H, W), got {tuple(logits.shape)}")
        if logits.shape[1] < 2:
            raise ValueError(f"Expected at least 2 classes, got logits shape {tuple(logits.shape)}")
        if target.ndim == 4 and target.shape[1] == 1:
            target = target[:, 0]
        if target.shape != logits.shape[0:1] + logits.shape[2:]:
            raise ValueError(f"Target shape {tuple(target.shape)} does not match logits {tuple(logits.shape)}")

        target = target.long()
        ce = F.cross_entropy(logits, target)
        probs = torch.softmax(logits, dim=1)
        target_1h = F.one_hot(target, num_classes=logits.shape[1]).permute(0, 3, 1, 2).float()
        dice_terms = {}
        dice_values = []
        for class_index in self.foreground_classes:
            pred_c = probs[:, class_index]
            target_c = target_1h[:, class_index]
            intersection = torch.sum(pred_c * target_c, dim=(0, 1, 2))
            denominator = torch.sum(pred_c + target_c, dim=(0, 1, 2))
            dice_c = (2.0 * intersection + self.eps) / (denominator + self.eps)
            dice_c = dice_c.mean()
            dice_terms[f"dice_class_{class_index}"] = dice_c
            dice_values.append(dice_c)
        foreground_dice = torch.stack(dice_values).mean()
        dice_loss = 1.0 - foreground_dice
        total = self.ce_weight * ce + self.dice_weight * dice_loss
        return {
            "loss": total,
            "ce": ce,
            "dice_loss": dice_loss,
            "foreground_dice_soft": foreground_dice,
            **dice_terms,
        }


class BinaryBCEDiceLoss(nn.Module):
    def __init__(self, bce_weight: float = 0.5, dice_weight: float = 0.5, eps: float = 1e-6):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.eps = eps

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> dict[str, torch.Tensor]:
        if logits.ndim != 4 or logits.shape[1] != 1:
            raise ValueError(f"Expected binary logits with shape (B, 1, H, W), got {tuple(logits.shape)}")
        target = target.float()
        if target.ndim == 3:
            target = target.unsqueeze(1)
        if target.shape != logits.shape:
            raise ValueError(f"Target shape {tuple(target.shape)} does not match logits {tuple(logits.shape)}")

        bce = F.binary_cross_entropy_with_logits(logits, target)
        probs = torch.sigmoid(logits)
        dims = (0, 2, 3)
        intersection = torch.sum(probs * target, dim=dims)
        denominator = torch.sum(probs + target, dim=dims)
        dice = (2.0 * intersection + self.eps) / (denominator + self.eps)
        dice_loss = 1.0 - dice.mean()
        total = self.bce_weight * bce + self.dice_weight * dice_loss
        return {"loss": total, "bce": bce, "dice_loss": dice_loss}
