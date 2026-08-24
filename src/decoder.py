from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class LinearMaskDecoder(nn.Module):
    def __init__(self, embed_dim: int, num_classes: int):
        super().__init__()
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(
        self,
        tokens: torch.Tensor,
        grid_size: tuple[int, int],
        output_size: tuple[int, int],
    ) -> torch.Tensor:
        batch, _, _ = tokens.shape
        height, width = grid_size
        logits = self.head(tokens)
        logits = logits.transpose(1, 2).reshape(batch, -1, height, width)
        return F.interpolate(logits, size=output_size, mode="bilinear", align_corners=False)

