from __future__ import annotations

import torch
from torch import nn


class PatchEmbedding2D(nn.Module):
    def __init__(self, in_channels: int, embed_dim: int, patch_size: int):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int]]:
        x = self.proj(x)
        grid_size = x.shape[-2:]
        tokens = x.flatten(2).transpose(1, 2)
        return tokens, grid_size

