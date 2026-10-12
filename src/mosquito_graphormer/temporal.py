"""Strictly left-causal depthwise-separable temporal encoder."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class CausalDepthwiseBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dilation: int, dropout: float):
        super().__init__()
        self.left_pad = dilation * (kernel_size - 1)
        self.depthwise = nn.Conv1d(channels, channels, kernel_size, dilation=dilation, groups=channels, bias=False)
        self.pointwise = nn.Conv1d(channels, channels, 1)
        # LayerNorm is applied independently at each time point. GroupNorm would
        # aggregate across the full temporal axis and leak future values.
        self.norm = nn.LayerNorm(channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.depthwise(F.pad(x, (self.left_pad, 0)))
        y = self.pointwise(y)
        return self.norm((x + self.dropout(F.gelu(y))).transpose(1, 2)).transpose(1, 2)


class CausalDilatedTCN(nn.Module):
    """Input/output `[B,T,N,C]`; no future values can affect time `t`."""
    def __init__(self, input_dim: int, hidden_dim: int, kernel_size: int = 3, dilations=(1, 2, 4), dropout: float = 0.1):
        super().__init__()
        self.input = nn.Conv1d(input_dim, hidden_dim, 1)
        self.blocks = nn.ModuleList(CausalDepthwiseBlock(hidden_dim, kernel_size, d, dropout) for d in dilations)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4: raise ValueError("Temporal input must be [B,T,N,F]")
        b, t, n, f = x.shape
        y = x.permute(0, 2, 3, 1).reshape(b * n, f, t)
        y = self.input(y)
        for block in self.blocks: y = block(y)
        return y.reshape(b, n, -1, t).permute(0, 3, 1, 2)
