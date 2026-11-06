"""Archived comparator architecture; caller supplies training-only scaled features."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

class _CityParallelCausalConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dilation: int, dropout: float) -> None:
        super().__init__()
        self.padding = 2 * dilation
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 1), dilation=(dilation, 1), padding=(self.padding, 0))
        self.norm = nn.GroupNorm(1, out_channels)
        self.dropout = nn.Dropout(dropout)
        self.residual = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.residual(x)
        value = self.conv(x)
        if self.padding:
            value = value[:, :, :-self.padding, :]
        batch, channels, history, cities = value.shape
        # Match GroupNorm applied independently to every city history while
        # retaining Conv2d's much faster city-parallel kernel on Tesla T4.
        value = self.norm(value.permute(0, 3, 1, 2).reshape(batch * cities, channels, history))
        value = value.reshape(batch, cities, channels, history).permute(0, 2, 3, 1)
        return self.dropout(F.gelu(value + residual))


class _TemporalEncoder(nn.Module):
    def __init__(self, num_features: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.blocks = nn.Sequential(
            _CityParallelCausalConvBlock(num_features, hidden_dim, 1, dropout),
            _CityParallelCausalConvBlock(hidden_dim, hidden_dim, 2, dropout),
            _CityParallelCausalConvBlock(hidden_dim, hidden_dim, 4, dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [B,T,N,F] -> city-parallel channels-first temporal maps [B,F,T,N].
        values = self.blocks(x.permute(0, 3, 1, 2))
        return values[:, :, -1, :].permute(0, 2, 1)


class TemporalTCNBaseline(nn.Module):
    """Graph-free temporal convolution baseline with a direct H1--H7 head."""

    def __init__(self, num_features: int, num_cities: int, horizon: int = 7, hidden_dim: int = 64, dropout: float = 0.1) -> None:
        super().__init__()
        self.temporal = _TemporalEncoder(num_features, hidden_dim, dropout)
        self.city_embedding = nn.Embedding(num_cities, hidden_dim)
        self.head = nn.Sequential(nn.Linear(2 * hidden_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, horizon))

    def forward(self, x: torch.Tensor, city_indices: torch.Tensor) -> torch.Tensor:
        temporal = self.temporal(x)
        city = self.city_embedding(city_indices).unsqueeze(0).expand(x.shape[0], -1, -1)
        return F.softplus(self.head(torch.cat((temporal, city), dim=-1)))
