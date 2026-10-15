from __future__ import annotations

import math
import torch
from torch import nn


class MobilityGeographicGraph(nn.Module):
    """One-normalisation mobility/GIS convex fusion with explicit local coupling."""
    def __init__(self, tau_initial=200.0, eta_initial=.70, gamma_initial=.15, lambda_initial=.50):
        super().__init__()
        tau_p = math.log((tau_initial - 20.0) / (500.0 - tau_initial))
        self.raw_tau_geo = nn.Parameter(torch.tensor(tau_p))
        self.raw_eta = nn.Parameter(torch.tensor(math.log(eta_initial / (1 - eta_initial))))
        self.raw_gamma = nn.Parameter(torch.tensor(math.log(gamma_initial / (1 - gamma_initial))))
        self.raw_lambda = nn.Parameter(torch.tensor(math.log(math.exp(lambda_initial) - 1.0)))

    def tau_geo(self): return 20.0 + 480.0 * torch.sigmoid(self.raw_tau_geo)
    def eta(self): return torch.sigmoid(self.raw_eta)
    def gamma(self): return 0.49 * torch.sigmoid(self.raw_gamma)
    def lambda_mobility(self): return torch.nn.functional.softplus(self.raw_lambda)

    def _row_normalize(self, x): return x / x.sum(-1, keepdim=True).clamp_min(1e-8)

    def cross_graph(self, mobility, distance, support, use_mobility=True, use_gis=True):
        b, lags, n, n2 = mobility.shape
        if n != n2 or distance.shape != (n, n): raise ValueError("invalid graph shapes")
        device, dtype = mobility.device, mobility.dtype
        offdiag = ~torch.eye(n, dtype=torch.bool, device=device)
        # Input history runs from oldest to newest; lag zero is the latest day.
        lag = torch.arange(lags - 1, -1, -1, device=device, dtype=dtype)
        weights = torch.softmax(-self.lambda_mobility() * lag, 0)
        mob = (mobility.clamp_min(0) * weights[None, :, None, None]).sum(1) if use_mobility else torch.zeros((b, n, n), device=device, dtype=dtype)
        mob = mob.masked_fill(~offdiag[None], 0.0)
        geo = torch.exp(-distance.to(device=device, dtype=dtype) / self.tau_geo()) if use_gis else torch.zeros((n, n), device=device, dtype=dtype)
        geo = geo.masked_fill(~offdiag, 0.0)
        geo_support = support.to(device=device, dtype=torch.bool).clone().masked_fill(~offdiag, False) if use_gis else torch.zeros((n, n), dtype=torch.bool, device=device)
        mob_support = mob > 0
        supported = mob_support | geo_support[None]
        mob = self._row_normalize(mob + mob_support.to(dtype) * 1e-8)
        geo = self._row_normalize(geo + geo_support.to(dtype) * 1e-8).expand(b, -1, -1)
        fused = self.eta() * mob + (1 - self.eta()) * geo
        fused = fused.masked_fill(~supported, 0.0)
        return self._row_normalize(fused + supported.to(dtype) * 1e-8).masked_fill(~supported, 0.0)

    def forward(self, mobility, distance, support, use_mobility=True, use_gis=True, no_spatial=False):
        b, _, n, _ = mobility.shape; eye = torch.eye(n, device=mobility.device, dtype=mobility.dtype).expand(b, -1, -1)
        if no_spatial: return eye
        cross = self.cross_graph(mobility, distance, support, use_mobility, use_gis)
        return (1 - self.gamma()) * eye + self.gamma() * cross

    def diagnostics(self, graph):
        off = ~torch.eye(graph.shape[-1], dtype=torch.bool, device=graph.device)
        cross = graph[..., off]
        return {"tau_geo": float(self.tau_geo().detach()), "eta_mobility": float(self.eta().detach()), "gamma_spatial": float(self.gamma().detach()), "lambda_mobility": float(self.lambda_mobility().detach()), "graph_row_entropy": float((-(graph.clamp_min(1e-12) * graph.clamp_min(1e-12).log()).sum(-1).mean()).detach()), "graph_off_diagonal_mass": float(cross.sum(-1).mean().detach()), "number_supported_edges": int((cross > 1e-8).sum().detach()), "mean_nonzero_cross_edge": float(cross[cross > 1e-8].mean().detach()) if (cross > 1e-8).any() else 0.0, "max_nonzero_cross_edge": float(cross.max().detach())}


class MaskedGraphTransformer(nn.Module):
    def __init__(self, hidden_dim: int, heads: int = 2, dropout: float = 0.1):
        super().__init__()
        if hidden_dim % heads: raise ValueError("hidden dimension must divide attention heads")
        self.heads, self.width = heads, hidden_dim // heads
        self.qkv = nn.Linear(hidden_dim, hidden_dim * 3, bias=False)
        self.out = nn.Linear(hidden_dim, hidden_dim)
        self.norm1, self.norm2 = nn.LayerNorm(hidden_dim), nn.LayerNorm(hidden_dim)
        self.ff = nn.Sequential(nn.Linear(hidden_dim, hidden_dim * 2), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim * 2, hidden_dim))
        self.dropout = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor, graph: torch.Tensor, return_attention: bool = False):
        b, n, d = h.shape
        q, k, v = self.qkv(h).chunk(3, -1)
        def split(x): return x.reshape(b, n, self.heads, self.width).transpose(1, 2)
        q, k, v = map(split, (q, k, v))
        support = graph > 0
        bias = torch.where(support, graph.clamp_min(1e-12).log(), torch.zeros_like(graph))
        scores = (q @ k.transpose(-1, -2)) / math.sqrt(self.width) + bias[:, None]
        scores = scores.masked_fill(~support[:, None], torch.finfo(scores.dtype).min)
        attention = torch.softmax(scores, -1)
        attention = attention.masked_fill(~support[:, None], 0.0)
        message = (attention @ v).transpose(1, 2).reshape(b, n, d)
        h = self.norm1(h + self.dropout(self.out(message)))
        h = self.norm2(h + self.dropout(self.ff(h)))
        return (h, attention) if return_attention else h


