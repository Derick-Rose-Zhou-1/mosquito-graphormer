"""Bounded effective-parameter heads with optional climate priors and city effects."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


PARAMETERS = ("a", "sigma_v", "mu_v", "M")


class EffectiveParameterHead(nn.Module):
    def __init__(self, hidden_dim: int, num_locations: int, mode: str = "FREE", location_effects: bool = False,
                 max_log_correction: float = .30, rho_min: float = .05, rho_max: float = 2.0):
        super().__init__(); self.mode, self.maximum, self.rho_min, self.rho_max = mode, max_log_correction, rho_min, rho_max
        self.delta = nn.Linear(hidden_dim, 4); self.global_raw = nn.Parameter(torch.zeros(4)); self.rho_raw = nn.Parameter(torch.tensor(0.0))
        self.location_effects = location_effects
        if location_effects: self.location_effect = nn.Embedding(num_locations, 5); nn.init.zeros_(self.location_effect.weight)

    def forward(self, h: torch.Tensor, priors: dict[str, torch.Tensor] | None = None, location_indices=None) -> dict[str, torch.Tensor]:
        raw = self.delta(h) + self.global_raw[None, None]
        if self.location_effects:
            location_effect = self.location_effect.weight if location_indices is None else self.location_effect.weight[location_indices]
            city = location_effect[None]
        else:
            city = torch.zeros(h.shape[0], h.shape[1], 5, device=h.device, dtype=h.dtype)
        values = {}
        for index, name in enumerate(PARAMETERS):
            base = F.softplus(self.global_raw[index]).add(0.05)
            prior = None if priors is None else priors.get(name)
            if self.mode == "CLIMATE_PRIOR" and prior is not None:
                value = prior.to(h).clamp_min(1e-8) * torch.exp(self.maximum * torch.tanh(raw[..., index] + city[..., index]))
            else:
                value = base * torch.exp(self.maximum * torch.tanh(raw[..., index] + city[..., index]))
            values[name] = value
        values["rho"] = self.rho_min + (self.rho_max - self.rho_min) * torch.sigmoid(self.rho_raw + city[..., 4])
        return values

    def regularisation(self) -> dict[str, torch.Tensor]:
        zero = self.global_raw.new_zeros(())
        return {"location_effect_regularisation": self.location_effect.weight.square().mean() if self.location_effects else zero,
                "climate_prior_correction_regularisation": self.delta.weight.square().mean()}


