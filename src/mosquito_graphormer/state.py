from __future__ import annotations
import torch
from torch import nn


class ScalePreservingStateInitialiser(nn.Module):
    def __init__(self, hidden_dim=32, num_locations=21, init_log_bound=1.0986122886681098, eps=.1, location_effects=False):
        super().__init__(); self.eps, self.bound = eps, init_log_bound
        self.delta = nn.Linear(hidden_dim, 2)
        self.location = nn.Embedding(num_locations, 2) if location_effects else None
        if self.location is not None: nn.init.zeros_(self.location.weight)

    def features(self, history):
        last, mean3, mean7 = history[:, -1], history[:, -3:].mean(1), history[:, -7:].mean(1)
        return torch.stack((torch.log1p(last), torch.log1p(mean3), torch.log1p(mean7), last - mean3), -1), last, mean3, mean7

    def forward(self, history, context, location_indices=None):
        features, last, mean3, mean7 = self.features(history)
        correction = self.delta(context)
        if self.location is not None:
            city_effect = self.location.weight if location_indices is None else self.location.weight[location_indices]
            correction = correction + city_effect[None]
        eh0 = (mean3 + self.eps) * torch.exp(self.bound * torch.tanh(correction[..., 0]))
        ih0 = (last + self.eps) * torch.exp(self.bound * torch.tanh(correction[..., 1]))
        return eh0, ih0, {"features": features, "last": last, "mean3": mean3, "mean7": mean7, "EH0": eh0, "IH0": ih0}
