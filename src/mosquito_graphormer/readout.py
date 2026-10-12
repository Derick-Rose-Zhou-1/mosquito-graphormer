from __future__ import annotations
import math
import torch
from torch import nn
from torch.nn import functional as F

def activity_gate(last_reported, k_activity, eps=0.1):
    return last_reported / (last_reported + k_activity + eps)


def relative_log_change(origin_intensity, future_intensity, eps_intensity=0.1):
    if eps_intensity <= 0:
        raise ValueError("eps_intensity must be positive")
    origin_intensity = origin_intensity.to(torch.float64)
    future_intensity = future_intensity.to(torch.float64)
    if origin_intensity.ndim == future_intensity.ndim - 1:
        origin_intensity = origin_intensity.unsqueeze(-1)
    return torch.log(future_intensity.clamp_min(0) + eps_intensity) - torch.log(
        origin_intensity.clamp_min(0) + eps_intensity
    )


def anchored_prediction(
    last_reported,
    origin_intensity,
    future_intensity,
    max_log_growth=2.0,
    eps_intensity=0.1,
    activity_gate_value=1.0,
):
    output_dtype = future_intensity.dtype
    last_reported, origin_intensity, future_intensity = [
        value.to(torch.float64) for value in (last_reported, origin_intensity, future_intensity)
    ]
    raw = relative_log_change(origin_intensity, future_intensity, eps_intensity)
    if torch.is_tensor(activity_gate_value) and activity_gate_value.ndim == raw.ndim - 1:
        activity_gate_value = activity_gate_value.unsqueeze(-1)
    delta = max_log_growth * torch.tanh(raw / max_log_growth) * activity_gate_value
    if last_reported.ndim == delta.ndim - 1:
        last_reported = last_reported.unsqueeze(-1)
    return torch.expm1(torch.log1p(last_reported.clamp_min(0)) + delta).clamp_min(0).to(output_dtype)


class ShrunkMechanisticObservationHead(nn.Module):
    """Two-coordinate, identity-centred smooth observation calibration."""

    def __init__(self, eps: float = 0.1, shrinkage_scale: float = 0.25):
        super().__init__()
        self.beta_slope = nn.Parameter(torch.zeros(()))
        self.beta_offset = nn.Parameter(torch.zeros(()))
        self.eps = float(eps)
        self.shrinkage_scale = float(shrinkage_scale)

    def regularisation(self) -> torch.Tensor:
        s = self.shrinkage_scale
        return 0.5 * ((self.beta_slope / s) ** 2 + (self.beta_offset / s) ** 2)

    def forward(self, last_reported: torch.Tensor, base_prediction: torch.Tensor) -> torch.Tensor:
        last = last_reported.to(torch.float64)
        base = base_prediction.to(torch.float64).clamp_min(0.0)
        base_delta = torch.log1p(base) - torch.log1p(last.clamp_min(0.0)).unsqueeze(-1)
        horizon = torch.linspace(0.0, 1.0, base.shape[-1], dtype=base.dtype, device=base.device)
        calibrated_delta = base_delta * torch.exp(horizon * self.beta_slope) + horizon * self.beta_offset
        return torch.expm1(torch.log1p(last.clamp_min(0.0)).unsqueeze(-1) + calibrated_delta).clamp_min(0.0).to(base_prediction.dtype)



class BoundedIncidenceAdapter(nn.Module):
    def __init__(self, horizon: int = 7, max_log_correction: float = 1.1):
        super().__init__()
        self.max_log_correction = max_log_correction
        self.weight = nn.Parameter(torch.zeros(horizon, 6))
        self.bias = nn.Parameter(torch.zeros(horizon))

    def forward(self, full: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        delta = self.max_log_correction * torch.tanh((features * self.weight).sum(-1) + self.bias)
        return torch.log1p(full.clamp_min(0)) + delta


class RecentHistoryTrendGate(nn.Module):
    """Bounded, learned log-count correction from causal 14-day trends."""

    def __init__(self, horizon: int = 7, max_log_correction: float = 1.1):
        super().__init__()
        self.max_log_correction = max_log_correction
        self.weight = nn.Parameter(torch.zeros(horizon, 3))
        self.bias = nn.Parameter(torch.zeros(horizon))

    def forward(self, history: torch.Tensor, horizon: int) -> torch.Tensor:
        if history.ndim != 3 or history.shape[1] != 14:
            raise ValueError("history must have shape [batch, 14, nodes]")
        if horizon != self.weight.shape[0]:
            raise ValueError("horizon differs from the seven-day forecast horizon")
        h = history.clamp_min(0)
        log_mean = lambda x: torch.log1p(x.mean(dim=1))
        last3 = log_mean(h[:, -3:])
        previous3 = log_mean(h[:, -6:-3])
        last7 = log_mean(h[:, -7:])
        previous7 = log_mean(h[:, :7])
        last = torch.log1p(h[:, -1])
        trends = torch.stack((last3 - previous3, last7 - previous7, last - last7), dim=-1)
        return self.max_log_correction * torch.tanh(
            torch.einsum("bnf,hf->bnh", trends, self.weight) + self.bias
        )


class ZeroLatestMechanismBridge(nn.Module):
    """Learn a reported-count link when the last observed count is zero.

    The mechanistic anchor multiplies its mechanism change by zero in this
    case, disconnecting all upstream gradients. This branch uses only the
    model's causal mechanism and the observed 14-day history.
    """

    def __init__(self, horizon: int = 7, initial_count: float = 0.1):
        super().__init__()
        self.bias = nn.Parameter(torch.full((horizon,), math.log(math.expm1(initial_count))))
        self.mechanism_weight = nn.Parameter(torch.ones(horizon))
        self.history_weight = nn.Parameter(torch.zeros(horizon, 2))

    def forward(self, mechanism: torch.Tensor, history: torch.Tensor) -> torch.Tensor:
        mean3 = torch.log1p(history[:, -3:].clamp_min(0).mean(1))
        mean7 = torch.log1p(history[:, -7:].clamp_min(0).mean(1))
        recent = torch.stack((mean3, mean7), dim=-1)
        logit = (self.bias + self.mechanism_weight * torch.log1p(mechanism.clamp_min(0))
                 + torch.einsum("bnf,hf->bnh", recent, self.history_weight))
        return F.softplus(logit)


