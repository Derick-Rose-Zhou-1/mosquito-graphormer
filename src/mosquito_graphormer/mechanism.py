"""Coupled human SEIR and mosquito SEI transitions."""
from __future__ import annotations

import torch
from torch import nn


STATE_NAMES = ("SM", "EM", "IM", "SH", "EH", "IH", "RH")


def hazard_probability(rate: torch.Tensor) -> torch.Tensor:
    return -torch.expm1(-rate)


class CoupledSEISEIR(nn.Module):
    def __init__(self, horizon: int, b_hv: float, b_vh: float, sigma_h: float, gamma_h: float):
        super().__init__(); self.horizon = horizon
        for name, value in (("b_hv", b_hv), ("b_vh", b_vh), ("sigma_h", sigma_h), ("gamma_h", gamma_h)):
            self.register_buffer(name, torch.tensor(float(value), dtype=torch.float64))

    def forward(self, parameters: dict[str, torch.Tensor], adjacency: torch.Tensor, population_log1p: torch.Tensor,
                initial_eh: torch.Tensor, initial_ih: torch.Tensor, update_callback=None) -> dict:
        nh = torch.expm1(population_log1p.double()).clamp_min(1.0); b, n = nh.shape
        A = adjacency.double();
        if A.ndim == 2: A = A[None].expand(b, -1, -1)
        if A.shape != (b, n, n): raise ValueError("A_eff must be [B,N,N]")
        params = {k: parameters[k].double() for k in ("a", "sigma_v", "mu_v", "M")}
        eh, ih = initial_eh.double().clamp_min(0), initial_ih.double().clamp_min(0)
        sh = (nh - eh - ih).clamp_min(0); rh = torch.zeros_like(nh)
        infectious_fraction = torch.exp(-params["mu_v"]) * hazard_probability(params["sigma_v"])
        im = params["M"] * infectious_fraction; em = torch.zeros_like(im); sm = params["M"] * (1 - infectious_fraction)
        states = {name: [value] for name, value in zip(STATE_NAMES, (sm, em, im, sh, eh, ih, rh))}
        flow_names = ("new_exposed_h", "new_infectious_h", "new_removed_h", "new_exposed_v", "new_infectious_v", "vector_deaths", "lambda_h", "lambda_v")
        flows = {name: [] for name in flow_names}
        for step in range(self.horizon):
            a, sigma_v, mu_v = params["a"], params["sigma_v"], params["mu_v"]
            visitor_prevalence = torch.einsum("bjn,bj->bn", A, ih / nh)
            lambda_v, lambda_h = a * self.b_hv * visitor_prevalence, a * self.b_vh * (im / nh)
            new_eh, new_ih, new_rh = sh * hazard_probability(lambda_h), eh * hazard_probability(self.sigma_h), ih * hazard_probability(self.gamma_h)
            total_s, total_e = lambda_v + mu_v, sigma_v + mu_v
            leave_s, leave_e = sm * hazard_probability(total_s), em * hazard_probability(total_e)
            new_em = leave_s * (lambda_v / total_s.clamp_min(1e-12)); new_im = leave_e * (sigma_v / total_e.clamp_min(1e-12))
            deaths = leave_s * (mu_v / total_s.clamp_min(1e-12)) + leave_e * (mu_v / total_e.clamp_min(1e-12)) + im * hazard_probability(mu_v)
            sm, em, im = sm * torch.exp(-total_s), em * torch.exp(-total_e) + new_em, im * torch.exp(-mu_v) + new_im
            sh, eh, ih, rh = sh * torch.exp(-lambda_h), eh * torch.exp(-self.sigma_h) + new_eh, ih * torch.exp(-self.gamma_h) + new_ih, rh + new_rh
            for name, value in zip(STATE_NAMES, (sm, em, im, sh, eh, ih, rh)): states[name].append(value)
            for name, value in zip(flow_names, (new_eh, new_ih, new_rh, new_em, new_im, deaths, lambda_h, lambda_v)): flows[name].append(value)
            if update_callback is not None and step < self.horizon - 1: params = update_callback(step, params, {"IH": ih, "EH": eh, "IM": im, "N_h": nh})
        states = {k: torch.stack(v, -1) for k, v in states.items()}; flows = {k: torch.stack(v, -1) for k, v in flows.items()}
        hmass = states["SH"] + states["EH"] + states["IH"] + states["RH"]
        minimum = min(float(x.detach().min()) for x in states.values())
        finite = all(bool(torch.isfinite(x).all()) for x in [*states.values(), *flows.values()])
        qc = {"states_min": minimum, "human_relative_mass_error": float(((hmass - nh[..., None]).abs() / nh[..., None]).detach().max()),
              "finite": finite, "prediction_nonnegative": bool((flows["new_infectious_h"] >= 0).all())}
        qc["pass"] = qc["finite"] and qc["states_min"] >= -1e-7 and qc["human_relative_mass_error"] < 1e-5 and qc["prediction_nonnegative"]
        return {"prediction": flows["new_infectious_h"].to(population_log1p.dtype), "states": states, "flows": flows, "numerical_qc": qc, "effective_parameters": params}


def intensity_from_state(exposed_h: torch.Tensor, sigma_h: torch.Tensor) -> torch.Tensor:
    return exposed_h * hazard_probability(sigma_h)


class SEISEIRHead(CoupledSEISEIR):
    """Expose observation intensities on X0..XH without changing transitions."""

    def forward(self, *args, **kwargs):
        result = super().forward(*args, **kwargs)
        state_intensities = intensity_from_state(result["states"]["EH"], self.sigma_h)
        output_dtype = result["prediction"].dtype
        result["transition_flows_new_infectious_h"] = result["prediction"]
        result["origin_intensity"] = state_intensities[..., 0].to(output_dtype)
        result["prediction"] = state_intensities[..., 1 : self.horizon + 1].to(output_dtype)
        result["origin_state_index"] = 0
        result["future_state_indices"] = tuple(range(1, self.horizon + 1))
        return result
