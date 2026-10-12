from __future__ import annotations

import math

import torch
from torch import nn

from .graph import MaskedGraphTransformer, MobilityGeographicGraph
from .parameters import EffectiveParameterHead
from .temporal import CausalDilatedTCN
from .state import ScalePreservingStateInitialiser

from .readout import activity_gate, anchored_prediction, ShrunkMechanisticObservationHead, BoundedIncidenceAdapter, RecentHistoryTrendGate, ZeroLatestMechanismBridge
from .mechanism import SEISEIRHead


class ForecastBackbone(nn.Module):
    def __init__(self, cfg, num_features, num_locations, feature_names):
        super().__init__()
        self.cfg, self.num_locations, self.feature_names = dict(cfg), num_locations, list(feature_names)
        names = cfg.get("temporal_feature_names", ["pre", "rh", "tmp", "vp", "dow_sin", "dow_cos", "mon_sin", "mon_cos"])
        self.feature_indices = [self.feature_names.index(name) for name in names]
        hidden = cfg["hidden_dim"]
        self.temporal = CausalDilatedTCN(len(self.feature_indices), hidden, 3, (1, 2, 4), 0.1) if cfg.get("use_tcn", True) else nn.Linear(len(self.feature_indices), hidden)
        self.case_projection = nn.Linear(4, hidden)
        self.graph = MobilityGeographicGraph(cfg["tau_geo_initial"], cfg["eta_initial"], cfg["gamma_spatial_initial"], cfg["lambda_mobility_initial"])
        self.graphormer = MaskedGraphTransformer(hidden, 2, 0.1) if cfg.get("use_graphormer", True) else nn.Identity()
        self.initialiser = ScalePreservingStateInitialiser(hidden, num_locations, cfg["init_log_bound"], cfg["eps"], cfg["use_location_effect"])
        self.parameter_head = EffectiveParameterHead(hidden, num_locations, cfg.get("parameter_mode", "CLIMATE_PRIOR"), cfg["use_location_effect"], 0.20, 0.05, 2.0)
        self.mechanism = SEISEIRHead(7, 0.35, 0.45, 0.2, 1 / 6)
        self.observation = ShrunkMechanisticObservationHead(cfg["eps"])
        self.raw_k_activity = nn.Parameter(torch.tensor(math.log(math.expm1(5.0 - cfg["eps"])), dtype=torch.float32))

    def k_activity(self):
        return torch.nn.functional.softplus(self.raw_k_activity) + self.cfg["eps"]

    def _latent(self, batch, no_spatial=False, no_mobility=False, no_gis=False):
        x = batch["x"][..., self.feature_indices]
        h = self.temporal(x)[:, -1] if self.cfg.get("use_tcn", True) else self.temporal(x[:, -1])
        state_features, *_ = self.initialiser.features(batch["history"])
        h = h + self.case_projection(state_features)
        use_mobility = self.cfg.get("use_mobility", True) if no_mobility is False else not no_mobility
        use_gis = self.cfg.get("use_gis", True) if no_gis is False else not no_gis
        spatial_off = not (use_mobility or use_gis) if no_spatial is False else no_spatial
        graph = self.graph(batch["mobility"], batch["distance"], batch["geo_support"], use_mobility, use_gis, spatial_off)
        return (self.graphormer(h, graph) if self.cfg.get("use_graphormer", True) else h), graph

    def forward(self, batch, no_spatial=False, no_mobility=False, no_gis=False):
        h, graph = self._latent(batch, no_spatial, no_mobility, no_gis)
        location_indices = batch.get("location_indices")
        eh0, ih0, init_diag = self.initialiser(batch["history"], h, location_indices)
        priors = {
            "a": batch["prior_a"],
            "sigma_v": batch["prior_sigma_v"],
            "mu_v": batch["prior_mu_v"],
            "M": 300.0 * torch.exp(self.cfg["m_proxy_scale"] * batch["prior_M_z"]).clamp(0.05, 20.0),
        }
        params = self.parameter_head(h, priors, location_indices)
        if not self.cfg.get("use_mechanism", True):
            future_intensity = ih0[..., None].expand(-1, -1, 7)
            result = {
                "prediction": future_intensity,
                "origin_intensity": ih0,
                "origin_state_index": 0,
                "future_state_indices": tuple(range(1, 8)),
                "states": {},
                "flows": {},
                "numerical_qc": {"pass": True},
            }
        else:
            result = self.mechanism(params, graph, batch["population"], eh0, ih0)
        future_intensity = result["prediction"].clamp_min(1e-8)
        if self.cfg.get("anchored", False):
            gate = activity_gate(init_diag["last"], self.k_activity(), self.cfg["eps"]) if self.cfg.get("use_activity_gate", True) else 1.0
            raw = anchored_prediction(init_diag["last"], result["origin_intensity"], future_intensity, self.cfg["max_log_growth"], self.cfg["eps"], gate)
            raw = self.observation(init_diag["last"], raw)
        else:
            raw = future_intensity
        result.update(
            {
                "raw_prediction": raw,
                "origin_intensity": result["origin_intensity"],
                "graph": graph,
                "latent": h,
                "effective_parameters": params,
                "parameter_regularisation": self.parameter_head.regularisation(),
                "state_initialisation": init_diag,
                "observation_regularisation": self.observation.regularisation(),
            }
        )
        return result




class HistoryMixture(nn.Module):
    def __init__(self):
        super().__init__()
        self.logits = nn.Parameter(torch.zeros(2))
    def forward(self, slopes):
        return (slopes * self.logits.softmax(0)).sum(-1)

class MechanismTrust(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(7, 5))
        self.bias = nn.Parameter(torch.zeros(7))
    def forward(self, features):
        return torch.sigmoid(torch.einsum('bnf,hf->bnh', features, self.weight) + self.bias)

class MosquitoGraphormer(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.adapter = BoundedIncidenceAdapter()
        self.trend_gate = RecentHistoryTrendGate()
        self.zero_latest_bridge = ZeroLatestMechanismBridge()
        self.history_mixture = HistoryMixture()
        self.mechanism_trust = MechanismTrust()

    def modified_backbone(self, output, history):
        last = history[:, -1].clamp_min(0)
        h = history.clamp_min(0)
        m3, p3 = torch.log1p(h[:, -3:].mean(1)), torch.log1p(h[:, -6:-3].mean(1))
        m7, p7 = torch.log1p(h[:, -7:].mean(1)), torch.log1p(h[:, :7].mean(1))
        slopes = torch.stack(((m3-p3)/3, (m7-p7)/7), -1)
        slope = self.history_mixture(slopes)
        G, eps = self.backbone.cfg['max_log_growth'], self.backbone.cfg['eps']
        mech = G * torch.tanh((torch.log(output['prediction'].clamp_min(0)+eps)
                    - torch.log(output['origin_intensity'].clamp_min(0)+eps)[..., None])/G)
        horizon = torch.arange(1, 8, dtype=h.dtype, device=h.device)
        empirical = G * torch.tanh(slope[..., None] * horizon/G)
        features = torch.stack((torch.log1p(last), m7,
                     torch.log1p(output['origin_intensity'].clamp_min(0)), m3-p3, m7-p7), -1)
        trust = self.mechanism_trust(features)
        activity = last/(last+self.backbone.k_activity()+eps)
        raw = torch.expm1(torch.log1p(last)[..., None]
                 + (trust*mech+(1-trust)*empirical)*activity[..., None]).clamp_min(0)
        output['mechanism_trust'] = trust
        return self.backbone.observation(last, raw)

    def forward(self, batch):
        # Exclude future targets from the causal model input.
        causal = {k:v for k,v in batch.items() if k != 'future_observed'}
        history = causal['history']
        if history.ndim != 3 or history.shape[1] != 14:
            raise ValueError('history must be [batch,14,nodes]')
        if history.device.type == 'cpu': torch.backends.mkldnn.enabled = False
        output = self.backbone(causal)
        backbone_raw = self.modified_backbone(output, history)
        zero_link = self.zero_latest_bridge(output['prediction'], history)
        full = torch.where(history[:, -1, :, None] <= 0, zero_link, backbone_raw)
        expand = lambda x: x[..., None].expand(-1, -1, 7)
        features = torch.stack((
            torch.log1p(expand(history[:, -1].clamp_min(0))),
            torch.log1p(expand(history[:, -3:].mean(1).clamp_min(0))),
            torch.log1p(expand(history[:, -7:].mean(1).clamp_min(0))),
            torch.log1p(full.clamp_min(0)), torch.log1p(output['prediction'].clamp_min(0)),
            torch.log1p(expand(output['origin_intensity'].clamp_min(0)))), -1)
        final = torch.expm1(self.adapter(full, features)+self.trend_gate(history, 7)).clamp_min(1e-8)
        result = {**output, 'backbone_prediction':backbone_raw,
                  'zero_latest_bridge_prediction':zero_link, 'raw_prediction':final}
        return result

