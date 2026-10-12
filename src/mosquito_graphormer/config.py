"""Architecture defaults; data splits and locations are supplied by the caller."""
from __future__ import annotations

FEATURE_NAMES = ("pre", "rh", "tmp", "vp", "dow_sin", "dow_cos", "mon_sin", "mon_cos")

def model_config() -> dict:
    return {
        "hidden_dim": 32, "history_days": 14, "horizon": 7,
        "use_tcn": True, "use_graphormer": True, "use_mobility": True,
        "use_gis": True, "use_location_effect": True, "anchored": True,
        "use_mechanism": True, "use_activity_gate": True,
        "parameter_mode": "CLIMATE_PRIOR", "temporal_feature_names": list(FEATURE_NAMES),
        "init_log_bound": 1.0986122886681098, "eps": 0.1,
        "max_log_growth": 2.0, "tau_geo_initial": 200.0,
        "eta_initial": 0.70, "gamma_spatial_initial": 0.15,
        "lambda_mobility_initial": 0.50, "m_proxy_scale": 0.20,
    }
