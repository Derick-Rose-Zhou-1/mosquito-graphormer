"""Single-component interventions from the matched-component experiment."""

from copy import deepcopy

ARMS = ("FULL", "NO_WEATHER_ENCODER", "NO_MOBILITY", "NO_GIS", "NO_TCN", "NO_GRAPHORMER")


def arm_config(base, arm="FULL"):
    if arm not in ARMS:
        raise ValueError(f"Unknown arm: {arm}; expected one of {ARMS}")
    cfg = deepcopy(base)
    if arm == "NO_WEATHER_ENCODER":
        # Climate-derived mechanistic priors are deliberately retained.
        cfg["temporal_feature_names"] = [
            name for name in cfg["temporal_feature_names"]
            if name not in {"pre", "rh", "tmp", "vp"}
        ]
    elif arm != "FULL":
        cfg[{"NO_MOBILITY": "use_mobility", "NO_GIS": "use_gis",
             "NO_TCN": "use_tcn", "NO_GRAPHORMER": "use_graphormer"}[arm]] = False
    return cfg
