"""Paired-seed summaries; never substitute ensemble metrics for seed means."""

import numpy as np
import pandas as pd

from .ablations import ARMS
from .evaluation import forecast_metrics

KEYS = ["origin", "target_date", "horizon", "city"]


def summarize_ablations(frame, expected_seeds):
    """Require all six arms, explicit seeds, identical targets and forecast slots.

    Return per-seed metrics, mean/sample-SD summary, paired differences and
    separately labelled ensemble metrics. Each slot receives equal weight.
    """
    columns = ["arm", "seed", *KEYS, "observed", "prediction"]
    if not set(columns).issubset(frame.columns):
        raise ValueError(f"Required columns: {columns}")
    data = frame[columns].copy()
    seeds = list(expected_seeds)
    if len(seeds) < 2 or len(set(seeds)) != len(seeds):
        raise ValueError("Supply at least two distinct expected seeds")
    if data.isna().any().any() or data.duplicated(["arm", "seed", *KEYS]).any():
        raise ValueError("Null or duplicate prediction keys/values")
    if set(data.arm) != set(ARMS) or set(data.seed) != set(seeds):
        raise ValueError("Arm or seed coverage differs from the requested experiment")
    values = data[["observed", "prediction"]].to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Counts must be finite and nonnegative")
    origin = pd.to_datetime(data.origin, errors="raise")
    target = pd.to_datetime(data.target_date, errors="raise")
    horizons = pd.to_numeric(data.horizon, errors="raise").to_numpy(dtype=float)
    if (not np.isfinite(horizons).all() or (horizons < 1).any()
            or (horizons != np.floor(horizons)).any()
            or not ((target - origin).dt.total_seconds() / 86400 == horizons).all()):
        raise ValueError("Target date must equal origin plus positive integer horizon")
    reference = data[(data.arm == "FULL") & (data.seed == seeds[0])].set_index(KEYS).sort_index()
    rows = []
    for arm in ARMS:
        for seed in seeds:
            group = data[(data.arm == arm) & (data.seed == seed)].set_index(KEYS).sort_index()
            if not group.index.equals(reference.index) or not np.array_equal(group.observed, reference.observed):
                raise ValueError(f"Unmatched forecast slots or observed values: {arm}, {seed}")
            rows.append(dict(arm=arm, seed=seed, n_slots=len(group),
                             **forecast_metrics(group.observed, group.prediction)))
    per_seed = pd.DataFrame(rows)
    metrics = ["mae", "rmse", "r2", "pearson", "ccc"]
    summary = []
    paired = []
    full = per_seed[per_seed.arm == "FULL"].set_index("seed")
    for arm in ARMS:
        group = per_seed[per_seed.arm == arm].set_index("seed")
        row = dict(arm=arm, n_seeds=len(seeds), slots_per_seed=len(reference))
        for metric in metrics:
            # NumPy intentionally propagates undefined metrics, rather than
            # silently excluding seeds with undefined correlations or R-squared.
            scores = group[metric].to_numpy()
            row[metric + "_mean"] = np.mean(scores)
            row[metric + "_sd"] = np.std(scores, ddof=1)
        summary.append(row)
        if arm != "FULL":
            for seed in seeds:
                paired.append(dict(arm=arm, seed=seed, **{
                    m + "_ablation_minus_full": group.loc[seed, m] - full.loc[seed, m]
                    for m in metrics}))
    ensemble = data.groupby(["arm", *KEYS], sort=True).agg(
        observed=("observed", "first"), prediction=("prediction", "mean")).reset_index()
    ensemble_metrics = pd.DataFrame([
        dict(arm=arm, **forecast_metrics(g.observed, g.prediction))
        for arm, g in ensemble.groupby("arm")])
    return {"per_seed": per_seed, "seed_mean_sd": pd.DataFrame(summary),
            "paired_differences": pd.DataFrame(paired), "ensemble_metrics": ensemble_metrics}
