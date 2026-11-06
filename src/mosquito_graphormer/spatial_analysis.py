"""Post hoc city-exclusion analysis of already aggregated city predictions."""

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from .evaluation import forecast_metrics


def spatial_sensitivity(frame, exclude_cities=()):
    """Input: panel, city, observed, predicted_mean; one row per panel/city.

    Does not refit a model or change aggregation/forecast horizons. P-values
    are nominal and do not account for spatial dependence or post hoc selection.
    """
    columns = ["panel", "city", "observed", "predicted_mean"]
    if not set(columns).issubset(frame.columns):
        raise ValueError(f"Required columns: {columns}")
    data = frame[columns].copy()
    if data.empty or data.isna().any().any() or data.duplicated(["panel", "city"]).any():
        raise ValueError("Empty input, null values or duplicate cities within a panel")
    values = data[["observed", "predicted_mean"]].to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Counts must be finite and nonnegative")
    excluded = set(exclude_cities)
    rows = []
    for panel, group in data.groupby("panel", sort=True):
        if not excluded.issubset(set(group.city)):
            raise ValueError(f"Excluded city missing in panel {panel}")
        selections = [("all", group)]
        if excluded:
            selections.append(("excluded", group[~group.city.isin(excluded)]))
        for selection, subset in selections:
            if len(subset) < 3:
                raise ValueError("At least three cities are required after exclusion")
            y, p = subset.observed.to_numpy(), subset.predicted_mean.to_numpy()
            variable = np.std(y) > 0 and np.std(p) > 0
            rho, spearman_p = spearmanr(y, p) if variable else (np.nan, np.nan)
            pearson_p = pearsonr(y, p).pvalue if variable else np.nan
            rows.append(dict(panel=panel, selection=selection, n_cities=len(subset),
                             **forecast_metrics(y, p), spearman=float(rho),
                             pearson_p_nominal=float(pearson_p),
                             spearman_p_nominal=float(spearman_p)))
    return pd.DataFrame(rows)
