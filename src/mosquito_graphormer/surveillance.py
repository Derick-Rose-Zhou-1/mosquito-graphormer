"""Reporting perturbations for externally supplied synthetic epidemics.

The fixed reference epidemic and its evaluation targets are kept outside this
module. Only the available case-input history is modified.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd


def apply_reporting_scenario(panel: pd.DataFrame, scenario: str, *,
                             known_start: str, known_days: int, seed: int,
                             core_sites: Iterable[str] = (),
                             focus_site: str | None = None) -> pd.DataFrame:
    required = {"site", "date", "cases", "population"}
    if not required.issubset(panel.columns):
        raise ValueError(f"Missing columns: {sorted(required - set(panel.columns))}")
    if known_days < 1:
        raise ValueError("known_days must be positive")
    out = panel.copy().sort_values(["date", "site"], kind="stable").reset_index(drop=True)
    out["date"] = pd.to_datetime(out["date"])
    if out.duplicated(["site", "date"]).any():
        raise ValueError("Duplicate site and date")
    values = out["cases"].to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Case inputs must be finite and nonnegative")
    start = pd.Timestamp(known_start)
    end = start + pd.Timedelta(days=known_days - 1)
    period = out["date"].between(start, end).to_numpy()
    if not period.any():
        raise ValueError("Known window does not overlap input panel")
    out["case_observed_mask"] = 1.0
    after = (out["date"] > end).to_numpy()
    out.loc[after, ["cases", "case_observed_mask"]] = 0.0
    rng = np.random.default_rng(seed)

    if scenario == "sparse_early":
        pass
    elif scenario == "missing_reports":
        eligible = period & (out["cases"].to_numpy(dtype=float) >= 1.0)
        chosen = np.flatnonzero(eligible)
        removed = chosen[rng.uniform(size=chosen.size) < 0.20]
        out.loc[removed, ["cases", "case_observed_mask"]] = 0.0
    elif scenario == "reporting_delay":
        for _, rows in out.loc[period].groupby("site", sort=False):
            ids = rows.sort_values("date").index
            old = out.loc[ids, "cases"].to_numpy(dtype=float)
            delayed = np.zeros_like(old)
            if old.size > 3:
                delayed[3:] = old[:-3]
            out.loc[ids, "cases"] = delayed
    elif scenario == "resource_underreporting":
        population = out.groupby("site", sort=False)["population"].median()
        median = float(population.median())
        rates = {
            str(site): (0.90 if site == focus_site else
                        (0.90 if value >= median else 0.70))
            for site, value in population.items()
        }
        ids = np.flatnonzero(period)
        counts = np.rint(out.loc[ids, "cases"].to_numpy(dtype=float)).astype(int)
        probabilities = out.loc[ids, "site"].map(rates).to_numpy(dtype=float)
        out.loc[ids, "cases"] = [rng.binomial(max(int(n), 0), p)
                                  for n, p in zip(counts, probabilities)]
    elif scenario == "spatial_heterogeneity":
        core = set(core_sites)
        for selected, rate in ((period & out["site"].isin(core).to_numpy(), 1.0),
                               (period & ~out["site"].isin(core).to_numpy(), 0.30)):
            ids = np.flatnonzero(selected)
            counts = np.rint(out.loc[ids, "cases"].to_numpy(dtype=float)).astype(int)
            out.loc[ids, "cases"] = [rng.binomial(max(int(n), 0), rate) for n in counts]
    else:
        raise ValueError(f"Unknown reporting scenario: {scenario}")
    out["cases_ln"] = np.log1p(out["cases"].to_numpy(dtype=float))
    return out
