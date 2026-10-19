"""Seed-level scenario metrics and intervals for externally supplied predictions.

The article's fifth analysis used one fixed synthetic epidemic and ten model
initialisations per reporting condition. This module accepts predictions from
the data holder; it does not distribute or reconstruct study inputs.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t as student_t

from .evaluation import top_k_capture


KEY = ("city", "origin_date", "target_date", "horizon")
REQUIRED = set(KEY) | {"observed", "predicted"}
METRICS = ("MAE", "RMSE", "Pearson", "R2", "Top10")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_prediction_file(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    missing = REQUIRED - set(frame.columns)
    if missing:
        raise ValueError(f"Missing prediction columns in {path}: {sorted(missing)}")
    if frame.duplicated(list(KEY)).any():
        raise ValueError(f"Repeated target key in {path}")
    frame = frame.sort_values(list(KEY), kind="stable").reset_index(drop=True)
    if not np.isfinite(frame[["observed", "predicted"]].to_numpy(dtype=float)).all():
        raise ValueError(f"Nonfinite target or prediction in {path}")
    if (frame[["observed", "predicted"]] < 0).to_numpy().any():
        raise ValueError(f"Negative target or prediction in {path}")
    return frame


def score_horizon(frame: pd.DataFrame, top_k: int = 10) -> dict[str, float]:
    # The archived run sorted city names before calculating each horizon's rank.
    frame = frame.sort_values("city", kind="stable")
    observed = frame["observed"].to_numpy(dtype=float)
    predicted = frame["predicted"].to_numpy(dtype=float)
    if len(observed) < top_k:
        raise ValueError("Fewer sites than the requested top-k cutoff")
    variance = float(np.square(observed - observed.mean()).sum())
    correlation = (float(np.corrcoef(observed, predicted)[0, 1])
                   if observed.std() > 0 and predicted.std() > 0 else float("nan"))
    return {
        "observed_total": float(observed.sum()),
        "predicted_total": float(predicted.sum()),
        "MAE": float(np.abs(observed - predicted).mean()),
        "RMSE": float(np.sqrt(np.square(observed - predicted).mean())),
        "Pearson": correlation,
        "R2": (float(1 - np.square(observed - predicted).sum() / variance)
               if variance > 0 else float("nan")),
        "Top10": float(top_k_capture(observed, predicted, top_k)),
    }


def t_interval(values: np.ndarray) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    if values.size < 2 or not np.isfinite(values).all():
        raise ValueError("At least two finite run-level metrics are required")
    mean = float(values.mean())
    sd = float(values.std(ddof=1))
    half = float(student_t.ppf(0.975, values.size - 1) * sd / np.sqrt(values.size))
    return mean, mean - half, mean + half


def summarise_scenarios(root: Path, output: Path, *, seeds: tuple[int, ...],
                        scenarios: tuple[str, ...], models: tuple[str, ...],
                        top_k: int = 10) -> dict:
    """Read the archived run layout and enforce identical test target keys.

    Expected input path: seed_<seed>/<scenario>/outputs/<model>/predictions.csv.
    All model/scenario/seed fits must share the same keys and observed values.
    """
    if len(set(seeds)) != len(seeds) or len(seeds) < 2:
        raise ValueError("Provide at least two distinct model seeds")
    if not scenarios or not models:
        raise ValueError("Provide reporting conditions and models")
    reference = None
    records, input_hashes = [], {}
    for seed in seeds:
        for scenario in scenarios:
            for model in models:
                path = root / f"seed_{seed}" / scenario / "outputs" / model / "predictions.csv"
                frame = read_prediction_file(path)
                targets = frame[list(KEY) + ["observed"]]
                if reference is None:
                    reference = targets
                elif not reference.equals(targets):
                    raise ValueError(f"Test target keys or observed values differ: {path}")
                input_hashes[str(path.relative_to(root))] = sha256_file(path)
                for horizon, subset in frame.groupby("horizon", sort=True):
                    records.append({"seed": seed, "scenario": scenario, "model": model,
                                    "horizon": int(horizon),
                                    **score_horizon(subset, top_k=top_k)})
    per_run = pd.DataFrame.from_records(records)
    summary = []
    for (scenario, model, horizon), group in per_run.groupby(
            ["scenario", "model", "horizon"], sort=False):
        if len(group) != len(seeds) or set(group.seed) != set(seeds):
            raise ValueError("Incomplete seed coverage")
        row = {"scenario": scenario, "model": model, "horizon": int(horizon),
               "n": len(group), "observed_total": float(group.observed_total.iloc[0])}
        for metric in METRICS:
            mean, low, high = t_interval(group[metric].to_numpy(dtype=float))
            row[f"{metric}_mean"] = mean
            row[f"{metric}_sd"] = float(group[metric].std(ddof=1))
            row[f"{metric}_low"] = low
            row[f"{metric}_high"] = high
        summary.append(row)
    output.mkdir(parents=True, exist_ok=True)
    per_run.to_csv(output / "PER_RUN.csv", index=False)
    pd.DataFrame(summary).to_csv(output / "MEAN_T_INTERVALS.csv", index=False)
    audit = {
        "model_runs": len(input_hashes),
        "fixed_epidemic_realisations": 1,
        "n_model_initialisations": len(seeds),
        "shared_target_keys": int(len(reference)),
        "scenarios": list(scenarios), "models": list(models),
        "interval": "two-sided 95% t interval for the mean run-level metric",
        "top_k": top_k,
        "tie_rule": "NumPy default argsort after sorting by city key, matching archived analysis",
        "input_sha256": input_hashes,
    }
    (output / "AUDIT.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    return audit
