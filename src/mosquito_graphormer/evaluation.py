"""Metrics for nonnegative case forecasts on caller-supplied targets."""
from __future__ import annotations

import numpy as np


def forecast_metrics(observed, predicted):
    y = np.asarray(observed, dtype=float).reshape(-1)
    p = np.asarray(predicted, dtype=float).reshape(-1)
    if y.shape != p.shape or y.size == 0 or not np.isfinite(y).all() or not np.isfinite(p).all():
        raise ValueError("Observed and predicted counts must be finite and aligned")
    mse = np.mean((p - y) ** 2)
    variance = np.sum((y - y.mean()) ** 2)
    correlation = np.corrcoef(y, p)[0, 1] if np.std(y) > 0 and np.std(p) > 0 else np.nan
    ccc_denominator = np.var(y) + np.var(p) + (y.mean() - p.mean()) ** 2
    concordance = (2 * np.cov(y, p, bias=True)[0, 1] / ccc_denominator
                   if ccc_denominator > 0 else np.nan)
    return {
        "mae": float(np.mean(np.abs(p - y))),
        "rmse": float(np.sqrt(mse)),
        "r2": float(1 - np.sum((p - y) ** 2) / variance) if variance > 0 else np.nan,
        "pearson": float(correlation),
        "ccc": float(concordance),
    }


def top_k_capture(observed, predicted, k):
    y = np.asarray(observed, dtype=float).reshape(-1)
    p = np.asarray(predicted, dtype=float).reshape(-1)
    if y.shape != p.shape or not 1 <= k <= y.size:
        raise ValueError("Expected aligned one-dimensional scores and a valid k")
    # Stable input order resolves ties; callers should pre-sort by a fixed location key.
    truth = set(np.argsort(-y, kind="stable")[:k])
    ranked = set(np.argsort(-p, kind="stable")[:k])
    return len(truth & ranked) / k
