"""Stateless input transforms and training-only scalers for external data."""

from __future__ import annotations

import numpy as np


def row_normalise_movement(outbound: np.ndarray) -> np.ndarray:
    """Return relative off-diagonal flows with origin rows and destination columns."""
    movement = np.asarray(outbound, dtype=np.float64).copy()
    if movement.ndim < 2 or movement.shape[-1] != movement.shape[-2]:
        raise ValueError("Movement matrices must be square")
    if not np.isfinite(movement).all() or (movement < 0).any():
        raise ValueError("Movement weights must be finite and nonnegative")
    diagonal = np.arange(movement.shape[-1])
    movement[..., diagonal, diagonal] = 0
    totals = movement.sum(axis=-1, keepdims=True)
    return np.divide(movement, totals, out=np.zeros_like(movement), where=totals > 0)


def temperature_proxy(temperature_kelvin: np.ndarray) -> dict[str, np.ndarray]:
    """Apply the fixed panel proxy equations; temperature is measured in K."""
    temperature = np.asarray(temperature_kelvin, dtype=np.float64)
    if not np.isfinite(temperature).all():
        raise ValueError("Temperature must be finite")
    z = np.clip((temperature - 300.0) / 10.0, -3.0, 3.0)
    return {
        "prior_a": np.clip(0.30 * np.exp(0.08 * z), 0.05, 1.5),
        "prior_mu_v": np.clip(0.10 * np.exp(-0.05 * z), 0.02, 0.5),
        "proxy_M": np.clip(300.0 * np.exp(0.12 * z), 30.0, 3000.0),
    }


def fit_standardiser(values: np.ndarray, training_origins: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fit one mean and SD per final feature using training origins only."""
    x = np.asarray(values, dtype=np.float64)
    train = np.asarray(training_origins, dtype=bool)
    if x.ndim < 2 or train.shape != (x.shape[0],) or not train.any():
        raise ValueError("Expected feature array and nonempty training-origin mask")
    subset = x[train]
    if not np.isfinite(subset).all():
        raise ValueError("Nonfinite training features")
    axes = tuple(range(subset.ndim - 1))
    mean = subset.mean(axis=axes)
    sd = subset.std(axis=axes)
    return mean, np.where(sd > 0, sd, 1.0)


def transform_standardised(values: np.ndarray, mean: np.ndarray, sd: np.ndarray) -> np.ndarray:
    x = np.asarray(values, dtype=np.float64)
    if np.any(np.asarray(sd) <= 0):
        raise ValueError("Standard deviations must be positive")
    return (x - mean) / sd


def log_population(population_persons: np.ndarray) -> np.ndarray:
    values = np.asarray(population_persons, dtype=np.float64)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Population must be finite and nonnegative")
    return np.log1p(values)
