"""Generic source-data plotting; no study inputs are bundled."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np


def plot_case_forecasts(observed, forecasts, output_path):
    """Save a compact observed-versus-forecast curve from aligned arrays."""
    y = np.asarray(observed, dtype=float)
    if y.ndim != 1:
        raise ValueError("Observed counts must be one-dimensional")
    figure, axis = plt.subplots(figsize=(7, 3.5))
    axis.plot(np.arange(y.size), y, label="Observed", color="black", linewidth=1.7)
    for label, values in forecasts.items():
        p = np.asarray(values, dtype=float)
        if p.shape != y.shape:
            raise ValueError("Forecasts must align with observed counts")
        axis.plot(np.arange(y.size), p, label=str(label), linewidth=1.5)
    axis.set(xlabel="Target index", ylabel="Reported cases")
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(output_path, dpi=300)
    plt.close(figure)
