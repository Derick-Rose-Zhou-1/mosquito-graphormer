"""Plot one generic horizon from an externally generated prediction file."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mosquito_graphormer.plotting import plot_case_forecasts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--horizon", type=int, choices=range(1, 8), default=1)
    args = parser.parse_args()
    with np.load(args.predictions, allow_pickle=False) as data:
        observed = data["observed"]
        predicted = data["predicted"]
        mask = data["mask"] if "mask" in data.files else np.ones_like(observed, dtype=bool)
    if observed.shape != predicted.shape or observed.ndim != 3:
        raise ValueError("Expected aligned [origins,nodes,horizons] arrays")
    if mask.shape != observed.shape:
        raise ValueError("Mask must align with observed and predicted arrays")
    index = args.horizon - 1
    valid = mask[:, :, index].any(axis=1)
    y = np.where(mask[:, :, index], observed[:, :, index], 0).sum(axis=1)[valid]
    p = np.where(mask[:, :, index], predicted[:, :, index], 0).sum(axis=1)[valid]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    plot_case_forecasts(y, {"Model": p}, args.output)
    print(args.output)


if __name__ == "__main__":
    main()
