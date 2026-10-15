"""Generate a tiny synthetic panel for exercising the public code path."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(19)
    origins, nodes = 8, 3
    history = rng.poisson(2, (origins, 14, nodes)).astype(np.float32)
    movement = rng.uniform(0, 1, (origins, 14, nodes, nodes)).astype(np.float32)
    movement[:, :, np.arange(nodes), np.arange(nodes)] = 0
    movement /= np.maximum(movement.sum(axis=-1, keepdims=True), 1e-8)
    target = np.repeat(history[:, -1, :, None], 7, axis=-1)
    target += np.arange(7, dtype=np.float32)[None, None, :] * 0.05
    np.savez_compressed(
        args.output,
        x=rng.normal(0, 0.2, (origins, 14, nodes, 8)).astype(np.float32),
        history=history, mobility=movement,
        distance=np.abs(np.subtract.outer(np.arange(nodes), np.arange(nodes))).astype(np.float32) * 20,
        geo_support=(1 - np.eye(nodes)).astype(bool),
        population=np.full((origins, nodes), np.log1p(1000), dtype=np.float32),
        prior_a=np.full((origins, nodes), 0.3, dtype=np.float32),
        prior_sigma_v=np.full((origins, nodes), 0.2, dtype=np.float32),
        prior_mu_v=np.full((origins, nodes), 0.1, dtype=np.float32),
        prior_M_z=np.zeros((origins, nodes), dtype=np.float32),
        target=target.astype(np.float32),
        split=np.array([0, 0, 0, 0, 1, 1, 2, 2], dtype=np.int64),
    )
    print(args.output)


if __name__ == "__main__":
    main()
