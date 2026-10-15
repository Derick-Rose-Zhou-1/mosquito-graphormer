"""Fit and evaluate an externally prepared panel without embedded locations or dates."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mosquito_graphormer.workflow import fit_and_evaluate, load_panel


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    panel = load_panel(args.panel)
    report = fit_and_evaluate(
        panel, args.output, args.seed,
        max_epochs=args.epochs, batch_size=args.batch_size,
        learning_rate=args.learning_rate, weight_decay=args.weight_decay,
        patience=args.patience, device_name=args.device,
    )
    print(f"seed={args.seed} best_epoch={report['best_epoch']} "
          f"development_rmse={report['development_rmse']:.5g}")


if __name__ == "__main__":
    main()
