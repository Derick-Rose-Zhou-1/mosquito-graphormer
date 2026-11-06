"""Summarize matched predictions supplied by the user; no training or data download."""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mosquito_graphormer.ablation_analysis import summarize_ablations


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seeds", required=True, nargs="+", type=int)
    args = parser.parse_args()
    results = summarize_ablations(pd.read_csv(args.input), args.seeds)
    args.output.mkdir(parents=True, exist_ok=True)
    for name, frame in results.items():
        frame.to_csv(args.output / f"{name}.csv", index=False)


if __name__ == "__main__":
    main()
