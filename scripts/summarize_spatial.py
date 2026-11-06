"""Evaluate spatial agreement before and after specified city exclusions."""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mosquito_graphormer.spatial_analysis import spatial_sensitivity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--exclude-city", action="append", default=[])
    args = parser.parse_args()
    result = spatial_sensitivity(pd.read_csv(args.input), args.exclude_city)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
