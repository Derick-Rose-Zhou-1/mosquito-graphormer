"""Summarise externally supplied scenario predictions without study data."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mosquito_graphormer.replicate_analysis import summarise_scenarios


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, action="append", required=True)
    parser.add_argument("--scenario", action="append", required=True)
    parser.add_argument("--model", action="append", required=True)
    args = parser.parse_args()
    audit = summarise_scenarios(args.runs, args.output,
                                seeds=tuple(args.seed), scenarios=tuple(args.scenario),
                                models=tuple(args.model))
    print(json.dumps({k: v for k, v in audit.items() if k != "input_sha256"}, indent=2))


if __name__ == "__main__":
    main()
