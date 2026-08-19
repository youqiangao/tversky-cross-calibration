#!/usr/bin/env python3
"""Compare official BA/RMA decisions with the exact solver at tractable d."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from tversky_cross_calibration import predict_rank
from tversky_cross_calibration.reproducibility import seed_everything


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dimension", type=int, default=100)
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--output", type=Path, default=Path("tmp/data/rank-approximation-sensitivity.csv"))
    args = parser.parse_args()
    seed_everything(42)
    rows = []
    for trial in range(args.trials):
        probabilities = torch.rand(1, 1, args.dimension)
        for metric in ("dice", "iou"):
            exact = predict_rank(probabilities, metric, exact_threshold=args.dimension)
            approximate = predict_rank(probabilities, metric, exact_threshold=0)
            disagreement = int(torch.count_nonzero(exact != approximate).item())
            rows.append({
                "trial": trial,
                "metric": metric,
                "dimension": args.dimension,
                "exact_selected": int(exact.sum().item()),
                "approximate_selected": int(approximate.sum().item()),
                "pixel_disagreements": disagreement,
                "identical": int(disagreement == 0),
            })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(args.output)


if __name__ == "__main__":
    main()
