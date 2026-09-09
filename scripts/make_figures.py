#!/usr/bin/env python3
"""Regenerate every figure from the computed metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from cdclp.figures.plots import CONTROL, DISTILLED, CurvePlot, IntervalPlot, PARENT


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--metrics", type=Path, default=Path("results/metrics.jsonl"))
    ap.add_argument("--stats", type=Path, default=Path("results/stats.json"))
    ap.add_argument("--out", type=Path, default=Path("figures"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    if args.stats.exists():
        stats = json.loads(args.stats.read_text())
        rows = [
            (k.replace("::", " vs "), 0.5 * (v["delta_kstar_interval"][0]
                                             + v["delta_kstar_interval"][1]),
             v["delta_kstar_interval"][0], v["delta_kstar_interval"][1])
            for k, v in sorted(stats["comparisons"].items())
        ]
        IntervalPlot().render(rows, str(args.out / "timing_differences.pdf"))
        print(f"wrote {args.out / 'timing_differences.pdf'}")

    print("done")


if __name__ == "__main__":
    main()
