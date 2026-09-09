#!/usr/bin/env python3
"""The confirmatory family: permutation tests plus Holm correction.

The family is fixed before the tests are run. Comparisons outside it are not
reported here, because adding them afterwards would inflate the error rate the
correction is meant to control.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np

from cdclp.crasp.prescribed import prescribed_curve
from cdclp.metrics.commitment import curve_from_indicators, unfaithfulness
from cdclp.metrics.inference import (
    bootstrap_statistic, dependence_robust_difference, holm_bonferroni,
    permutation_test,
)
from cdclp.tasks.base import REGISTRY
import cdclp.tasks.formal  # noqa: F401


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", type=Path, default=Path("runs"))
    ap.add_argument("--treatment", required=True,
                    help="model whose commitment is being tested")
    ap.add_argument("--controls", nargs="+", required=True)
    ap.add_argument("--tasks", nargs="+", required=True)
    ap.add_argument("--permutations", type=int, default=10_000)
    ap.add_argument("--resamples", type=int, default=2000)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--out", type=Path, default=Path("results/stats.json"))
    args = ap.parse_args()

    def load(model, task):
        hits = sorted(args.runs.glob(f"{model}__{task}__*.meta.json"))
        if not hits:
            raise SystemExit(f"no run for {model} on {task}")
        meta = json.loads(hits[0].read_text())
        ind = np.load(str(hits[0]).replace(".meta.json", ".indicators.npy"))
        return ind, meta

    raw_p, deltas, intervals = {}, {}, {}
    for task, ctrl in itertools.product(args.tasks, args.controls):
        a, ma = load(args.treatment, task)
        b, mb = load(ctrl, task)
        grid = np.linspace(0.0, 1.0, ma["bins"])
        pres = prescribed_curve(REGISTRY.build(task), np.random.default_rng(0),
                                n_bins=ma["bins"], n_instances=2000)

        def u_of(ind, m=ma["m"]):
            return unfaithfulness(curve_from_indicators(ind, m, grid).values, pres.values)

        key = f"{task}::{ctrl}"
        obs, p = permutation_test(a, b, u_of, n_permutations=args.permutations,
                                  rng=np.random.default_rng(0))
        raw_p[key], deltas[key] = p, obs

        def k_of(ind, m=ma["m"]):
            return curve_from_indicators(ind, m, grid).commitment_step() or 1.0

        sa = bootstrap_statistic(a, k_of, args.resamples, np.random.default_rng(1))
        sb = bootstrap_statistic(b, k_of, args.resamples, np.random.default_rng(2))
        iv = dependence_robust_difference(sa, sb)
        intervals[key] = (iv.lo, iv.hi)
        print(f"{key:<34}dU={obs:+.3f}  p={p:.4f}  "
              f"dk* in [{iv.lo:+.3f}, {iv.hi:+.3f}]")

    adjusted = holm_bonferroni(raw_p, alpha=args.alpha)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "family_size": len(raw_p),
        "alpha": args.alpha,
        "comparisons": {
            k: {"delta_U": deltas[k], "raw_p": raw_p[k],
                "holm_p": adjusted[k][0], "significant": bool(adjusted[k][1]),
                "delta_kstar_interval": intervals[k]}
            for k in sorted(raw_p)
        },
    }, indent=2))
    n_sig = sum(v[1] for v in adjusted.values())
    print(f"\n{n_sig}/{len(raw_p)} survive Holm at alpha={args.alpha}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
