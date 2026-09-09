#!/usr/bin/env python3
"""Turn raw flip indicators into curves, U, k*, and intervals.

Reads the ``.indicators.npy`` / ``.meta.json`` pairs written by ``run_sweep.py``,
combines components with their weights, compares against the prescribed
trajectory, and writes one row per (model, task).
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from cdclp.crasp.prescribed import prescribed_curve
from cdclp.metrics.commitment import (
    combine_components, curve_from_indicators, unfaithfulness, violation_area,
)
from cdclp.metrics.inference import bootstrap_statistic, percentile_interval
from cdclp.tasks.base import REGISTRY
import cdclp.tasks.formal  # noqa: F401


def load_runs(run_dir: Path):
    groups = defaultdict(dict)
    for meta_path in sorted(run_dir.glob("*.meta.json")):
        meta = json.loads(meta_path.read_text())
        ind = np.load(str(meta_path).replace(".meta.json", ".indicators.npy"))
        groups[(meta["model"], meta["task"], meta["seed"])][meta["component"]] = (ind, meta)
    return groups


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", type=Path, default=Path("runs"))
    ap.add_argument("--out", type=Path, default=Path("results/metrics.jsonl"))
    ap.add_argument("--resamples", type=int, default=2000)
    ap.add_argument("--prescribed-instances", type=int, default=4000)
    args = ap.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    groups = load_runs(args.runs)
    if not groups:
        raise SystemExit(f"no runs found under {args.runs}")

    rows = []
    for (model, task_name, seed), comps in sorted(groups.items()):
        any_meta = next(iter(comps.values()))[1]
        grid = np.linspace(0.0, 1.0, any_meta["bins"])
        task = REGISTRY.build(task_name)
        pres = prescribed_curve(
            task, np.random.default_rng(0),
            n_bins=any_meta["bins"], n_instances=args.prescribed_instances,
        )

        curves, weights = {}, {}
        for name, (ind, meta) in comps.items():
            curves[name] = curve_from_indicators(ind, meta["m"], grid, model, task_name)
            weights[name] = meta["weight"]
        combined = combine_components(curves, weights)

        u = unfaithfulness(combined.values, pres.values)
        area = violation_area(combined.values, pres.values, grid)
        kstar = combined.commitment_step()

        # Bootstrap the dominant component; a joint resample across components
        # would need the pair identities to be shared, which they are not.
        biggest = max(comps, key=lambda n: weights[n])
        ind, meta = comps[biggest]
        samples_k = bootstrap_statistic(
            ind,
            lambda a: curve_from_indicators(a, meta["m"], grid).commitment_step() or 1.0,
            n_resamples=args.resamples, rng=np.random.default_rng(seed),
        )
        ci = percentile_interval(samples_k)

        rows.append({
            "model": model, "task": task_name, "seed": seed,
            "U": round(u, 6), "area": round(area, 6),
            "kstar": None if kstar is None else round(kstar, 6),
            "kstar_lo": round(ci.lo, 6), "kstar_hi": round(ci.hi, 6),
            "prescribed_crossing": pres.crossing(),
            "n_pairs": int(ind.shape[1]),
        })
        print(f"{model:<28}{task_name:<8}U={u:.3f}  k*={kstar}  "
              f"CI=[{ci.lo:.3f},{ci.hi:.3f}]")

    with args.out.open("w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    print(f"\nwrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
