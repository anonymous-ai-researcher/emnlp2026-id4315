#!/usr/bin/env python3
"""Run the Commitment Depth sweep for one model on one task.

Writes per-pair flip indicators, not just their means. Everything downstream
(bootstrap, permutation tests, paired analyses) needs the raw indicators; a run
that saves only curve summaries cannot be reanalyzed later.

Example
-------
    python scripts/run_sweep.py \\
        --model meta-llama/Llama-3.1-8B \\
        --task a5s5 --pairs 2400 --seed 73 --out runs/
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from cdclp.clp.intervention import ResidualStreamPatcher, sweep_step
from cdclp.clp.readout import choose_readout
from cdclp.tasks.base import REGISTRY
import cdclp.tasks.formal  # noqa: F401  (registers the tasks)


def build_pairs(task, rng, component, n_pairs):
    """Draw base and source independently, keeping both kinds of pair.

    Restricting to different-answer pairs would change the estimand: the flip
    rate would then be conditional on a redirection being possible, which is a
    different quantity from the one the bound constrains.
    """
    pairs = []
    for _ in range(n_pairs):
        b = task.sample(rng, component)
        s = task.sample(rng, component)
        pairs.append((b.prompt, s.prompt, b.answer, s.answer))
    return pairs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--task", required=True, choices=REGISTRY.names())
    ap.add_argument("--pairs", type=int, default=2400)
    ap.add_argument("--bins", type=int, default=24)
    ap.add_argument("--seed", type=int, default=73)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", type=Path, default=Path("runs"))
    ap.add_argument("--dry-run", action="store_true",
                    help="exercise the pipeline with a stub model")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    task = REGISTRY.build(args.task)
    readout = choose_readout(args.task)
    grid = np.linspace(0.0, 1.0, args.bins)

    if args.dry_run:
        from cdclp.clp.stub import StubRunner
        runner = StubRunner(rng=rng)
    else:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from cdclp.clp.backend_hf import HFRunner
        tok = AutoTokenizer.from_pretrained(args.model)
        mdl = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype="float16", device_map=args.device
        )
        runner = HFRunner(model=mdl, tokenizer=tok, device=args.device)

    patcher = ResidualStreamPatcher(runner.capture, runner.inject)
    args.out.mkdir(parents=True, exist_ok=True)

    for comp in task.components:
        pairs = build_pairs(task, rng, comp.name, args.pairs)
        indicators = np.zeros((args.bins, args.pairs), dtype=np.int8)
        for b, frac in enumerate(grid):
            step = int(round(frac * 24))
            indicators[b] = sweep_step(patcher, runner.generate, readout, pairs, step)
            print(f"[{args.task}/{comp.name}] bin {b + 1}/{args.bins} "
                  f"rate={indicators[b].mean():.3f}", flush=True)

        stem = f"{Path(args.model).name}__{args.task}__{comp.name}__seed{args.seed}"
        np.save(args.out / f"{stem}.indicators.npy", indicators)
        (args.out / f"{stem}.meta.json").write_text(json.dumps({
            "model": args.model, "task": args.task, "component": comp.name,
            "m": comp.m, "weight": comp.weight, "seed": args.seed,
            "pairs": args.pairs, "bins": args.bins,
        }, indent=2))
        print(f"  wrote {stem}")


if __name__ == "__main__":
    main()
