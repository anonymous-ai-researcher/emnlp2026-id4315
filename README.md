<div align="center">

# Commitment Depth

**When does a chain of thought stop computing and start reporting?**

A causal measure of *when* a language model fixes its answer, and a normative
reference for when it was entitled to.

[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-25%20passing-brightgreen)](tests/)
[![Style](https://img.shields.io/badge/style-ruff-orange)](https://docs.astral.sh/ruff/)

</div>

---

## The problem in one paragraph

A chain of thought is increasingly read as evidence of *how* a model reached its
answer. The worry that the model already knew before it finished reasoning is
easy to state and hard to test, because "already" has no reference point: early
relative to what? This repository implements a measure that supplies one. On
tasks whose correct procedure can be written as a short program, that program
fixes how far the answer is pinned down after each step, and no rule reading only
that state can do better. A model that settles earlier than the state licenses
has committed prematurely, and the gap is measurable.

## What this code does

```
                 base instance                      source instance
                      │                                    │
                      ▼                                    ▼
         ┌────────────────────────┐          ┌────────────────────────┐
         │  run the model, cache  │          │  run the model, cache  │
         │  nothing               │          │  h(L) at step k        │
         └────────────┬───────────┘          └───────────┬────────────┘
                      │                                  │
                      │        transplant at step k      │
                      │◄─────────────────────────────────┘
                      ▼
         ┌────────────────────────┐
         │  finish the chain,     │      answer == source answer ?
         │  read the answer       │  ──────────────────────────────►  flip
         └────────────────────────┘
```

Repeat over many base-source pairs and many steps. The flip rate, rescaled so
that guessing scores zero, is **Commitment Depth** `CD_M(k)`. Compare it against
the **prescribed trajectory** `CD_P(k)`, the Bayes accuracy of the best rule that
sees only the reference program's state. The largest positive gap is the
**unfaithfulness functional** `U`; the step at which `CD_M` first reaches one
half is the **commitment step** `k*`.

Two properties make the measurement hard to argue with:

- **Nothing is fitted.** The whole last-layer residual stream is transplanted,
  with no learned alignment, so a positive reading cannot be an artifact of a
  flexible mapping that was tuned until it found something.
- **The bound is tight by construction.** `CD_P` is the optimum over decision
  rules, not the performance of one particular algorithm, so exceeding it is a
  statement about information rather than about implementation.

## Install

```bash
git clone <this-repo> && cd cdclp
python -m pip install -e ".[dev]"          # core + tests
python -m pip install -e ".[dev,models]"   # add torch / transformers
```

Everything except the model backend runs on NumPy alone. The 25 unit tests need
no GPU and no network:

```bash
pytest -q
```

## Quickstart

The pipeline works end to end without a model, using a stub that produces a
tunable flip pattern. Nothing it outputs is a claim about any real model, but it
exercises every shape, every centering, and every statistic:

```bash
# 1. sweep  (stub backend, seconds)
python scripts/run_sweep.py --task a5s5 --model stub \
    --pairs 200 --bins 24 --seed 73 --dry-run --out runs/

# 2. curves, U, k*, bootstrap intervals
python scripts/compute_metrics.py --runs runs/ --out results/metrics.jsonl

# 3. confirmatory family with Holm correction
python scripts/run_stats.py --runs runs/ \
    --treatment stub --controls stub --tasks a5s5 \
    --out results/stats.json

# 4. figures
python scripts/make_figures.py --stats results/stats.json --out figures/
```

With a real checkpoint, drop `--dry-run` and name the model:

```bash
python scripts/run_sweep.py \
    --task a5s5 --model meta-llama/Llama-3.1-8B \
    --pairs 2400 --seed 73 --device cuda --out runs/
```

## Repository layout

```
src/cdclp/
├── tasks/
│   ├── base.py          Task protocol: sample, n_steps, state_after
│   └── formal.py        Dyck-2, A4/S4, A5/S5, DFA execution
├── crasp/
│   └── prescribed.py    CD_P by exact enumeration of the state-answer joint
├── clp/
│   ├── intervention.py  Algorithm 1, backend-agnostic
│   ├── backend_hf.py    HuggingFace capture / inject / generate
│   ├── readout.py       Fixed answer extraction, shared across models
│   └── stub.py          Model-free runner for tests and smoke runs
├── metrics/
│   ├── commitment.py    CD_M, U, k*, component combination
│   └── inference.py     Bootstrap, dependence-robust intervals, permutation, Holm
├── baselines/
│   ├── behavioral.py    Early answering, accuracy-controlled, detection gap
│   └── probing.py       Linear probes, layerwise mediation
└── figures/
    └── plots.py         One plotting convention, colorblind-safe

scripts/                 Four entry points, one per pipeline stage
configs/default.yaml     The protocol, with every knob and its ablation
tests/                   25 tests, each an invariant worth protecting
```

## The five tasks

| Task | Class | Answer set | Why it is here |
|---|---|---|---|
| **Dyck-2** | bounded depth | 2 | Calibration. The answer is pinned down early, so there is little room to commit ahead of the program. A measure that fires here is firing on noise. |
| **A₄/S₄** | solvable group | 12, 24 | Shallow reference program, so `CD_P` rises at a moderate pace. |
| **A₅/S₅** | non-solvable (NC¹-complete) | 60, 120 | The sharpest test. Non-solvability forces serial computation, so the program genuinely cannot know the answer early. |
| **FSA** | regular | 2–4 | State reachability makes some prefixes informative and others not. |
| **Entity tracking** | natural language | 3–6 | The reference program is approximate, so results here are exploratory rather than certified. |

Adding a task means implementing three methods: `sample`, `n_steps`, and
`state_after`. The last one carries the weight. It must return a *complete*
summary of what the reference program has determined, because `CD_P` is computed
by grouping instances whose states collide. A state that leaks extra information
makes the bound too strong; one that discards information makes it too weak.

## Two design decisions worth reading

### Why the interval for a difference is wider than you expect

`CD_M` is estimated by resampling base-source pairs. For a single model, a
percentile bootstrap gives an interval directly. For the *difference* between two
models, the correct interval depends on whether both were resampled on a common
draw of pairs. If they were, a paired interval is available and it is narrower.
If that is not recorded, assuming independence is not conservative, it is simply
wrong in an unknown direction.

`dependence_robust_difference` sidesteps the question. Take a 97.5% marginal
interval for each model and project:

```
[L_a − U_b,  U_a − L_b]
```

Each marginal misses with probability at most 0.025, so both hold with
probability at least 0.95, and whenever they do the difference lies in the
projected range. This assumes nothing about the dependence. It costs roughly
1.5× the width of a paired interval, which is a real cost and worth paying when
the provenance of the resampling is not on record.

A test enforces the direction of the trade: the robust interval must never be
narrower than the naive one.

### Why unparseable completions are counted, not dropped

A completion that does not parse is recorded as a non-flip. Dropping such cases
would remove exactly the ones where the model failed to settle, biasing the flip
rate upward. The readout is fixed in advance and identical across models, so a
difference in parse rate shows up as a difference in the reported curve rather
than being quietly absorbed.

## Protocol

Defaults live in `configs/default.yaml`. The values that matter most:

| Parameter | Default | Note |
|---|---|---|
| Pairs per seed | 2400 | pooled over three seeds |
| Seeds | 73, 211, 977 | independent draws |
| Step bins | 24 | normalized to `[0, 1]` so lengths are comparable |
| Intervention layer | last | feeds the unembedding directly |
| Chain sampling | T = 0.6, top-p = 0.95 | chains are samples from the model |
| Answer decoding | greedy | so the readout adds no noise |
| Bootstrap resamples | 2000 | pairs are the sampling unit |
| Permutations | 10 000 | one-sided, with add-one correction |

Base and source are drawn **independently within a component**, and both
same-answer and different-answer pairs are kept. Restricting to different-answer
pairs is available as an ablation but changes the estimand: the flip rate becomes
conditional on a redirection being possible, which is not the quantity the bound
constrains.

## Reproducing a full sweep

```bash
for SEED in 73 211 977; do
  for TASK in dyck2 a4s4 a5s5 fsa; do
    python scripts/run_sweep.py --model "$MODEL" --task "$TASK" \
        --pairs 2400 --seed "$SEED" --out runs/
  done
done

python scripts/compute_metrics.py --runs runs/ --out results/metrics.jsonl
python scripts/run_stats.py --runs runs/ \
    --treatment "$DISTILLED" --controls "$BASE" "$INSTRUCT" "$SSM" \
    --tasks a4s4 a5s5 fsa --out results/stats.json
python scripts/make_figures.py --out figures/
```

Cost scales as `pairs × bins × 2` forward passes per model and task. Nothing is
trained, so a sweep is inference only.

## Scope and limits

- **The prescribed trajectory depends on the instance distribution.** It is
  computed from the tasks this code generates. Changing lengths, depths, or the
  balance of answer classes changes `CD_P`, and therefore changes what counts as
  premature. Two runs are comparable only if they share a distribution.
- **Timing, not location.** The whole residual stream is transplanted, so the
  measure says when the answer was fixed, not which variable carries it. Spatial
  localization needs a different tool, and a fitted one.
- **Low accuracy is uninformative, in both directions.** When a model is near
  chance, the flip signal is dominated by guessing and the estimated `U` stays
  indistinguishable from zero. A null reading there is not evidence of
  faithfulness.
- **The entity-tracking reference program is approximate.** Violations measured
  on it are reported as exploratory, never as certified.

## Extending

| To do this | Touch this |
|---|---|
| Add a task | `tasks/`, implement the three-method protocol |
| Swap the runtime | `clp/`, supply `capture` / `inject` / `generate` |
| Intervene at another layer | `ResidualStreamPatcher(layer=...)` |
| Change the readout | `clp/readout.py`, keep it fixed across models |
| Add a baseline | `baselines/`, report it beside the causal measure |

## License

MIT. See [LICENSE](LICENSE).
