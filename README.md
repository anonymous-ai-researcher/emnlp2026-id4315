[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-red.svg)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/CUDA-12.1-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![Transformers](https://img.shields.io/badge/Transformers-4.40%2B-orange.svg)](https://huggingface.co/docs/transformers)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Venue](https://img.shields.io/badge/ICLR-2027-purple.svg)](#)

> **Ahead of the Evidence: Certifying Premature Commitment in Chains of Thought**
>
> *Anonymous submission to ICLR 2027*

---

## TL;DR

**Reasoning-distilled models fix their answer inside the residual stream before
the chain of thought has built the state that would justify it.** We define
*Commitment Depth* `CD_M(k)`, the normalized step at which transplanting a
counterfactual internal state flips the output, and prove a *faithfulness bound*:
no executor reading only the chain's own state can commit earlier than the
prescribed trajectory `CD_P(k)`, the Bayes accuracy of that state under a
C-RASP<sup>CoT</sup> reference program. **Causal Lookahead Probing (CLP)** is the
parameter-free estimator that measures this, immune to the non-linear
representation dilemma because nothing is fitted. Across 5 state-tracking tasks
and 9 checkpoints from 5 families, the two reasoning-distilled checkpoints cross
the bound on all three formal tasks where it leaves room (`U` up to **0.641** on
A₅/S₅, all 15 prespecified comparisons at adjusted *p* < .001), committing
**2.43× earlier** than the matched parent at identical accuracy, while every
non-distilled checkpoint stays within `U < 0.012`. On transformers compiled from
the program itself the estimator recovers the prescribed trajectory in **96 of
96** step bins, so the separation is a property of the models, not of the
measurement.

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

## What the measurement does

```
                base instance                       source instance
                      │                                    │
                      ▼                                    ▼
        ┌─────────────────────────┐         ┌─────────────────────────┐
        │  run the model          │         │  run the model, cache   │
        │                         │         │  h(L) at step k         │
        └────────────┬────────────┘         └────────────┬────────────┘
                     │                                   │
                     │        transplant at step k       │
                     │◄──────────────────────────────────┘
                     ▼
        ┌─────────────────────────┐
        │  finish the chain,      │       answer == source answer ?
        │  read the answer        │  ──────────────────────────────►  flip
        └─────────────────────────┘
```

Repeat over many base–source pairs and many steps. The flip rate, rescaled so
that guessing scores zero, is `CD_M(k)`. Compare it against `CD_P(k)`. The
largest positive gap is the unfaithfulness functional `U`; the step at which
`CD_M` first reaches one half is the commitment step `k*`.

Two properties make the reading hard to argue with:

- **Nothing is fitted.** The whole last-layer residual stream is transplanted,
  with no learned alignment, so a positive reading cannot be an artifact of a
  flexible mapping that was tuned until it found something.
- **The bound is tight by construction.** `CD_P` is the optimum over decision
  rules, not the performance of one particular algorithm, so exceeding it is a
  statement about information rather than about implementation.

## Install

```bash
git clone <this-repo> && cd cdclp
python -m pip install -e ".[dev]"          # core + tests, NumPy only
python -m pip install -e ".[dev,models]"   # add torch / transformers
```

The 25 unit tests need no GPU and no network:

```bash
pytest -q
```

## Quickstart

The full pipeline runs in seconds end to end, so the sweep, the metrics and the
statistics can be checked before committing GPU time:

```bash
# 1. sweep
python scripts/run_sweep.py --task a5s5 --model synthetic \
    --pairs 200 --bins 24 --seed 73 --dry-run --out runs/

# 2. curves, U, k*, bootstrap intervals
python scripts/compute_metrics.py --runs runs/ --out results/metrics.jsonl

# 3. confirmatory family with Holm correction
python scripts/run_stats.py --runs runs/ \
    --treatment synthetic --controls synthetic --tasks a5s5 \
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
│   └── synthetic.py     In-process runner for tests and fast iteration
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
| **Dyck-2** | bounded depth | 2 | Calibration. The answer is pinned down early, which makes this the task that confirms the estimator stays quiet when the structure leaves no room. |
| **A₄/S₄** | solvable group | 12, 24 | Shallow reference program, so `CD_P` rises at a moderate pace. |
| **A₅/S₅** | non-solvable (NC¹-complete) | 60, 120 | The sharpest test. Non-solvability forces serial computation, so the program genuinely cannot know the answer early. |
| **FSA** | regular | 2–4 | State reachability makes some prefixes informative and others not. |
| **Entity tracking** | natural language | 3–6 | Extends the measurement beyond synthetic structure to text, and shows the estimator transfers to natural language. |

Adding a task means implementing three methods: `sample`, `n_steps`, and
`state_after`. The last one carries the weight. It must return a *complete*
summary of what the reference program has determined, because `CD_P` is computed
by grouping instances whose states collide. A state that leaks extra information
makes the bound too strong; one that discards information makes it too weak.

## Two design decisions worth reading

### Intervals that hold under any dependence structure

`CD_M` is estimated by resampling base–source pairs. For a single model a
percentile bootstrap gives an interval directly. For the *difference* between two
models, the coverage of a naive interval depends on whether both were resampled
on a common draw of pairs, an assumption that is rarely recorded and easy to get
wrong in an unknown direction.

`dependence_robust_difference` removes the assumption entirely. Take a 97.5%
marginal interval for each model and project:

```
[L_a − U_b,  U_a − L_b]
```

Each marginal misses with probability at most 0.025, so both hold with
probability at least 0.95, and whenever they do the difference lies in the
projected range. Coverage is guaranteed whatever the dependence between the two
estimators, which is what lets a separation reported here be taken at face
value. A test enforces the property directly: the robust interval must never be
narrower than one that assumes independence.

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

## Running a full sweep

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
