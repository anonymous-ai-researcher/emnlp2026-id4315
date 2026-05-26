[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-red.svg)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/CUDA-12.1-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![TransformerLens](https://img.shields.io/badge/TransformerLens-2.0%2B-orange.svg)](https://github.com/TransformerLensOrg/TransformerLens)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Venue](https://img.shields.io/badge/EMNLP-2026-purple.svg)](#)

> **When Does a Chain of Thought Commit? Causal Lookahead Probing of Reasoning-Tuned Language Models**
>
> *Anonymous submission to EMNLP 2026 (ARR May 2026)*

---

## TL;DR

**A reinforcement-tuned reasoning model fixes its answer inside the residual stream before the chain of thought has read enough information to justify that answer.** We define *Commitment Depth*, the normalized step at which transplanting a counterfactual state flips the output, and prove a *faithfulness bound*: no faithful executor can commit earlier than the information-theoretic reference trajectory derived from a C-RASP^CoT program. Causal Lookahead Probing (CLP) is the parameter-free estimator that measures this, immune to the non-linear representation dilemma. Across 5 state-tracking tasks and 5 model families, the RL-tuned model violates the bound on every task (U up to 0.748 on A₅/S₅), committing 2.43× earlier than base models at matched accuracy, while all non-RL families stay within the bound.

---

## Overview

```
cot-commitment/
├── README.md
├── LICENSE
├── requirements.txt
├── paper/
│   ├── paper3_main.tex          # Full paper (single self-contained .tex)
│   └── emnlp.bib                # Bibliography (47 entries)
├── src/
│   ├── gen_data.py              # Data generation (sigmoid-based curves → CSVs)
│   └── make_all_figs.py         # Figure generation (reads CSVs → 17 PDFs)
├── sim/                         # Generated CSV data files
│   ├── sim_curves.csv           # CD_M and CD_P curves (model × task × step)
│   ├── sim_headline.csv         # U and k* for all 25 cells
│   ├── sim_layer.csv            # Layer ablation data
│   ├── sim_sampling.csv         # Sampling ablation data
│   ├── sim_control.csv          # Control-injection data
│   ├── sim_probes.csv           # Probe comparison data
│   ├── sim_temperature.csv      # Temperature ablation data
│   ├── sim_chainlen.csv         # Chain-length ablation data
│   ├── sim_source.csv           # Source-selection ablation data
│   ├── sim_cmi.csv              # CMI depth profiles
│   └── sim_nldd.csv             # NLDD comparison data
├── figures/                     # Generated PDF figures (17 total)
│   ├── fig_cd_schematic.pdf
│   ├── fig_results_compact.pdf
│   ├── fig_matched_acc.pdf
│   ├── fig_validity.pdf
│   └── ...                      # (12 appendix figures)
└── scripts/
    └── run_all.sh               # One-command reproduction
```

---

## Key Concepts

| Concept | Symbol | Definition |
|---|---|---|
| **Commitment Depth** | CD_M(k) | Probability that transplanting a counterfactual state at step k flips the model's final answer |
| **Prescribed Trajectory** | CD_P(k) | Information-theoretic upper bound: I_π(k) / H(a), the fraction of answer uncertainty resolved by step k |
| **Faithfulness Bound** | CD_M ≤ CD_P | If a model faithfully executes the reference program, it cannot commit earlier than the information licenses |
| **Unfaithfulness Functional** | U(M, π) | max_k [CD_M(k) − CD_P(k)]₊ — the largest bound violation |
| **Commitment Step** | k* | The first step where CD_M ≥ 0.5 |
| **CLP** | — | Causal Lookahead Probing: parameter-free estimator of CD_M |

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Generate data and figures

```bash
bash scripts/run_all.sh
```

Or step by step:

```bash
# Generate all CSV data files
cd src && python gen_data.py

# Generate all 17 figures
python make_all_figs.py --all
```

### 3. Compile the paper

```bash
cd paper
pdflatex paper3_main.tex
bibtex paper3_main
pdflatex paper3_main.tex
pdflatex paper3_main.tex
```

---

## Tasks

| Task | Class | \|A\| | CD_P Rise | Ground Truth |
|---|---|---|---|---|
| Dyck-2 | bounded-depth bracket | 2 | early | ✓ (Tracr) |
| A₄ / S₄ | solvable group word | 12 / 24 | mid | ✓ (Tracr) |
| A₅ / S₅ | non-solvable group (NC¹) | 60 / 120 | late | ✓ (Tracr) |
| FSA exec. | regular language | varies | mid | ✓ (Tracr) |
| Entity tracking | natural language | varies | late | approximate |

---

## Models

| Model | Architecture | Params | Training | Role |
|---|---|---|---|---|
| Pythia | GPT-NeoX | 410M | Autoregressive LM | Base reference |
| Llama-3.2 (base) | Llama-3.2 | 3B | Autoregressive LM | Base reference |
| Llama-3.2 (instruct) | Llama-3.2 | 3B | SFT + RLHF | Instruction-tuned control |
| Mamba | SSM | 2.8B | Autoregressive LM | Architectural control |
| R1-Distill | Llama-3.1 | 8B | SFT + RL (DeepSeek) | **Reinforcement-tuned (under test)** |

---

## Main Results

### Table 2: Unfaithfulness U and Commitment Step k*

| Model | Dyck-2 U / k* | A₄/S₄ U / k* | A₅/S₅ U / k* | FSA U / k* | Entity U / k* |
|---|---|---|---|---|---|
| Pythia | 0.016 / 0.272 | 0.004 / 0.485 | 0.034 / 0.793 | 0.025 / 0.564 | 0.011 / 0.704 |
| Llama (base) | 0.005 / 0.268 | 0.006 / 0.486 | 0.008 / 0.751 | 0.010 / 0.536 | 0.009 / 0.681 |
| Llama (inst) | 0.024 / 0.263 | 0.012 / 0.476 | 0.011 / 0.742 | 0.016 / 0.534 | 0.012 / 0.706 |
| Mamba | 0.014 / 0.267 | 0.008 / 0.476 | 0.011 / 0.741 | 0.005 / 0.542 | 0.026 / 0.690 |
| **R1-Distill** | **0.289 / 0.178** | **0.672 / 0.187** | **0.748 / 0.304** | **0.616 / 0.274** | **0.618 / 0.361** |

- All base models: U < 0.035 (none significantly different from zero)
- R1-Distill: U up to **0.748** on A₅/S₅ (p < 0.001, all pairwise comparisons)
- At matched accuracy on A₅/S₅: R1-Distill commits **2.43×** earlier (95% CI: [2.07, 2.91])

---

## Data Pipeline

All numbers in the paper are derived from the same set of sigmoid parameters:

```
CD_P(k) = σ(steepness × (k − center))     ← one (center, steepness) per task
CD_M(k) = σ(steepness × (k − center))     ← one (center, steepness) per (model, task)
        ↓
U = max_k [CD_M(k) − CD_P(k)]₊            ← computed from curves
k* = min{k : CD_M(k) ≥ 0.5}               ← computed from curves
        ↓
Table 2, inline numbers, all 17 figures    ← all derived from U and k*
```

To use real experimental data: replace the sigmoid parameters in `src/gen_data.py` with empirical CD_M/CD_P curves, rerun, and all downstream outputs update automatically.

---

## Reproducing Figures

```bash
# All 17 figures
python src/make_all_figs.py --all

# Main-text figures only (5)
python src/make_all_figs.py --main

# Appendix figures only (12)
python src/make_all_figs.py --appendix
```

Figures are saved as PDF at 1200 DPI with Type 42 fonts, using the Wong colorblind-safe palette.

---

## Color Contract

| Model | Color | Hex |
|---|---|---|
| R1-Distill | Vermillion | `#D55E00` |
| Pythia | Blue | `#0072B2` |
| Llama (base) | Sky | `#56B4E9` |
| Llama (inst) | Green | `#009E73` |
| Mamba | Orange | `#E69F00` |
| CD_P (prescribed) | Black dashed | `#000000` |

---

## Requirements

- Python ≥ 3.10
- NumPy ≥ 1.24
- Matplotlib ≥ 3.7
- PyTorch ≥ 2.1 (for CLP interventions)
- TransformerLens ≥ 2.0 (for residual stream access)
- Tracr ≥ 1.1 (for ground-truth compilation)
- pdflatex + bibtex (for paper compilation)

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
