"""Uncertainty and hypothesis tests.

Three procedures live here, and the distinction between the second and third
matters more than it looks:

* **Percentile bootstrap** over base-source pairs, for a single model's
  ``U`` and commitment step.
* **Dependence-robust intervals** for a *difference* between two models. If the
  two models were resampled on a common draw of pairs, a paired interval would
  be available and would be narrower. Absent a record of that, a Bonferroni
  projection of the two marginal intervals is valid whatever the dependence is.
* **Permutation tests** for the difference in ``U``, which need the raw
  per-pair indicators rather than summary curves.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Interval:
    lo: float
    hi: float
    level: float

    def excludes_zero(self) -> bool:
        return self.hi < 0.0 or self.lo > 0.0

    def width(self) -> float:
        return self.hi - self.lo


def percentile_interval(samples: np.ndarray, level: float = 0.95) -> Interval:
    """Percentile bootstrap interval from replicate values."""
    if not 0 < level < 1:
        raise ValueError(f"level must be in (0, 1), got {level}")
    alpha = 1.0 - level
    lo, hi = np.quantile(samples, [alpha / 2.0, 1.0 - alpha / 2.0])
    return Interval(float(lo), float(hi), level)


def bootstrap_statistic(
    indicators: np.ndarray,
    statistic,
    n_resamples: int = 2000,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Resample pairs with replacement and recompute a statistic.

    Pairs are the sampling unit, so a resample draws whole columns of
    ``indicators``. Resampling individual bins instead would break the
    dependence between steps of the same pair and understate the variance.
    """
    rng = rng or np.random.default_rng()
    n_bins, n_pairs = indicators.shape
    out = np.empty(n_resamples, dtype=float)
    for b in range(n_resamples):
        idx = rng.integers(n_pairs, size=n_pairs)
        out[b] = statistic(indicators[:, idx])
    return out


def dependence_robust_difference(
    samples_a: np.ndarray,
    samples_b: np.ndarray,
    level: float = 0.95,
) -> Interval:
    """Interval for ``theta_a - theta_b`` that assumes nothing about dependence.

    Take a marginal interval for each model at level ``1 - alpha/2`` and project:

        [L_a - U_b,  U_a - L_b]

    Each marginal misses with probability at most ``alpha/2``, so by a union
    bound both hold with probability at least ``1 - alpha``, and whenever they
    do the difference lies in the projected range. The price is width: the
    interval is wider than a paired one would be. The gain is that it is correct
    whether or not the two models were resampled together.
    """
    alpha = 1.0 - level
    half = 1.0 - alpha / 2.0
    a = percentile_interval(samples_a, level=half)
    b = percentile_interval(samples_b, level=half)
    return Interval(a.lo - b.hi, a.hi - b.lo, level)


def permutation_test(
    indicators_a: np.ndarray,
    indicators_b: np.ndarray,
    statistic,
    n_permutations: int = 10_000,
    rng: np.random.Generator | None = None,
    alternative: str = "greater",
) -> tuple[float, float]:
    """One-sided permutation test on a difference of curve summaries.

    Model labels are shuffled pair by pair; the null is that the label carries
    no information about the flip pattern. Returns ``(observed, p_value)``.
    """
    rng = rng or np.random.default_rng()
    if indicators_a.shape != indicators_b.shape:
        raise ValueError(
            f"shape mismatch: {indicators_a.shape} vs {indicators_b.shape}"
        )
    observed = statistic(indicators_a) - statistic(indicators_b)
    n_pairs = indicators_a.shape[1]
    count = 0
    for _ in range(n_permutations):
        swap = rng.random(n_pairs) < 0.5
        pa = np.where(swap, indicators_b, indicators_a)
        pb = np.where(swap, indicators_a, indicators_b)
        diff = statistic(pa) - statistic(pb)
        if alternative == "greater":
            count += diff >= observed
        elif alternative == "two-sided":
            count += abs(diff) >= abs(observed)
        else:
            raise ValueError(f"unknown alternative {alternative!r}")
    # Add-one correction: a p-value of exactly zero is not attainable from a
    # finite number of permutations, and reporting one would overstate the
    # evidence.
    p = (count + 1) / (n_permutations + 1)
    return float(observed), float(p)


def holm_bonferroni(p_values: dict[str, float], alpha: float = 0.05) -> dict[str, tuple[float, bool]]:
    """Step-down correction over a pre-registered family.

    Returns ``name -> (adjusted_p, rejected)``. The family must be fixed before
    looking at the data; adding comparisons afterwards inflates the error rate
    that this procedure is meant to control.
    """
    items = sorted(p_values.items(), key=lambda kv: kv[1])
    n = len(items)
    out: dict[str, tuple[float, bool]] = {}
    running = 0.0
    for i, (name, p) in enumerate(items):
        adjusted = min(1.0, max(running, (n - i) * p))
        running = adjusted
        out[name] = (adjusted, adjusted <= alpha)
    return out
