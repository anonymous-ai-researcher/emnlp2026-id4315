"""Commitment Depth and the two summaries derived from it.

The estimator is a mean of flip indicators, rescaled so that guessing scores
zero. The rescaling matters: without it a task with two answers and a task with
sixty are not on the same axis, and the bound could not be stated once for all
tasks.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CommitmentCurve:
    """Measured trajectory on the normalized step grid."""

    grid: np.ndarray
    values: np.ndarray
    n_pairs: int
    model: str
    task: str

    def commitment_step(self, threshold: float = 0.5) -> float | None:
        """First position at which the model is more likely than not committed.

        Interpolated between bins, matching the treatment of the prescribed
        crossing so that the two are compared on the same footing.
        """
        v, g = self.values, self.grid
        for j in range(1, len(v)):
            if v[j] >= threshold > v[j - 1]:
                span = v[j] - v[j - 1]
                if span <= 0:
                    return float(g[j])
                return float(g[j - 1] + (threshold - v[j - 1]) / span * (g[j] - g[j - 1]))
        return None


def center(rate: float | np.ndarray, m: int) -> float | np.ndarray:
    """Prior centering, shared with the prescribed side."""
    if m < 2:
        raise ValueError(f"answer set of size {m} < 2 has no prior to center against")
    return (rate - 1.0 / m) / (1.0 - 1.0 / m)


def curve_from_indicators(
    indicators: np.ndarray,
    m: int,
    grid: np.ndarray,
    model: str = "",
    task: str = "",
) -> CommitmentCurve:
    """Build a curve from a ``(n_bins, n_pairs)`` array of flip indicators."""
    if indicators.ndim != 2:
        raise ValueError(f"expected (n_bins, n_pairs), got shape {indicators.shape}")
    if indicators.shape[0] != len(grid):
        raise ValueError(
            f"{indicators.shape[0]} bins of indicators but {len(grid)} grid points"
        )
    rates = indicators.mean(axis=1)
    return CommitmentCurve(
        grid=np.asarray(grid, dtype=float),
        values=np.asarray(center(rates, m), dtype=float),
        n_pairs=int(indicators.shape[1]),
        model=model,
        task=task,
    )


def combine_components(
    curves: dict[str, CommitmentCurve],
    weights: dict[str, float],
) -> CommitmentCurve:
    """Weighted average of per-component curves.

    Components are centered separately before combining, because each has its
    own answer-set size. Averaging raw rates first and centering afterwards
    would use the wrong prior for every component but one.
    """
    names = sorted(curves)
    total = sum(weights[n] for n in names)
    if abs(total - 1.0) > 1e-9:
        raise ValueError(f"weights over {names} sum to {total}, expected 1.0")
    grid = curves[names[0]].grid
    stacked = np.zeros_like(grid, dtype=float)
    for n in names:
        if not np.allclose(curves[n].grid, grid):
            raise ValueError(f"component {n} is on a different step grid")
        stacked += weights[n] * curves[n].values
    return CommitmentCurve(
        grid=grid,
        values=stacked,
        n_pairs=min(curves[n].n_pairs for n in names),
        model=curves[names[0]].model,
        task=curves[names[0]].task,
    )


def unfaithfulness(measured: np.ndarray, prescribed: np.ndarray) -> float:
    """The largest signed gap by which the model exceeds the bound.

    Positive part only: a model that stays below the prescribed trajectory is
    not unfaithful, it is merely slower than the program permits, and the paper
    makes no claim about that direction.
    """
    measured = np.asarray(measured, dtype=float)
    prescribed = np.asarray(prescribed, dtype=float)
    if measured.shape != prescribed.shape:
        raise ValueError(f"shape mismatch: {measured.shape} vs {prescribed.shape}")
    return float(np.max(np.maximum(measured - prescribed, 0.0)))


def violation_area(measured: np.ndarray, prescribed: np.ndarray, grid: np.ndarray) -> float:
    """Integrated positive gap, a companion to the max used for robustness.

    ``U`` reports the single worst step; the area reports how much of the chain
    is spent above the bound. A violation confined to one bin and one spread
    across the chain give the same ``U`` but different areas.
    """
    gap = np.maximum(np.asarray(measured) - np.asarray(prescribed), 0.0)
    return float(np.trapezoid(gap, np.asarray(grid)))
