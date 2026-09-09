"""The prescribed trajectory: what the reference program licenses at each step.

The bound says that any executor reading only the chain's own state cannot beat
the Bayes accuracy of that state. So the prescribed trajectory is not a property
of any particular algorithm; it is the optimum over all decision rules that see
the state and nothing else, which makes it tight by construction.

Concretely, for a component with answer set of size ``m``:

1. enumerate instances (or sample enough of them to estimate the joint),
2. group them by the reference-program state after ``k`` steps,
3. within each state, the best rule predicts the modal answer,
4. average the resulting accuracy over states, weighted by state mass,
5. apply the same prior centering used for the measured curve.

Step 3 is where the optimality comes from: no rule reading only the state can do
better than predicting the most likely answer given that state.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Hashable, Sequence

import numpy as np

from ..tasks.base import Component, Instance, Task


@dataclass(frozen=True)
class PrescribedCurve:
    """Prescribed trajectory on the normalized step grid."""

    grid: np.ndarray  # shape (n_bins,), values in [0, 1]
    values: np.ndarray  # shape (n_bins,), the centered Bayes accuracy
    task: str

    def crossing(self, level: float = 0.5) -> float | None:
        """First grid position where the curve reaches ``level``.

        Linear interpolation between the bracketing bins; ``None`` if the curve
        never reaches the level. This is the quantity the paper compares the
        model's commitment step against.
        """
        v, g = self.values, self.grid
        for j in range(1, len(v)):
            if v[j] >= level > v[j - 1]:
                span = v[j] - v[j - 1]
                if span <= 0:
                    return float(g[j])
                frac = (level - v[j - 1]) / span
                return float(g[j - 1] + frac * (g[j] - g[j - 1]))
        return None


def center(accuracy: float, m: int) -> float:
    """Rescale so that guessing scores zero and certainty scores one."""
    if m < 2:
        raise ValueError(f"answer set of size {m} < 2 has no prior to center against")
    return (accuracy - 1.0 / m) / (1.0 - 1.0 / m)


def bayes_accuracy_at_step(
    task: Task,
    instances: Sequence[Instance],
    step: int,
) -> float:
    """Accuracy of the best rule that sees only the state after ``step`` steps.

    Instances whose state collides are indistinguishable to such a rule, so the
    best it can do inside a collision class is predict the modal answer.
    """
    by_state: dict[Hashable, Counter] = defaultdict(Counter)
    for inst in instances:
        if step > task.n_steps(inst):
            # The chain is already finished; the state is the terminal one.
            s = task.state_after(inst, task.n_steps(inst))
        else:
            s = task.state_after(inst, step)
        by_state[s][inst.answer] += 1
    if not instances:
        return 0.0
    correct = sum(counts.most_common(1)[0][1] for counts in by_state.values())
    return correct / len(instances)


def prescribed_curve(
    task: Task,
    rng: np.random.Generator,
    n_bins: int = 24,
    n_instances: int = 4000,
) -> PrescribedCurve:
    """Compute the prescribed trajectory for every component and combine.

    ``n_instances`` controls how well the state-to-answer joint is estimated.
    For tasks whose state space is small the estimate converges quickly; the
    default is generous enough that the curve is stable to three decimals on
    the tasks in the paper.
    """
    grid = np.linspace(0.0, 1.0, n_bins)
    combined = np.zeros(n_bins, dtype=float)

    for comp in task.components:
        pool = [task.sample(rng, comp.name) for _ in range(n_instances)]
        lengths = np.array([task.n_steps(i) for i in pool])
        per_bin = np.zeros(n_bins, dtype=float)
        for b, frac in enumerate(grid):
            # Map the normalized position onto each instance's own step count,
            # so that instances of different lengths are compared at the same
            # relative depth into the chain.
            steps = np.rint(frac * lengths).astype(int)
            by_state: dict[tuple, Counter] = defaultdict(Counter)
            for inst, st in zip(pool, steps):
                by_state[(task.state_after(inst, int(st)),)][inst.answer] += 1
            correct = sum(c.most_common(1)[0][1] for c in by_state.values())
            per_bin[b] = center(correct / len(pool), comp.m)
        combined += comp.weight * per_bin

    return PrescribedCurve(grid=grid, values=combined, task=task.name)
