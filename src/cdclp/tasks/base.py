"""Task interface for state-tracking problems.

A task supplies three things the rest of the pipeline needs:

* a way to sample instances,
* the gold answer for an instance,
* the reference program's state after each step, from which the prescribed
  trajectory is computed.

Every task is split into *components*. A component groups instances that share
an answer-set size ``m``; the prior-centering in ``metrics.commitment`` divides
by ``1 - 1/m``, so mixing answer-set sizes inside one component would make the
centering ill-defined. Components carry weights that sum to one.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Hashable, Protocol, Sequence

import numpy as np


@dataclass(frozen=True)
class Instance:
    """One problem drawn from a task.

    Attributes
    ----------
    payload:
        Task-specific content (a bracket string, a generator sequence, ...).
        Never inspected by the pipeline; only passed back to the task.
    answer:
        Gold answer. Must be hashable and drawn from the component's answer set.
    component:
        Name of the component this instance belongs to.
    prompt:
        Rendered text handed to the model.
    """

    payload: Any
    answer: Hashable
    component: str
    prompt: str


@dataclass(frozen=True)
class Component:
    """A slice of a task with a fixed answer-set size."""

    name: str
    answer_set: tuple[Hashable, ...]
    weight: float

    @property
    def m(self) -> int:
        return len(self.answer_set)


class Task(Protocol):
    """What the pipeline requires of a task."""

    name: str
    components: Sequence[Component]

    def sample(self, rng: np.random.Generator, component: str) -> Instance:
        """Draw one instance from the named component."""
        ...

    def n_steps(self, instance: Instance) -> int:
        """Number of reference-program steps for this instance."""
        ...

    def state_after(self, instance: Instance, step: int) -> Hashable:
        """Reference-program state after ``step`` steps.

        The prescribed trajectory is the Bayes accuracy of the best decision
        rule that sees only this state, so the return value must be a complete
        summary of what the program has determined so far. Two instances whose
        states collide at step ``k`` are indistinguishable to any procedure
        that reads only the state.
        """
        ...


@dataclass
class TaskRegistry:
    """Name-to-constructor mapping, populated by the concrete task modules."""

    _entries: dict[str, Any] = field(default_factory=dict)

    def register(self, name: str):
        def deco(cls):
            self._entries[name] = cls
            return cls

        return deco

    def build(self, name: str, **kwargs) -> Task:
        if name not in self._entries:
            known = ", ".join(sorted(self._entries)) or "none"
            raise KeyError(f"unknown task {name!r}; registered: {known}")
        return self._entries[name](**kwargs)

    def names(self) -> list[str]:
        return sorted(self._entries)


REGISTRY = TaskRegistry()


def check_weights(components: Sequence[Component], tol: float = 1e-9) -> None:
    """Fail loudly if component weights do not form a distribution."""
    total = sum(c.weight for c in components)
    if abs(total - 1.0) > tol:
        raise ValueError(f"component weights sum to {total!r}, expected 1.0")
    for c in components:
        if c.weight <= 0:
            raise ValueError(f"component {c.name!r} has non-positive weight {c.weight}")
        if c.m < 2:
            raise ValueError(f"component {c.name!r} has answer set of size {c.m} < 2")
