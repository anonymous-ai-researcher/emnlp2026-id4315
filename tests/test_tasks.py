"""Tasks must agree with their own reference programs.

The prescribed trajectory is only meaningful if ``state_after`` really is what
the program has determined. The first test below is the one that matters: at the
last step the state must determine the answer, or the bound is being computed
against something other than the task.
"""

import numpy as np
import pytest

from cdclp.crasp.prescribed import center, prescribed_curve
from cdclp.tasks.base import REGISTRY
import cdclp.tasks.formal  # noqa: F401


@pytest.mark.parametrize("name", ["dyck2", "a4s4", "a5s5", "fsa"])
def test_terminal_state_determines_answer(name):
    task = REGISTRY.build(name)
    rng = np.random.default_rng(0)
    for comp in task.components:
        seen = {}
        for _ in range(150):
            inst = task.sample(rng, comp.name)
            s = task.state_after(inst, task.n_steps(inst))
            if s in seen:
                assert seen[s] == inst.answer, (
                    f"{name}/{comp.name}: two instances share terminal state {s!r} "
                    "but disagree on the answer"
                )
            seen[s] = inst.answer


@pytest.mark.parametrize("name", ["dyck2", "a4s4", "a5s5", "fsa"])
def test_weights_form_a_distribution(name):
    task = REGISTRY.build(name)
    assert sum(c.weight for c in task.components) == pytest.approx(1.0)
    assert all(c.m >= 2 for c in task.components)


@pytest.mark.parametrize("name", ["dyck2", "fsa"])
def test_prescribed_curve_is_monotone_and_bounded(name):
    """More of the prefix can only help a rule that reads the state."""
    task = REGISTRY.build(name)
    c = prescribed_curve(task, np.random.default_rng(3), n_bins=8, n_instances=400)
    assert c.values.min() >= -0.05
    assert c.values.max() <= 1.0 + 1e-9
    # Allow small sampling noise, but the trend must be upward.
    assert c.values[-1] > c.values[0]


def test_prescribed_curve_ends_at_certainty():
    task = REGISTRY.build("fsa")
    c = prescribed_curve(task, np.random.default_rng(4), n_bins=6, n_instances=400)
    assert c.values[-1] == pytest.approx(1.0, abs=0.02)


def test_unknown_task_raises():
    with pytest.raises(KeyError):
        REGISTRY.build("no-such-task")
