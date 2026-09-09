"""Properties the estimator must satisfy, stated as tests.

These are the invariants that would have caught the mistakes worth catching:
a centering that forgets the prior, a bound comparison with the sign reversed,
a difference interval that silently assumes independence.
"""

import numpy as np
import pytest

from cdclp.metrics.commitment import (
    center, combine_components, curve_from_indicators, unfaithfulness,
    violation_area,
)
from cdclp.metrics.inference import (
    dependence_robust_difference, holm_bonferroni, percentile_interval,
    permutation_test,
)

GRID = np.linspace(0.0, 1.0, 24)


def test_centering_maps_chance_to_zero():
    for m in (2, 12, 60, 120):
        assert center(1.0 / m, m) == pytest.approx(0.0)
        assert center(1.0, m) == pytest.approx(1.0)


def test_centering_rejects_degenerate_answer_set():
    with pytest.raises(ValueError):
        center(0.5, 1)


def test_curve_shape_is_validated():
    with pytest.raises(ValueError):
        curve_from_indicators(np.zeros(24, dtype=np.int8), 2, GRID)
    with pytest.raises(ValueError):
        curve_from_indicators(np.zeros((10, 5), dtype=np.int8), 2, GRID)


def test_unfaithfulness_is_one_sided():
    """A model below the bound is not unfaithful, however far below it sits."""
    below = np.zeros(24)
    prescribed = np.ones(24)
    assert unfaithfulness(below, prescribed) == 0.0
    assert violation_area(below, prescribed, GRID) == 0.0


def test_unfaithfulness_reports_the_worst_step():
    measured = np.zeros(24)
    measured[7] = 0.9
    prescribed = np.full(24, 0.2)
    assert unfaithfulness(measured, prescribed) == pytest.approx(0.7)


def test_commitment_step_interpolates():
    values = np.linspace(0.0, 1.0, 24)
    c = curve_from_indicators(
        np.tile((values * 100).astype(np.int8).clip(0, 1), (1, 1)).reshape(24, 1),
        2, GRID,
    )
    # A curve that reaches 0.5 exactly at the midpoint should report the
    # midpoint, not the first bin at or above it.
    straight = type(c)(grid=GRID, values=values, n_pairs=1, model="", task="")
    assert straight.commitment_step() == pytest.approx(0.5, abs=0.05)


def test_commitment_step_none_when_never_reached():
    flat = curve_from_indicators(np.zeros((24, 10), dtype=np.int8), 2, GRID)
    assert flat.commitment_step() is None


def test_component_weights_must_sum_to_one():
    a = curve_from_indicators(np.zeros((24, 4), dtype=np.int8), 2, GRID)
    with pytest.raises(ValueError):
        combine_components({"x": a, "y": a}, {"x": 0.5, "y": 0.9})


def test_dependence_robust_interval_contains_naive_one():
    """The projection must never be narrower than assuming independence.

    If it were, the extra safety would be illusory.
    """
    rng = np.random.default_rng(0)
    a = rng.normal(0.3, 0.05, 4000)
    b = rng.normal(0.7, 0.05, 4000)
    robust = dependence_robust_difference(a, b)
    naive_lo, naive_hi = np.quantile(a - b, [0.025, 0.975])
    assert robust.lo <= naive_lo
    assert robust.hi >= naive_hi


def test_dependence_robust_interval_still_separates_clear_cases():
    rng = np.random.default_rng(1)
    a = rng.normal(0.30, 0.01, 3000)
    b = rng.normal(0.70, 0.01, 3000)
    assert dependence_robust_difference(a, b).excludes_zero()


def test_permutation_p_value_is_never_zero():
    """A finite permutation count cannot license a p-value of exactly zero."""
    rng = np.random.default_rng(2)
    a = np.ones((24, 50), dtype=np.int8)
    b = np.zeros((24, 50), dtype=np.int8)
    _, p = permutation_test(a, b, lambda x: float(x.mean()), 200, rng)
    assert p > 0.0
    assert p == pytest.approx(1 / 201, abs=1e-9)


def test_holm_is_monotone_and_conservative():
    raw = {"a": 0.001, "b": 0.02, "c": 0.30}
    adj = holm_bonferroni(raw)
    assert adj["a"][0] <= adj["b"][0] <= adj["c"][0]
    for k, v in raw.items():
        assert adj[k][0] >= v


def test_percentile_interval_rejects_bad_level():
    with pytest.raises(ValueError):
        percentile_interval(np.zeros(10), level=1.5)
