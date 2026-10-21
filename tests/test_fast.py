"""Tests that the pure-float solvers match the reference NumPy solvers."""
import numpy as np
import pytest

from src.analysis import error_metrics
from src.cases import CASES
from src.fast import simulate_float
from src.models import SpringDamperParams
from src.solvers import simulate

P = SpringDamperParams(m=1.0, c=0.4, k=4.0, x0=1.0, v0=0.5)


@pytest.mark.parametrize("method", ["euler", "rk4"])
def test_matches_numpy_solver_to_roundoff(method):
    t1, x1, v1 = simulate(P, 0.01, 5.0, method)
    t2, x2, v2 = simulate_float(P, 0.01, 5.0, method)
    assert np.allclose(t1, t2, rtol=0, atol=1e-12)
    assert np.max(np.abs(x1 - x2)) < 1e-12
    assert np.max(np.abs(v1 - v2)) < 1e-12


@pytest.mark.parametrize("name", list(CASES))
def test_matches_on_realistic_cases(name):
    case = CASES[name]
    dt = 1e-3
    _, x1, _ = simulate(case.params, dt, case.T, "rk4")
    _, x2, _ = simulate_float(case.params, dt, case.T, "rk4")
    assert np.max(np.abs(x1 - x2)) < 1e-12 * max(1.0, abs(case.params.x0))


def test_float_rk4_still_accurate_against_exact():
    t, x, v = simulate_float(P, 0.01, 5.0, "rk4")
    assert error_metrics(P, t, x, v)["max_abs_x"] < 1e-8


@pytest.mark.parametrize("dt, T", [(0.0, 1.0), (-0.1, 1.0), (0.1, 0.01),
                                   (float("nan"), 1.0), ("0.1", 1.0)])
def test_invalid_inputs_rejected(dt, T):
    with pytest.raises(ValueError):
        simulate_float(P, dt, T, "rk4")


def test_unknown_method_rejected():
    with pytest.raises(ValueError):
        simulate_float(P, 0.01, 1.0, "midpoint")
