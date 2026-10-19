"""Tests for the time integrators, convergence, stability and metrics."""
import numpy as np
import pytest

from src.analysis import (convergence_study, error_metrics,
                          euler_max_stable_dt, time_method)
from src.models import SpringDamperParams, energy
from src.solvers import simulate, simulate_reference

P = SpringDamperParams(m=1.0, c=0.4, k=4.0, x0=1.0, v0=0.0)


def test_unknown_method_rejected():
    with pytest.raises(ValueError):
        simulate(P, 0.01, 1.0, "midpoint")


@pytest.mark.parametrize("dt, T", [(0.0, 1.0), (-0.1, 1.0), (0.1, 0.01),
                                   (float("nan"), 1.0), (0.1, float("inf")),
                                   ("0.1", 1.0)])
def test_invalid_time_inputs_rejected(dt, T):
    with pytest.raises(ValueError):
        simulate(P, dt, T, "rk4")


def test_output_shapes_and_initial_values():
    t, x, v = simulate(P, 0.1, 2.0, "euler")
    assert t.shape == x.shape == v.shape == (21,)
    assert (t[0], x[0], v[0]) == (0.0, 1.0, 0.0)


def test_rk4_matches_exact_closely():
    t, x, v = simulate(P, 0.01, 5.0, "rk4")
    assert error_metrics(P, t, x, v)["max_abs_x"] < 1e-8


def test_euler_error_larger_than_rk4():
    errs = {}
    for m in ("euler", "rk4"):
        t, x, v = simulate(P, 0.01, 5.0, m)
        errs[m] = error_metrics(P, t, x, v)["max_abs_x"]
    assert errs["rk4"] < errs["euler"]


def test_agrees_with_scipy_reference():
    t, x, v = simulate(P, 0.01, 5.0, "rk4")
    xr, vr = simulate_reference(P, t)
    assert np.max(np.abs(x - xr)) < 1e-7


def test_euler_convergence_order_is_one():
    _, orders = convergence_study(P, [0.04, 0.02, 0.01, 0.005], 5.0, "euler")
    assert np.all(np.abs(orders - 1.0) < 0.15)


def test_rk4_convergence_order_is_four():
    _, orders = convergence_study(P, [0.2, 0.1, 0.05, 0.025], 5.0, "rk4")
    assert np.all(np.abs(orders - 4.0) < 0.5)


def test_euler_gains_energy_without_damping():
    p = SpringDamperParams(1.0, 0.0, 4.0)
    t, x, v = simulate(p, 0.01, 20.0, "euler")
    e = energy(p, x, v)
    assert e[-1] > 1.5 * e[0]


def test_stability_limit_is_sharp():
    """Euler decays just below the analytical limit and grows just above it."""
    dt_max = euler_max_stable_dt(P)
    assert dt_max == pytest.approx(2 * P.zeta / P.omega_n)
    T = 400.0
    _, x_lo, v_lo = simulate(P, 0.9 * dt_max, T, "euler")
    _, x_hi, v_hi = simulate(P, 1.1 * dt_max, T, "euler")
    assert energy(P, x_lo[-1], v_lo[-1]) < energy(P, x_lo[0], v_lo[0])
    assert energy(P, x_hi[-1], v_hi[-1]) > energy(P, x_hi[0], v_hi[0])


def test_stability_limit_zero_when_undamped():
    assert euler_max_stable_dt(SpringDamperParams(1.0, 0.0, 4.0)) == 0.0


def test_time_method_returns_positive_seconds():
    assert time_method(P, 0.01, 1.0, "rk4", repeats=2) > 0.0
