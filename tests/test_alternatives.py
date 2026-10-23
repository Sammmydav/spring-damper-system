"""Tests for the alternative integrators."""
import numpy as np
import pytest

from src.alternatives import simulate_alternative, simulate_scipy
from src.analysis import error_metrics
from src.cases import LIGHTLY_DAMPED_SPECIMEN
from src.models import SpringDamperParams, energy

P = SpringDamperParams(m=1.0, c=0.4, k=4.0, x0=1.0, v0=0.0)
P0 = SpringDamperParams(m=1.0, c=0.0, k=4.0, x0=1.0, v0=0.0)


@pytest.mark.parametrize("method", ["backward_euler", "symplectic_euler"])
@pytest.mark.parametrize("dt, T", [(0.0, 1.0), (-0.1, 1.0), (0.1, 0.01),
                                   (float("nan"), 1.0), ("0.1", 1.0)])
def test_invalid_time_inputs_rejected(method, dt, T):
    with pytest.raises(ValueError):
        simulate_alternative(P, dt, T, method)


def test_unknown_methods_rejected():
    with pytest.raises(ValueError):
        simulate_alternative(P, 0.01, 1.0, "midpoint")
    with pytest.raises(ValueError):
        simulate_scipy(P, 0.01, 1.0, "LSODA")


@pytest.mark.parametrize("method", ["backward_euler", "symplectic_euler"])
def test_output_shapes_and_initial_values(method):
    t, x, v = simulate_alternative(P, 0.1, 2.0, method)
    assert t.shape == x.shape == v.shape == (21,)
    assert (x[0], v[0]) == (1.0, 0.0)


def _orders(method):
    errs = []
    dts = [0.02, 0.01, 0.005, 0.0025]
    for dt in dts:
        t, x, v = simulate_alternative(P, dt, 5.0, method)
        errs.append(error_metrics(P, t, x, v)["max_abs_x"])
    errs = np.array(errs)
    return np.log(errs[:-1] / errs[1:]) / np.log(2.0)


@pytest.mark.parametrize("method", ["backward_euler", "symplectic_euler"])
def test_first_order_convergence(method):
    assert np.all(np.abs(_orders(method) - 1.0) < 0.15)


def test_backward_euler_energy_never_increases():
    for p in (P, P0):
        t, x, v = simulate_alternative(p, 0.05, 50.0, "backward_euler")
        assert np.all(np.diff(energy(p, x, v)) <= 1e-12)


def test_backward_euler_stable_above_forward_euler_limit():
    p = LIGHTLY_DAMPED_SPECIMEN.params           # forward Euler limit is about 3e-4 s
    t, x, v = simulate_alternative(p, 1e-2, 5.0, "backward_euler")
    e = energy(p, x, v)
    assert np.all(np.isfinite(e)) and e[-1] < e[0]


def test_symplectic_energy_bounded_without_damping():
    t, x, v = simulate_alternative(P0, 0.01, 100.0, "symplectic_euler")
    dev = np.abs(energy(P0, x, v) / energy(P0, 1.0, 0.0) - 1.0)
    n = len(dev) // 3
    assert dev.max() < 0.05
    assert dev[-n:].max() < 1.5 * dev[:n].max()   # no secular growth


def test_scipy_dop853_matches_exact():
    t, x, v, nfev = simulate_scipy(P, 0.01, 5.0, "DOP853", rtol=1e-10, atol=1e-12)
    assert error_metrics(P, t, x, v)["max_abs_x"] < 1e-8
    assert nfev > 0


def test_scipy_rk45_meets_loose_tolerance():
    t, x, v, nfev = simulate_scipy(P, 0.01, 5.0, "RK45", rtol=1e-6, atol=1e-9)
    assert error_metrics(P, t, x, v)["max_abs_x"] < 1e-4
