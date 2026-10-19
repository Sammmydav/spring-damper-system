"""Tests for the realistic named cases."""
import numpy as np
import pytest

from src.analysis import convergence_study, error_metrics, euler_max_stable_dt
from src.cases import CASES, LIGHTLY_DAMPED_SPECIMEN, QUARTER_CAR
from src.models import energy
from src.solvers import simulate


def test_quarter_car_parameters_within_published_ranges():
    p = QUARTER_CAR.params
    assert 250 <= p.m <= 340
    assert 5000 <= p.k <= 20000
    assert 500 <= p.c <= 2000
    assert p.regime == "underdamped"


def test_lightly_damped_specimen_matches_published_damping_range():
    p = LIGHTLY_DAMPED_SPECIMEN.params
    assert p.m == 1.0
    assert 0.0042 <= p.zeta <= 0.0055


@pytest.mark.parametrize("name", list(CASES))
def test_rk4_accurate_on_realistic_cases(name):
    case = CASES[name]
    dt = 1e-3 if name == "quarter_car" else 1e-4
    t, x, v = simulate(case.params, dt, case.T, "rk4")
    err = error_metrics(case.params, t, x, v)["max_abs_x"]
    assert err < 1e-4 * abs(case.params.x0)


@pytest.mark.parametrize("name", list(CASES))
def test_energy_never_increases_with_rk4_small_dt(name):
    case = CASES[name]
    t, x, v = simulate(case.params, 1e-4, 1.0, "rk4")
    e = energy(case.params, x, v)
    assert np.all(np.diff(e) <= 1e-9 * e[0])


def test_euler_unstable_above_limit_on_lightly_damped_specimen():
    p = LIGHTLY_DAMPED_SPECIMEN.params
    dt_max = euler_max_stable_dt(p)
    t, x, v = simulate(p, 2.0 * dt_max, 5.0, "euler")
    e = energy(p, x, v)
    assert e[-1] > e[0]


def test_quarter_car_euler_converges_first_order():
    _, orders = convergence_study(QUARTER_CAR.params,
                                  [0.004, 0.002, 0.001, 0.0005], 5.0, "euler")
    assert np.all(np.abs(orders - 1.0) < 0.2)
