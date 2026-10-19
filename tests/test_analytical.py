"""Tests for the exact solution."""
import numpy as np
import pytest

from src.analytical import exact_solution
from src.models import SpringDamperParams, energy

CASES = [
    SpringDamperParams(1.0, 0.0, 4.0, 1.0, 0.5),    # undamped
    SpringDamperParams(1.0, 1.0, 4.0, 1.0, 0.5),    # underdamped
    SpringDamperParams(1.0, 4.0, 4.0, 1.0, 0.5),    # critical
    SpringDamperParams(1.0, 10.0, 4.0, 1.0, 0.5),   # overdamped
]


@pytest.mark.parametrize("p", CASES)
def test_initial_conditions(p):
    x, v = exact_solution(p, [0.0])
    assert x[0] == pytest.approx(p.x0)
    assert v[0] == pytest.approx(p.v0)


@pytest.mark.parametrize("p", CASES)
def test_satisfies_ode(p):
    """Residual m x'' + c x' + k x ~ 0, with x'' from finite differences of v."""
    t = np.linspace(0.0, 3.0, 3001)
    x, v = exact_solution(p, t)
    a = np.gradient(v, t)
    residual = p.m * a + p.c * v + p.k * x
    assert np.max(np.abs(residual[2:-2])) < 1e-3


def test_undamped_conserves_energy():
    p = CASES[0]
    t = np.linspace(0.0, 20.0, 500)
    x, v = exact_solution(p, t)
    e = energy(p, x, v)
    assert np.allclose(e, e[0], rtol=1e-12)


@pytest.mark.parametrize("p", CASES[1:])
def test_damped_energy_decays(p):
    t = np.linspace(0.0, 10.0, 500)
    x, v = exact_solution(p, t)
    e = energy(p, x, v)
    assert np.all(np.diff(e) <= 1e-12)


def test_negative_time_rejected():
    with pytest.raises(ValueError):
        exact_solution(CASES[0], [-1.0])
