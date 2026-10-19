"""Tests for input validation and derived quantities."""
import math

import pytest

from src.models import SpringDamperParams, energy


@pytest.mark.parametrize("kwargs", [
    dict(m=0, c=1, k=1), dict(m=-1, c=1, k=1),
    dict(m=1, c=-0.1, k=1), dict(m=1, c=1, k=0), dict(m=1, c=1, k=-5),
    dict(m=float("nan"), c=1, k=1), dict(m=1, c=float("inf"), k=1),
    dict(m="1", c=1, k=1), dict(m=1, c=1, k=1, x0=None),
])
def test_invalid_parameters_raise(kwargs):
    with pytest.raises(ValueError):
        SpringDamperParams(**kwargs)


def test_derived_quantities():
    p = SpringDamperParams(m=1.0, c=2.0, k=4.0)
    assert math.isclose(p.omega_n, 2.0)
    assert math.isclose(p.zeta, 0.5)
    assert p.regime == "underdamped"


@pytest.mark.parametrize("c, regime", [(0.0, "undamped"), (1.0, "underdamped"),
                                       (4.0, "critical"), (10.0, "overdamped")])
def test_regime_classification(c, regime):
    assert SpringDamperParams(m=1.0, c=c, k=4.0).regime == regime


def test_energy_value():
    p = SpringDamperParams(m=2.0, c=0.0, k=8.0)
    assert math.isclose(energy(p, 1.0, 2.0), 0.5 * 2 * 4 + 0.5 * 8 * 1)
