"""Time integrators: explicit Euler (baseline) and classical RK4.

The second-order ODE is written as a first-order system for
``y = [x, v]``:  ``y' = [v, -(c v + k x)/m]``.
"""
from __future__ import annotations

import math

import numpy as np

from .models import SpringDamperParams


def _rhs(params: SpringDamperParams, y: np.ndarray) -> np.ndarray:
    """Right-hand side f(y) of the first-order system (one evaluation)."""
    x, v = y
    return np.array([v, -(params.c * v + params.k * x) / params.m])


def euler_step(params: SpringDamperParams, y: np.ndarray, dt: float) -> np.ndarray:
    """One explicit (forward) Euler step. 1 RHS evaluation, 1st-order accurate."""
    return y + dt * _rhs(params, y)


def rk4_step(params: SpringDamperParams, y: np.ndarray, dt: float) -> np.ndarray:
    """One classical Runge-Kutta step. 4 RHS evaluations, 4th-order accurate."""
    k1 = _rhs(params, y)
    k2 = _rhs(params, y + 0.5 * dt * k1)
    k3 = _rhs(params, y + 0.5 * dt * k2)
    k4 = _rhs(params, y + dt * k3)
    return y + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6.0


METHODS = {"euler": euler_step, "rk4": rk4_step}


def simulate(params: SpringDamperParams, dt: float, T: float, method: str = "rk4"):
    """Integrate the oscillator from t = 0 to ``T`` with a fixed step.

    Parameters
    ----------
    params : SpringDamperParams
        Oscillator parameters and initial conditions.
    dt : float
        Time step in s. Must be finite and > 0.
    T : float
        End time in s. Must be finite and >= dt.
    method : {'euler', 'rk4'}
        Integration method.

    Returns
    -------
    t, x, v : numpy.ndarray
        Time grid ``0, dt, 2 dt, ...`` (n+1 points, n = round(T/dt)),
        displacement [m] and velocity [m/s].

    Raises
    ------
    ValueError
        For an unknown method or invalid ``dt`` / ``T``.
    """
    if method not in METHODS:
        raise ValueError(f"unknown method {method!r}; choose from {sorted(METHODS)}")
    for name, val in (("dt", dt), ("T", T)):
        if isinstance(val, bool) or not isinstance(val, (int, float)) or not math.isfinite(val):
            raise ValueError(f"{name} must be a finite real number, got {val!r}")
    if dt <= 0:
        raise ValueError(f"dt must be > 0, got {dt}")
    if T < dt:
        raise ValueError(f"T must be >= dt, got T={T}, dt={dt}")

    step = METHODS[method]
    n = int(round(T / dt))
    t = dt * np.arange(n + 1)
    y = np.empty((n + 1, 2))
    y[0] = (params.x0, params.v0)
    for i in range(n):
        y[i + 1] = step(params, y[i], dt)
    return t, y[:, 0], y[:, 1]


def simulate_reference(params: SpringDamperParams, t):
    """Trusted reference using SciPy's adaptive ``solve_ivp`` (tight tolerances).

    Parameters
    ----------
    params : SpringDamperParams
        Oscillator parameters.
    t : array_like
        Output times in s (monotonic, starting at t[0] = 0).

    Returns
    -------
    x, v : numpy.ndarray
        Displacement and velocity at ``t``.
    """
    from scipy.integrate import solve_ivp

    t = np.asarray(t, dtype=float)
    sol = solve_ivp(
        lambda _t, y: _rhs(params, y),
        (t[0], t[-1]),
        (params.x0, params.v0),
        t_eval=t,
        method="DOP853",
        rtol=1e-12,
        atol=1e-14,
    )
    return sol.y[0], sol.y[1]
