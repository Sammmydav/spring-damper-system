"""Alternative integrators used to compare against Euler (baseline) and RK4.

* ``backward_euler``    implicit Euler (A-stable, first order, numerically dissipative).
* ``symplectic_euler``  semi-implicit Euler (first order, keeps energy bounded for c = 0).
* SciPy adaptive solvers (``RK45`` and ``DOP853``) through :func:`simulate_scipy`.

For this linear problem the implicit step reduces to a 2 x 2 linear solve, which is
written out in closed form, so no iteration is needed.
"""
from __future__ import annotations

import math

import numpy as np

from .models import SpringDamperParams

ALT_METHODS = ("backward_euler", "symplectic_euler")


def _check_time_inputs(dt: float, T: float) -> None:
    """Raise ValueError unless dt > 0, T >= dt and both are finite real numbers."""
    for name, val in (("dt", dt), ("T", T)):
        if isinstance(val, bool) or not isinstance(val, (int, float)) or not math.isfinite(val):
            raise ValueError(f"{name} must be a finite real number, got {val!r}")
    if dt <= 0:
        raise ValueError(f"dt must be > 0, got {dt}")
    if T < dt:
        raise ValueError(f"T must be >= dt, got T={T}, dt={dt}")


def simulate_alternative(params: SpringDamperParams, dt: float, T: float,
                         method: str = "backward_euler"):
    """Integrate with a fixed step using an alternative Euler-type method.

    Parameters
    ----------
    params : SpringDamperParams
        Oscillator parameters and initial conditions.
    dt : float
        Time step in s. Must be finite and > 0.
    T : float
        End time in s. Must be finite and >= dt.
    method : {'backward_euler', 'symplectic_euler'}
        ``backward_euler`` solves ``y1 = y0 + dt A y1`` exactly (2 x 2 solve per step).
        ``symplectic_euler`` updates ``v`` first (damping treated explicitly) and then
        uses the new ``v`` to update ``x``.

    Returns
    -------
    t, x, v : numpy.ndarray
        Same meaning and shape as :func:`src.solvers.simulate`.

    Raises
    ------
    ValueError
        For an unknown method or invalid ``dt`` / ``T``.
    """
    if method not in ALT_METHODS:
        raise ValueError(f"unknown method {method!r}; choose from {list(ALT_METHODS)}")
    _check_time_inputs(dt, T)

    c_m, k_m = params.c / params.m, params.k / params.m
    n = int(round(T / dt))
    xs = [0.0] * (n + 1)
    vs = [0.0] * (n + 1)
    x, v = float(params.x0), float(params.v0)
    xs[0], vs[0] = x, v

    if method == "backward_euler":
        det = 1.0 + dt * c_m + dt * dt * k_m           # det(I - dt A)
        for i in range(n):
            x, v = ((1.0 + dt * c_m) * x + dt * v) / det, (v - dt * k_m * x) / det
            xs[i + 1], vs[i + 1] = x, v
    else:  # symplectic_euler
        for i in range(n):
            v = v - dt * (c_m * v + k_m * x)
            x = x + dt * v
            xs[i + 1], vs[i + 1] = x, v

    return dt * np.arange(n + 1), np.array(xs), np.array(vs)


def simulate_scipy(params: SpringDamperParams, dt: float, T: float,
                   method: str = "DOP853", rtol: float = 1e-6, atol: float = 1e-9):
    """Adaptive-step integration with SciPy, sampled on the grid ``0, dt, 2 dt, ...``.

    Parameters
    ----------
    params : SpringDamperParams
        Oscillator parameters and initial conditions.
    dt : float
        Output sampling interval in s (not the internal step size).
    T : float
        End time in s.
    method : {'RK45', 'DOP853'}
        SciPy explicit Runge-Kutta method.
    rtol, atol : float
        Relative and absolute error tolerances passed to ``solve_ivp``.

    Returns
    -------
    t, x, v : numpy.ndarray
        Sampled time grid, displacement and velocity.
    nfev : int
        Number of right-hand-side evaluations used (machine-independent cost).

    Raises
    ------
    ValueError
        For an unknown method, invalid ``dt`` / ``T``, or a failed integration.
    """
    from scipy.integrate import solve_ivp

    if method not in ("RK45", "DOP853"):
        raise ValueError(f"unknown method {method!r}; choose 'RK45' or 'DOP853'")
    _check_time_inputs(dt, T)
    n = int(round(T / dt))
    t = dt * np.arange(n + 1)
    c_m, k_m = params.c / params.m, params.k / params.m
    sol = solve_ivp(lambda _t, y: [y[1], -(c_m * y[1] + k_m * y[0])], (0.0, t[-1]),
                    [params.x0, params.v0], t_eval=t, method=method, rtol=rtol, atol=atol)
    if not sol.success:
        raise ValueError(f"solve_ivp failed: {sol.message}")
    return t, sol.y[0], sol.y[1], int(sol.nfev)
