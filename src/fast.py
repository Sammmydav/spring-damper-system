"""Pure-Python-float versions of the Euler and RK4 integrators.

The reference solvers in :mod:`src.solvers` allocate a small NumPy array on
every right-hand-side evaluation. For a 2-element state the arithmetic is
trivial, so profiling shows the cost is dominated by that array overhead.
These versions keep ``x`` and ``v`` as plain floats and write the
arithmetic out explicitly. They implement the *same* mathematics, so results
agree with :func:`src.solvers.simulate` to round-off.
"""
from __future__ import annotations

import math

import numpy as np

from .models import SpringDamperParams
from .solvers import METHODS


def simulate_float(params: SpringDamperParams, dt: float, T: float,
                   method: str = "rk4"):
    """Integrate the oscillator using scalar floats instead of arrays.

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
        Same meaning and shape as :func:`src.solvers.simulate`.

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

    c_m, k_m = params.c / params.m, params.k / params.m   # acceleration = -(c/m) v - (k/m) x
    n = int(round(T / dt))
    xs = [0.0] * (n + 1)
    vs = [0.0] * (n + 1)
    x, v = float(params.x0), float(params.v0)
    xs[0], vs[0] = x, v

    if method == "euler":
        for i in range(n):
            a = -(c_m * v + k_m * x)
            x, v = x + dt * v, v + dt * a
            xs[i + 1], vs[i + 1] = x, v
    else:  # rk4
        h2, h6 = 0.5 * dt, dt / 6.0
        for i in range(n):
            k1x, k1v = v, -(c_m * v + k_m * x)
            xa, va = x + h2 * k1x, v + h2 * k1v
            k2x, k2v = va, -(c_m * va + k_m * xa)
            xb, vb = x + h2 * k2x, v + h2 * k2v
            k3x, k3v = vb, -(c_m * vb + k_m * xb)
            xc, vc = x + dt * k3x, v + dt * k3v
            k4x, k4v = vc, -(c_m * vc + k_m * xc)
            x += h6 * (k1x + 2.0 * k2x + 2.0 * k3x + k4x)
            v += h6 * (k1v + 2.0 * k2v + 2.0 * k3v + k4v)
            xs[i + 1], vs[i + 1] = x, v

    return dt * np.arange(n + 1), np.array(xs), np.array(vs)
