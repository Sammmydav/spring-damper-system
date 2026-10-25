"""Exact solution of the free damped oscillator (all damping regimes)."""
from __future__ import annotations

import numpy as np

from .models import SpringDamperParams


def exact_solution(params: SpringDamperParams, t):
    """Closed-form displacement and velocity.

    Parameters
    ----------
    params : SpringDamperParams
        Oscillator parameters and initial conditions.
    t : array_like
        Times in seconds (>= 0).

    Returns
    -------
    x, v : numpy.ndarray
        Displacement [m] and velocity [m/s] at each time in ``t``.

    Notes
    -----
    With w = omega_n and zeta the damping ratio:

    * undamped / underdamped (zeta < 1):
      ``x = e^{-s t}(A cos wd t + B sin wd t)``, ``s = zeta w``,
      ``wd = w sqrt(1 - zeta^2)``, ``A = x0``, ``B = (v0 + s x0)/wd``.
    * critical (zeta = 1): ``x = e^{-w t}(x0 + (v0 + w x0) t)``.
    * overdamped (zeta > 1): ``x = C1 e^{r1 t} + C2 e^{r2 t}`` with
      ``r = -w (zeta +- sqrt(zeta^2 - 1))``.
    """
    t = np.asarray(t, dtype=float)
    # The initial-value problem is only defined for t >= 0.
    if np.any(t < 0):
        raise ValueError("times must be >= 0")
    w, z = params.omega_n, params.zeta
    x0, v0 = params.x0, params.v0
    regime = params.regime

    # zeta = 0 is the limit of the underdamped formula (wd = w), so both cases share this
    # branch.
    if regime in ("undamped", "underdamped"):
        # s is the exponential decay rate; wd is the damped natural frequency.
        s = z * w
        wd = w * np.sqrt(1.0 - z**2)
        # A and B come from matching x(0) = x0 and x'(0) = v0.
        a, b = x0, (v0 + s * x0) / wd
        e, cs, sn = np.exp(-s * t), np.cos(wd * t), np.sin(wd * t)
        x = e * (a * cs + b * sn)
        # v is the analytic derivative of x, so velocity can be validated as well as
        # displacement.
        v = e * ((-s * a + wd * b) * cs + (-s * b - wd * a) * sn)
    elif regime == "critical":
        # A repeated root r = -w gives (C1 + C2 t) e^(-w t); C2 comes from x'(0) = v0.
        c1, c2 = x0, v0 + w * x0
        e = np.exp(-w * t)
        x = e * (c1 + c2 * t)
        v = e * (c2 - w * c1 - w * c2 * t)
    else:  # overdamped
        # Two distinct real roots r1 and r2; both are negative, so the response decays without
        # oscillating.
        root = w * np.sqrt(z**2 - 1.0)
        r1, r2 = -w * z + root, -w * z - root
        # C1 and C2 solve the 2 x 2 system from x(0) = x0 and x'(0) = v0.
        c1 = (v0 - r2 * x0) / (r1 - r2)
        c2 = x0 - c1
        x = c1 * np.exp(r1 * t) + c2 * np.exp(r2 * t)
        v = c1 * r1 * np.exp(r1 * t) + c2 * r2 * np.exp(r2 * t)
    return x, v
