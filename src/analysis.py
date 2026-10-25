"""Validation metrics, convergence study, stability limit and timing."""
from __future__ import annotations

import time

import numpy as np

from .analytical import exact_solution
from .models import SpringDamperParams
from .solvers import simulate


def error_metrics(params: SpringDamperParams, t, x, v) -> dict:
    """Compare a numerical solution with the exact one.

    Returns
    -------
    dict
        ``max_abs_x``, ``rmse_x``, ``max_abs_v``, ``rmse_v`` (m, m, m/s, m/s).
    """
    # Signed differences from the exact solution: max shows the worst point, RMSE the typical
    # size.
    xe, ve = exact_solution(params, t)
    ex, ev = np.asarray(x) - xe, np.asarray(v) - ve
    return {
        "max_abs_x": float(np.max(np.abs(ex))),
        "rmse_x": float(np.sqrt(np.mean(ex**2))),
        "max_abs_v": float(np.max(np.abs(ev))),
        "rmse_v": float(np.sqrt(np.mean(ev**2))),
    }


def convergence_study(params: SpringDamperParams, dts, T: float, method: str):
    """Max displacement error for a sequence of time steps.

    Parameters
    ----------
    dts : sequence of float
        Time steps, ideally each half the previous one.

    Returns
    -------
    errors : numpy.ndarray
        Max absolute displacement error for each dt.
    orders : numpy.ndarray
        Observed order ``log(e_i / e_{i+1}) / log(dt_i / dt_{i+1})``
        (length ``len(dts) - 1``).
    """
    dts = np.asarray(dts, dtype=float)
    errors = []
    for dt in dts:
        t, x, v = simulate(params, float(dt), T, method)
        errors.append(error_metrics(params, t, x, v)["max_abs_x"])
    errors = np.array(errors)
    # If e ~ C dt^p then p = log(e_i / e_(i+1)) / log(dt_i / dt_(i+1)).
    orders = np.log(errors[:-1] / errors[1:]) / np.log(dts[:-1] / dts[1:])
    return errors, orders


def euler_max_stable_dt(params: SpringDamperParams) -> float:
    """Largest dt for which explicit Euler does not amplify the solution.

    Euler multiplies each eigen-mode by ``1 + lambda dt``; stability needs
    ``|1 + lambda dt| <= 1`` for both eigenvalues of the system matrix, which
    gives ``dt <= -2 Re(lambda) / |lambda|^2``.

    Returns
    -------
    float
        Stability limit in s. ``0.0`` for the undamped case (Euler is
        never stable there: ``|1 + i w dt| > 1`` for every dt > 0).
    """
    # y' = A y with y = [x, v]. Euler multiplies each eigenmode by (1 + lambda dt) every step.
    a = np.array([[0.0, 1.0], [-params.k / params.m, -params.c / params.m]])
    lam = np.linalg.eigvals(a)
    # |1 + lambda dt| <= 1 reduces to dt <= -2 Re(lambda) / |lambda|^2; the smaller of the two
    # limits governs.
    limits = [-2.0 * l.real / abs(l) ** 2 for l in lam]
    # Clamp at zero: with c = 0 the limit is zero, meaning Euler is never stable.
    return float(max(0.0, min(limits)))


def time_method(params: SpringDamperParams, dt: float, T: float, method: str,
                repeats: int = 5) -> float:
    """Best-of-``repeats`` wall-clock time of one simulation, in seconds."""
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        simulate(params, dt, T, method)
        # Keep the minimum: it is the run least disturbed by other processes on the machine.
        best = min(best, time.perf_counter() - t0)
    return best
