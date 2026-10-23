"""Compare alternative integrators with the baseline (Euler) and the selected method (RK4).

Usage (from the repository root)::

    python -m scripts.alternatives_study

Prints three tables: (1) accuracy and cost on the main case, (2) energy behaviour of
the undamped oscillator, (3) robustness on the lightly damped specimen.
Cost is reported as right-hand-side (RHS) evaluations, which does not depend on the
machine; wall-clock time is shown for reference only.
"""
from __future__ import annotations

import time

import numpy as np

from src.alternatives import simulate_alternative, simulate_scipy
from src.analysis import error_metrics
from src.cases import LIGHTLY_DAMPED_SPECIMEN
from src.fast import simulate_float
from src.models import SpringDamperParams, energy

MAIN = SpringDamperParams(m=1.0, c=0.4, k=4.0, x0=1.0, v0=0.0)
UNDAMPED = SpringDamperParams(m=1.0, c=0.0, k=4.0, x0=1.0, v0=0.0)
T = 5.0


def best_time(fn, repeats: int = 5) -> float:
    """Best-of-``repeats`` wall-clock time of ``fn()`` in seconds."""
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def fixed_step_runs(p, dt, T, method):
    """Return (t, x, v) for one fixed-step method name used in this study."""
    if method in ("euler", "rk4"):
        return simulate_float(p, dt, T, method)
    return simulate_alternative(p, dt, T, method)


def main() -> None:
    print("TABLE 1: main case (zeta = 0.1), T = 5 s, error = max |x - exact| [m]")
    print(f"{'method':32s} {'max error':>11s} {'RHS evals':>10s} {'time [ms]':>10s}")
    rows = [("Euler (dt = 0.001)", "euler", 0.001, 1), ("Backward Euler (dt = 0.001)", "backward_euler", 0.001, 1),
            ("Symplectic Euler (dt = 0.001)", "symplectic_euler", 0.001, 1), ("RK4 (dt = 0.05)", "rk4", 0.05, 4)]
    for label, m, dt, per_step in rows:
        t, x, v = fixed_step_runs(MAIN, dt, T, m)
        err = error_metrics(MAIN, t, x, v)["max_abs_x"]
        ms = best_time(lambda: fixed_step_runs(MAIN, dt, T, m)) * 1e3
        print(f"{label:32s} {err:11.2e} {per_step * int(round(T / dt)):10d} {ms:10.3f}")
    for label, m, rtol in (("SciPy RK45 (rtol = 1e-6)", "RK45", 1e-6), ("SciPy DOP853 (rtol = 1e-9)", "DOP853", 1e-9)):
        t, x, v, nfev = simulate_scipy(MAIN, 0.01, T, m, rtol=rtol, atol=1e-12)
        err = error_metrics(MAIN, t, x, v)["max_abs_x"]
        ms = best_time(lambda: simulate_scipy(MAIN, 0.01, T, m, rtol=rtol, atol=1e-12)) * 1e3
        print(f"{label:32s} {err:11.2e} {nfev:10d} {ms:10.3f}")

    print("\nTABLE 2: undamped oscillator, dt = 0.01 s, relative energy change after 50 s")
    print(f"{'method':20s} {'(E_end - E_0) / E_0':>20s}")
    for m in ("euler", "backward_euler", "symplectic_euler", "rk4"):
        t, x, v = fixed_step_runs(UNDAMPED, 0.01, 50.0, m)
        e = energy(UNDAMPED, x, v)
        print(f"{m:20s} {(e[-1] - e[0]) / e[0]:20.3e}")

    print("\nTABLE 3: lightly damped specimen (zeta ~ 0.005), dt = 1e-3 s, T = 5 s")
    p = LIGHTLY_DAMPED_SPECIMEN.params
    print(f"{'method':20s} {'max |x err| / |x0|':>20s}")
    for m in ("euler", "backward_euler", "symplectic_euler", "rk4"):
        t, x, v = fixed_step_runs(p, 1e-3, 5.0, m)
        err = error_metrics(p, t, x, v)["max_abs_x"] / abs(p.x0)
        print(f"{m:20s} {err:20.3e}")


if __name__ == "__main__":
    main()
