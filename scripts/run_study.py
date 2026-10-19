"""Run the validation, convergence, stability and timing studies.

Usage (from the repository root)::

    python -m scripts.run_study

Prints a summary table and, if matplotlib is installed, saves figures to
``figures/``.
"""
from __future__ import annotations

import os

import numpy as np

from src.analysis import (convergence_study, error_metrics,
                          euler_max_stable_dt, time_method)
from src.models import SpringDamperParams, energy
from src.solvers import simulate

P = SpringDamperParams(m=1.0, c=0.4, k=4.0, x0=1.0, v0=0.0)
T = 5.0


def main() -> None:
    os.makedirs("figures", exist_ok=True)
    print(f"Case: m={P.m} c={P.c} k={P.k}  zeta={P.zeta:.3f} ({P.regime})")

    # 1. Convergence study
    dts = [0.2, 0.1, 0.05, 0.025, 0.0125, 0.00625]
    results = {m: convergence_study(P, dts, T, m) for m in ("euler", "rk4")}
    print("\nConvergence (max |x error|):")
    for m, (err, orders) in results.items():
        print(f"  {m:5s} errors={np.array2string(err, precision=2)} "
              f"orders={np.array2string(orders, precision=2)}")

    # 2. Stability limit
    print(f"\nEuler stability limit dt_max = {euler_max_stable_dt(P):.4f} s")

    # 3. Cost vs accuracy: time each method at the same dt
    print("\nTiming (dt=0.01, T=5 s):")
    for m in ("euler", "rk4"):
        print(f"  {m:5s} {time_method(P, 0.01, T, m) * 1e3:.2f} ms")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\nmatplotlib not installed - skipping figures.")
        return

    plt.figure()
    for m, (err, _) in results.items():
        plt.loglog(dts, err, "o-", label=m)
    plt.xlabel("dt [s]"); plt.ylabel("max |x error| [m]"); plt.legend()
    plt.title("Convergence"); plt.grid(True, which="both", alpha=0.3)
    plt.savefig("figures/convergence.png", dpi=150); plt.close()

    p0 = SpringDamperParams(1.0, 0.0, 4.0)
    plt.figure()
    for m in ("euler", "rk4"):
        t, x, v = simulate(p0, 0.01, 20.0, m)
        plt.plot(t, energy(p0, x, v), label=m)
    plt.xlabel("t [s]"); plt.ylabel("energy [J]"); plt.legend()
    plt.title("Energy drift, undamped case"); plt.grid(alpha=0.3)
    plt.savefig("figures/energy_drift.png", dpi=150); plt.close()
    print("\nFigures saved to figures/")


if __name__ == "__main__":
    main()
