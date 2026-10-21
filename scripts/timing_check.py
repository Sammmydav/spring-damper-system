import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import time
from src.models import SpringDamperParams
from src.solvers import simulate
from src.fast import simulate_float

p = SpringDamperParams(1.0, 0.4, 4.0, 1.0, 0.0)

def best(f, *a):
    times = []
    for _ in range(7):
        t0 = time.perf_counter()
        f(*a)
        times.append(time.perf_counter() - t0)
    return min(times)

for m in ("euler", "rk4"):
    a = best(simulate, p, 0.001, 5.0, m)
    b = best(simulate_float, p, 0.001, 5.0, m)
    print(m, f"NumPy {a*1e3:.2f} ms, float {b*1e3:.2f} ms, speedup {a/b:.1f}x")