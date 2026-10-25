# Spring-Mass-Damper Simulation

MMA3001 Numerical Methods and Machine Learning, individual project.

Author: Samuel Davis 36136603

Numerical simulation of the free damped oscillator

    m x'' + c x' + k x = 0,   x(0) = x0,  x'(0) = v0

Explicit Euler (baseline) is compared with RK4 and with alternative integrators (backward Euler, symplectic (semi-implicit) Euler, SciPy RK45 and DOP853). All are validated against the exact analytical solution and analysed for convergence, stability, cost and performance.

## Engineering context

Spring-mass-damper systems model many mechanical and aerospace components, such as vehicle suspensions, vibration isolators and landing-gear struts. Engineers simulate them to predict how quickly vibrations decay and whether a design overshoots or rings. A time-stepping method that is too inaccurate gives wrong predictions, and one that is unstable can produce results that grow without bound even though the real system loses energy. This project quantifies those trade-offs so a method and time step can be chosen using evidence.

**Intended use:** a validated reference implementation for choosing an integration method and time step for linear single-degree-of-freedom vibration problems.

**Scope and limitations:** linear spring, viscous (linear) damping, no external forcing, one degree of freedom, fixed time step. It does not model nonlinear springs, Coulomb friction, forced vibration or multi-body systems.

## Inputs and outputs

| Input | Symbol | Unit | Valid domain |
|---|---|---|---|
| Mass | `m` | kg | finite, > 0 |
| Damping coefficient | `c` | N s/m | finite, >= 0 |
| Spring stiffness | `k` | N/m | finite, > 0 |
| Initial displacement | `x0` | m | finite |
| Initial velocity | `v0` | m/s | finite |
| Time step | `dt` | s | finite, > 0 |
| End time | `T` | s | finite, >= dt |
| Method | `method` | n/a | `"euler"`, `"rk4"`, `"backward_euler"`, `"symplectic_euler"` |

| Output | Unit | Meaning |
|---|---|---|
| `t` | s | time grid 0, dt, 2dt, ... |
| `x` | m | displacement at each time |
| `v` | m/s | velocity at each time |
| error metrics | m, m/s | max absolute error and RMSE against the exact solution |
| observed order | n/a | convergence rate from halving dt |
| Euler stability limit | s | largest dt for which explicit Euler does not amplify the solution |
| runtime | s | wall-clock time of a simulation |

**Invalid input handling:** non-numeric, non-finite or out-of-domain values (for example `m <= 0`, `c < 0`, `dt <= 0`) and unknown method names raise `ValueError` with a descriptive message. This behaviour is covered by the tests.

**Named test cases:** `src/cases.py` provides two cases with realistic parameters chosen inside published ranges (a quarter-car suspension and a lightly damped steel specimen), with sources cited in the file. These are representative values, not measurements, so they test the solvers at realistic scales but do not validate the physical model against real data.

## Setup

Requires Python 3.10 or newer.

```bash
git clone https://github.com/Sammmydav/spring-damper-system
cd spring-damper-system
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

```python
from src.models import SpringDamperParams
from src.solvers import simulate
from src.analysis import error_metrics

p = SpringDamperParams(m=1.0, c=0.4, k=4.0, x0=1.0, v0=0.0)
t, x, v = simulate(p, dt=0.01, T=5.0, method="rk4")
print(error_metrics(p, t, x, v))
```

A faster pure-float implementation with identical results is also provided:

```python
from src.fast import simulate_float
t, x, v = simulate_float(p, dt=0.01, T=5.0, method="rk4")
```

Alternative integrators (`src/alternatives.py`):

```python
from src.alternatives import simulate_alternative, simulate_scipy

t, x, v = simulate_alternative(p, dt=0.001, T=5.0, method="backward_euler")
t, x, v, nfev = simulate_scipy(p, dt=0.01, T=5.0, method="DOP853")
```

Run the full study (convergence, stability, timing, figures saved to `figures/`):

```bash
python -m scripts.run_study
python -m scripts.alternatives_study   # comparison of alternative integrators
```

The notebook `notebooks/analysis.ipynb` reproduces all plots used in the report.

## Tests

```bash
python -m pytest --html=docs/test_report.html --self-contained-html
```

The suite checks input validation, the analytical solution (initial conditions, the ODE residual, energy behaviour), agreement of both solvers with the exact and SciPy solutions, the observed convergence orders (about 1 for Euler and 4 for RK4), and the Euler stability limit. It also checks the realistic cases, that the pure-float solvers match the NumPy ones to round-off, and the alternative integrators (convergence, energy behaviour, stability).

## HTML documentation

```bash
python -m pdoc src -o docs/api
```

Open `docs/api/index.html` in a browser.

## Repository layout

```
src/         models, analytical solution, solvers, fast solvers, alternatives, analysis, named cases
tests/       pytest suite
scripts/     run_study.py, alternatives_study.py, timing_check.py
notebooks/   analysis.ipynb (experiments and plots)
docs/        generated API docs, test report, report outline
```

## Key results

Measured on a 12th Gen Intel(R) Core(TM) i5-1235U, Python 3.10.11; timings vary between machines.

- Observed convergence orders: Euler ~ 1.02, RK4 ~ 4.00 (finest step).
- Euler stability limit for the test case: dt <= 0.1 s (2*zeta/omega_n).
- Per step (NumPy solvers, dt = 0.001 s): Euler 3.6 us, RK4 17.5 us.
- Time to reach a max error of 1e-6 m (NumPy solvers): RK4 2.47 ms; Euler about 93 s (extrapolated, as Euler never reaches 1e-6 m in the tested range).
- Pure-float solvers are 13.0x (Euler) and 24.1x (RK4) faster than the NumPy versions.
- 84 automated tests pass.
- Main case at dt = 0.001 s: Euler and backward Euler both reach about 3.7e-3 m; symplectic Euler 9.9e-4 m. RK4 reaches 3.0e-6 m with 400 RHS evaluations (dt = 0.05 s). SciPy RK45 reaches 5.4e-7 m (362 evaluations) and DOP853 1.1e-9 m (404 evaluations) but is slower in wall-clock time.
- Undamped energy after 50 s: Euler +639%, backward Euler -86%, symplectic Euler +0.9%, RK4 -4.4e-9 (relative).

## AI use

- **Tools:** Claude (Anthropic, Claude Sonnet 5.5) through claude.ai for most of the work; Google Gemini for general queries, such as explaining concepts.
- **Used for:** explaining the brief, suggesting topics, and generating the project scaffold (solvers, exact solution, analysis functions, tests, notebook, scripts, README and report drafts, figures). It also explained Git and GitHub and proposed the pure-float optimisation.
- **Contribution:** the initial code and tests were AI-generated and I modified only a small part of the final code. The report and README were drafted by AI and then rewritten or completed by me, including the engineering context, key results and this AI declaration.
- **Checking:** I ran the full test suite after every change, reproduced the convergence orders in the notebook, compared results with SciPy's `solve_ivp`, and checked the cited parameter ranges against the source papers.
- **My decisions:** the topic, adding realistic parameter cases, requesting a dedicated alternative-solutions section, the report structure and referencing, and approving each addition.
- **AI errors found:** a Git commit grouping that would have failed, a time-to-accuracy calculation that understated RK4's advantage, timings measured on a different machine, and some incorrect source details.

The full AI reflection is in the project report.

## Licence

MIT. See `LICENSE`.
