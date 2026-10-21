# Spring-Mass-Damper Simulation



MMA3001 Numerical Methods and Machine Learning, individual project.

Author: Samuel Davis 36136603



Numerical simulation of the free damped oscillator



&#x20;   m x'' + c x' + k x = 0,   x(0) = x0,  x'(0) = v0



Explicit Euler (baseline) is compared with classical fourth-order Runge-Kutta (RK4). Both are validated against the exact analytical solution and an independent SciPy reference, and analysed for convergence, stability and performance.



### Engineering context



Spring-mass-damper systems model many mechanical and aerospace components, such as vehicle suspensions, vibration isolators and landing-gear struts. Engineers simulate them to predict how quickly vibrations decay and whether a design overshoots or rings. A time-stepping method that is too inaccurate gives wrong predictions, and one that is unstable can produce results that grow without bound even though the real system loses energy. This project quantifies those trade-offs so a method and time step can be chosen using evidence.



**Intended use:** a validated reference implementation for choosing an integration method and time step for linear single-degree-of-freedom vibration problems.



**Scope and limitations:** linear spring, viscous (linear) damping, no external forcing, one degree of freedom, fixed time step. It does not model nonlinear springs, Coulomb friction, forced vibration or multi-body systems.



### Inputs and outputs



|**Input**|**Symbol**|**Unit**|**Valid domain**|
|-|-|-|-|
|Mass|'m'|kg|finite > 0|
|Damping coefficient|'c'|Ns/m|finite, >= 0|
|Spring stiffness|'k'|Nm|finite, > 0|
|Initial velocity|'v0'|m/s|finite|
|Initial displacement|'x0'|m|finite|
|Time step|'dt'|s|finite > 0|
|End time|'T'|s|finite, >= dt|
|Method|'method'|n/a|"euler" or "rk4"|



|**Output**|**Unit**|**Meaning**|
|-|-|-|
|'t'|s|time grid 0, dt, 2dt, ...|
|'x'|m|displacement at each time|
|'v'|m/s|velocity at each time|
|error metrics|m, m/s|max absolute error and RMSE against the exact solution|
|observed error|n/a|convergence rate from halving dt|
|Euler stability limit|s|largest dt for which explicit Euler does no amplify the solution|
|runtime|s|wall-clock time of a simulation|



**Invalid input handling:** non-numeric, non-finite or out-of-domain values (for example `m <= 0`, `c < 0`, `dt <= 0`) and unknown method names raise `ValueError` with a descriptive message. This behaviour is covered by the tests.



**Named test cases:** `src/cases.py` provides two cases with realistic parameters chosen inside published ranges (a quarter-car suspension and a lightly damped steel specimen), with sources cited in the file. These are representative values, not measurements, so they test the solvers at realistic scales but do not validate the physical model against real data.



### Setup



Requires Python 3.10 or newer.



```bash

git clone: https://github.com/Sammmydav/spring-damper-system

cd spring-damper

python -m venv .venv

source .venv/bin/activate        # Windows: .venv\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\Scripts\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\activate

pip install -r requirements.txt

```



### Usage



```python

from src.models import SpringDamperParams

from src.solvers import simulate

from src.analysis import error\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\_metrics



p = SpringDamperParams(m=1.0, c=0.4, k=4.0, x0=1.0, v0=0.0)

t, x, v = simulate(p, dt=0.01, T=5.0, method="rk4")

print(error\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\_metrics(p, t, x, v))

```



A faster pure-float implementation with identical results is also provided:



```python

from src.fast import simulate\\\_float

t, x, v = simulate\\\_float(p, dt=0.01, T=5.0, method="rk4")

```



Run the full study (convergence, stability, timing, figures saved to `figures/`):



```bash

python -m scripts.run\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\_study

```



The notebook `notebooks/analysis.ipynb` reproduces all plots used in the report.



### Tests



```bash

python -m pytest --html=docs/test\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\_report.html --self-contained-html

```



The suite checks input validation, the analytical solution (initial conditions, the ODE residual, energy behaviour), agreement of both solvers with the exact and SciPy solutions, the observed convergence orders (about 1 for Euler and 4 for RK4), and the Euler stability limit. It also checks the realistic cases and that the pure-float solvers match the NumPy ones to round-off.



### HTML documentation



```bash

python -m pdoc src -o docs/api

```



Open `docs/api/index.html` in a browser.



### Repository layout



```

src/         models, analytical solution, solvers, fast solvers, analysis, named cases

tests/       pytest suite

scripts/     run\\\_study.py (convergence, stability, timing, figures)

notebooks/   analysis.ipynb (experiments and plots)

docs/        generated API docs, test report, report outline
```



### Key Results



Measured on \[12th Gen intel(R) Core(TM) i5-1235U, Python 3.10.11]; timings vary between machines.



* Observed convergence orders: Euler \~ 1.02, RK4 \~ 4.00 (finest step).
* Euler stability limit for the test case: dt <= 0.1 s (2\*zeta/omega\_n).
* Per step (NumPy solvers, dt = 0.001 s): Euler 3.6us, RK4 17.5 us.
* Time to reach max error of 1e-6 m (NumPy solvers): RK4 2.47 ms; Euler about s (extrapolated, as Euler never reaches 1e-6 m in the tested range)
* Pure-float solvers are 13.0x (Euler) and 24.1x (RK4) faster than the NumPy versions.
* 64 automated tests pass.



### AI use



\*(Rewrite this in your own words as you work. Keep it specific.)\*



\- **Tools used:** Claude AI

\- **Used for:** Generating basic ideas that could be used for project. Scaffolding code and checking/fixing problems endured throughout coding process. Produced outlines/formatting/structure for documents such as README.md and the report. Performed checks over project to ensure guideline was being followed, and project worked correctly as designed.

\- **How output was checked: F**or example, rerunning tests, hand-checking formulas

\- **Errors or unhelpful suggestions:** Generated code trying to fix an issue that couldn't correctly or performed incorrect task that was wasn't required or wrong. Suggested ideas as part of the project that fell outside of guidelines or we physically impossible to do. Incorrectly predicted/calculated results and timings, so I had to create a file to fix this.

\- **Decisions that were mine:** The topic that was chosen for the project, improvements that should be made and implemented for the project, contents of documents such as README.md and report.



The full AI reflection is in the project report.



### Licence



MIT. See `LICENSE`.

