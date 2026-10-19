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
|Initial displacement |'x0'|m|finite|
|Time step|'dt'|s|finite > 0|
|End time|'T' |s|finite, >= dt|
|Method |'method'|n/a|"euler" or "rk4"|



|**Output**|**Unit** |**Meaning**|
|-|-|-|
|'t'|s|time grid 0, dt, 2dt, ...|
|'x'|m|displacement at each time|
|'v'|m/s|velocity at each time|
|error metrics|m, m/s|max absolute error and RMSE against the exact solution|
|observed error |n/a |convergence rate from halving dt|
|Euler stability limit|s|largest dt for which explicit Euler does no amplify the solution|
|runtime |s |wall-clock time of a simulation|



**Invalid input handling:** non-numeric, non-finite or out-of-domain values (for example `m <= 0`, `c < 0`, `dt <= 0`) and unknown method names raise `ValueError` with a descriptive message. This behaviour is covered by the tests.



### Setup



Requires Python 3.10 or newer.



```bash

git clone <YOUR REPO URL>

cd spring-damper

python -m venv .venv

source .venv/bin/activate        # Windows: .venv\\\\Scripts\\\\activate

pip install -r requirements.txt

```



\## Usage



```python

from src.models import SpringDamperParams

from src.solvers import simulate

from src.analysis import error\\\_metrics



p = SpringDamperParams(m=1.0, c=0.4, k=4.0, x0=1.0, v0=0.0)

t, x, v = simulate(p, dt=0.01, T=5.0, method="rk4")

print(error\\\_metrics(p, t, x, v))

```



Run the full study (convergence, stability, timing, figures saved to `figures/`):



```bash

python -m scripts.run\\\_study

```



The notebook `notebooks/analysis.ipynb` reproduces all plots used in the report.



### Tests



```bash

python -m pytest --html=docs/test\\\_report.html --self-contained-html

```



The suite checks input validation, the analytical solution (initial conditions, the ODE residual, energy behaviour), agreement of both solvers with the exact and SciPy solutions, the observed convergence orders (about 1 for Euler and 4 for RK4), and the Euler stability limit.



### HTML documentation



```bash

python -m pdoc src -o docs/api

```



Open `docs/api/index.html` in a browser.



### Repository layout



```

src/         models, analytical solution, solvers, analysis

tests/       pytest suite

scripts/     run\\\_study.py (convergence, stability, timing, figures)

notebooks/   analysis.ipynb (experiments and plots)

docs/        generated API docs, test report, report outline

```



### Key results



\*(Fill in from your own runs.)\*



\- Observed convergence orders: Euler \~ \_\_\_, RK4 \~ \_\_\_.

\- Euler stability limit for the test case: dt <= \_\_\_ s.

\- Time to reach a max error of 1e-6 m: Euler \_\_\_ ms, RK4 \_\_\_ ms.



### AI use



\*(Rewrite this in your own words as you work. Keep it specific.)\*



\- \*\*Tools used:\*\* \_\_\_

\- \*\*Used for:\*\* \_\_\_ (for example, scaffolding code, tests and documentation)

\- \*\*How output was checked:\*\* \_\_\_ (for example, rerunning tests, hand-checking formulas, comparing with `solve\\\_ivp`)

\- \*\*Errors or unhelpful suggestions:\*\* \_\_\_

\- \*\*Decisions that were mine:\*\* \_\_\_



The full AI reflection is in the project report.



### Licence



MIT. See `LICENSE`.

