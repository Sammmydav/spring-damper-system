# Report outline (5-10 pages; references and appendices don't count)

Suggested page budget in brackets. Write it in your own words: you are questioned on it in Week 14 with no AI.

## 1\. Engineering problem \[0.75 p]

* Context: where a spring-mass-damper model appears (suspension, vibration isolator, landing-gear strut). Pick ONE and say why it matters.
* Problem: simulate the free response accurately and stably; choose a method with evidence.
* Intended use of the code (who runs it, for what).
* Scope limits: linear spring, viscous damping, no forcing, 1 DOF. State what it cannot model.

## 2\. Inputs, outputs and data \[0.75 p]

* Table: every input with symbol, unit, valid domain (see README).
* Outputs: t, x, v, error metrics, observed order, stability limit, runtime, with meaning and use.
* Invalid input handling: which cases raise `ValueError` (point to the tests).

## 3\. Computational approach \[1.5 p]

* Governing equation, then first-order system y' = f(y).
* Explicit Euler (baseline) and RK4: update formulas, order of accuracy, RHS evaluations per step.
* Why these two: simple baseline vs standard higher-order method.
* Assumptions, parameters (dt, T, test case values), implementation notes (structure of `src/`).
* Known limitations.

## 4\. Validation \[2 p]

* What was validated: solver correctness (not the physical model itself).
* Benchmarks: exact solution (all four regimes) + independent `solve\\\_ivp` reference.
* Metrics and acceptance criteria: max abs error, RMSE; state your thresholds BEFORE quoting results (e.g. RK4 max error < 1e-6 at dt = 0.01).
* Convergence study: log-log plot, observed orders \~1 and \~4.
* Physical checks: undamped energy conservation, monotone energy decay when damped.
* Verification via tests: pytest summary (counts, what each group covers).
* What validation cannot establish: real-world parameter accuracy, nonlinear effects, forcing.
* Did validation influence method choice? Say yes/no honestly.

## 5\. Comparison and optimisation \[2 p]

* Baseline (Euler) vs selected (RK4): accuracy, runtime, memory, stability, complexity.
* Error-vs-runtime plot; time to reach a target error (the notebook computes this).
* Stability: analytical limit dt <= 2\*zeta/omega\_n (underdamped), numerical test either side, undamped energy growth.
* Profiling: cProfile output summary, FLOP estimate per step, arithmetic intensity, memory growth with 1/dt; conclude where time actually goes.
* Optional improvement with before/after numbers (e.g. pure-float step instead of NumPy arrays). Only include it if you measure it.
* Final recommendation and trade-offs.

## 6\. Limitations and lessons learned \[0.5 p]

* Limitations of the model, benchmarks and tests.
* Bugs or surprises you hit and how you found them.
* What you would do next (e.g. forcing, nonlinear spring, implicit methods, adaptive step).

## 7\. AI-use reflection \[0.75 p]

Answer every point from the brief:

* Tools used and what for.
* Approximate AI contribution (e.g. scaffold of code/tests/docs vs your own analysis).
* Why you used AI for those tasks.
* How you checked AI output (reran tests, hand-derived formulas, compared to solve\_ivp, read every function).
* Decisions that stayed yours (topic, parameters, acceptance criteria, conclusions).
* Errors or unhelpful suggestions from the AI (keep a running list as you go).
* Effect on your understanding and workflow.

## 8\. References

* Textbook/lecture notes for the exact solution and RK4; SciPy docs for `solve\\\_ivp`; any paper or source you cite.

## Appendix (free of page limit)

* Repo link and structure, test report, extra plots.

## Before submitting

* \[ ] Repo: README, LICENSE, pdoc HTML in `docs/api`, pytest report in `docs/`, functional main branch, commit history.
* \[ ] Zipped CLONE of the repo + report uploaded to Moodle.
* \[ ] Practice the 5-minute talk and the sample questions from the brief (stability, convergence criteria, error sources, how you verified correctness).

