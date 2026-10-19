"""Named test cases with physically realistic parameters.

These cases complement the normalised cases used in the unit tests
(m = 1 kg, k = 4 N/m). Values are **representative, not measured on a
specific vehicle or specimen**: they are chosen inside ranges reported in
the published sources listed below. Check each source yourself before
citing it in the report.

Sources
-------
[1] "Modelling, Stability Analysis, and PID Control of a Quarter-Car
    Suspension System Using Frequency and State-Space Methods", CDSR 2026
    proceedings, paper 151 (Avestia).
    https://www.avestia.com/CDSR2026_Proceedings/files/paper/CDSR_151.pdf
    Reports, for compact vehicles, a quarter-car sprung mass of 250-340 kg,
    suspension stiffness 5,000-20,000 N/m and damping 500-2,000 N s/m.
[2] M. Acharya, "Study on Damping Property and Vibration Behaviour of SDOF
    System", Kathmandu University (nepjol.info, article 89608).
    https://www.nepjol.info/index.php/jsce/article/view/89608
    Reports damping ratios of 0.0042-0.0055 for steel cantilever
    specimens carrying a 1 kg lumped mass (free-vibration tests).
"""
from __future__ import annotations

from dataclasses import dataclass

from .models import SpringDamperParams


@dataclass(frozen=True)
class NamedCase:
    """A named parameter set with provenance and a sensible simulation length.

    Attributes
    ----------
    name : str
        Short identifier.
    params : SpringDamperParams
        Physical parameters and initial conditions (SI units).
    T : float
        Suggested simulation end time in s (several decay time constants).
    note : str
        What the case represents and where the values come from.
    """

    name: str
    params: SpringDamperParams
    T: float
    note: str


QUARTER_CAR = NamedCase(
    name="quarter_car",
    params=SpringDamperParams(m=300.0, c=1000.0, k=16000.0, x0=0.05, v0=0.0),
    T=5.0,
    note=("Passive suspension of a compact car, one quarter of the body. "
          "m, k, c each chosen inside the ranges in [1]; 50 mm initial "
          "displacement is an illustrative bump. zeta ~ 0.23."),
)

LIGHTLY_DAMPED_SPECIMEN = NamedCase(
    name="lightly_damped_specimen",
    params=SpringDamperParams(m=1.0, c=0.3162, k=1000.0, x0=0.002, v0=0.0),
    T=5.0,
    note=("1 kg lumped mass and zeta ~ 0.005 follow the specimens in [2]; "
          "k = 1000 N/m is illustrative (not taken from the source). "
          "Stress test: Euler is only stable for very small dt."),
)

CASES = {c.name: c for c in (QUARTER_CAR, LIGHTLY_DAMPED_SPECIMEN)}
