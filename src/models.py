"""Physical model and input validation for the free spring-mass-damper.

The governing equation is

    m x'' + c x' + k x = 0,   x(0) = x0,  x'(0) = v0

with mass ``m`` [kg], viscous damping coefficient ``c`` [N s/m] and spring
stiffness ``k`` [N/m].  Displacement ``x`` is in metres and velocity ``v``
in metres per second.

Scope / assumptions: linear spring, viscous (linear) damping, no external
forcing, one degree of freedom.
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class SpringDamperParams:
    """Physical parameters and initial conditions of the oscillator.

    Parameters
    ----------
    m : float
        Mass in kg. Must be finite and > 0.
    c : float
        Damping coefficient in N s/m. Must be finite and >= 0.
    k : float
        Spring stiffness in N/m. Must be finite and > 0.
    x0 : float, optional
        Initial displacement in m (default 1.0).
    v0 : float, optional
        Initial velocity in m/s (default 0.0).

    Raises
    ------
    ValueError
        If any value is non-numeric, non-finite or outside its valid domain.
    """

    m: float
    c: float
    k: float
    x0: float = 1.0
    v0: float = 0.0

    def __post_init__(self) -> None:
        for name in ("m", "c", "k", "x0", "v0"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{name} must be a real number, got {value!r}")
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite, got {value!r}")
        if self.m <= 0:
            raise ValueError(f"mass m must be > 0, got {self.m}")
        if self.k <= 0:
            raise ValueError(f"stiffness k must be > 0, got {self.k}")
        if self.c < 0:
            raise ValueError(f"damping c must be >= 0, got {self.c}")

    @property
    def omega_n(self) -> float:
        """Undamped natural frequency sqrt(k/m) in rad/s."""
        return math.sqrt(self.k / self.m)

    @property
    def zeta(self) -> float:
        """Damping ratio c / (2 sqrt(k m)) (dimensionless)."""
        return self.c / (2.0 * math.sqrt(self.k * self.m))

    @property
    def regime(self) -> str:
        """Damping regime: 'undamped', 'underdamped', 'critical' or 'overdamped'."""
        z = self.zeta
        if z == 0.0:
            return "undamped"
        if abs(z - 1.0) < 1e-9:
            return "critical"
        return "underdamped" if z < 1.0 else "overdamped"


def energy(params: SpringDamperParams, x, v):
    """Total mechanical energy 0.5 m v^2 + 0.5 k x^2 in joules.

    Parameters
    ----------
    params : SpringDamperParams
        Oscillator parameters.
    x, v : float or numpy.ndarray
        Displacement [m] and velocity [m/s].

    Returns
    -------
    float or numpy.ndarray
        Energy, same shape as the inputs.
    """
    return 0.5 * params.m * v**2 + 0.5 * params.k * x**2
