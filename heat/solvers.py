"""Finite-difference time-stepping schemes for the 1-D heat equation.

All three schemes discretise

.. math:: u_t = \\alpha u_{xx}

on a uniform grid with second-order central differences in space. They differ
only in how the spatial term is evaluated in time, which is what sets their
stability limit and their order of accuracy in time:

=================  ======================  ==================  ===============
Scheme             Spatial term at         Stability           Order (t, x)
=================  ======================  ==================  ===============
Explicit (FTCS)    old time level          r <= 1/2            1, 2
Implicit (BTCS)    new time level          unconditional       1, 2
Crank-Nicolson     average of both         unconditional       2, 2
=================  ======================  ==================  ===============

where ``r = alpha * dt / dx**2`` is the mesh Fourier number.

The two implicit schemes solve a tridiagonal system each step. That system is
built once and factorised once, not rebuilt per step: the matrix depends only
on ``r``, which is constant for a fixed grid, so rebuilding it every step would
repeat identical work for no benefit.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import solve_banded


@dataclass(frozen=True)
class Grid:
    """A uniform spatial grid with Dirichlet ends.

    ``nx`` counts the interior points. The two boundary values are held
    separately rather than stored in the solution vector, so the linear systems
    below are exactly the interior problem and carry no identity rows.
    """

    nx: int
    length: float = 1.0

    @property
    def dx(self) -> float:
        return self.length / (self.nx + 1)

    @property
    def x_interior(self) -> np.ndarray:
        """Interior node positions, excluding both boundaries."""
        return np.linspace(0.0, self.length, self.nx + 2)[1:-1]

    @property
    def x_full(self) -> np.ndarray:
        """All node positions, boundaries included."""
        return np.linspace(0.0, self.length, self.nx + 2)


def fourier_number(alpha: float, dt: float, dx: float) -> float:
    """Mesh Fourier number ``r = alpha * dt / dx**2``.

    This single number decides whether the explicit scheme is stable, and it is
    why halving dx forces a four-fold cut in dt to hold r fixed.
    """
    return alpha * dt / dx**2


EXPLICIT_STABILITY_LIMIT = 0.5
"""Largest r for which the explicit scheme is stable.

Von Neumann analysis gives an amplification factor
``g(k) = 1 - 4 r sin^2(k dx / 2)``. The worst case is the highest representable
wavenumber, where ``sin^2`` reaches 1 and ``g = 1 - 4r``. Requiring ``|g| <= 1``
gives ``r <= 1/2``. Above it the shortest wavelength on the grid grows by
``|1 - 4r| > 1`` every step, which is why the blow-up is so abrupt rather than
a gradual loss of accuracy.
"""


def _tridiagonal_banded(nx: int, lower: float, diag: float, upper: float) -> np.ndarray:
    """Build the banded (``ab``) form scipy's ``solve_banded`` expects.

    Row 0 holds the superdiagonal, row 1 the diagonal, row 2 the subdiagonal,
    with the documented one-element offsets at the ends.
    """
    ab = np.zeros((3, nx))
    ab[0, 1:] = upper
    ab[1, :] = diag
    ab[2, :-1] = lower
    return ab


def step_explicit(u: np.ndarray, r: float, left: float = 0.0, right: float = 0.0) -> np.ndarray:
    """One FTCS step. Evaluates the spatial term entirely at the old level."""
    padded = np.concatenate(([left], u, [right]))
    return u + r * (padded[2:] - 2.0 * padded[1:-1] + padded[:-2])


def solve(
    *,
    scheme: str,
    initial: np.ndarray,
    alpha: float,
    dt: float,
    dx: float,
    steps: int,
    left: float = 0.0,
    right: float = 0.0,
    check_stability: bool = True,
) -> np.ndarray:
    """March ``initial`` forward ``steps`` steps and return the interior values.

    ``scheme`` is one of ``"explicit"``, ``"implicit"`` or ``"crank-nicolson"``.

    ``check_stability`` guards only the explicit scheme, and only against its
    known limit. It is a keyword so the stability study can switch it off
    deliberately and watch the blow-up happen; nothing else should need to.
    """
    r = fourier_number(alpha, dt, dx)
    u = np.asarray(initial, dtype=float).copy()
    nx = u.size

    if scheme == "explicit":
        if check_stability and r > EXPLICIT_STABILITY_LIMIT:
            raise ValueError(
                f"explicit scheme is unstable at r={r:.4f} (limit {EXPLICIT_STABILITY_LIMIT}). "
                f"Reduce dt below {EXPLICIT_STABILITY_LIMIT * dx**2 / alpha:.3e}, "
                f"or use scheme='implicit' / 'crank-nicolson', which carry no limit."
            )
        for _ in range(steps):
            u = step_explicit(u, r, left, right)
        return u

    if scheme == "implicit":
        # (I + r A) u^{n+1} = u^n, with A the negative second-difference operator.
        ab = _tridiagonal_banded(nx, lower=-r, diag=1.0 + 2.0 * r, upper=-r)
        boundary = np.zeros(nx)
        boundary[0] = r * left
        boundary[-1] = r * right
        for _ in range(steps):
            u = solve_banded((1, 1), ab, u + boundary)
        return u

    if scheme == "crank-nicolson":
        # Average the spatial term across the two time levels:
        #   (I - (r/2) L) u^{n+1} = (I + (r/2) L) u^n
        half = 0.5 * r
        ab = _tridiagonal_banded(nx, lower=-half, diag=1.0 + r, upper=-half)
        # The old level's boundary values arrive through the padded stencil
        # below; this carries the NEW level's, which the matrix moved to the
        # right-hand side. Both are weighted r/2, hence one term each.
        boundary = np.zeros(nx)
        boundary[0] = half * left
        boundary[-1] = half * right
        for _ in range(steps):
            padded = np.concatenate(([left], u, [right]))
            explicit_part = u + half * (padded[2:] - 2.0 * padded[1:-1] + padded[:-2])
            u = solve_banded((1, 1), ab, explicit_part + boundary)
        return u

    raise ValueError(f"unknown scheme {scheme!r}; expected explicit, implicit or crank-nicolson")
