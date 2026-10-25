"""Convergence and stability studies.

Each study answers one question and prints the number that answers it, so the
claims in the README are reproducible rather than asserted:

* :func:`temporal_order` -- how fast does error fall as dt shrinks?
* :func:`spatial_order`  -- how fast does error fall as dx shrinks?
* :func:`stability_sweep` -- where exactly does the explicit scheme break?
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from heat.analytical import semi_discrete_sine_mode, sine_mode
from heat.solvers import EXPLICIT_STABILITY_LIMIT, Grid, fourier_number, solve

SCHEMES = ("explicit", "implicit", "crank-nicolson")


@dataclass(frozen=True)
class Refinement:
    """One row of a refinement table."""

    parameter: int
    r: float
    error: float
    order: float | None

    def __str__(self) -> str:
        order = "    -" if self.order is None else f"{self.order:5.2f}"
        return f"  {self.parameter:6d}   r={self.r:8.4f}   err={self.error:.4e}   order={order}"


def _orders(errors: list[float]) -> list[float | None]:
    """Observed order between consecutive refinements, each a factor of 2."""
    out: list[float | None] = [None]
    for prev, cur in zip(errors, errors[1:]):
        out.append(float("nan") if cur == 0 else np.log2(prev / cur))
    return out


def temporal_order(
    scheme: str, *, alpha: float = 1.0, t_end: float = 0.02, nx: int = 40, base_steps: int = 100, levels: int = 4
) -> list[Refinement]:
    """Refine dt at fixed dx, measuring against the semi-discrete solution.

    The reference matters. Against the true solution this table flattens out
    once the time error drops below the fixed spatial error, and the observed
    order collapses towards 0 for a scheme that is working perfectly. Measuring
    against the spatially-discrete exact solution isolates the time integrator.
    """
    grid = Grid(nx=nx)
    dx, x = grid.dx, grid.x_interior
    u0 = sine_mode(x, 0.0, alpha)

    rows, errors, params, rs = [], [], [], []
    for level in range(levels):
        steps = base_steps * 2**level
        dt = t_end / steps
        u = solve(scheme=scheme, initial=u0, alpha=alpha, dt=dt, dx=dx, steps=steps)
        exact = semi_discrete_sine_mode(x, t_end, alpha, dx)
        errors.append(float(np.max(np.abs(u - exact))))
        params.append(steps)
        rs.append(fourier_number(alpha, dt, dx))

    for p, r, e, o in zip(params, rs, errors, _orders(errors)):
        rows.append(Refinement(p, r, e, o))
    return rows


def spatial_order(
    scheme: str, *, alpha: float = 1.0, t_end: float = 0.02, base_nx: int = 10, levels: int = 4, r: float = 0.25
) -> list[Refinement]:
    """Refine dx with r held fixed, measuring against the true solution.

    Holding r fixed ties dt to dx**2. For Crank-Nicolson that makes the time
    error O(dx**4), far below the spatial O(dx**2) being measured, so the table
    shows the spatial order cleanly. It also keeps the explicit scheme inside
    its stability limit at every level, which a fixed dt would not.
    """
    rows, errors, params, rs = [], [], [], []
    for level in range(levels):
        nx = base_nx * 2**level
        grid = Grid(nx=nx)
        dx, x = grid.dx, grid.x_interior
        dt = r * dx**2 / alpha
        steps = max(1, int(round(t_end / dt)))
        dt = t_end / steps  # land exactly on t_end
        u = solve(scheme=scheme, initial=sine_mode(x, 0.0, alpha), alpha=alpha, dt=dt, dx=dx, steps=steps)
        errors.append(float(np.max(np.abs(u - sine_mode(x, t_end, alpha)))))
        params.append(nx)
        rs.append(fourier_number(alpha, dt, dx))

    for p, rr, e, o in zip(params, rs, errors, _orders(errors)):
        rows.append(Refinement(p, rr, e, o))
    return rows


def stability_sweep(
    *,
    alpha: float = 1.0,
    t_end: float = 0.02,
    nx: int = 40,
    r_values: tuple[float, ...] = (0.25, 0.49, 0.50, 0.51, 0.55, 0.75, 1.00),
    perturbation: float = 1e-10,
) -> list[tuple[float, float, float, float, int]]:
    """Run every scheme either side of r = 1/2 and report the peak magnitude.

    Returns ``(r, explicit_peak, implicit_peak)``. The explicit stability check
    is switched off here on purpose: the point is to watch it fail, not to be
    stopped before it does.

    The initial condition carries a tiny alternating perturbation, and without
    it this sweep shows nothing. Von Neumann analysis gives
    ``g(k) = 1 - 4 r sin^2(k dx / 2)``, so the amplification that exceeds 1 for
    r > 1/2 belongs to the *highest* wavenumber on the grid. A pure low sine
    mode has almost no energy there: ``sin^2(pi dx / 2)`` is ~1e-3 on this
    grid, so a smooth start stays bounded well past the limit and the scheme
    looks stable when it is not. The alternating (+1, -1, +1, ...) pattern is
    exactly that worst-case mode, seeded at 1e-10 so it changes nothing when
    the scheme is stable and grows as |1 - 4r|^n when it is not.
    """
    grid = Grid(nx=nx)
    dx, x = grid.dx, grid.x_interior
    u0 = sine_mode(x, 0.0, alpha) + perturbation * (-1.0) ** np.arange(x.size)

    out = []
    for r in r_values:
        dt = r * dx**2 / alpha
        steps = max(1, int(round(t_end / dt)))
        kwargs = dict(initial=u0, alpha=alpha, dt=dt, dx=dx, steps=steps)
        with np.errstate(over="ignore", invalid="ignore"):
            ex = solve(scheme="explicit", check_stability=False, **kwargs)
        im = solve(scheme="implicit", **kwargs)
        # Predicted growth of the worst-case mode over this run, |1 - 4r|^steps.
        # Reported because it explains the rows that look stable but are not:
        # at r = 0.51 the seed grows by only ~1.09 over these steps, so a short
        # horizon hides an instability that is already present.
        growth = abs(1.0 - 4.0 * r) ** steps
        out.append((r, float(np.max(np.abs(ex))), float(np.max(np.abs(im))), float(growth), steps))
    return out


def main() -> None:
    print("1-D heat equation: u_t = alpha u_xx, u(x,0) = sin(pi x), u(0,t) = u(1,t) = 0\n")

    print("TEMPORAL CONVERGENCE  (dx fixed, dt halved; vs semi-discrete exact)")
    for scheme in SCHEMES:
        print(f"\n  {scheme}")
        for row in temporal_order(scheme):
            print(row)

    print("\n\nSPATIAL CONVERGENCE  (r fixed at 0.25, nx doubled; vs analytical)")
    for scheme in SCHEMES:
        print(f"\n  {scheme}")
        for row in spatial_order(scheme):
            print(row)

    print(f"\n\nSTABILITY  (explicit limit is r = {EXPLICIT_STABILITY_LIMIT})")
    print("  initial condition seeded with a 1e-10 alternating mode, the worst case for |g| = |1 - 4r|\n")
    print("       r    |g|=|1-4r|   |g|^steps     explicit peak |u|   implicit peak |u|")
    for r, ex, im, growth, steps in stability_sweep():
        flag = "   <-- blown up" if ex > 1.0 else ""
        print(
            f"  {r:6.2f}    {abs(1 - 4 * r):8.2f}   {growth:10.2e}      {ex:14.4e}    {im:14.4e}{flag}"
        )
    print(
        "\n  Every row with |g| > 1 is unstable, including r = 0.51. It stays invisible until"
        "\n  |g|^steps outgrows the 1e-10 seed, which needs ~1e10: r = 0.51 reaches only ~1e1"
        "\n  over this run. Instability is a property of r, not of whether a short run shows it."
    )


if __name__ == "__main__":
    main()
