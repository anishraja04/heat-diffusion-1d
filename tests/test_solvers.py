"""Tests that check numerical behaviour, not just that the code runs.

Each test asserts a property the method is supposed to have: the observed
convergence order, the stability threshold, conservation of the steady state.
A test that only checked the solver returns an array of the right shape would
pass just as happily on a wrong discretisation.
"""

from __future__ import annotations

import numpy as np
import pytest

from heat.analytical import semi_discrete_sine_mode, sine_mode
from heat.solvers import EXPLICIT_STABILITY_LIMIT, Grid, fourier_number, solve
from heat.study import spatial_order, temporal_order

SCHEMES = ("explicit", "implicit", "crank-nicolson")
ALPHA = 1.0
T_END = 0.02


def _run(scheme, nx=40, steps=200, t_end=T_END, **kw):
    grid = Grid(nx=nx)
    dx, x = grid.dx, grid.x_interior
    dt = t_end / steps
    u = solve(
        scheme=scheme, initial=sine_mode(x, 0.0, ALPHA), alpha=ALPHA, dt=dt, dx=dx, steps=steps, **kw
    )
    return x, dx, dt, u


@pytest.mark.parametrize("scheme", SCHEMES)
def test_matches_analytical_solution(scheme):
    """Every scheme lands on the closed-form answer to a sensible tolerance."""
    x, _, _, u = _run(scheme)
    exact = sine_mode(x, T_END, ALPHA)
    assert np.max(np.abs(u - exact)) < 5e-4


@pytest.mark.parametrize(
    "scheme,expected",
    [("explicit", 1.0), ("implicit", 1.0), ("crank-nicolson", 2.0)],
)
def test_observed_temporal_order(scheme, expected):
    """Order in time, measured against the semi-discrete exact solution.

    Crank-Nicolson is second order because it centres the spatial term in
    time; the other two evaluate it at one end of the step and are first order.
    """
    orders = [row.order for row in temporal_order(scheme) if row.order is not None]
    assert np.allclose(orders, expected, atol=0.1), f"{scheme}: {orders}"


@pytest.mark.parametrize("scheme", SCHEMES)
def test_observed_spatial_order(scheme):
    """Second-order central differencing gives second-order spatial accuracy."""
    orders = [row.order for row in spatial_order(scheme) if row.order is not None]
    assert orders[-1] == pytest.approx(2.0, abs=0.15), f"{scheme}: {orders}"


def test_explicit_rejects_unstable_step():
    """The guard fires on the wrong side of r = 1/2, and not on the right side."""
    grid = Grid(nx=40)
    dx, x = grid.dx, grid.x_interior
    u0 = sine_mode(x, 0.0, ALPHA)

    safe_dt = 0.4 * dx**2 / ALPHA
    solve(scheme="explicit", initial=u0, alpha=ALPHA, dt=safe_dt, dx=dx, steps=5)

    unsafe_dt = 0.6 * dx**2 / ALPHA
    with pytest.raises(ValueError, match="unstable"):
        solve(scheme="explicit", initial=u0, alpha=ALPHA, dt=unsafe_dt, dx=dx, steps=5)

    assert fourier_number(ALPHA, safe_dt, dx) < EXPLICIT_STABILITY_LIMIT
    assert fourier_number(ALPHA, unsafe_dt, dx) > EXPLICIT_STABILITY_LIMIT


def test_explicit_actually_diverges_past_the_limit():
    """Past r = 1/2 the worst-case mode grows instead of decaying.

    Seeded with the alternating pattern, which is the mode whose amplification
    factor |1 - 4r| exceeds 1 first. A smooth initial condition would stay
    bounded here and make the scheme look stable when it is not.
    """
    grid = Grid(nx=40)
    dx, x = grid.dx, grid.x_interior
    u0 = sine_mode(x, 0.0, ALPHA) + 1e-10 * (-1.0) ** np.arange(x.size)

    r = 1.0
    dt = r * dx**2 / ALPHA
    steps = int(round(T_END / dt))
    with np.errstate(over="ignore", invalid="ignore"):
        unstable = solve(
            scheme="explicit", initial=u0, alpha=ALPHA, dt=dt, dx=dx, steps=steps, check_stability=False
        )
    stable = solve(scheme="implicit", initial=u0, alpha=ALPHA, dt=dt, dx=dx, steps=steps)

    assert np.max(np.abs(unstable)) > 1e3
    assert np.max(np.abs(stable)) < 1.0


@pytest.mark.parametrize("scheme", ["implicit", "crank-nicolson"])
def test_implicit_schemes_are_unconditionally_stable(scheme):
    """A step far past the explicit limit stays bounded and still decays."""
    grid = Grid(nx=40)
    dx, x = grid.dx, grid.x_interior
    dt = 50.0 * dx**2 / ALPHA  # r = 50, a hundred times the explicit limit
    u = solve(scheme=scheme, initial=sine_mode(x, 0.0, ALPHA), alpha=ALPHA, dt=dt, dx=dx, steps=40)
    assert np.all(np.isfinite(u))
    assert np.max(np.abs(u)) < 1.0


@pytest.mark.parametrize("scheme", SCHEMES)
def test_zero_initial_condition_stays_zero(scheme):
    """With zero data and zero boundaries nothing should appear from nowhere."""
    grid = Grid(nx=30)
    dx = grid.dx
    u = solve(
        scheme=scheme, initial=np.zeros(30), alpha=ALPHA, dt=0.2 * dx**2, dx=dx, steps=25
    )
    assert np.allclose(u, 0.0, atol=1e-14)


@pytest.mark.parametrize("scheme", SCHEMES)
def test_constant_dirichlet_reaches_its_steady_state(scheme):
    """Held at 1.0 on both ends, the bar warms to a uniform 1.0.

    This exercises the non-zero boundary terms, which the homogeneous tests
    above never touch, and the steady answer is known exactly without any
    series truncation.

    t_end = 2.0 is chosen, not guessed. The transient dies at the rate of the
    slowest mode, exp(-alpha pi^2 t): at t = 0.5 that is still 7e-3, which is
    the solution being 0.991 rather than 1.0 and has nothing to do with the
    discretisation. At t = 2.0 it is ~3e-9, below the tolerance asserted here.
    """
    grid = Grid(nx=25)
    dx, x = grid.dx, grid.x_interior
    dt = 0.25 * dx**2 / ALPHA
    steps = int(round(2.0 / dt))
    u = solve(
        scheme=scheme,
        initial=np.zeros(x.size),
        alpha=ALPHA,
        dt=dt,
        dx=dx,
        steps=steps,
        left=1.0,
        right=1.0,
    )
    assert np.allclose(u, 1.0, atol=1e-6), f"{scheme}: max deviation {np.max(np.abs(u - 1.0)):.2e}"


def test_grid_spacing_and_nodes_are_consistent():
    grid = Grid(nx=9, length=1.0)
    assert grid.dx == pytest.approx(0.1)
    assert grid.x_interior.size == 9
    assert grid.x_full.size == 11
    assert grid.x_full[0] == 0.0 and grid.x_full[-1] == pytest.approx(1.0)


def test_unknown_scheme_is_rejected():
    with pytest.raises(ValueError, match="unknown scheme"):
        solve(scheme="runge-kutta", initial=np.zeros(5), alpha=1.0, dt=1e-4, dx=0.1, steps=1)


def test_semi_discrete_reference_approaches_the_true_solution():
    """The discrete eigenvalue tends to the continuous one as dx shrinks.

    Guards the reference itself: if this drifted, every temporal order measured
    against it would be wrong in a way the other tests could not see.
    """
    errors = []
    for nx in (20, 40, 80):
        grid = Grid(nx=nx)
        dx, x = grid.dx, grid.x_interior
        errors.append(
            np.max(np.abs(semi_discrete_sine_mode(x, T_END, ALPHA, dx) - sine_mode(x, T_END, ALPHA)))
        )
    order = np.log2(errors[0] / errors[1])
    assert order == pytest.approx(2.0, abs=0.1), f"errors {errors}"
