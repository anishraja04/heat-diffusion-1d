"""Closed-form solutions used as the reference for every numerical scheme here.

The whole point of this repository is that the numerical answers are checked
against something known rather than against each other, so these functions are
the only place a "correct" answer is defined.
"""

from __future__ import annotations

import numpy as np


def sine_mode(x: np.ndarray, t: float, alpha: float, mode: int = 1, length: float = 1.0) -> np.ndarray:
    r"""Exact solution of the 1-D heat equation for a single sine mode.

    Solves

    .. math:: u_t = \alpha u_{xx},\quad 0 < x < L

    with homogeneous Dirichlet ends :math:`u(0,t) = u(L,t) = 0` and the initial
    condition :math:`u(x,0) = \sin(n \pi x / L)`.

    Separation of variables gives a single decaying mode:

    .. math:: u(x,t) = \sin\!\left(\frac{n \pi x}{L}\right)
              \exp\!\left(-\alpha \left(\frac{n \pi}{L}\right)^{2} t\right)

    A single mode is deliberate. It stays smooth for all t > 0, so the error a
    scheme shows is its own truncation error and not the Gibbs ringing that a
    discontinuous initial condition would add on top.
    """
    k = mode * np.pi / length
    return np.sin(k * x) * np.exp(-alpha * k**2 * t)


def step_initial_condition(
    x: np.ndarray, t: float, alpha: float, length: float = 1.0, terms: int = 200
) -> np.ndarray:
    r"""Exact solution for a unit step initial condition, as a Fourier series.

    Initial condition :math:`u(x,0) = 1` on the interior with
    :math:`u(0,t) = u(L,t) = 0`, giving

    .. math:: u(x,t) = \sum_{n \text{ odd}} \frac{4}{n\pi}
              \sin\!\left(\frac{n \pi x}{L}\right)
              \exp\!\left(-\alpha \left(\frac{n\pi}{L}\right)^{2} t\right)

    Included to exercise the schemes on an initial condition that is
    discontinuous at the boundaries. The series is truncated at ``terms``, so
    near t = 0 this is itself only an approximation: use it for qualitative
    comparison, not for measuring convergence order.
    """
    total = np.zeros_like(x, dtype=float)
    for n in range(1, 2 * terms, 2):
        k = n * np.pi / length
        total += (4.0 / (n * np.pi)) * np.sin(k * x) * np.exp(-alpha * k**2 * t)
    return total


def semi_discrete_sine_mode(
    x: np.ndarray, t: float, alpha: float, dx: float, mode: int = 1, length: float = 1.0
) -> np.ndarray:
    r"""Exact-in-time solution of the *spatially discretised* problem.

    Central differencing is exact on a sine mode up to its eigenvalue: applying
    the second-difference operator to :math:`\sin(kx)` returns

    .. math:: -\frac{4}{\Delta x^{2}} \sin^{2}\!\left(\frac{k \Delta x}{2}\right) \sin(kx)

    so the method-of-lines system has the closed-form solution

    .. math:: u_j(t) = \sin(k x_j) \exp(-\alpha \lambda t), \quad
              \lambda = \frac{4}{\Delta x^{2}} \sin^{2}\!\left(\frac{k \Delta x}{2}\right)

    Measuring a scheme against this instead of against :func:`sine_mode`
    removes the spatial truncation error entirely, leaving only the error of
    the time integrator. Without it a temporal refinement study stalls at the
    spatial error floor and reports an order of 0 once dt is small enough,
    which looks like a broken scheme rather than a saturated measurement.
    """
    k = mode * np.pi / length
    lam = (4.0 / dx**2) * np.sin(k * dx / 2.0) ** 2
    return np.sin(k * x) * np.exp(-alpha * lam * t)
