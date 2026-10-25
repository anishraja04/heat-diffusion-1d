# 1-D heat diffusion: explicit vs implicit time stepping

Finite-difference solvers for

$$u_t = \alpha\, u_{xx}, \qquad 0 < x < L$$

with Dirichlet boundaries, written to compare three time-stepping schemes and to
check each one against a closed-form solution rather than against the others.

The question the repository answers: **when is the cheap scheme the wrong
choice?** Explicit stepping costs one array update per step and no linear solve,
which looks like the obvious option until the stability limit forces a time step
so small that it is slower than the implicit scheme it was meant to beat.

| Scheme | Spatial term evaluated at | Stability | Order (time, space) | Cost per step |
| --- | --- | --- | --- | --- |
| Explicit (FTCS) | old time level | `r <= 1/2` | 1, 2 | one array update |
| Implicit (BTCS) | new time level | unconditional | 1, 2 | one tridiagonal solve |
| Crank-Nicolson | average of both | unconditional | **2**, 2 | one tridiagonal solve |

where `r = alpha*dt/dx^2` is the mesh Fourier number.

## Running it

```bash
pip install -e ".[dev]"
python -m heat.study     # the full convergence and stability report
pytest -q                # 22 tests
```

## Results

All numbers below are printed by `python -m heat.study`. Nothing is quoted from
a textbook.

### Convergence in time

Measured against the **semi-discrete** exact solution, which is exact in space
and so isolates the time integrator. Against the true solution this table
flattens out once the time error falls below the fixed spatial error, and a
perfectly correct scheme then reports an order near 0, which is a saturated
measurement rather than a broken method.

```
  explicit                          crank-nicolson
     100   err=1.5986e-04              100   err=5.2501e-08
     200   err=7.9880e-05  order=1.00  200   err=1.3125e-08  order=2.00
     400   err=3.9928e-05  order=1.00  400   err=3.2813e-09  order=2.00
     800   err=1.9961e-05  order=1.00  800   err=8.2031e-10  order=2.00
```

Explicit and implicit are first order; Crank-Nicolson is second, because it
centres the spatial term in time. At 800 steps Crank-Nicolson is already about
**24,000x more accurate** than either first-order scheme on the same grid.

### Convergence in space

`r` held at 0.25 while `nx` doubles, so `dt` scales with `dx^2` and the time
error stays below the spatial error being measured. All three converge at second
order, which is the central difference in space they share.

```
  explicit        10 -> 80 nodes, order 1.68 -> 1.95 -> 1.98
  implicit        10 -> 80 nodes, order 1.80 -> 1.93 -> 1.97
  crank-nicolson  10 -> 80 nodes, order 1.85 -> 1.93 -> 1.96
```

### Stability

Von Neumann analysis gives an amplification factor `g(k) = 1 - 4r sin^2(k dx/2)`.
The worst case is the highest wavenumber the grid can hold, where `g = 1 - 4r`,
so `|g| <= 1` requires `r <= 1/2`.

The initial condition here carries a `1e-10` alternating perturbation, and
**without it this table shows nothing**. That mode is where the instability
lives, and a smooth sine start has almost no energy in it, so the explicit
scheme stays bounded well past its limit and looks stable when it is not.

```
       r    |g|=|1-4r|   |g|^steps     explicit peak |u|   implicit peak |u|
    0.25        0.00     0.00e+00          8.2080e-01        8.2104e-01
    0.49        0.96     5.98e-02          8.1920e-01        8.1966e-01
    0.50        1.00     1.00e+00          8.2069e-01        8.2116e-01
    0.51        1.04     1.33e+01          8.1991e-01        8.2039e-01
    0.55        1.20     6.76e+04          8.2043e-01        8.2094e-01
    0.75        2.00     3.52e+13          3.5167e+03        8.2008e-01   <-- blown up
    1.00        3.00     1.67e+16          1.6677e+06        8.1900e-01   <-- blown up
```

Every row with `|g| > 1` is unstable, `r = 0.51` included. It stays invisible
until `|g|^steps` outgrows the `1e-10` seed, which needs about `1e10`, and
`r = 0.51` reaches only `1e1` over this run. **Instability is a property of `r`,
not of whether a short run happens to reveal it** — which is exactly how it
reaches production and fails later on a finer mesh.

### What this costs in practice

Halving `dx` forces a four-fold cut in `dt` to hold `r` fixed, so refining an
explicit solve by 2x in space costs 8x the work. The implicit schemes carry no
such constraint: `test_implicit_schemes_are_unconditionally_stable` runs them at
`r = 50`, a hundred times the explicit limit, and both stay bounded and still
decay.

## Layout

```
heat/analytical.py   closed-form solutions: single sine mode, step IC,
                     and the semi-discrete reference used for time-order studies
heat/solvers.py      the three schemes; tridiagonal system built and factorised
                     once per grid rather than per step
heat/study.py        convergence and stability studies that print the tables above
tests/               22 tests asserting convergence orders, the stability
                     threshold, and steady-state behaviour
```

## Notes on the implementation

**Boundary values are held outside the solution vector.** The linear systems are
exactly the interior problem, with no identity rows for known values.

**The Crank-Nicolson right-hand side carries the boundary twice at half weight,
once per time level.** Getting this wrong is quiet: the scheme still converges,
just at first order instead of second. The observed order of exactly 2.00 is the
check that it is right.

**The explicit stability guard is a keyword argument, not a hard rule.** The
stability study switches it off deliberately, because the point there is to watch
the method fail.

**`test_constant_dirichlet_reaches_its_steady_state` runs to `t = 2.0`, not
`t = 0.5`.** At 0.5 the slowest mode still contributes `exp(-alpha pi^2 t)`, about
`7e-3`, so the answer sits at 0.991 and looks like a boundary bug. It is the
physics, not the discretisation.

## Licence

MIT.
