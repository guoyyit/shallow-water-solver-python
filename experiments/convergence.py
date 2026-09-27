"""Advection-style convergence check on the linearised shallow-water system.

A smooth periodic surface perturbation advects/relaxes; here we measure
self-convergence of the scheme (fine vs coarse) to quantify the spatial order
of accuracy and document the expected discretisation error.

Run:  uv run python experiments/convergence.py
"""

from __future__ import annotations

import numpy as np

from shallow_water.models import (
    BoundaryCondition,
    BoundaryType,
)
from shallow_water.solver import ShallowWaterSolver
from tests.conftest import make_request


def run(n: int, T: float = 0.05) -> np.ndarray:
    x = np.linspace(0.0, 1.0, n, endpoint=False)
    y = np.linspace(0.0, 1.0, n, endpoint=False)
    X, Y = np.meshgrid(x, y)
    # Small smooth perturbation around a 1 m lake: keeps the solution in the
    # linear, near-lake-at-rest regime.
    eta = 1.0 + 0.01 * np.sin(2 * np.pi * X) * np.cos(2 * np.pi * Y)
    b = np.zeros((n, n))
    bcs = {s: BoundaryCondition(type=BoundaryType.PERIODIC)
           for s in ("west", "east", "south", "north")}
    dx = 1.0 / n
    req = make_request(eta, b, bcs, nx=n, ny=n, dx=dx, dy=dx,
                       target_time=T, cfl=0.5)
    res = ShallowWaterSolver.from_request(req).run()
    return np.array(res.eta.values).reshape(n, n)


def main() -> None:
    ns = [16, 32, 64, 128]
    fields = {n: run(n) for n in ns}
    ref = fields[ns[-1]]
    print(f"{'nx':>4} {'L_inf vs finest':>16} {'L2 vs finest':>16} {'order(L2)':>11}")
    prev = None
    for n in ns:
        if n == ns[-1]:
            print(f"{n:>4} {'(reference)':>16} {'':>16}")
            continue
        # Restrict finest solution to the coarse grid by subsampling.
        step = ns[-1] // n
        coarse_ref = ref[::step, ::step]
        linfty = float(np.max(np.abs(fields[n] - coarse_ref)))
        l2 = float(np.sqrt(np.mean((fields[n] - coarse_ref) ** 2)))
        order = "" if prev is None else f"{np.log(prev / l2) / np.log(2):.2f}"
        print(f"{n:>4} {linfty:>16.3e} {l2:>16.3e} {order:>11}")
        prev = l2


if __name__ == "__main__":
    main()
