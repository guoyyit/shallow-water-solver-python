"""Shared pytest fixtures and scenario builders."""

from __future__ import annotations

import numpy as np
import pytest

from shallow_water.models import (
    BoundaryCondition,
    BoundaryType,
    GridSpec,
    SolveRequest,
    SolverConfig,
)


@pytest.fixture
def walls() -> dict[str, BoundaryCondition]:
    return {s: BoundaryCondition(type=BoundaryType.WALL)
            for s in ("west", "east", "south", "north")}


def make_request(
    eta,
    b,
    boundaries,
    *,
    nx=None,
    ny=None,
    dx=0.1,
    dy=0.1,
    target_time=0.2,
    cfl=0.5,
    checkpoint_every=100,
    max_steps=1_000_000,
    noise_amplitude=0.0,
    seed=0,
    g=9.81,
) -> SolveRequest:
    eta = np.asarray(eta, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if eta.ndim == 1:
        assert nx is not None and ny is not None
    else:
        ny, nx = eta.shape
    return SolveRequest(
        grid=GridSpec(nx=nx, ny=ny, dx=dx, dy=dy),
        initial_level=eta.ravel().tolist(),
        bathymetry=b.ravel().tolist(),
        boundaries=boundaries,
        config=SolverConfig(
            target_time=target_time,
            cfl=cfl,
            checkpoint_every=checkpoint_every,
            max_steps=max_steps,
            noise_amplitude=noise_amplitude,
            seed=seed,
            g=g,
        ),
    )


def dambreak(n=24, left=2.0, right=1.0):
    eta = np.ones((n, n)) * right
    eta[:, : n // 2] = left
    b = np.zeros((n, n))
    return eta, b
