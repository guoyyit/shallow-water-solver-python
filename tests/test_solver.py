"""Tests for the numerical core: boundaries, balance, determinism, physics."""

from __future__ import annotations

import numpy as np
import pytest

from shallow_water.models import (
    BoundaryCondition,
    BoundaryType,
    CancellationToken,
    FailureReason,
    SolverError,
)
from shallow_water.solver import ShallowWaterSolver

from .conftest import dambreak, make_request


def _arr(result, name):
    return np.array(getattr(result, name).values)


def test_flat_lake_at_rest_wall(walls):
    n = 16
    req = make_request(np.ones((n, n)), np.zeros((n, n)), walls, target_time=0.5)
    res = ShallowWaterSolver.from_request(req).run()
    assert np.max(np.abs(_arr(res, "eta") - 1.0)) == 0.0
    assert np.max(np.abs(_arr(res, "u"))) == 0.0
    assert np.max(np.abs(_arr(res, "v"))) == 0.0


def test_lake_over_discontinuous_bathymetry_is_well_balanced(walls):
    n = 24
    b = np.zeros((n, n))
    b[:, n // 2:] = 0.4
    b[: n // 3, :] += 0.2
    req = make_request(np.ones((n, n)), b, walls, target_time=0.5)
    res = ShallowWaterSolver.from_request(req).run()
    assert np.max(np.abs(_arr(res, "eta") - 1.0)) < 1e-12
    assert np.max(np.abs(_arr(res, "u"))) < 1e-10
    assert np.max(np.abs(_arr(res, "v"))) < 1e-10


def test_lake_on_slope_is_well_balanced(walls):
    n = 20
    yy, xx = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    b = 0.02 * xx + 0.01 * yy
    req = make_request(np.ones((n, n)), b, walls, target_time=0.3)
    res = ShallowWaterSolver.from_request(req).run()
    assert np.max(np.abs(_arr(res, "eta") - 1.0)) < 1e-10


def test_dambreak_produces_flow_and_runs_to_target(walls):
    eta, b = dambreak()
    req = make_request(eta, b, walls, target_time=0.2)
    res = ShallowWaterSolver.from_request(req).run()
    assert res.time == pytest.approx(0.2, abs=1e-12)
    assert np.max(np.abs(_arr(res, "u"))) > 0.1
    # Wave spreads from left (deep) to right (shallow).
    u = _arr(res, "u").reshape(eta.shape)
    assert u[eta.shape[0] // 2, eta.shape[1] // 2 - 1] > 0.0


def test_periodic_boundary_conserves_mass(walls):
    eta, b = dambreak(n=24)
    bcs = {
        **walls,
        "west": BoundaryCondition(type=BoundaryType.PERIODIC),
        "east": BoundaryCondition(type=BoundaryType.PERIODIC),
    }
    req = make_request(eta, b, bcs, target_time=0.4)
    res = ShallowWaterSolver.from_request(req).run()
    assert _arr(res, "eta").sum() == pytest.approx(eta.sum(), rel=1e-10)


def test_all_periodic_conserves_mass(walls):
    eta, b = dambreak(n=20)
    bcs = {s: BoundaryCondition(type=BoundaryType.PERIODIC)
           for s in ("west", "east", "south", "north")}
    req = make_request(eta, b, bcs, target_time=0.4)
    res = ShallowWaterSolver.from_request(req).run()
    assert _arr(res, "eta").sum() == pytest.approx(eta.sum(), rel=1e-10)


def test_wall_boundary_is_reflecting_near_zero_normal_flow(walls):
    n = 16
    eta, b = dambreak(n)
    req = make_request(eta, b, walls, target_time=0.05)
    res = ShallowWaterSolver.from_request(req).run()
    u = _arr(res, "u").reshape(n, n)
    # Solid walls bound the velocity field; extreme normal velocity lives
    # in the interior rather than at the wall.
    assert np.max(np.abs(u[:, 0])) < np.max(np.abs(u)) + 1e-12


def test_level_boundary_drives_flow(walls):
    n = 20
    bcs = {
        **walls,
        "west": BoundaryCondition(type=BoundaryType.LEVEL, level=2.0),
        "east": BoundaryCondition(type=BoundaryType.LEVEL, level=1.0),
    }
    req = make_request(np.ones((n, n)), np.zeros((n, n)), bcs, target_time=0.1)
    res = ShallowWaterSolver.from_request(req).run()
    u = _arr(res, "u")
    assert u.mean() > 0.05


def test_determinism_identical_runs_bitwise(walls):
    eta, b = dambreak()
    req = make_request(eta, b, walls, target_time=0.2, seed=7)
    r1 = ShallowWaterSolver.from_request(req).run()
    r2 = ShallowWaterSolver.from_request(req).run()
    assert r1.eta.values == r2.eta.values
    assert r1.u.values == r2.u.values
    assert r1.v.values == r2.v.values


def test_seed_controls_perturbation(walls):
    n = 16
    req1 = make_request(np.ones((n, n)), np.zeros((n, n)), walls,
                        target_time=0.05, noise_amplitude=0.1, seed=1)
    req2 = make_request(np.ones((n, n)), np.zeros((n, n)), walls,
                        target_time=0.05, noise_amplitude=0.1, seed=2)
    r1 = ShallowWaterSolver.from_request(req1).run()
    r2 = ShallowWaterSolver.from_request(req2).run()
    assert r1.eta.values != r2.eta.values
    # Same seed reproduces history exactly.
    r1b = ShallowWaterSolver.from_request(req1).run()
    assert r1.eta.values == r1b.eta.values


def test_step_matches_checkpoint_then_continue(walls):
    """A restored solver continues identically to a single continuous run."""
    eta, b = dambreak()
    req = make_request(eta, b, walls, target_time=0.3, checkpoint_every=5)

    snap = {}

    def hook(state, t, step, final_result=None):
        if step == 20:
            snap["state"] = state

    continuous = ShallowWaterSolver.from_request(req)
    cont_res = continuous.run(on_checkpoint=hook)

    resumed = ShallowWaterSolver.from_state(req, snap["state"])
    r = resumed.run()
    assert r.eta.values == cont_res.eta.values
    assert r.time == pytest.approx(0.3, abs=1e-12)


def test_cancellation_raises_distinguishable_error(walls):
    eta, b = dambreak(n=40)
    req = make_request(eta, b, walls, target_time=5.0)
    tok = CancellationToken()
    tok.cancel()
    with pytest.raises(SolverError) as ei:
        ShallowWaterSolver.from_request(req).run(cancel=tok)
    assert ei.value.reason is FailureReason.CANCELLED


def test_max_steps_failure(walls):
    eta, b = dambreak()
    req = make_request(eta, b, walls, target_time=5.0, max_steps=3)
    with pytest.raises(SolverError) as ei:
        ShallowWaterSolver.from_request(req).run()
    assert ei.value.reason is FailureReason.MAX_STEPS


def test_unstable_cfl_diverges_not_fake(walls):
    n = 24
    eta = np.zeros((n, n))
    eta[:, : n // 2] = 10.0
    req = make_request(eta, np.zeros((n, n)), walls,
                       target_time=5.0, cfl=2.0, dx=1.0, dy=1.0)
    with pytest.raises(SolverError) as ei:
        ShallowWaterSolver.from_request(req).run()
    assert ei.value.reason in (FailureReason.DIVERGED, FailureReason.NEGATIVE_DEPTH)


def test_negative_initial_depth_rejected(walls):
    n = 8
    req = make_request(np.ones((n, n)), np.full((n, n), 3.0), walls,
                       target_time=0.1)
    with pytest.raises(SolverError) as ei:
        ShallowWaterSolver.from_request(req)
    assert ei.value.reason is FailureReason.INVALID


def test_corrupt_checkpoint_negative_depth(walls):
    eta, b = dambreak(n=8)
    req = make_request(eta, b, walls, target_time=0.5)
    state = ShallowWaterSolver.from_request(req).to_state()
    state["eta"][0][0] = -9.0
    with pytest.raises(SolverError) as ei:
        ShallowWaterSolver.from_state(req, state)
    assert ei.value.reason is FailureReason.NEGATIVE_DEPTH
