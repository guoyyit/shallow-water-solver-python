"""2D shallow-water numerical core.

Governing equations (conservative form, bed elevation b, surface eta, depth
h = eta - b)::

    h_t  + (h u)_x + (h v)_y = 0
    (hu)_t + (h u^2 + g h^2/2)_x + (h u v)_y = -g h b_x
    (hv)_t + (h u v)_x + (h v^2 + g h^2/2)_y = -g h b_y

Discretisation
--------------
* Cell-centred finite volume on a uniform rectangular grid (ny rows, nx cols,
  row-major layout, arrays shape (ny, nx)).
* First-order Rusanov (local Lax-Friedrichs) flux with hydrostatic
  reconstruction (Audusse et al., 2004) so a flat lake at rest is preserved
  even over discontinuous bathymetry.
* Explicit Euler in time; global CFL-controlled step
  ``dt = cfl / (cx/dx + cy/dy)``, clipped so the run ends exactly at the
  target time.
* Boundaries are one ghost cell thick: WALL mirrors the normal discharge,
  PERIODIC wraps the opposite interior cell, LEVEL prescribes a still ghost
  cell at the requested surface elevation.

Determinism: float operation order is fixed; the only randomness is the
optional initial perturbation, drawn from ``np.random.default_rng(seed)`` and
applied once at construction (never on resume).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from .models import (
    CancellationToken,
    FailureReason,
    BoundaryType,
    SolverError,
    SolveRequest,
    SolveResult,
    FieldResult,
)

_DRY = 1.0e-6          # depth below which a cell is treated as dry
_NEG_TOL = 1.0e-6      # |negative h| tolerated after a step (clipped)

CheckpointHook = Callable[[dict[str, Any], float, int], None]


class ShallowWaterSolver:
    """Stateful cell-centred solver whose state can be checkpointed/restored."""

    def __init__(self, request: SolveRequest) -> None:
        self.request = request
        g = request.grid
        shape = (g.ny, g.nx)
        self.eta: np.ndarray = np.zeros(shape, dtype=np.float64)
        self.hu: np.ndarray = np.zeros(shape, dtype=np.float64)
        self.hv: np.ndarray = np.zeros(shape, dtype=np.float64)
        self.b: np.ndarray = np.zeros(shape, dtype=np.float64)
        self.time: float = 0.0
        self.step: int = 0
        self._h_scale = 1.0
        self._h_limit = 1.0e6
        self._v_limit = 1.0e4

    # -- construction -------------------------------------------------------
    @classmethod
    def from_request(cls, request: SolveRequest) -> "ShallowWaterSolver":
        g = request.grid
        n = g.n_cells
        if len(request.initial_level) != n or len(request.bathymetry) != n:
            raise SolverError(
                FailureReason.INVALID,
                f"arrays must have length nx*ny={n}",
            )
        s = cls(request)
        s.eta = np.asarray(request.initial_level, dtype=np.float64).reshape(g.ny, g.nx).copy()
        s.b = np.asarray(request.bathymetry, dtype=np.float64).reshape(g.ny, g.nx).copy()
        if not (np.all(np.isfinite(s.eta)) and np.all(np.isfinite(s.b))):
            raise SolverError(FailureReason.INVALID, "initial fields contain non-finite values")
        h0_raw = s.eta - s.b
        if float(h0_raw.min()) < -_NEG_TOL:
            raise SolverError(
                FailureReason.INVALID,
                f"initial depth is negative ({float(h0_raw.min()):.6g}); "
                "bed elevation must not exceed the water level",
            )
        amp = request.config.noise_amplitude
        if amp > 0.0:
            rng = np.random.default_rng(request.config.seed)
            s.eta = s.eta + amp * (rng.random(s.eta.shape) - 0.5)
            s.eta = np.ascontiguousarray(s.eta)
        # Dry initial bed: no flow.
        h0 = np.maximum(s.eta - s.b, 0.0)
        s.hu.fill(0.0)
        s.hv.fill(0.0)
        levels = [float(h0.max()), 1.0]
        levels.extend(
            abs(bc.level) for bc in request.boundaries.values()
            if bc.type is BoundaryType.LEVEL and bc.level is not None
        )
        s._h_scale = max(levels)
        s._h_limit = max(1.0e6, 1.0e4 * s._h_scale)
        s._v_limit = max(1.0e4, 100.0 * np.sqrt(request.config.g * s._h_scale))
        return s

    # -- state (de)serialisation -------------------------------------------
    def to_state(self) -> dict[str, Any]:
        return {
            "eta": self.eta.tolist(),
            "hu": self.hu.tolist(),
            "hv": self.hv.tolist(),
            "b": self.b.tolist(),
            "time": float(self.time),
            "step": int(self.step),
        }

    @classmethod
    def from_state(cls, request: SolveRequest, state: dict[str, Any]) -> "ShallowWaterSolver":
        s = cls.from_request(request)  # builds scales; does NOT re-apply noise
        g = request.grid
        try:
            s.eta = np.asarray(state["eta"], dtype=np.float64).reshape(g.ny, g.nx).copy()
            s.hu = np.asarray(state["hu"], dtype=np.float64).reshape(g.ny, g.nx).copy()
            s.hv = np.asarray(state["hv"], dtype=np.float64).reshape(g.ny, g.nx).copy()
            s.b = np.asarray(state["b"], dtype=np.float64).reshape(g.ny, g.nx).copy()
            s.time = float(state["time"])
            s.step = int(state["step"])
        except (KeyError, TypeError, ValueError) as exc:
            raise SolverError(FailureReason.INVALID, f"corrupt checkpoint: {exc}") from exc
        if not (np.all(np.isfinite(s.eta)) and np.all(np.isfinite(s.hu))
                and np.all(np.isfinite(s.hv)) and np.all(np.isfinite(s.b))):
            raise SolverError(FailureReason.DIVERGED, "checkpoint contains non-finite fields")
        if float((s.eta - s.b).min()) < -_NEG_TOL:
            raise SolverError(FailureReason.NEGATIVE_DEPTH, "checkpoint has negative depth")
        if s.time < 0.0 or s.step < 0 or s.time > request.config.target_time + 1e-9:
            raise SolverError(FailureReason.INVALID, "checkpoint time/step out of range")
        return s

    # -- helpers ------------------------------------------------------------
    def _depth(self) -> np.ndarray:
        return self.eta - self.b

    def _check_state(self) -> None:
        h = self._depth()
        if not (np.all(np.isfinite(self.eta)) and np.all(np.isfinite(self.hu))
                and np.all(np.isfinite(self.hv))):
            raise SolverError(FailureReason.DIVERGED, f"non-finite state at t={self.time}")
        if float(h.min()) < -_NEG_TOL:
            raise SolverError(
                FailureReason.NEGATIVE_DEPTH,
                f"negative depth {float(h.min()):.6g} at t={self.time}",
            )
        if float(h.max()) > self._h_limit:
            raise SolverError(FailureReason.DIVERGED, f"depth {float(h.max()):.6g} blew up")
        wet = h > _DRY
        if wet.any():
            spd = np.maximum(
                np.abs(self.hu[wet]) / h[wet], np.abs(self.hv[wet]) / h[wet]
            )
            if float(spd.max()) > self._v_limit:
                raise SolverError(FailureReason.DIVERGED, f"speed {float(spd.max()):.6g} blew up")

    def _clip_dry(self) -> None:
        h = self._depth()
        dry = h <= _DRY
        if dry.any():
            self.eta[dry] = self.b[dry] + _DRY
            self.hu[dry] = 0.0
            self.hv[dry] = 0.0

    # -- CFL step -----------------------------------------------------------
    def compute_dt(self) -> float:
        cfg = self.request.config
        h = np.maximum(self._depth(), 0.0)
        wet = h > _DRY
        cx = cy = 0.0
        if wet.any():
            c = np.sqrt(cfg.g * h[wet])
            u = np.zeros_like(h)
            v = np.zeros_like(h)
            u[wet] = self.hu[wet] / h[wet]
            v[wet] = self.hv[wet] / h[wet]
            cx = float(np.max(np.abs(u[wet]) + c))
            cy = float(np.max(np.abs(v[wet]) + c))
        # Include boundary ghost wave speeds (driven LEVEL boundaries).
        cx = max(cx, self._boundary_speed(axis=0))
        cy = max(cy, self._boundary_speed(axis=1))
        denom = cx / self.request.grid.dx + cy / self.request.grid.dy
        if denom <= 0.0:
            return 0.0  # stationary state
        return cfg.cfl / denom

    def _boundary_speed(self, axis: int) -> float:
        """Max gravity-wave speed that a LEVEL boundary ghost can inject."""
        cfg = self.request.config
        bnd = self.request.boundaries
        if axis == 0:
            sides = ("west", "east")
            edge_h = (self._depth()[:, 0], self._depth()[:, -1])
        else:
            sides = ("south", "north")
            edge_h = (self._depth()[0, :], self._depth()[-1, :])
        speed = 0.0
        for side, hh in zip(sides, edge_h):
            bc = bnd[side]
            if bc.type is BoundaryType.LEVEL and bc.level is not None:
                hg = max(0.0, float(bc.level) - 0.0)  # bed copy => relative depth
                speed = max(speed, np.sqrt(cfg.g * max(hg, float(np.maximum(hh, 0.0).max()))))
        return float(speed)

    # -- one finite-volume step --------------------------------------------
    def advance(self, dt: float) -> None:
        h = np.maximum(self._depth(), 0.0)
        h_new = h.copy()
        hu_new = self.hu.copy()
        hv_new = self.hv.copy()

        self._sweep_x(h, h_new, hu_new, hv_new, dt)
        self._sweep_y(h, h_new, hu_new, hv_new, dt)

        self.eta = self.b + h_new
        self.hu = hu_new
        self.hv = hv_new
        self.time += dt
        self._clip_dry()
        self._check_state()

    def _sweep_x(
        self,
        h: np.ndarray,
        h_new: np.ndarray,
        hu_new: np.ndarray,
        hv_new: np.ndarray,
        dt: float,
    ) -> None:
        cfg = self.request.config
        ny, nx = h.shape
        # Padded arrays: interior columns 1..nx.
        H = np.zeros((ny, nx + 2))
        HU = np.zeros_like(H)
        HV = np.zeros_like(H)
        B = np.zeros_like(H)
        H[:, 1:-1] = h
        HU[:, 1:-1] = self.hu
        HV[:, 1:-1] = self.hv
        B[:, 1:-1] = self.b
        self._fill_x_ghosts(H, HU, HV, B)

        HL, HR = H[:, :-1], H[:, 1:]
        BL, BR = B[:, :-1], B[:, 1:]
        etaL, etaR = BL + HL, BR + HR
        UL = np.divide(HU[:, :-1], HL, out=np.zeros_like(HL), where=HL > _DRY)
        UR = np.divide(HU[:, 1:], HR, out=np.zeros_like(HR), where=HR > _DRY)
        VL = np.divide(HV[:, :-1], HL, out=np.zeros_like(HL), where=HL > _DRY)
        VR = np.divide(HV[:, 1:], HR, out=np.zeros_like(HR), where=HR > _DRY)

        zf = np.maximum(BL, BR)
        hL = np.clip(etaL - zf, 0.0, None)
        hR = np.clip(etaR - zf, 0.0, None)
        c = np.maximum(
            np.sqrt(UL**2 + VL**2) + np.sqrt(cfg.g * hL),
            np.sqrt(UR**2 + VR**2) + np.sqrt(cfg.g * hR),
        )

        qL, qR = hL * UL, hR * UR
        Fm = 0.5 * (qL + qR) - 0.5 * c * (hR - hL)
        Fx = (
            0.5 * (hL * UL**2 + 0.5 * cfg.g * hL**2
                   + hR * UR**2 + 0.5 * cfg.g * hR**2)
            - 0.5 * c * (qR - qL)
        )
        Fy = 0.5 * (hL * UL * VL + hR * UR * VR) - 0.5 * c * (hR * VR - hL * VL)

        # Hydrostatic-reconstruction balance source gathered per cell:
        # +left-face contribution, -right-face contribution.
        Sx = np.zeros_like(H)
        Sx[:, 1:] += 0.5 * cfg.g * (HR**2 - hR**2)
        Sx[:, :-1] -= 0.5 * cfg.g * (HL**2 - hL**2)

        r = dt / self.request.grid.dx
        h_new -= r * (Fm[:, 1:] - Fm[:, :-1])
        hu_new += -r * (Fx[:, 1:] - Fx[:, :-1]) + r * Sx[:, 1:-1]
        hv_new += -r * (Fy[:, 1:] - Fy[:, :-1])

    def _fill_x_ghosts(
        self, H: np.ndarray, HU: np.ndarray, HV: np.ndarray, B: np.ndarray
    ) -> None:
        bnd = self.request.boundaries
        west, east = bnd["west"], bnd["east"]
        # west ghost (column 0)
        if west.type is BoundaryType.WALL:
            H[:, 0] = H[:, 1]; HU[:, 0] = -HU[:, 1]; HV[:, 0] = HV[:, 1]; B[:, 0] = B[:, 1]
        elif west.type is BoundaryType.PERIODIC:
            H[:, 0] = H[:, -2]; HU[:, 0] = HU[:, -2]; HV[:, 0] = HV[:, -2]; B[:, 0] = B[:, -2]
        else:
            B[:, 0] = B[:, 1]
            H[:, 0] = np.clip(float(west.level) - B[:, 1], 0.0, None)
            HU[:, 0] = 0.0; HV[:, 0] = 0.0
        # east ghost (column -1)
        if east.type is BoundaryType.WALL:
            H[:, -1] = H[:, -2]; HU[:, -1] = -HU[:, -2]; HV[:, -1] = HV[:, -2]; B[:, -1] = B[:, -2]
        elif east.type is BoundaryType.PERIODIC:
            H[:, -1] = H[:, 1]; HU[:, -1] = HU[:, 1]; HV[:, -1] = HV[:, 1]; B[:, -1] = B[:, 1]
        else:
            B[:, -1] = B[:, -2]
            H[:, -1] = np.clip(float(east.level) - B[:, -2], 0.0, None)
            HU[:, -1] = 0.0; HV[:, -1] = 0.0

    def _sweep_y(
        self,
        h: np.ndarray,
        h_new: np.ndarray,
        hu_new: np.ndarray,
        hv_new: np.ndarray,
        dt: float,
    ) -> None:
        cfg = self.request.config
        ny, nx = h.shape
        H = np.zeros((ny + 2, nx))
        HU = np.zeros_like(H)
        HV = np.zeros_like(H)
        B = np.zeros_like(H)
        H[1:-1, :] = h
        HU[1:-1, :] = self.hu
        HV[1:-1, :] = self.hv
        B[1:-1, :] = self.b
        self._fill_y_ghosts(H, HU, HV, B)

        HL, HR = H[:-1, :], H[1:, :]
        BL, BR = B[:-1, :], B[1:, :]
        etaL, etaR = BL + HL, BR + HR
        UL = np.divide(HU[:-1, :], HL, out=np.zeros_like(HL), where=HL > _DRY)
        UR = np.divide(HU[1:, :], HR, out=np.zeros_like(HR), where=HR > _DRY)
        VL = np.divide(HV[:-1, :], HL, out=np.zeros_like(HL), where=HL > _DRY)
        VR = np.divide(HV[1:, :], HR, out=np.zeros_like(HR), where=HR > _DRY)

        zf = np.maximum(BL, BR)
        hL = np.clip(etaL - zf, 0.0, None)
        hR = np.clip(etaR - zf, 0.0, None)
        c = np.maximum(
            np.sqrt(UL**2 + VL**2) + np.sqrt(cfg.g * hL),
            np.sqrt(UR**2 + VR**2) + np.sqrt(cfg.g * hR),
        )

        qL, qR = hL * VL, hR * VR
        Fm = 0.5 * (qL + qR) - 0.5 * c * (hR - hL)
        Fy = (
            0.5 * (hL * VL**2 + 0.5 * cfg.g * hL**2
                   + hR * VR**2 + 0.5 * cfg.g * hR**2)
            - 0.5 * c * (qR - qL)
        )
        Fx = 0.5 * (hL * UL * VL + hR * UR * VR) - 0.5 * c * (hR * UR - hL * UL)

        Sy = np.zeros_like(H)
        Sy[1:, :] += 0.5 * cfg.g * (HR**2 - hR**2)
        Sy[:-1, :] -= 0.5 * cfg.g * (HL**2 - hL**2)

        r = dt / self.request.grid.dy
        h_new -= r * (Fm[1:, :] - Fm[:-1, :])
        hv_new += -r * (Fy[1:, :] - Fy[:-1, :]) + r * Sy[1:-1, :]
        hu_new += -r * (Fx[1:, :] - Fx[:-1, :])

    def _fill_y_ghosts(
        self, H: np.ndarray, HU: np.ndarray, HV: np.ndarray, B: np.ndarray
    ) -> None:
        bnd = self.request.boundaries
        south, north = bnd["south"], bnd["north"]
        if south.type is BoundaryType.WALL:
            H[0, :] = H[1, :]; HU[0, :] = HU[1, :]; HV[0, :] = -HV[1, :]; B[0, :] = B[1, :]
        elif south.type is BoundaryType.PERIODIC:
            H[0, :] = H[-2, :]; HU[0, :] = HU[-2, :]; HV[0, :] = HV[-2, :]; B[0, :] = B[-2, :]
        else:
            B[0, :] = B[1, :]
            H[0, :] = np.clip(float(south.level) - B[1, :], 0.0, None)
            HU[0, :] = 0.0; HV[0, :] = 0.0
        if north.type is BoundaryType.WALL:
            H[-1, :] = H[-2, :]; HU[-1, :] = HU[-2, :]; HV[-1, :] = -HV[-2, :]; B[-1, :] = B[-2, :]
        elif north.type is BoundaryType.PERIODIC:
            H[-1, :] = H[1, :]; HU[-1, :] = HU[1, :]; HV[-1, :] = HV[1, :]; B[-1, :] = B[1, :]
        else:
            B[-1, :] = B[-2, :]
            H[-1, :] = np.clip(float(north.level) - B[-2, :], 0.0, None)
            HU[-1, :] = 0.0; HV[-1, :] = 0.0

    # -- time integration ---------------------------------------------------
    def run(
        self,
        cancel: CancellationToken | None = None,
        deadline: float | None = None,
        on_checkpoint: CheckpointHook | None = None,
    ) -> SolveResult:
        import time as _time

        cfg = self.request.config
        target = cfg.target_time

        def _cancelled() -> bool:
            if cancel is not None and cancel.cancelled:
                raise SolverError(FailureReason.CANCELLED, "job cancelled")
            if deadline is not None and _time.monotonic() >= deadline:
                raise SolverError(FailureReason.TIMEOUT, f"deadline exceeded at t={self.time}")
            return False

        _cancelled()
        if self.time < target:
            dt0 = self.compute_dt()
            if dt0 == 0.0:
                # Entirely stationary (flat lake, no driving boundary): nothing
                # can ever change; jump to the requested time.
                self.time = target
        while self.time < target:
            _cancelled()
            dt = self.compute_dt()
            if dt <= 0.0:
                self.time = target
                break
            dt = min(dt, target - self.time)
            self.advance(dt)
            self.step += 1
            if self.step >= cfg.max_steps and self.time < target:
                raise SolverError(
                    FailureReason.MAX_STEPS,
                    f"max_steps={cfg.max_steps} reached at t={self.time}",
                )
            if on_checkpoint is not None and (
                self.step % cfg.checkpoint_every == 0 or self.time >= target
            ):
                on_checkpoint(self.to_state(), self.time, self.step)
        return self.result()

    def result(self) -> SolveResult:
        g = self.request.grid
        h = np.maximum(self._depth(), 0.0)
        u = np.divide(self.hu, h, out=np.zeros_like(h), where=h > _DRY)
        v = np.divide(self.hv, h, out=np.zeros_like(h), where=h > _DRY)

        def field(a: np.ndarray) -> FieldResult:
            return FieldResult(shape=[g.ny, g.nx], values=a.ravel().tolist())

        return SolveResult(
            time=float(self.time),
            step=int(self.step),
            eta=field(self.eta),
            u=field(u),
            v=field(v),
        )


def validate_request(request: SolveRequest) -> None:
    """Synchronously validate physical inputs before a job is accepted.

    Performs the cheap allocations/checks that would otherwise surface
    asynchronously inside the worker (where they could only be reported as a
    failed job). Raises SolverError(INVALID) on bad input.
    """
    ShallowWaterSolver.from_request(request)


def solve(
    request: SolveRequest,
    state: dict[str, Any] | None = None,
    cancel: CancellationToken | None = None,
    deadline: float | None = None,
    on_checkpoint: CheckpointHook | None = None,
) -> SolveResult:
    """Construct (or restore) a solver and integrate to the target time."""
    solver = (
        ShallowWaterSolver.from_state(request, state)
        if state is not None
        else ShallowWaterSolver.from_request(request)
    )
    return solver.run(cancel=cancel, deadline=deadline, on_checkpoint=on_checkpoint)
