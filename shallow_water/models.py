"""Data models: boundary-condition types, requests, results, job records.

Only type definitions live here; numerical and persistence logic belongs to
other modules.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field, field_validator


class BoundaryType(str, Enum):
    """Supported boundary-condition families."""

    WALL = "wall"        # reflecting solid wall (zero normal flow)
    PERIODIC = "periodic"  # periodic wrap
    LEVEL = "level"      # prescribed water-surface elevation


class BoundaryCondition(BaseModel):
    """Boundary condition for one of the four sides (west/east/south/north)."""

    type: BoundaryType
    level: float | None = None  # required (and only used) for LEVEL


class GridSpec(BaseModel):
    """Rectangular uniform grid specification."""

    nx: int = Field(gt=1)
    ny: int = Field(gt=1)
    dx: float = Field(gt=0)
    dy: float = Field(gt=0)

    @property
    def n_cells(self) -> int:
        return self.nx * self.ny


class SolverConfig(BaseModel):
    """Numerical configuration."""

    target_time: float = Field(gt=0)
    # CFL number. Keep <= 0.9 for stable integration; values > 1 are allowed
    # (e.g. stress tests) but the explicit scheme is then unstable and a run
    # is expected to fail with DIVERGED rather than return fake results.
    cfl: float = Field(default=0.5, gt=0)
    max_steps: int = Field(default=1_000_000, gt=0)
    g: float = Field(default=9.81, gt=0)
    # Checkpoint every N accepted steps (>=1). Also enforced after final step.
    checkpoint_every: int = Field(default=100, ge=1)
    # Optional deterministic perturbation of the initial surface field.
    noise_amplitude: float = Field(default=0.0, ge=0)
    seed: int = Field(default=0)


class SolveRequest(BaseModel):
    """Request payload for POST /jobs."""

    grid: GridSpec
    # Flattened row-major arrays of length nx*ny.
    initial_level: list[float]
    bathymetry: list[float]
    boundaries: dict[str, BoundaryCondition]
    config: SolverConfig = Field(default_factory=SolverConfig)
    # If supplied, resume from the most recent committed checkpoint of that job.
    resume_from: str | None = None

    @field_validator("boundaries")
    @classmethod
    def _check_sides(cls, v: dict[str, BoundaryCondition]) -> dict[str, BoundaryCondition]:
        sides = {"west", "east", "south", "north"}
        if set(v.keys()) != sides:
            raise ValueError(f"boundaries must define exactly {sorted(sides)}")
        for bc in v.values():
            if bc.type is BoundaryType.LEVEL and bc.level is None:
                raise ValueError("level boundary requires a 'level' value")
        return v


class FailureReason(str, Enum):
    """Distinguishable failure categories (never a fake convergence)."""

    NEGATIVE_DEPTH = "negative_depth"
    DIVERGED = "diverged"
    MAX_STEPS = "max_steps_exceeded"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"
    INVALID = "invalid_input"


class SolverError(Exception):
    """Raised by the numerical core on unrecoverable failure."""

    def __init__(self, reason: FailureReason, message: str):
        self.reason = reason
        super().__init__(message)


class FieldResult(BaseModel):
    """A single output field: flattened row-major array of nx*ny values."""

    shape: list[int]
    values: list[float]


class SolveResult(BaseModel):
    """Final output: surface level eta and the two velocity components."""

    time: float
    step: int
    eta: FieldResult
    u: FieldResult
    v: FieldResult


class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class JobRecord(BaseModel):
    """Serializable job metadata persisted next to its checkpoints."""

    job_id: str
    status: JobStatus
    created_at: float
    updated_at: float
    # Progress of the most recent committed checkpoint.
    time: float = 0.0
    step: int = 0
    result: SolveResult | None = None
    failure_reason: FailureReason | None = None
    failure_message: str | None = None
    request: SolveRequest | None = None


class CancellationToken:
    """Cooperative cancellation flag (per job, no shared mutable state).

    ``cause`` records who requested the stop: ``"user"`` (explicit cancel),
    ``"shutdown"`` (the service is stopping and the job should remain an
    interruptible RUNNING job for the next process) or ``None``.
    """

    def __init__(self) -> None:
        self._cancelled = False
        self.cause: str | None = None

    def cancel(self, cause: str = "user") -> None:
        self._cancelled = True
        if self.cause is None:
            self.cause = cause

    @property
    def cancelled(self) -> bool:
        return self._cancelled
