"""Job manager: concurrency, per-job isolation, cancellation, timeout, resume.

Concurrency model
-----------------
* A bounded ``ThreadPoolExecutor`` runs jobs. The numerical core releases the
  GIL during the numpy-heavy sweeps, so jobs compute in parallel.
* Each job owns its own solver instance, cancellation token and deadline.
  Jobs never share mutable numerical state.
* A single lock guards only the small in-memory bookkeeping dictionaries;
  it is never held while a step integrates, so a slow/failed job cannot block
  others. A failure inside one job is caught at its boundary and therefore
  cannot affect other jobs or their persisted files.

Checkpoint / resume
-------------------
The solver hands the manager every committed checkpoint; the manager
atomically persists it together with the (immutable) request and advances the
record to the committed time/step. On process restart ``start()`` finds jobs
left in RUNNING and resumes them from the newest checkpoint -- never from the
initial condition, so no committed work is recomputed. Cancellation/timeout
keep the last checkpoint; the job lands in CANCELLED with that progress and
can be resumed later.
"""

from __future__ import annotations

import threading
import time
import uuid
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor

from .models import (
    CancellationToken,
    FailureReason,
    JobRecord,
    JobStatus,
    SolveRequest,
    SolveResult,
    SolverError,
)
from .solver import solve
from .solver import validate_request
from .storage import JobStore


class JobManager:
    """Owns the worker pool and in-memory/persistent job bookkeeping."""

    def __init__(
        self,
        store: JobStore,
        max_workers: int = 4,
        default_timeout: float | None = None,
        on_progress: "Callable[[str, float, int], None] | None" = None,
    ) -> None:
        self.store = store
        self.max_workers = max_workers
        self.default_timeout = default_timeout
        self.on_progress = on_progress
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._lock = threading.Lock()
        self._jobs: dict[str, JobRecord] = {}
        self._tokens: dict[str, CancellationToken] = {}
        self._futures: dict[str, Future] = {}

    # -- lifecycle ----------------------------------------------------------
    def start(self) -> None:
        """Load persisted records and resume interrupted RUNNING jobs."""
        for rec in self.store.all_records():
            with self._lock:
                self._jobs[rec.job_id] = rec
            if rec.status is JobStatus.RUNNING:
                # A previous process died/stopped mid-run; continue from the
                # newest checkpoint. Explicitly CANCELLED jobs are terminal and
                # are only resumed via the explicit /resume endpoint.
                self._resume(rec.job_id)

    def shutdown(self, wait: bool = True) -> None:
        """Signal every running job to stop, then (by default) join its thread.

        A graceful shutdown guarantees no worker thread is left writing to the
        store, so another manager can safely take over the same directory.
        Each job's last committed checkpoint is kept and remains resumable.
        """
        with self._lock:
            tokens = list(self._tokens.values())
        for token in tokens:
            token.cancel("shutdown")
        self._executor.shutdown(wait=wait, cancel_futures=False)

    # -- public API ---------------------------------------------------------
    def submit(
        self,
        request: SolveRequest,
        job_id: str | None = None,
        timeout: float | None = None,
    ) -> JobRecord:
        job_id = job_id or uuid.uuid4().hex
        timeout = timeout if timeout is not None else self.default_timeout

        resume_from = request.resume_from
        resume_state = None
        start_time = start_step = 0
        existing: JobRecord | None = None

        with self._lock:
            existing = self._jobs.get(job_id)
        if existing is not None:
            # Idempotent resubmission: only a stopped job with a checkpoint may
            # be continued via the same id.
            if existing.status is JobStatus.RUNNING:
                return existing
            if existing.status in (JobStatus.COMPLETED, JobStatus.FAILED):
                return existing
        if resume_from is not None:
            loaded = self.store.load_checkpoint(resume_from)
            if loaded is None:
                raise KeyError(f"no checkpoint for job {resume_from!r}")
            res_req, resume_state, start_time, start_step, _ = loaded
            # Resume must use the original physical configuration.
            request = self._merge_resume_request(request, res_req)

        # Synchronous up-front validation: bad physics must be rejected at
        # submission time, not discovered later in the background.
        validate_request(request)

        now = time.time()
        rec = JobRecord(
            job_id=job_id,
            status=JobStatus.RUNNING,
            created_at=existing.created_at if existing else now,
            updated_at=now,
            time=start_time,
            step=start_step,
            request=request,
        )
        self._commit(rec)

        token = CancellationToken()
        deadline = (time.monotonic() + timeout) if timeout else None
        with self._lock:
            self._jobs[job_id] = rec
            self._tokens[job_id] = token
        fut = self._executor.submit(
            self._run_job, job_id, request, token, deadline, resume_state
        )
        with self._lock:
            self._futures[job_id] = fut
        return self.get(job_id)  # type: ignore[return-value]

    def get(self, job_id: str) -> JobRecord | None:
        with self._lock:
            return self._jobs.get(job_id)

    def cancel(self, job_id: str) -> bool:
        """Cooperatively request cancellation. Returns False if not running."""
        with self._lock:
            rec = self._jobs.get(job_id)
            token = self._tokens.get(job_id)
            if rec is None or rec.status is not JobStatus.RUNNING or token is None:
                return False
            token.cancel()
        return True

    # -- internals ----------------------------------------------------------
    @staticmethod
    def _merge_resume_request(new_req: SolveRequest, stored_req: SolveRequest) -> SolveRequest:
        """Keep the original physics/grid; only target_time may be advanced."""
        if new_req.config.target_time < stored_req.config.target_time:
            return stored_req
        # Preserve all physical fields of the original request but allow the
        # caller to extend the integration target and tune scheduling knobs.
        return stored_req.model_copy(
            update={
                "config": stored_req.config.model_copy(
                    update={
                        "target_time": new_req.config.target_time,
                        "cfl": new_req.config.cfl,
                        "checkpoint_every": new_req.config.checkpoint_every,
                        "max_steps": new_req.config.max_steps,
                    }
                )
            }
        )

    def _resume(self, job_id: str) -> None:
        loaded = self.store.load_checkpoint(job_id)
        if loaded is None:
            # Marked running but never committed a checkpoint: restart fresh.
            rec = self.get(job_id)
            if rec is not None and rec.request is not None:
                request = rec.request
                state = None
            else:
                return
        else:
            request, state, _, _, _ = loaded
        token = CancellationToken()
        with self._lock:
            self._tokens[job_id] = token
            rec = self._jobs[job_id]
            rec.status = JobStatus.RUNNING
            rec.updated_at = time.time()
            rec.failure_reason = None
            rec.failure_message = None
        self._commit(rec)
        fut = self._executor.submit(self._run_job, job_id, request, token, None, state)
        with self._lock:
            self._futures[job_id] = fut

    def _run_job(
        self,
        job_id: str,
        request: SolveRequest,
        token: CancellationToken,
        deadline: float | None,
        state: dict | None,
    ) -> None:
        """Boundary of a job: every outcome is captured and persisted."""

        def on_checkpoint(
            cstate: dict, ctime: float, cstep: int, final_result: SolveResult | None = None
        ) -> None:
            self.store.save_checkpoint(job_id, request, cstate, ctime, cstep)
            if self.on_progress is not None:
                try:
                    self.on_progress(job_id, ctime, cstep)
                except Exception:
                    pass
            if final_result is not None:
                # Terminal checkpoint: publish the completed record (with the
                # result) in the same logical commit as the state file.
                self._patch(
                    job_id,
                    time=ctime,
                    step=cstep,
                    status=JobStatus.COMPLETED,
                    result=final_result,
                    failure_reason=None,
                    failure_message=None,
                    updated_at=time.time(),
                )
            else:
                self._patch(
                    job_id,
                    time=ctime,
                    step=cstep,
                    status=JobStatus.RUNNING,
                    updated_at=time.time(),
                )

        try:
            result = solve(
                request,
                state=state,
                cancel=token,
                deadline=deadline,
                on_checkpoint=on_checkpoint,
            )
        except SolverError as exc:
            if exc.reason is FailureReason.CANCELLED and token.cause == "shutdown":
                # Service shutdown: keep the job RUNNING (interrupted) with its
                # last committed checkpoint so the next process auto-resumes it
                # rather than treating it as a terminal user cancellation.
                self._patch(
                    job_id,
                    status=JobStatus.RUNNING,
                    failure_reason=None,
                    failure_message=None,
                    updated_at=time.time(),
                )
                return
            # User cancellation or timeout are terminal stops; the last
            # committed checkpoint remains on disk for an explicit /resume.
            if exc.reason in (FailureReason.CANCELLED, FailureReason.TIMEOUT):
                terminal = JobStatus.CANCELLED
            else:
                terminal = JobStatus.FAILED
            self._patch(
                job_id,
                status=terminal,
                failure_reason=exc.reason,
                failure_message=str(exc),
                updated_at=time.time(),
            )
            return
        except Exception as exc:  # defensive: never escape the job boundary
            self._patch(
                job_id,
                status=JobStatus.FAILED,
                failure_reason=FailureReason.DIVERGED,
                failure_message=f"unexpected error: {exc!r}",
                updated_at=time.time(),
            )
            return

        # run() invoked on_checkpoint at the final step, so the newest
        # committed checkpoint already equals the final state; just publish the
        # terminal record carrying the result.
        self._patch(
            job_id,
            status=JobStatus.COMPLETED,
            time=result.time,
            step=result.step,
            result=result,
            failure_reason=None,
            failure_message=None,
            updated_at=time.time(),
        )

    def _patch(self, job_id: str, **changes: object) -> JobRecord:
        with self._lock:
            rec = self._jobs[job_id]
            updated = rec.model_copy(update=changes)
            # Persist BEFORE publishing the new state in memory: anyone that
            # observes the terminal status via get() is guaranteed the durable
            # record already matches (no window where memory says done but the
            # file on disk still shows the previous status).
            self.store.save_record(updated)
            self._jobs[job_id] = updated
        return updated

    def _commit(self, record: JobRecord) -> None:
        # Persistence is per-job; an I/O error here only affects this job.
        self.store.save_record(record)
