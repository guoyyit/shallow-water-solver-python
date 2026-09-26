"""FastAPI HTTP layer for the shallow-water solver service."""

from __future__ import annotations

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .manager import JobManager
from .models import (
    JobRecord,
    SolveRequest,
    SolveResult,
    SolverError,
    FailureReason,
)
from .storage import JobStore


class ResumeOptions(BaseModel):
    """Optional overrides when continuing a checkpointed job."""

    target_time: float | None = Field(default=None, gt=0)
    timeout: float | None = Field(default=None, gt=0)


def create_app(
    store_dir: str | None = None,
    max_workers: int | None = None,
    default_timeout: float | None = None,
) -> FastAPI:
    store_dir = store_dir or os.environ.get("SWS_STORE", "./.sws_jobs")
    max_workers = max_workers or int(os.environ.get("SWS_WORKERS", "4"))
    manager = JobManager(
        JobStore(store_dir),
        max_workers=max_workers,
        default_timeout=default_timeout,
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        manager.start()
        yield
        manager.shutdown()

    app = FastAPI(
        title="2D Shallow Water Solver",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.state.manager = manager

    @app.post("/jobs", response_model=JobRecord, status_code=201)
    def create_job(request: SolveRequest, timeout: float | None = None) -> JobRecord:
        try:
            return manager.submit(request, timeout=timeout)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc).strip("'"))
        except SolverError as exc:
            if exc.reason is FailureReason.INVALID:
                raise HTTPException(status_code=400, detail=str(exc))
            raise

    @app.get("/jobs", response_model=list[JobRecord])
    def list_jobs() -> list[JobRecord]:
        return manager.store.all_records()

    @app.get("/jobs/{job_id}", response_model=JobRecord)
    def get_job(job_id: str) -> JobRecord:
        rec = manager.get(job_id)
        if rec is None:
            raise HTTPException(status_code=404, detail="job not found")
        return rec

    @app.get("/jobs/{job_id}/result", response_model=SolveResult)
    def get_result(job_id: str) -> SolveResult:
        rec = manager.get(job_id)
        if rec is None:
            raise HTTPException(status_code=404, detail="job not found")
        if rec.result is None:
            raise HTTPException(
                status_code=409,
                detail={
                    "status": rec.status.value,
                    "failure_reason": rec.failure_reason.value
                    if rec.failure_reason
                    else None,
                    "message": rec.failure_message,
                    "committed_time": rec.time,
                },
            )
        return rec.result

    @app.post("/jobs/{job_id}/cancel", response_model=JobRecord)
    def cancel_job(job_id: str) -> JobRecord:
        if not manager.cancel(job_id):
            rec = manager.get(job_id)
            if rec is None:
                raise HTTPException(status_code=404, detail="job not found")
            return rec  # already stopped: idempotent
        rec = manager.get(job_id)
        assert rec is not None
        return rec

    @app.post("/jobs/{job_id}/resume", response_model=JobRecord)
    def resume_job(job_id: str, options: ResumeOptions | None = None) -> JobRecord:
        rec = manager.get(job_id)
        if rec is None:
            raise HTTPException(status_code=404, detail="job not found")
        if rec.status.value == "running":
            return rec
        if rec.request is None:
            raise HTTPException(status_code=409, detail="job has no stored request")
        options = options or ResumeOptions()
        request = rec.request.model_copy(deep=True)
        if options.target_time is not None:
            request.config.target_time = options.target_time
        request.resume_from = job_id
        try:
            return manager.submit(
                request, job_id=job_id, timeout=options.timeout
            )
        except SolverError as exc:
            if exc.reason is FailureReason.INVALID:
                raise HTTPException(status_code=400, detail=str(exc))
            raise

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    return app
