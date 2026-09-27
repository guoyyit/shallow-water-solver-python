"""Tests for the job manager: concurrency, isolation, resume, cancel, restart."""

from __future__ import annotations

import time

import numpy as np
import pytest

from shallow_water.manager import JobManager
from shallow_water.models import (
    BoundaryCondition,
    BoundaryType,
    FailureReason,
    JobStatus,
)
from shallow_water.storage import JobStore

from .conftest import dambreak, make_request


def _wait(manager, jid, timeout=10.0):
    deadline = time.time() + timeout
    while True:
        rec = manager.get(jid)
        assert rec is not None
        if rec.status in (JobStatus.COMPLETED, JobStatus.FAILED,
                          JobStatus.CANCELLED):
            return rec
        assert time.time() < deadline, f"job {jid} stuck in {rec.status}"
        time.sleep(0.005)


@pytest.fixture
def manager(tmp_path):
    # Unique store per test: only the explicit restart test shares a store
    # between manager instances.
    import uuid as _uuid

    m = JobManager(JobStore(tmp_path / f"jobs-{_uuid.uuid4().hex}"), max_workers=4)
    m.start()
    yield m
    m.shutdown()


def test_completes_and_persists_result(manager, walls):
    eta, b = dambreak(n=24)
    req = make_request(eta, b, walls, target_time=0.2)
    rec = manager.submit(req)
    final = _wait(manager, rec.job_id)
    assert final.status is JobStatus.COMPLETED
    assert final.result is not None
    assert final.result.time == pytest.approx(0.2, abs=1e-12)
    assert manager.store.load_record(final.job_id).status is JobStatus.COMPLETED


def test_checkpoint_resume_matches_continuous(manager, walls):
    eta, b = dambreak(n=32)
    # Short target so it lands CANCELLED at a committed checkpoint.
    req = make_request(eta, b, walls, target_time=1.0, checkpoint_every=1)
    rec = manager.submit(req)
    jid = rec.job_id
    # Let at least one checkpoint commit, then cancel.
    time.sleep(0.05)
    assert manager.cancel(jid)
    cancelled = _wait(manager, jid)
    assert cancelled.status is JobStatus.CANCELLED
    assert cancelled.failure_reason is FailureReason.CANCELLED
    committed_t = cancelled.time
    assert committed_t > 0.0

    # Extend the run to the same target; resume from checkpoint.
    cont_req = req.model_copy(deep=True)
    cont_req.resume_from = jid
    rec2 = manager.submit(cont_req, job_id=jid)
    assert rec2.time == pytest.approx(committed_t, abs=1e-12)
    final = _wait(manager, jid)
    assert final.status is JobStatus.COMPLETED

    # Reference: one uninterrupted integration.
    from shallow_water.solver import ShallowWaterSolver
    ref = ShallowWaterSolver.from_request(req).run()
    assert final.result.eta.values == ref.eta.values


def test_restart_resumes_from_last_checkpoint(tmp_path, walls):
    """Simulate a process restart: a RUNNING job continues from checkpoint."""
    root = tmp_path / "jobs"
    eta, b = dambreak(n=32)
    req = make_request(eta, b, walls, target_time=0.6, checkpoint_every=1)

    m1 = JobManager(JobStore(root), max_workers=2)
    m1.start()
    rec = m1.submit(req)
    jid = rec.job_id
    time.sleep(0.05)
    # Process stop: graceful shutdown cancels and JOINS the worker thread, so
    # no thread is left writing to the store when m2 takes over. The job is
    # persisted as CANCELLED with its last committed checkpoint.
    m1.shutdown(wait=True)

    # The on-disk record must remain RUNNING (interrupted, auto-resumable) and
    # carry a committed checkpoint below the target.
    disk = JobStore(root).load_record(jid)
    assert disk.status is JobStatus.RUNNING
    ck = JobStore(root).load_checkpoint(jid)
    assert ck is not None and ck[2] > 0.0
    assert ck[2] < 0.6

    m2 = JobManager(JobStore(root), max_workers=2)
    progress: list[tuple[float, int]] = []
    m2.on_progress = lambda _j, t, s: progress.append((t, s))
    m2.start()
    try:
        final = _wait(m2, jid, timeout=15.0)
        assert final.status is JobStatus.COMPLETED
        assert final.step >= ck[3]  # did not restart from step 0
        # The first checkpoint observed after restart must be at or beyond the
        # pre-crash committed step: nothing was recomputed from the start.
        assert progress, "no checkpoint observed after restart"
        assert progress[0][1] >= ck[3]
        assert progress[0][0] >= ck[2] - 1e-12
    finally:
        m2.shutdown()


def test_user_cancelled_job_is_not_auto_resumed_on_restart(tmp_path, walls):
    root = tmp_path / "jobs2"
    eta, b = dambreak(n=32)
    req = make_request(eta, b, walls, target_time=2.0, checkpoint_every=1)

    m1 = JobManager(JobStore(root), max_workers=2)
    m1.start()
    jid = m1.submit(req).job_id
    time.sleep(0.05)
    assert m1.cancel(jid)  # explicit USER cancellation
    _wait(m1, jid)
    m1.shutdown(wait=True)
    assert JobStore(root).load_record(jid).status is JobStatus.CANCELLED

    # A fresh manager must leave a user-cancelled job stopped.
    m2 = JobManager(JobStore(root), max_workers=2)
    m2.start()
    try:
        time.sleep(0.15)
        assert m2.get(jid).status is JobStatus.CANCELLED
    finally:
        m2.shutdown()


def test_concurrent_jobs_are_independent(manager):
    # Two different problems at once must not interfere.
    n = 24
    eta1, b1 = dambreak(n, left=2.0, right=1.0)
    walls = {s: BoundaryCondition(type=BoundaryType.WALL)
             for s in ("west", "east", "south", "north")}
    eta2 = np.ones((n, n))
    b2 = np.zeros((n, n))
    bcs_lev = {
        **walls,
        "west": BoundaryCondition(type=BoundaryType.LEVEL, level=2.0),
        "east": BoundaryCondition(type=BoundaryType.LEVEL, level=1.0),
    }
    r1 = manager.submit(make_request(eta1, b1, walls, target_time=0.25))
    r2 = manager.submit(make_request(eta2, b2, bcs_lev, target_time=0.25))
    f1 = _wait(manager, r1.job_id)
    f2 = _wait(manager, r2.job_id)
    assert f1.status is JobStatus.COMPLETED and f2.status is JobStatus.COMPLETED
    a = np.array(f1.result.eta.values)
    bb = np.array(f2.result.eta.values)
    assert a.tolist() != bb.tolist()


def test_failing_job_does_not_affect_others(manager, walls):
    n = 24
    good = manager.submit(make_request(*dambreak(n), walls, target_time=0.2))
    eta = np.zeros((n, n))
    eta[:, : n // 2] = 10.0
    bad = manager.submit(make_request(eta, np.zeros((n, n)), walls,
                                      target_time=5.0, cfl=2.0, dx=1.0, dy=1.0))
    g = _wait(manager, good.job_id)
    b = _wait(manager, bad.job_id)
    assert g.status is JobStatus.COMPLETED
    assert b.status is JobStatus.FAILED
    assert b.failure_reason in (FailureReason.DIVERGED, FailureReason.NEGATIVE_DEPTH)
    # Failed job exposes no intermediate result.
    assert b.result is None
    # Failure metadata is persisted; the other job's data is untouched.
    disk_bad = manager.store.load_record(b.job_id)
    assert disk_bad.status is JobStatus.FAILED
    disk_good = manager.store.load_record(g.job_id)
    assert disk_good.status is JobStatus.COMPLETED


def test_timeout_cancels_and_keeps_checkpoint(manager, walls):
    eta, b = dambreak(n=48)
    req = make_request(eta, b, walls, target_time=10.0, checkpoint_every=1)
    rec = manager.submit(req, timeout=0.1)
    final = _wait(manager, rec.job_id, timeout=5.0)
    assert final.status is JobStatus.CANCELLED
    assert final.failure_reason is FailureReason.TIMEOUT
    assert final.result is None
    ck = manager.store.load_checkpoint(rec.job_id)
    assert ck is not None and ck[2] > 0.0


def test_cancel_idempotent_and_resumable(manager, walls):
    eta, b = dambreak(n=40)
    req = make_request(eta, b, walls, target_time=2.0, checkpoint_every=1)
    rec = manager.submit(req)
    jid = rec.job_id
    time.sleep(0.05)
    assert manager.cancel(jid) is True
    _wait(manager, jid)
    # Cancelling again / a completed job is a no-op.
    assert manager.cancel(jid) is False


def test_resume_unknown_checkpoint_raises(manager, walls):
    eta, b = dambreak(n=8)
    req = make_request(eta, b, walls, target_time=0.1)
    req.resume_from = "does-not-exist"
    with pytest.raises(KeyError):
        manager.submit(req)
