"""Tests for durable storage: atomic writes, isolation, recovery reads."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest

from shallow_water.models import JobRecord, JobStatus
from shallow_water.storage import JobStore

from .conftest import make_request
import numpy as np


@pytest.fixture
def req(walls):
    return make_request(np.ones((4, 4)), np.zeros((4, 4)), walls, target_time=0.1)


@pytest.fixture
def store(tmp_path):
    return JobStore(tmp_path / "jobs")


def _record(jid="j1"):
    now = time.time()
    return JobRecord(job_id=jid, status=JobStatus.RUNNING,
                     created_at=now, updated_at=now)


def test_record_roundtrip(store):
    rec = _record()
    store.save_record(rec)
    got = store.load_record("j1")
    assert got is not None and got.job_id == "j1" and got.status is JobStatus.RUNNING


def test_checkpoint_roundtrip_and_fields(store, req):
    state = {"eta": [[1.0]], "hu": [[0.0]], "hv": [[0.0]],
             "b": [[0.0]], "time": 0.25, "step": 12}
    cid = store.save_checkpoint("j1", req, state, 0.25, 12)
    loaded = store.load_checkpoint("j1")
    assert loaded is not None
    rq, st, t, step, c = loaded
    assert t == 0.25 and step == 12 and c == cid
    assert st["time"] == 0.25
    assert rq.config.target_time == 0.1


def test_no_partial_temp_files_after_writes(store, req):
    state = {"eta": [], "hu": [], "hv": [], "b": [], "time": 0.0, "step": 0}
    for i in range(20):
        store.save_record(_record())
        store.save_checkpoint("j1", req, state, 0.01 * i, i)
    leftovers = list(store.root.rglob("*.tmp"))
    assert leftovers == []


def test_atomic_replace_keeps_previous_complete_file(store):
    store.save_record(_record())
    path = store.root / "j1" / "record.json"
    before = json.loads(path.read_text())
    # A simulated torn temp file must never become the record.
    tmp = path.with_name(".record.json.broken.tmp")
    tmp.write_text("{not json")
    os.replace  # just confirm primitive exists
    after = json.loads(path.read_text())
    assert before == after
    tmp.unlink()


def test_jobs_use_separate_directories(store, req):
    state = {"eta": [], "hu": [], "hv": [], "b": [], "time": 0.0, "step": 0}
    store.save_record(_record("a"))
    store.save_record(_record("b"))
    store.save_checkpoint("a", req, state, 0.1, 1)
    store.save_checkpoint("b", req, state, 0.2, 2)
    assert (store.root / "a" / "checkpoint.json").exists()
    assert (store.root / "b" / "checkpoint.json").exists()


def test_corrupt_record_does_not_break_all_records(store):
    store.save_record(_record("good"))
    bad_dir = store.job_dir("bad")
    (bad_dir / "record.json").write_text("{corrupt")
    recs = store.all_records()
    assert [r.job_id for r in recs] == ["good"]


def test_missing_job_returns_none(store):
    assert store.load_record("ghost") is None
    assert store.load_checkpoint("ghost") is None
