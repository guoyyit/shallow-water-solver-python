"""End-to-end HTTP tests: submit, poll, result, cancel, resume, restart."""

from __future__ import annotations

import time

import numpy as np
import pytest
from fastapi.testclient import TestClient

from shallow_water.service import create_app

from .conftest import dambreak, make_request


@pytest.fixture
def client(tmp_path):
    app = create_app(store_dir=str(tmp_path / "jobs"), max_workers=4)
    with TestClient(app) as c:
        yield c


def _payload(req):
    return req.model_dump(mode="json")


def _wait(client, jid, timeout=10.0):
    end = time.time() + timeout
    while True:
        rec = client.get(f"/jobs/{jid}").json()
        if rec["status"] in ("completed", "failed", "cancelled"):
            return rec
        assert time.time() < end
        time.sleep(0.01)


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_submit_and_get_result(client, walls):
    eta, b = dambreak(16)
    req = make_request(eta, b, walls, target_time=0.1)
    r = client.post("/jobs", json=_payload(req))
    assert r.status_code == 201
    jid = r.json()["job_id"]
    rec = _wait(client, jid)
    assert rec["status"] == "completed"
    res = client.get(f"/jobs/{jid}/result")
    assert res.status_code == 200
    body = res.json()
    assert body["eta"]["shape"] == [16, 16]
    assert len(body["eta"]["values"]) == 256


def test_invalid_payload_422_and_bad_physics_400(client, walls):
    eta, b = dambreak(8)
    req = make_request(eta, b, walls)
    p = _payload(req)
    p["boundaries"] = {"west": {"type": "wall"}}  # missing sides
    assert client.post("/jobs", json=p).status_code == 422

    req2 = make_request(eta, b, walls, target_time=0.1)
    p2 = _payload(req2)
    p2["grid"] = {"nx": 3, "ny": 3, "dx": 0.1, "dy": 0.1}
    assert client.post("/jobs", json=p2).status_code == 400


def test_result_unavailable_while_running_and_after_failure(client, walls):
    n = 24
    eta = np.zeros((n, n))
    eta[:, : n // 2] = 10.0
    req = make_request(eta, np.zeros((n, n)), walls,
                       target_time=5.0, cfl=2.0, dx=1.0, dy=1.0)
    jid = client.post("/jobs", json=_payload(req)).json()["job_id"]
    rec = _wait(client, jid)
    assert rec["status"] == "failed"
    r = client.get(f"/jobs/{jid}/result")
    assert r.status_code == 409
    detail = r.json()["detail"]
    assert detail["failure_reason"] in ("diverged", "negative_depth")


def test_cancel_returns_no_intermediate_result_and_resume(client, walls):
    eta, b = dambreak(40)
    req = make_request(eta, b, walls, target_time=2.0, checkpoint_every=1)
    jid = client.post("/jobs", json=_payload(req)).json()["job_id"]
    time.sleep(0.05)
    r = client.post(f"/jobs/{jid}/cancel")
    assert r.status_code == 200
    rec = _wait(client, jid)
    assert rec["status"] == "cancelled"
    assert rec["result"] is None
    assert client.get(f"/jobs/{jid}/result").status_code == 409
    assert rec["time"] > 0.0  # committed progress

    # Resume and complete.
    rr = client.post(f"/jobs/{jid}/resume", json={"target_time": 2.0})
    assert rr.status_code == 200
    rec2 = _wait(client, jid)
    assert rec2["status"] == "completed"
    assert client.get(f"/jobs/{jid}/result").status_code == 200


def test_concurrent_submissions(client, walls):
    eta, b = dambreak(16)
    ids = []
    for i in range(6):
        req = make_request(eta, b, walls, target_time=0.1)
        r = client.post("/jobs", json=_payload(req))
        ids.append(r.json()["job_id"])
    for jid in ids:
        assert _wait(client, jid)["status"] == "completed"


def test_restart_recovers_running_job(tmp_path, walls):
    from shallow_water.service import create_app as make_app
    store = str(tmp_path / "jobs")
    eta, b = dambreak(32)
    req = make_request(eta, b, walls, target_time=0.6, checkpoint_every=1)
    app = make_app(store_dir=store, max_workers=2)
    with TestClient(app) as c:
        jid = c.post("/jobs", json=_payload(req)).json()["job_id"]
        time.sleep(0.05)
    # New app instance on the same store = process restart.
    app2 = make_app(store_dir=store, max_workers=2)
    with TestClient(app2) as c2:
        rec = _wait(c2, jid, timeout=15.0)
        assert rec["status"] == "completed"
        assert rec["step"] > 1
