"""Durable job storage.

Layout (one directory per job, no shared files between jobs)::

    <root>/<job_id>/record.json        # JobRecord metadata (atomically replaced)
    <root>/<job_id>/checkpoint.json    # newest committed checkpoint (atomic)

Every write goes to a unique temp file in the same directory and is published
with ``os.replace``, which is atomic on POSIX: a reader (or a process restart)
can only ever observe a fully written file or the previous one, never a torn
partial write.
"""

from __future__ import annotations

import json
import os
import tempfile
import uuid
from pathlib import Path
from typing import Any

from .models import JobRecord, SolveRequest


class JobStore:
    """File-system backed store with atomic JSON writes."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def job_dir(self, job_id: str) -> Path:
        d = self.root / job_id
        d.mkdir(parents=True, exist_ok=True)
        return d

    # -- atomic primitives --------------------------------------------------
    @staticmethod
    def _atomic_write_json(path: Path, payload: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        # Unique name prevents concurrent (re)writes from clobbering each
        # other's temp files; fsync before rename guarantees durability.
        fd, tmp_name = tempfile.mkstemp(
            prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
        )
        tmp = Path(tmp_name)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, separators=(",", ":"), allow_nan=False)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, path)
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise

    @staticmethod
    def _read_json(path: Path) -> Any:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)

    # -- records ------------------------------------------------------------
    def _record_path(self, job_id: str) -> Path:
        return self.job_dir(job_id) / "record.json"

    def save_record(self, record: JobRecord) -> None:
        self._atomic_write_json(
            self._record_path(record.job_id), record.model_dump(mode="json")
        )

    def load_record(self, job_id: str) -> JobRecord | None:
        path = self._record_path(job_id)
        if not path.exists():
            return None
        return JobRecord.model_validate(self._read_json(path))

    def all_records(self) -> list[JobRecord]:
        out: list[JobRecord] = []
        if not self.root.exists():
            return out
        for child in sorted(self.root.iterdir()):
            if not child.is_dir():
                continue
            rec_path = child / "record.json"
            if rec_path.exists():
                try:
                    out.append(JobRecord.model_validate(self._read_json(rec_path)))
                except (OSError, ValueError):
                    # A corrupt record must not take down other jobs.
                    continue
        return out

    # -- checkpoints --------------------------------------------------------
    def _checkpoint_path(self, job_id: str) -> Path:
        return self.job_dir(job_id) / "checkpoint.json"

    def save_checkpoint(
        self,
        job_id: str,
        request: SolveRequest,
        state: dict[str, Any],
        time: float,
        step: int,
        checkpoint_id: str | None = None,
    ) -> str:
        """Atomically persist the newest checkpoint. Returns checkpoint id."""
        cid = checkpoint_id or uuid.uuid4().hex
        payload = {
            "checkpoint_id": cid,
            "time": float(time),
            "step": int(step),
            "request": request.model_dump(mode="json"),
            "state": state,
        }
        self._atomic_write_json(self._checkpoint_path(job_id), payload)
        return cid

    def load_checkpoint(self, job_id: str) -> tuple[SolveRequest, dict[str, Any], float, int, str] | None:
        path = self._checkpoint_path(job_id)
        if not path.exists():
            return None
        data = self._read_json(path)
        return (
            SolveRequest.model_validate(data["request"]),
            data["state"],
            float(data["time"]),
            int(data["step"]),
            str(data.get("checkpoint_id", "")),
        )

    def delete_checkpoint(self, job_id: str) -> None:
        self._checkpoint_path(job_id).unlink(missing_ok=True)
