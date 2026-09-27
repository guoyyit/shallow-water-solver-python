"""任务持久化：原子写入，杜绝部分写入（partial write）。

存储布局（每个任务一个目录 ``<store>/<task_id>/``）::

    request.json     创建任务时的完整请求（不可变）
    meta.json        任务状态、失败原因、结果等元数据
    checkpoint.tmp   检查点临时文件（写入中）
    checkpoint.json  最近一次已提交检查点（rename 原子替换）

所有写入流程均为 ``写临时文件 -> fsync -> os.replace 原子改名``，
崩溃后要么看到上一份完整文件，要么看到新的完整文件，不会读到半成品。
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

from shallow_water.models import CreateTaskRequest
from shallow_water.numerics import CheckpointData


def _atomic_write_json(path: Path, payload: Any) -> None:
    """原子写 JSON：临时文件 + fsync + os.replace。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_name, path)
        # 目录项也需 fsync 才能保证 rename 在崩溃后可见
        try:
            dir_fd = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)
        except OSError:
            pass
    except BaseException:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


class TaskStore:
    """文件系统任务存储。每个任务独立目录，任务间无共享可变数据。"""

    REQUEST_FILE = "request.json"
    META_FILE = "meta.json"
    CHECKPOINT_FILE = "checkpoint.json"

    def __init__(self, root: str | os.PathLike):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def task_dir(self, task_id: str) -> Path:
        self._validate_id(task_id)
        return self.root / task_id

    @staticmethod
    def _validate_id(task_id: str) -> None:
        if not task_id or "/" in task_id or task_id in (".", "..") \
                or os.sep in task_id:
            raise ValueError(f"非法任务 id: {task_id!r}")

    # ------------------------------------------------------------------ #
    def init_task(self, task_id: str, request: CreateTaskRequest) -> None:
        d = self.task_dir(task_id)
        d.mkdir(parents=True, exist_ok=True)
        _atomic_write_json(d / self.REQUEST_FILE, request.model_dump(mode="json"))

    def save_meta(self, task_id: str, meta: dict) -> None:
        _atomic_write_json(self.task_dir(task_id) / self.META_FILE, meta)

    def load_meta(self, task_id: str) -> dict | None:
        p = self.task_dir(task_id) / self.META_FILE
        return _read_json(p) if p.exists() else None

    def load_request(self, task_id: str) -> CreateTaskRequest:
        return CreateTaskRequest.model_validate(
            _read_json(self.task_dir(task_id) / self.REQUEST_FILE)
        )

    def exists(self, task_id: str) -> bool:
        return (self.task_dir(task_id) / self.REQUEST_FILE).exists()

    def list_tasks(self) -> list[str]:
        return sorted(
            p.name for p in self.root.iterdir()
            if p.is_dir() and (p / self.REQUEST_FILE).exists()
        )

    # ------------------------------------------------------------------ #
    def save_checkpoint(self, task_id: str, cp: CheckpointData) -> None:
        """原子提交检查点。"""
        payload = {"checkpoint": cp.to_dict()}
        _atomic_write_json(
            self.task_dir(task_id) / self.CHECKPOINT_FILE, payload
        )

    def load_checkpoint(self, task_id: str) -> CheckpointData | None:
        p = self.task_dir(task_id) / self.CHECKPOINT_FILE
        if not p.exists():
            return None
        data = _read_json(p)
        return CheckpointData.from_dict(data["checkpoint"])

    def clear_checkpoint(self, task_id: str) -> None:
        p = self.task_dir(task_id) / self.CHECKPOINT_FILE
        try:
            p.unlink()
        except FileNotFoundError:
            pass
