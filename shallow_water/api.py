"""FastAPI 路由：提交任务、查询状态/结果、取消、续算。"""
from __future__ import annotations

import os
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request

from shallow_water.models import CreateTaskRequest, TaskResponse
from shallow_water.runner import TaskManager
from shallow_water.storage import TaskStore

router = APIRouter()

DEFAULT_STORE = Path(
    os.environ.get("SW_DATA_DIR", str(Path.cwd() / "data"))
)


def get_manager(request: Request) -> TaskManager:
    mgr = getattr(request.app.state, "task_manager", None)
    if mgr is None:  # pragma: no cover - lifespan 总是创建
        raise HTTPException(status_code=503, detail="服务尚未就绪")
    return mgr


def _to_response(meta: dict) -> TaskResponse:
    return TaskResponse(**meta)


@router.post("/tasks", response_model=TaskResponse, status_code=201)
def create_task(req: CreateTaskRequest, request: Request) -> TaskResponse:
    mgr = get_manager(request)
    task_id = mgr.create_task(req)
    return _to_response(mgr.get_meta(task_id))


@router.get("/tasks/{task_id}", response_model=TaskResponse)
def get_task(task_id: str, request: Request) -> TaskResponse:
    mgr = get_manager(request)
    meta = mgr.get_meta(task_id)
    if meta is None:
        raise HTTPException(status_code=404, detail="任务不存在")
    return _to_response(meta)


@router.post("/tasks/{task_id}/cancel", response_model=TaskResponse)
def cancel_task(task_id: str, request: Request) -> TaskResponse:
    mgr = get_manager(request)
    if not mgr.is_known(task_id):
        raise HTTPException(status_code=404, detail="任务不存在")
    ok = mgr.cancel_task(task_id)
    meta = mgr.get_meta(task_id)
    if not ok:
        raise HTTPException(
            status_code=409,
            detail={"message": "任务不在可取消状态", "task": meta},
        )
    return _to_response(meta)


@router.post("/tasks/{task_id}/resume", response_model=TaskResponse)
async def resume_task(task_id: str, request: Request) -> TaskResponse:
    mgr = get_manager(request)
    if not mgr.is_known(task_id):
        raise HTTPException(status_code=404, detail="任务不存在")
    ok = await mgr.resume_task(task_id)
    if not ok:
        raise HTTPException(
            status_code=409,
            detail="任务不存在、已完成或仍在运行，无法续算",
        )
    return _to_response(mgr.get_meta(task_id))


def build_manager(store_root: str | os.PathLike | None = None,
                  max_workers: int | None = None,
                  autostart_persisted: bool = True) -> TaskManager:
    root = Path(store_root) if store_root is not None else DEFAULT_STORE
    return TaskManager(
        TaskStore(root),
        max_workers=max_workers,
        autostart_persisted=autostart_persisted,
    )
