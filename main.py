"""本地二维浅水方程数值求解服务入口。

启动::

    uvicorn main:app --port 8000

环境变量:
    SW_DATA_DIR      任务持久化目录（默认 ./data）
    SW_MAX_WORKERS   并发求解线程数（默认 min(32, cpu+4)）
"""
from __future__ import annotations

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI

from shallow_water.api import DEFAULT_STORE, build_manager, router


@asynccontextmanager
async def lifespan(app: FastAPI):
    max_workers = int(os.environ.get("SW_MAX_WORKERS", "0")) or None
    mgr = build_manager(
        DEFAULT_STORE,
        max_workers=max_workers,
        autostart_persisted=True,
    )
    mgr.attach_loop(asyncio.get_running_loop())
    app.state.task_manager = mgr
    resumed = mgr.resume_persisted()
    if resumed:
        app.state.resumed_tasks = resumed
    try:
        yield
    finally:
        mgr.shutdown(wait=False)


import asyncio  # noqa: E402  (lifespan 内使用)

app = FastAPI(
    title="二维浅水方程数值求解服务",
    version="0.1.0",
    lifespan=lifespan,
)
app.include_router(router)


@app.get("/")
def health() -> dict:
    return {"service": "shallow-water-solver", "status": "ok"}
