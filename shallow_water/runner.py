"""任务调度：并发执行、取消/超时、检查点续算。

并发模型
========
* 每个任务拥有独立的 ``SWSolver`` 实例与 numpy 数组，任务间不共享可变状态；
  numpy 随机数生成器在求解器内独立创建（固定 seed 即可复现）。
* 计算在线程池执行器中进行（numpy 热点释放 GIL，可真正并行），
  默认 ``min(32, cpu+4)`` 个工作线程，可在构造时配置。
* 计算循环每一步都检查取消事件与墙钟超时截止时刻，因此取消/超时只在
  时间步边界生效：已完成的步骤已随检查点原子提交，未提交的中间状态
  仅存在于该任务私有的内存中，绝不会作为结果返回。

断点续算
========
* 任务创建后先提交 ``t=0`` 初始检查点，随后按 ``checkpoint_interval``
  （模拟时间）间隔原子提交。
* 进程重启时 ``resume_persisted`` 会把所有未完成任务重新调度，求解器从
  最近一次已提交检查点恢复，绝不从初始状态重复计算。
"""
from __future__ import annotations

import asyncio
import logging
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import numpy as np

from shallow_water.models import (
    BoundarySide,
    CreateTaskRequest,
    FailureReason,
    FieldOutput,
    TaskStatus,
)
from shallow_water.numerics import SWSolver, SimulationError
from shallow_water.storage import TaskStore

logger = logging.getLogger("shallow_water.runner")


class JobCancelled(Exception):
    """任务在时间步边界被取消。"""


class JobTimeout(Exception):
    """任务在时间步边界检测到超时。"""


@dataclass
class Job:
    task_id: str
    request: CreateTaskRequest
    cancel_event: threading.Event = field(default_factory=threading.Event)
    meta_lock: threading.Lock = field(default_factory=threading.Lock)
    meta: dict | None = None
    future: asyncio.Future | None = None
    started_at: float = 0.0


def _now() -> float:
    return time.time()


def _build_solver(req: CreateTaskRequest) -> SWSolver:
    boundaries = {side: req.boundaries[side] for side in BoundarySide}
    return SWSolver(
        spec=req.grid,
        eta0=np.asarray(req.eta0, dtype=np.float64),
        bottom=np.asarray(req.bottom, dtype=np.float64),
        boundaries=boundaries,
    )


class TaskManager:
    """管理仿真任务的生命周期。"""

    # 步数硬上限，防止 dt 趋 0 的伪稳定（实际发散）
    MAX_STEPS = 20_000_000

    def __init__(
        self,
        store: TaskStore,
        max_workers: int | None = None,
    ):
        self.store = store
        self._jobs: dict[str, Job] = {}
        self._jobs_lock = threading.Lock()
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="sw-solve",
        )
        self._asyncio_loop: asyncio.AbstractEventLoop | None = None

    # ------------------------------------------------------------------ #
    # 生命周期
    # ------------------------------------------------------------------ #
    def attach_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._asyncio_loop = loop

    def resume_persisted(self) -> list[str]:
        """启动时恢复：重新调度所有未完成任务（从检查点继续）。"""
        resumed: list[str] = []
        for task_id in self.store.list_tasks():
            try:
                meta = self.store.load_meta(task_id)
            except (OSError, ValueError):
                logger.exception("读取任务 %s 元数据失败", task_id)
                continue
            if not meta:
                continue
            if meta.get("status") in (
                TaskStatus.PENDING.value,
                TaskStatus.RUNNING.value,
            ):
                try:
                    req = self.store.load_request(task_id)
                except (OSError, ValueError):
                    logger.exception("任务 %s 请求损坏，跳过", task_id)
                    continue
                self._schedule(task_id, req)
                resumed.append(task_id)
        return resumed

    def shutdown(self, wait: bool = False) -> None:
        """关闭线程池（不强制打断正在原子提交的步骤）。"""
        self._executor.shutdown(wait=wait, cancel_futures=not wait)

    # ------------------------------------------------------------------ #
    def create_task(self, req: CreateTaskRequest) -> str:
        task_id = uuid.uuid4().hex
        now = _now()
        meta = {
            "task_id": task_id,
            "status": TaskStatus.PENDING.value,
            "progress_time": 0.0,
            "t_end": req.t_end,
            "result": None,
            "failure_reason": None,
            "failure_detail": None,
            "created_at": now,
            "updated_at": now,
        }
        self.store.init_task(task_id, req)
        self.store.save_meta(task_id, meta)
        self._schedule(task_id, req)
        return task_id

    def _schedule(self, task_id: str, req: CreateTaskRequest) -> Job:
        loop = self._asyncio_loop or asyncio.get_running_loop()
        job = Job(
            task_id=task_id,
            request=req,
            meta=self.store.load_meta(task_id),
        )
        with self._jobs_lock:
            self._jobs[task_id] = job
        job.future = loop.run_in_executor(
            self._executor, self._run_in_thread, job
        )

        def _done(fut: asyncio.Future) -> None:
            if not fut.cancelled():
                exc = fut.exception()
                if exc is not None:
                    logger.exception("任务 %s 异常退出", task_id, exc_info=exc)

        job.future.add_done_callback(_done)
        return job

    # ------------------------------------------------------------------ #
    # 查询 / 取消
    # ------------------------------------------------------------------ #
    def get_meta(self, task_id: str) -> dict | None:
        with self._jobs_lock:
            job = self._jobs.get(task_id)
        if job is not None:
            with job.meta_lock:
                return dict(job.meta) if job.meta else None
        # 进程内无记录（如重启后、尚未被查询触发恢复）：读磁盘
        return self.store.load_meta(task_id)

    def cancel(self, task_id: str) -> bool:
        """请求取消。仅在时间步边界生效，返回是否找到了运行中的任务。"""
        with self._jobs_lock:
            job = self._jobs.get(task_id)
        if job is None:
            return False
        job.cancel_event.set()
        return True

    # ------------------------------------------------------------------ #
    # 线程内执行
    # ------------------------------------------------------------------ #
    def _set_meta(self, job: Job, **patch) -> None:
        patch["updated_at"] = _now()
        with job.meta_lock:
            if job.meta is None:
                job.meta = self.store.load_meta(job.task_id) or {}
            job.meta.update(patch)
            self.store.save_meta(job.task_id, job.meta)

    def _run_in_thread(self, job: Job) -> None:
        task_id = job.task_id
        req = job.request
        try:
            self._set_meta(job, status=TaskStatus.RUNNING.value)
            job.started_at = _now()

            # 1) 优先从最近一次已提交检查点恢复；否则构建并先提交 t=0
            cp = self.store.load_checkpoint(task_id)
            solver = _build_solver(req)
            if cp is not None:
                solver.restore(cp)
            else:
                self.store.save_checkpoint(task_id, solver.checkpoint())

            if job.cancel_event.is_set():
                raise JobCancelled()

            # 2) 时间步进
            interval = (
                req.checkpoint_interval
                if req.checkpoint_interval > 0
                else None
            )
            next_cp = solver.t if interval is None else solver.t + interval
            t_end = req.t_end
            timeout = req.timeout_seconds if req.timeout_seconds > 0 else None
            last_meta_t = -1.0

            while solver.t < t_end - 1e-12:
                if job.cancel_event.is_set():
                    raise JobCancelled()
                if timeout is not None and _now() - job.started_at > timeout:
                    raise JobTimeout()
                if solver.step_count >= self.MAX_STEPS:
                    raise SimulationError(
                        FailureReason.DIVERGED.value,
                        f"超过最大步数 {self.MAX_STEPS}，疑似 dt 趋零的伪稳定",
                    )

                dt = solver.cfl_timestep()
                dt = min(dt, t_end - solver.t)
                if interval is not None:
                    dt = min(dt, max(next_cp - solver.t, 0.0) or dt)

                solver.advance(dt)

                # 3) 检查点提交（原子替换）。取消只在步间生效，因此磁盘上
                #    的检查点永远对应某个完整时间步。
                commit = False
                if interval is None:
                    commit = True
                elif solver.t + 1e-12 >= next_cp or solver.t >= t_end - 1e-12:
                    commit = True
                    next_cp += interval
                if commit:
                    self.store.save_checkpoint(task_id, solver.checkpoint())
                    if solver.t - last_meta_t > 1e-9:
                        self._set_meta(
                            job,
                            progress_time=solver.t,
                            status=TaskStatus.RUNNING.value,
                        )
                        last_meta_t = solver.t

            # 4) 完成：产出结果；绝不带失败标记返回中间状态
            fields = solver.snapshot_fields()
            result = FieldOutput(
                time=solver.t,
                eta=fields["eta"].tolist(),
                u=fields["u"].tolist(),
                v=fields["v"].tolist(),
            ).model_dump(mode="json")
            self._set_meta(
                job,
                status=TaskStatus.COMPLETED.value,
                progress_time=solver.t,
                result=result,
                failure_reason=None,
                failure_detail=None,
            )
            self.store.clear_checkpoint(task_id)

        except JobCancelled:
            # 已提交检查点保留且可恢复；不写入任何中间结果
            self._set_meta(
                job,
                status=TaskStatus.CANCELLED.value,
                failure_reason=FailureReason.CANCELLED.value,
                failure_detail="任务已取消",
                result=None,
            )
        except JobTimeout:
            self._set_meta(
                job,
                status=TaskStatus.FAILED.value,
                failure_reason=FailureReason.TIMEOUT.value,
                failure_detail=f"超过墙钟超时 {req.timeout_seconds:g}s",
                result=None,
            )
        except SimulationError as exc:
            try:
                reason = FailureReason(exc.reason)
            except ValueError:
                reason = FailureReason.DIVERGED
            self._set_meta(
                job,
                status=TaskStatus.FAILED.value,
                failure_reason=reason.value,
                failure_detail=exc.detail,
                result=None,
            )
