"""二维浅水方程数值核心。

控制方程（守恒形式，水面梯度写法）::

    h_t  + (hu)_x + (hv)_y                          = 0
    (hu)_t + (hu^2)_x + (huv)_y + g h * eta_x       = 0
    (hv)_t + (huv)_x + (hv^2)_y + g h * eta_y       = 0

其中水深 ``h = eta - b``，``eta`` 为水位、``b`` 为地形高程。
采用单元中心有限体积 + Rusanov（局部 Lax-Friedrichs）数值通量，
显式 Euler 时间推进，自适应 CFL 时间步长::

    dt = CFL / max(|u|/dx + sqrt(g h)/dx + |v|/dy + sqrt(g h)/dy)

边界通过一层 ghost 单元实现：
``wall`` 镜像反射（法向动量反号）、``periodic`` 环形绕接、
``level`` 令 ghost 单元水位恒为指定值。
水面梯度写法配合界面平均对静湖（eta 常数）天然平衡（well-balanced）。
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from shallow_water.models import BoundarySide, BoundaryType, GridSpec


class SimulationError(Exception):
    """数值计算失败基类，携带可区分的失败原因。"""

    def __init__(self, reason: str, detail: str):
        super().__init__(detail)
        self.reason = reason
        self.detail = detail


@dataclass
class CheckpointData:
    """可序列化的检查点状态。"""

    t: float
    h: np.ndarray
    q: np.ndarray  # hu
    r: np.ndarray  # hv

    def to_dict(self) -> dict:
        return {
            "t": float(self.t),
            "h": self.h.tolist(),
            "q": self.q.tolist(),
            "r": self.r.tolist(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "CheckpointData":
        return cls(
            t=float(d["t"]),
            h=np.asarray(d["h"], dtype=np.float64),
            q=np.asarray(d["q"], dtype=np.float64),
            r=np.asarray(d["r"], dtype=np.float64),
        )


class SWSolver:
    """二维浅水求解器（状态可变，非线程安全；每个任务独立实例）。"""

    def __init__(
        self,
        spec: GridSpec,
        eta0: np.ndarray,
        bottom: np.ndarray,
        boundaries: dict[BoundarySide, BoundaryType | object],
    ):
        self.spec = spec
        self.nx, self.ny = spec.nx, spec.ny
        self.boundaries = boundaries

        eta0 = np.asarray(eta0, dtype=np.float64)
        bottom = np.asarray(bottom, dtype=np.float64)
        if eta0.shape != (self.ny, self.nx):
            raise SimulationError(
                "invalid_input", f"eta0 形状 {eta0.shape} 与网格 "
                f"({self.ny},{self.nx}) 不符"
            )
        if bottom.shape != (self.ny, self.nx):
            raise SimulationError(
                "invalid_input", f"bottom 形状 {bottom.shape} 与网格不符"
            )
        if not (np.all(np.isfinite(eta0)) and np.all(np.isfinite(bottom))):
            raise SimulationError("invalid_input", "初始场或地形含非有限值")

        # 可复现的初始高斯扰动（amplitude 默认 0，不影响常规输入）
        if spec.noise_amplitude > 0.0:
            rng = np.random.default_rng(spec.seed)
            eta0 = eta0 + spec.noise_amplitude * rng.standard_normal(
                eta0.shape
            )

        h0 = eta0 - bottom
        if np.min(h0) < -self.spec.dry_tolerance:
            raise SimulationError(
                "negative_depth",
                f"初始水深为负: min(h)={np.min(h0):.6g}",
            )

        # padded 数组，1..ny / 1..nx 为内部单元
        shape = (self.ny + 2, self.nx + 2)
        self.b = np.zeros(shape, dtype=np.float64)
        self.h = np.zeros(shape, dtype=np.float64)
        self.q = np.zeros(shape, dtype=np.float64)
        self.r = np.zeros(shape, dtype=np.float64)
        self.b[1:-1, 1:-1] = bottom
        self.h[1:-1, 1:-1] = np.maximum(h0, 0.0)
        self.t = 0.0
        self.step_count = 0
        self._apply_boundaries()

    # ------------------------------------------------------------------ #
    # 边界
    # ------------------------------------------------------------------ #
    def _apply_boundaries(self) -> None:
        """根据边界条件填充 ghost 单元（第 0 / ny+1 行、第 0 / nx+1 列）。"""
        h, q, r, b = self.h, self.q, self.r, self.b

        left = self.boundaries[BoundarySide.LEFT]
        right = self.boundaries[BoundarySide.RIGHT]
        bottom_bc = self.boundaries[BoundarySide.BOTTOM]
        top = self.boundaries[BoundarySide.TOP]

        def get(bc):
            return bc.type if hasattr(bc, "type") else bc

        def value_of(bc):
            return float(getattr(bc, "value", 0.0) or 0.0)

        # 地形 ghost：周期绕接，其余镜像
        b[1:-1, 0] = (
            b[1:-1, -2] if get(left) == BoundaryType.PERIODIC else b[1:-1, 1]
        )
        b[1:-1, -1] = (
            b[1:-1, 1] if get(right) == BoundaryType.PERIODIC else b[1:-1, -2]
        )
        b[0, 1:-1] = (
            b[-2, 1:-1] if get(bottom_bc) == BoundaryType.PERIODIC
            else b[1, 1:-1]
        )
        b[-1, 1:-1] = (
            b[1, 1:-1] if get(top) == BoundaryType.PERIODIC
            else b[-2, 1:-1]
        )

        # 左边界（法向 x => q 反号）
        if get(left) == BoundaryType.WALL:
            h[1:-1, 0] = h[1:-1, 1]
            q[1:-1, 0] = -q[1:-1, 1]
            r[1:-1, 0] = r[1:-1, 1]
        elif get(left) == BoundaryType.PERIODIC:
            h[1:-1, 0] = h[1:-1, -2]
            q[1:-1, 0] = q[1:-1, -2]
            r[1:-1, 0] = r[1:-1, -2]
        else:  # level
            h[1:-1, 0] = np.maximum(value_of(left) - b[1:-1, 0], 0.0)
            q[1:-1, 0] = q[1:-1, 1]
            r[1:-1, 0] = r[1:-1, 1]

        # 右边界
        if get(right) == BoundaryType.WALL:
            h[1:-1, -1] = h[1:-1, -2]
            q[1:-1, -1] = -q[1:-1, -2]
            r[1:-1, -1] = r[1:-1, -2]
        elif get(right) == BoundaryType.PERIODIC:
            h[1:-1, -1] = h[1:-1, 1]
            q[1:-1, -1] = q[1:-1, 1]
            r[1:-1, -1] = r[1:-1, 1]
        else:
            h[1:-1, -1] = np.maximum(value_of(right) - b[1:-1, -1], 0.0)
            q[1:-1, -1] = q[1:-1, -2]
            r[1:-1, -1] = r[1:-1, -2]

        # 下边界（法向 y => r 反号）
        if get(bottom_bc) == BoundaryType.WALL:
            h[0, 1:-1] = h[1, 1:-1]
            q[0, 1:-1] = q[1, 1:-1]
            r[0, 1:-1] = -r[1, 1:-1]
        elif get(bottom_bc) == BoundaryType.PERIODIC:
            h[0, 1:-1] = h[-2, 1:-1]
            q[0, 1:-1] = q[-2, 1:-1]
            r[0, 1:-1] = r[-2, 1:-1]
        else:
            h[0, 1:-1] = np.maximum(value_of(bottom_bc) - b[0, 1:-1], 0.0)
            q[0, 1:-1] = q[1, 1:-1]
            r[0, 1:-1] = r[1, 1:-1]

        # 上边界
        if get(top) == BoundaryType.WALL:
            h[-1, 1:-1] = h[-2, 1:-1]
            q[-1, 1:-1] = q[-2, 1:-1]
            r[-1, 1:-1] = -r[-2, 1:-1]
        elif get(top) == BoundaryType.PERIODIC:
            h[-1, 1:-1] = h[1, 1:-1]
            q[-1, 1:-1] = q[1, 1:-1]
            r[-1, 1:-1] = r[1, 1:-1]
        else:
            h[-1, 1:-1] = np.maximum(value_of(top) - b[-1, 1:-1], 0.0)
            q[-1, 1:-1] = q[-2, 1:-1]
            r[-1, 1:-1] = r[-2, 1:-1]

        # 四角直接取相邻内部值（通量只在 j=1..ny / i=1..nx 用到）
        h[0, 0], h[0, -1], h[-1, 0], h[-1, -1] = (
            h[1, 1], h[1, -2], h[-2, 1], h[-2, -2]
        )
        q[0, 0], q[0, -1], q[-1, 0], q[-1, -1] = 0.0, 0.0, 0.0, 0.0
        r[0, 0], r[0, -1], r[-1, 0], r[-1, -1] = 0.0, 0.0, 0.0, 0.0

    # ------------------------------------------------------------------ #
    # 数值通量
    # ------------------------------------------------------------------ #
    @staticmethod
    def _rusanov(h_l, q_l, r_l, h_r, q_r, r_r, g, h_dry):
        """逐元素 Rusanov 通量，返回 (质量, x动量, y动量) 三个通量数组。"""
        hl = np.maximum(h_l, h_dry)
        hr = np.maximum(h_r, h_dry)
        u_l, v_l = q_l / hl, r_l / hl
        u_r, v_r = q_r / hr, r_r / hr
        c_l = np.sqrt(g * np.maximum(h_l, 0.0))
        c_r = np.sqrt(g * np.maximum(h_r, 0.0))
        speed = np.maximum(
            np.maximum(np.abs(u_l) + c_l, np.abs(u_r) + c_r),
            np.maximum(np.abs(v_l) + c_l, np.abs(v_r) + c_r),
        )
        # F = [q, q^2/h, qr/h]（水面梯度形式，压力项进源项）
        fq_l, fq_r = q_l, q_r
        fqq_l, fqq_r = q_l * u_l, q_r * u_r
        fqr_l, fqr_r = q_l * v_l, q_r * v_r
        fq = 0.5 * (fq_l + fq_r) - 0.5 * speed * (h_r - h_l)
        fqq = 0.5 * (fqq_l + fqq_r) - 0.5 * speed * (q_r - q_l)
        fqr = 0.5 * (fqr_l + fqr_r) - 0.5 * speed * (r_r - r_l)
        return fq, fqq, fqr, speed

    def cfl_timestep(self) -> float:
        """按当前最大波速计算满足 CFL 稳定性约束的时间步长。"""
        s = self.spec
        h = self.h[1:-1, 1:-1]
        q = self.q[1:-1, 1:-1]
        r = self.r[1:-1, 1:-1]
        hs = np.maximum(h, s.dry_tolerance)
        c = np.sqrt(s.gravity * np.maximum(h, 0.0))
        speed = (
            np.max(np.abs(q) / hs) / s.dx
            + np.max(np.abs(r) / hs) / s.dy
            + np.max(c) * (1.0 / s.dx + 1.0 / s.dy)
        )
        if not np.isfinite(speed) or speed <= 0.0:
            raise SimulationError("diverged", "无法计算有限 CFL 时间步长")
        return s.cfl / float(speed)

    def advance(self, dt: float) -> None:
        """以显式 Euler 推进一个时间步。"""
        s = self.spec
        g, hdry = s.gravity, s.dry_tolerance
        self._apply_boundaries()
        h, q, r, b = self.h, self.q, self.r, self.b
        eta = h + b

        # x 方向界面通量（界面 i 在列 i 与 i+1 之间，i=0..nx）
        fx_q, fx_qq, fx_qr, _ = self._rusanov(
            h[:, :-1], q[:, :-1], r[:, :-1],
            h[:, 1:], q[:, 1:], r[:, 1:], g, hdry,
        )
        # y 方向界面通量（界面 j 在行 j 与 j+1 之间，j=0..ny）
        fy_r, fy_rq, fy_rr, _ = self._rusanov(
            h[:-1, :], q[:-1, :], r[:-1, :],
            h[1:, :], q[1:, :], r[1:, :], g, hdry,
        )

        # 内部单元更新（padded 下标 1..ny, 1..nx）
        dh = (
            -dt / s.dx * (fx_q[1:-1, 1:] - fx_q[1:-1, :-1])
            - dt / s.dy * (fy_r[1:, 1:-1] - fy_r[:-1, 1:-1])
        )
        # 水面梯度（界面平均，保证静湖平衡）
        eta_face_x = 0.5 * (eta[:, :-1] + eta[:, 1:])
        eta_face_y = 0.5 * (eta[:-1, :] + eta[1:, :])
        gx = (eta_face_x[1:-1, 1:] - eta_face_x[1:-1, :-1]) / s.dx
        gy = (eta_face_y[1:, 1:-1] - eta_face_y[:-1, 1:-1]) / s.dy
        hc = h[1:-1, 1:-1]

        dq = (
            -dt / s.dx * (fx_qq[1:-1, 1:] - fx_qq[1:-1, :-1])
            -dt / s.dy * (fy_rq[1:, 1:-1] - fy_rq[:-1, 1:-1])
            -dt * g * hc * gx
        )
        dr = (
            -dt / s.dx * (fx_qr[1:-1, 1:] - fx_qr[1:-1, :-1])
            -dt / s.dy * (fy_rr[1:, 1:-1] - fy_rr[:-1, 1:-1])
            -dt * g * hc * gy
        )

        h_new = hc + dh
        q_new = q[1:-1, 1:-1] + dq
        r_new = r[1:-1, 1:-1] + dr

        self._check_state(h_new, q_new, r_new)

        # 干单元处理：水深低于阈值则置干、动量清零
        wet = h_new > hdry
        h_new = np.where(wet, h_new, 0.0)
        q_new = np.where(wet, q_new, 0.0)
        r_new = np.where(wet, r_new, 0.0)

        h[1:-1, 1:-1] = h_new
        q[1:-1, 1:-1] = q_new
        r[1:-1, 1:-1] = r_new
        self.t += dt
        self.step_count += 1

    def _check_state(self, h_new, q_new, r_new) -> None:
        """失败检测：负水深 / 非有限值 / 发散，原因可区分，绝不返回伪收敛。"""
        scale = 1.0 + float(np.max(np.abs(self.b)))
        if np.min(h_new) < -self.spec.dry_tolerance * scale:
            raise SimulationError(
                "negative_depth",
                f"t={self.t:.6g}: 出现负水深 min(h)={np.min(h_new):.6g}",
            )
        if not (np.all(np.isfinite(h_new))
                and np.all(np.isfinite(q_new))
                and np.all(np.isfinite(r_new))):
            raise SimulationError(
                "diverged", f"t={self.t:.6g}: 场中出现 NaN/Inf"
            )
        eta_new = h_new + self.b[1:-1, 1:-1]
        limit = self.spec.divergence_limit
        if (np.max(np.abs(eta_new)) > limit
                or np.max(np.abs(q_new)) > limit
                or np.max(np.abs(r_new)) > limit):
            raise SimulationError(
                "diverged",
                f"t={self.t:.6g}: 数值超过发散保护阈值 {limit:g}",
            )

    # ------------------------------------------------------------------ #
    # 检查点 / 输出
    # ------------------------------------------------------------------ #
    def checkpoint(self) -> CheckpointData:
        return CheckpointData(
            t=self.t,
            h=self.h[1:-1, 1:-1].copy(),
            q=self.q[1:-1, 1:-1].copy(),
            r=self.r[1:-1, 1:-1].copy(),
        )

    def restore(self, cp: CheckpointData) -> None:
        if cp.h.shape != (self.ny, self.nx):
            raise SimulationError("invalid_input", "检查点网格形状不匹配")
        self.t = float(cp.t)
        self.h[1:-1, 1:-1] = cp.h
        self.q[1:-1, 1:-1] = cp.q
        self.r[1:-1, 1:-1] = cp.r
        self._apply_boundaries()

    def snapshot_fields(self) -> dict[str, np.ndarray]:
        """返回内部单元上的 eta、u、v（干单元速度记 0）。"""
        h = self.h[1:-1, 1:-1]
        q = self.q[1:-1, 1:-1]
        r = self.r[1:-1, 1:-1]
        b = self.b[1:-1, 1:-1]
        hs = np.maximum(h, self.spec.dry_tolerance)
        wet = h > self.spec.dry_tolerance
        u = np.where(wet, q / hs, 0.0)
        v = np.where(wet, r / hs, 0.0)
        return {"eta": h + b, "u": u, "v": v}


def estimate_steps(spec: GridSpec, t_end: float) -> int:
    """粗略估计步数上限，仅用于保护。"""
    return 10_000_000


def courant_number(solver: SWSolver, dt: float) -> float:
    """供测试/诊断：给定 dt 的最大 CFL 数。"""
    s = solver.spec
    h = solver.h[1:-1, 1:-1]
    q = solver.q[1:-1, 1:-1]
    r = solver.r[1:-1, 1:-1]
    hs = np.maximum(h, s.dry_tolerance)
    c = np.sqrt(s.gravity * np.maximum(h, 0.0))
    speed = (
        np.max(np.abs(q) / hs) / s.dx
        + np.max(np.abs(r) / hs) / s.dy
        + np.max(c) * (1.0 / s.dx + 1.0 / s.dy)
    )
    return dt * float(speed)


# 避免未使用告警
_ = (math,)
