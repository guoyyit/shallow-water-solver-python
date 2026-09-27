"""API 数据模型与枚举定义。

网格采用 Arakawa-C 交错网格：
* 水位 ``eta``、地形高程 ``b``、水深 ``h = eta - b`` 定义在单元中心 (i, j)；
* x 方向单宽流量 ``hu`` 定义在单元东界面 (i+1/2, j)；
* y 方向单宽流量 ``hv`` 定义在单元北界面 (i, j+1/2)。

输入的 ``eta0``、``b`` 为按 ``[ny][nx]`` 行优先排列的二维 JSON 数组。
"""
from __future__ import annotations

from enum import Enum
from typing import Annotated

from pydantic import BaseModel, Field


class BoundaryType(str, Enum):
    """边界条件类型。"""

    WALL = "wall"          # 固壁：法向流量为 0
    PERIODIC = "periodic"  # 周期：界面变量环形绕接
    LEVEL = "level"        # 指定水位：法向界面外水位恒为 value


class BoundarySide(str, Enum):
    LEFT = "left"
    RIGHT = "right"
    BOTTOM = "bottom"
    TOP = "top"


class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class FailureReason(str, Enum):
    NEGATIVE_DEPTH = "negative_depth"  # 出现负水深
    DIVERGED = "diverged"              # 数值发散（NaN/Inf 或超限）
    INVALID_INPUT = "invalid_input"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"


class BoundaryCondition(BaseModel):
    """单条边界条件。``value`` 仅在 ``type == level`` 时使用。"""

    type: BoundaryType
    value: float | None = None


class GridSpec(BaseModel):
    """矩形网格与物理/数值配置。"""

    nx: Annotated[int, Field(ge=2)]
    ny: Annotated[int, Field(ge=2)]
    dx: Annotated[float, Field(gt=0)]
    dy: Annotated[float, Field(gt=0)]
    gravity: Annotated[float, Field(gt=0)] = 9.81
    cfl: Annotated[float, Field(gt=0, lt=1.0)] = 0.5
    # 干单元阈值：水深低于该值视为干地，通量置零
    dry_tolerance: Annotated[float, Field(ge=0.0)] = 1e-6
    # 发散保护：水深/水位绝对值超过该阈值判定为发散
    divergence_limit: Annotated[float, Field(gt=0.0)] = 1.0e6
    # 初始扰动：可复现的高斯噪声（乘到水位上），seed 固定时结果确定
    noise_amplitude: Annotated[float, Field(ge=0.0)] = 0.0
    seed: int = 0


class CreateTaskRequest(BaseModel):
    """创建仿真任务请求。"""

    grid: GridSpec
    eta0: list[list[float]]
    bottom: list[list[float]]
    t_end: Annotated[float, Field(gt=0.0)]
    boundaries: Annotated[
        dict[BoundarySide, BoundaryCondition],
        Field(min_length=4, max_length=4),
    ]
    # 检查点落盘的时间间隔（模拟时间）；<=0 表示每个时间步都落盘
    checkpoint_interval: float = 0.1
    # 墙钟超时（秒）；<=0 表示不超时
    timeout_seconds: float = 0.0


class FieldOutput(BaseModel):
    """某一时刻的完整场输出。"""

    time: float
    eta: list[list[float]]
    u: list[list[float]]
    v: list[list[float]]


class TaskResponse(BaseModel):
    """任务状态与（成功时）结果。"""

    task_id: str
    status: TaskStatus
    progress_time: float
    t_end: float
    result: FieldOutput | None = None
    failure_reason: FailureReason | None = None
    failure_detail: str | None = None
    created_at: float
    updated_at: float
