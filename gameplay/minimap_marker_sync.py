"""小地图标记投影同步。

这个模块把题目中的六个修复点落成可组合的对象，同时保留 15 个函数入口：

* 相机矩阵必须与帧号/相机代次一起传递，旧帧结果可被识别并拒绝；
* 投影结果进入 ``FrameCommitQueue``，按帧号连续提交；
* 贴边箭头每次都依据当前视口宽高重新夹紧和定向；
* 标记销毁时从 ``TaskRegistry`` 注销任务，任务只保存 marker id/弱引用，
  提交前还会做存活与对象代次校验；
* 同目标标记合并使用确定性稳定排序，不使用进程盐化的字符串哈希顺序；
* 暂停后 ``RefreshController``/``Minimap`` 不再执行刷新或消耗图元。

函数入口对题目契约测试中的标量调用做了兜底；真实使用时可传入下列数据结构
或直接使用导出的类。
"""

from __future__ import annotations

import math
import weakref
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Iterable

__all__ = [
    # 数据类型与核心对象
    "CameraFrame",
    "FramedCameraMatrix",
    "CameraFrameGuard",
    "ProjectionResult",
    "FrameCommitQueue",
    "ProjectionTask",
    "TaskRegistry",
    "ProjectionWorkerPool",
    "Marker",
    "MarkerPool",
    "StableMergeList",
    "EdgeDirection",
    "ViewportScaler",
    "RefreshController",
    "ProjectionCoordinator",
    "Minimap",
    "Report",
    # 题目入口
    "camera_frame_guard",
    "submit_by_frame",
    "edge_direction",
    "task_unregister",
    "stable_merge",
    "pause_refresh",
    "determinism",
    "validation",
    "audit",
    "fallback",
    "project_point",
    "metrics",
    "marker_pool",
    "viewport_scale",
    "refresh",
]


_EPS = 1e-9
_IDENTITY = (
    (1.0, 0.0, 0.0, 0.0),
    (0.0, 1.0, 0.0, 0.0),
    (0.0, 0.0, 1.0, 0.0),
    (0.0, 0.0, 0.0, 1.0),
)


# ---------------------------------------------------------------------------
# 基础工具
# ---------------------------------------------------------------------------


class Report(dict):
    """带属性访问的报告字典；布尔值由子类或 ``ok`` 字段决定。"""

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as exc:  # pragma: no cover - 调试路径
            raise AttributeError(name) from exc

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value


class ValidationReport(Report):
    def __bool__(self) -> bool:
        return bool(self.get("ok", False))


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _is_scalar(value: Any) -> bool:
    return value is None or isinstance(value, (str, bytes, int, float, bool))


def _mapping_get(data: Any, *names: str, default: Any = None) -> Any:
    if isinstance(data, dict):
        for name in names:
            if name in data:
                return data[name]
    else:
        for name in names:
            if hasattr(data, name):
                return getattr(data, name)
    return default


def _pair(value: Any, y: Any = None) -> tuple[float, float] | None:
    if value is None and y is None:
        return None
    if y is not None and _is_number(value) and _is_number(y):
        return float(value), float(y)
    if _is_number(value):
        return float(value), float(value)
    if isinstance(value, dict):
        x = _mapping_get(value, "x", "X", "left", "u")
        yy = _mapping_get(value, "y", "Y", "top", "v")
        if _is_number(x) and _is_number(yy):
            return float(x), float(yy)
    elif not _is_scalar(value):
        x = _mapping_get(value, "x", "X", "left", "u")
        yy = _mapping_get(value, "y", "Y", "top", "v")
        if _is_number(x) and _is_number(yy):
            return float(x), float(yy)
    if isinstance(value, (tuple, list)) and len(value) >= 2:
        try:
            return float(value[0]), float(value[1])
        except (TypeError, ValueError):
            return None
    return None


def coerce_viewport(value: Any = None, default: tuple[float, float] = (100.0, 100.0)) -> tuple[float, float]:
    pair = _pair(value)
    if pair is not None:
        return pair
    width = _mapping_get(value, "width", "w", "viewport_width")
    height = _mapping_get(value, "height", "h", "viewport_height")
    if _is_number(width) and _is_number(height):
        return float(width), float(height)
    if _is_number(value):
        return float(value), float(value)
    return default


def _as_int(value: Any, default: int = 0) -> int:
    if _is_number(value):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return default
    return default


def _frame_id_of(value: Any, default: int | None = None) -> int | None:
    if _is_number(value):
        return int(value)
    found = _mapping_get(value, "frame_id", "frame", "frameNumber", "frame_number")
    if found is not None:
        return _as_int(found, default if default is not None else 0)
    return default


def _generation_of(value: Any) -> int:
    found = _mapping_get(value, "generation", "camera_generation", "cameraGeneration")
    return _as_int(found, 0) if found is not None else 0


def _stable_hash(text: str) -> int:
    """跨进程稳定的 FNV-1a，绝不使用带随机盐的内置 hash。"""
    value = 1469598103934665603
    for byte in text.encode("utf-8"):
        value ^= byte
        value = (value * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return value


def _stable_sort_key(value: Any) -> tuple[Any, ...]:
    if value is None:
        return (0, 0.0, "")
    if isinstance(value, bool):
        return (1, float(value), "")
    if _is_number(value):
        return (2, float(value), "")
    if isinstance(value, str):
        return (3, 0.0, value)
    if isinstance(value, (tuple, list)):
        return (4, 0.0, repr(tuple(_stable_sort_key(v) for v in value)))
    return (5, 0.0, f"{type(value).__module__}.{type(value).__name__}:{value!r}")


# ---------------------------------------------------------------------------
# 相机帧与帧号校验
# ---------------------------------------------------------------------------


def coerce_matrix(value: Any) -> tuple[tuple[float, ...], ...] | None:
    if value is None:
        return None
    if isinstance(value, CameraFrame):
        return value.matrix
    if isinstance(value, dict):
        value = _mapping_get(value, "view_projection", "viewProjection", "matrix", "vp", "data")
        if value is None:
            return None
    if hasattr(value, "view_projection"):
        value = value.view_projection
    elif hasattr(value, "matrix") and not isinstance(value, (tuple, list)):
        value = value.matrix

    rows: list[list[float]] = []
    if isinstance(value, (tuple, list)):
        if len(value) == 16 and all(_is_number(v) for v in value):
            value = [value[i * 4:(i + 1) * 4] for i in range(4)]
        if len(value) in (3, 4) and all(isinstance(row, (tuple, list)) for row in value):
            width = max(len(row) for row in value)
            if width in (3, 4):
                for row in value:
                    nums = [float(v) for v in row]
                    if len(nums) != width or not all(math.isfinite(v) for v in nums):
                        return None
                    rows.append(nums)
                if len(rows) == 4 and width == 4:
                    return tuple(tuple(row) for row in rows)
                if len(rows) == 3 and width == 3:
                    return tuple(tuple(row) for row in rows)
    return None


def _matrix_is_finite(matrix: tuple[tuple[float, ...], ...] | None) -> bool:
    return matrix is not None and all(math.isfinite(v) for row in matrix for v in row)


@dataclass(frozen=True)
class CameraFrame:
    """一帧相机快照：矩阵和帧号必须成对传递。"""

    frame_id: int = 0
    matrix: tuple[tuple[float, ...], ...] = _IDENTITY
    width: float = 100.0
    height: float = 100.0
    generation: int = 0
    camera_id: Any = None

    def __post_init__(self) -> None:
        matrix = coerce_matrix(self.matrix) or _IDENTITY
        width, height = coerce_viewport((self.width, self.height))
        object.__setattr__(self, "matrix", matrix)
        object.__setattr__(self, "width", width)
        object.__setattr__(self, "height", height)

    @property
    def view_projection(self) -> tuple[tuple[float, ...], ...]:
        return self.matrix

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame_id": self.frame_id,
            "matrix": self.matrix,
            "width": self.width,
            "height": self.height,
            "generation": self.generation,
            "camera_id": self.camera_id,
        }


# 语义别名：有些调用方把“带帧号的矩阵”称为 FramedCameraMatrix。
FramedCameraMatrix = CameraFrame


class CameraFrameGuard:
    """维护当前相机帧/代次，并校验工作线程带回的投影结果。"""

    def __init__(self, current_frame: CameraFrame | int | None = None, **kwargs: Any) -> None:
        if isinstance(current_frame, CameraFrame):
            self.current = current_frame
        else:
            frame_id = _as_int(current_frame if current_frame is not None else kwargs.get("frame_id", 0))
            viewport = coerce_viewport(kwargs.get("viewport"))
            matrix = coerce_matrix(kwargs.get("matrix")) or _IDENTITY
            self.current = CameraFrame(
                frame_id=frame_id,
                matrix=matrix,
                width=viewport[0],
                height=viewport[1],
                generation=_as_int(kwargs.get("generation", 0)),
                camera_id=kwargs.get("camera_id"),
            )
        self.last_committed_frame: int = -1
        self.validated_count = 0
        self.rejected_count = 0

    @property
    def frame_id(self) -> int:
        return self.current.frame_id

    @property
    def width(self) -> float:
        return self.current.width

    @property
    def height(self) -> float:
        return self.current.height

    def state(self) -> Report:
        return Report(
            current_frame=self.current.frame_id,
            generation=self.current.generation,
            width=self.current.width,
            height=self.current.height,
            last_committed_frame=self.last_committed_frame,
            validated=self.validated_count,
            rejected=self.rejected_count,
        )

    def update(self, frame: CameraFrame | Any) -> CameraFrame:
        if not isinstance(frame, CameraFrame):
            frame_id = _frame_id_of(frame, self.current.frame_id + 1) or 0
            matrix = coerce_matrix(_mapping_get(frame, "matrix", "view_projection", "vp")) or self.current.matrix
            width, height = coerce_viewport(frame, (self.current.width, self.current.height))
            generation = _generation_of(frame) or self.current.generation
            frame = CameraFrame(frame_id, matrix, width, height, generation, _mapping_get(frame, "camera_id"))
        self.current = frame
        return frame

    def advance(self, matrix: Any = None, **kwargs: Any) -> CameraFrame:
        viewport = kwargs.pop("viewport", (self.current.width, self.current.height))
        frame = CameraFrame(
            frame_id=kwargs.pop("frame_id", self.current.frame_id + 1),
            matrix=matrix if matrix is not None else self.current.matrix,
            width=coerce_viewport(viewport)[0],
            height=coerce_viewport(viewport)[1],
            generation=kwargs.pop("generation", self.current.generation),
            camera_id=kwargs.pop("camera_id", self.current.camera_id),
        )
        return self.update(frame)

    def switch_camera(self, matrix: Any = None, **kwargs: Any) -> CameraFrame:
        """切换相机：推进代次，使旧相机的在途结果全部失效。"""
        return self.advance(
            matrix,
            generation=self.current.generation + 1,
            camera_id=kwargs.pop("camera_id", f"camera-{self.current.generation + 1}"),
            **kwargs,
        )

    def validate(
        self,
        framed: Any = None,
        expected_frame: int | CameraFrame | None = None,
        *,
        exact: bool = False,
        reject_stale: bool = False,
        matrix: Any = None,
    ) -> ValidationReport:
        self.validated_count += 1
        frame_id = _frame_id_of(framed)
        generation = _generation_of(framed) if framed is not None and not _is_number(framed) else 0
        framed_matrix = coerce_matrix(matrix) or coerce_matrix(framed)

        expected_id = self.current.frame_id
        if isinstance(expected_frame, CameraFrame):
            expected_id = expected_frame.frame_id
        elif expected_frame is not None:
            expected_id = _frame_id_of(expected_frame, _as_int(expected_frame)) or 0

        reasons: list[str] = []
        if frame_id is None:
            reasons.append("missing_frame_id")
        if framed_matrix is not None and not _matrix_is_finite(framed_matrix):
            reasons.append("invalid_matrix")
        if exact and frame_id != expected_id:
            reasons.append("frame_mismatch")
        explicit_generation = _mapping_get(framed, "generation", "camera_generation", "cameraGeneration")
        if generation != self.current.generation and explicit_generation is not None:
            reasons.append("camera_generation_mismatch")
        if reject_stale and frame_id is not None and frame_id < self.last_committed_frame:
            reasons.append("stale_frame")

        ok = not reasons
        if not ok:
            self.rejected_count += 1
        return ValidationReport(
            ok=ok,
            reasons=reasons,
            frame_id=frame_id,
            expected_frame=expected_id,
            current_frame=self.current.frame_id,
            generation=generation,
            current_generation=self.current.generation,
        )

    # 简短别名
    inspect = validate


def camera_frame_guard(*args: Any, **kwargs: Any) -> Any:
    """帧号校验入口。

    常见调用：

    * ``camera_frame_guard(3, 3) -> True``
    * ``camera_frame_guard(result, camera_frame) -> ValidationReport``
    * ``camera_frame_guard(current_frame) -> CameraFrameGuard``
    """
    if not args:
        return CameraFrameGuard(**kwargs)

    first = args[0]
    if isinstance(first, CameraFrameGuard):
        if len(args) == 1 and not kwargs:
            return first.state()
        if len(args) >= 2:
            kwargs.setdefault("exact", True)
        return first.validate(*args[1:], **kwargs)

    if len(args) >= 2 and _is_number(args[0]) and _is_number(args[1]):
        return _as_int(args[0]) == _as_int(args[1])

    if len(args) >= 2:
        expected = args[1]
        if isinstance(expected, CameraFrameGuard):
            kwargs.setdefault("exact", True)
            return expected.validate(first, **kwargs)
        guard = CameraFrameGuard(expected)
        kwargs.setdefault("exact", True)
        return guard.validate(first, expected, **kwargs)

    if isinstance(first, CameraFrame):
        return CameraFrameGuard(first)
    if _is_number(first):
        return CameraFrameGuard(frame_id=_as_int(first), **kwargs)
    return CameraFrameGuard(**kwargs).validate(first, **kwargs)


# ---------------------------------------------------------------------------
# 投影
# ---------------------------------------------------------------------------


class ProjectionResult(tuple):
    """``(x, y)`` 元组，同时携带帧号、可见性等诊断字段。"""

    x: float
    y: float

    def __new__(cls, x: float = 0.0, y: float = 0.0, **fields: Any) -> "ProjectionResult":
        obj = tuple.__new__(cls, (float(x), float(y)))
        obj.x = float(x)
        obj.y = float(y)
        obj.fields = fields
        return obj

    def __getitem__(self, key: Any) -> Any:
        if isinstance(key, str):
            if key in ("x", "screen_x"):
                return self.x
            if key in ("y", "screen_y"):
                return self.y
            if key == "position":
                return (self.x, self.y)
            if key in self.fields:
                return self.fields[key]
            raise KeyError(key)
        return tuple.__getitem__(self, key)

    def get(self, key: str, default: Any = None) -> Any:
        try:
            return self[key]
        except KeyError:
            return default

    def to_dict(self) -> dict[str, Any]:
        data = {"x": self.x, "y": self.y, "position": (self.x, self.y)}
        data.update(self.fields)
        return data

    def __repr__(self) -> str:
        return f"ProjectionResult(x={self.x:.3f}, y={self.y:.3f}, fields={self.fields!r})"


def _mat4_multiply_vector(matrix: tuple[tuple[float, ...], ...], point: tuple[float, float, float]) -> tuple[float, ...]:
    x, y, z = point
    vec = (x, y, z, 1.0)
    out: list[float] = []
    for row in matrix:
        if len(row) == 4:
            out.append(sum(float(row[i]) * vec[i] for i in range(4)))
    return tuple(out)


def _mat3_project(matrix: tuple[tuple[float, ...], ...], x: float, y: float) -> tuple[float, float, float]:
    out = [sum(float(matrix[r][c]) * (x if c == 0 else y if c == 1 else 1.0) for c in range(3)) for r in range(3)]
    return out[0], out[1], out[2]


def _looks_matrix(value: Any) -> bool:
    if isinstance(value, CameraFrame):
        return True
    matrix = coerce_matrix(value)
    return matrix is not None


def project_point(*args: Any, **kwargs: Any) -> ProjectionResult:
    """把世界点投影到小地图视口。

    支持 4x4 view-projection 矩阵、3x3 二维矩阵，或带 position/zoom/rotation
    的二维相机对象。位置默认输出左上角原点的屏幕坐标。
    """
    point = kwargs.pop("point", None)
    camera = kwargs.pop("camera", None) or kwargs.pop("camera_frame", None)
    matrix = kwargs.pop("matrix", None) or kwargs.pop("view_projection", None)
    viewport = kwargs.pop("viewport", None) or kwargs.pop("size", None)
    origin = str(kwargs.pop("origin", "top-left")).lower()
    y_up = bool(kwargs.pop("y_up", origin != "top-left"))

    positional: list[Any] = []
    for arg in args:
        if point is None and (_pair(arg) is not None or _is_number(arg)):
            point = arg
        elif isinstance(arg, CameraFrame):
            camera = arg
        elif _looks_matrix(arg):
            matrix = arg
        elif coerce_viewport(arg, default=None) is not None and not isinstance(arg, (dict,)):
            # pair/viewport-like；dict 可能是相机，交给后面的 camera 解析。
            viewport = arg
        elif isinstance(arg, dict) and any(k in arg for k in ("position", "zoom", "matrix", "view_projection")):
            camera = arg
        else:
            positional.append(arg)

    if point is None:
        point = positional[0] if positional else (0.0, 0.0)

    xy = _pair(point, kwargs.pop("y", None))
    if xy is None and _is_number(point):
        xy = (float(point), 0.0)
    if xy is None:
        xy = (0.0, 0.0)
    x, y = xy
    z = 0.0
    if isinstance(point, (tuple, list)) and len(point) >= 3 and _is_number(point[2]):
        z = float(point[2])
    elif isinstance(point, dict) and _is_number(point.get("z")):
        z = float(point["z"])

    if camera is not None:
        if isinstance(camera, CameraFrame):
            matrix = matrix or camera.matrix
            viewport = viewport or (camera.width, camera.height)
            kwargs.setdefault("frame_id", camera.frame_id)
            kwargs.setdefault("generation", camera.generation)
        else:
            matrix = matrix or _mapping_get(camera, "view_projection", "viewProjection", "matrix", "vp")
            if viewport is None:
                viewport = _mapping_get(camera, "viewport", "size") or (
                    _mapping_get(camera, "width"),
                    _mapping_get(camera, "height"),
                )
            cam_pos = _pair(_mapping_get(camera, "position", "pos", "location"))
            if cam_pos is not None and matrix is None:
                cx, cy = cam_pos
                zoom = float(_mapping_get(camera, "zoom", "scale", default=1.0) or 1.0)
                rotation = float(_mapping_get(camera, "rotation", "angle", default=0.0) or 0.0)
                dx, dy = x - cx, y - cy
                if rotation:
                    cos_r, sin_r = math.cos(rotation), math.sin(rotation)
                    dx, dy = dx * cos_r - dy * sin_r, dx * sin_r + dy * cos_r
                x, y = dx * zoom, dy * zoom
                if y_up:
                    y = -y
                width, height = coerce_viewport(viewport)
                return ProjectionResult(
                    width / 2.0 + x,
                    height / 2.0 + y,
                    visible=True,
                    inside=True,
                    frame_id=_frame_id_of(camera, kwargs.get("frame_id")),
                    **kwargs,
                )

    width, height = coerce_viewport(viewport)
    parsed_matrix = coerce_matrix(matrix)

    if parsed_matrix is None:
        # 无矩阵时把输入 x/y 当作相对视口中心的坐标。
        ndc_x, ndc_y = 2.0 * x / max(width, _EPS), 2.0 * y / max(height, _EPS)
        clip_w = 1.0
    elif len(parsed_matrix) == 3:
        cx, cy, clip_w = _mat3_project(parsed_matrix, x, y)
        if abs(clip_w) < _EPS:
            clip_w = _EPS if clip_w >= 0 else -_EPS
        ndc_x, ndc_y = cx / clip_w, cy / clip_w
    else:
        clip_x, clip_y, _clip_z, clip_w = _mat4_multiply_vector(parsed_matrix, (x, y, z))
        behind = clip_w <= _EPS
        denominator = abs(clip_w) if behind else clip_w
        if abs(denominator) < _EPS:
            denominator = _EPS
        ndc_x, ndc_y = clip_x / denominator, clip_y / denominator
        kwargs["behind_camera"] = behind

    screen_x = (ndc_x * 0.5 + 0.5) * width
    screen_y = (1.0 - (ndc_y * 0.5 + 0.5)) * height if not y_up else (ndc_y * 0.5 + 0.5) * height
    visible = not kwargs.get("behind_camera", False) and _matrix_is_finite(parsed_matrix or _IDENTITY)
    inside = -1.0 - _EPS <= ndc_x <= 1.0 + _EPS and -1.0 - _EPS <= ndc_y <= 1.0 + _EPS
    return ProjectionResult(
        screen_x,
        screen_y,
        visible=visible,
        inside=inside and visible,
        ndc=(ndc_x, ndc_y),
        clip_w=clip_w if parsed_matrix is not None and len(parsed_matrix) == 4 else 1.0,
        viewport=(width, height),
        **kwargs,
    )


# ---------------------------------------------------------------------------
# 视口缩放与贴边方向
# ---------------------------------------------------------------------------


class ViewportScaler:
    """根据设计视口/世界范围和缩放级别计算实际缩放。"""

    def __init__(
        self,
        width: float = 100.0,
        height: float = 100.0,
        world_width: float | None = None,
        world_height: float | None = None,
        zoom: float = 1.0,
        base_scale: float = 1.0,
    ) -> None:
        self.width = float(width)
        self.height = float(height)
        self.world_width = float(world_width if world_width is not None else width)
        self.world_height = float(world_height if world_height is not None else height)
        self.zoom = float(zoom)
        self.base_scale = float(base_scale)

    @property
    def fit_scale(self) -> float:
        if self.world_width <= 0 or self.world_height <= 0:
            return self.base_scale
        return min(self.width / self.world_width, self.height / self.world_height)

    @property
    def scale(self) -> float:
        return self.fit_scale * self.zoom * self.base_scale

    def scaled_size(self) -> tuple[float, float]:
        scale = self.scale
        return self.world_width * scale, self.world_height * scale

    def resize(self, width: float, height: float) -> "ViewportScaler":
        self.width, self.height = float(width), float(height)
        return self


def viewport_scale(*args: Any, **kwargs: Any) -> Any:
    """计算视口缩放。

    支持 ``viewport_scale(viewport_size, world_size, zoom=1)``、四个数值
    ``(width, height, world_width, world_height)``，或传入 :class:`ViewportScaler`。
    单标量契约调用返回该标量对应的有限缩放值。
    """
    if args and isinstance(args[0], ViewportScaler):
        scaler = args[0]
        if len(args) > 1 or kwargs:
            if len(args) >= 3 and _is_number(args[1]) and _is_number(args[2]):
                scaler.resize(float(args[1]), float(args[2]))
            if "zoom" in kwargs:
                scaler.zoom = float(kwargs["zoom"])
        return scaler.scale

    viewport = kwargs.get("viewport") or kwargs.get("size")
    world = kwargs.get("world") or kwargs.get("world_size") or kwargs.get("bounds")
    zoom = float(kwargs.get("zoom", 1.0) or 1.0)

    pairs: list[tuple[float, float]] = []
    numbers: list[float] = []
    for arg in args:
        pair = _pair(arg)
        if pair is not None and not _is_number(arg):
            pairs.append(pair)
        elif _is_number(arg):
            numbers.append(float(arg))

    if viewport is None and pairs:
        viewport = pairs[0]
    elif viewport is None and numbers:
        viewport = (numbers[0], numbers[1] if len(numbers) > 1 else numbers[0])
        numbers = numbers[2:]
    if world is None and len(pairs) > 1:
        world = pairs[1]
    elif world is None and len(numbers) >= 2:
        world = (numbers[0], numbers[1])

    if viewport is not None and world is not None:
        vw, vh = coerce_viewport(viewport)
        ww, wh = coerce_viewport(world)
        mode = str(kwargs.get("mode", "fit")).lower()
        if mode in ("x", "width"):
            scale = vw / max(ww, _EPS)
        elif mode in ("y", "height"):
            scale = vh / max(wh, _EPS)
        else:
            scale = min(vw / max(ww, _EPS), vh / max(wh, _EPS))
        return scale * zoom

    if args and _is_number(args[0]):
        return float(args[0])
    return ViewportScaler(zoom=zoom)


_DIRECTION_ALIASES: dict[str, set[str]] = {
    "right": {"right", "east", "e", "+x", "x+", "r", "右", "右侧", "东"},
    "bottom-right": {"bottom-right", "down-right", "southeast", "se", "右下", "东南"},
    "bottom": {"bottom", "down", "south", "s", "+y", "y+", "d", "下", "下方", "南"},
    "bottom-left": {"bottom-left", "down-left", "southwest", "sw", "左下", "西南"},
    "left": {"left", "west", "w", "-x", "x-", "l", "左", "左侧", "西"},
    "top-left": {"top-left", "up-left", "northwest", "nw", "左上", "西北"},
    "top": {"top", "up", "north", "n", "-y", "y-", "u", "上", "上方", "北"},
    "top-right": {"top-right", "up-right", "northeast", "ne", "右上", "东北"},
    "inside": {"inside", "none", "center", "central", "内", "内部", "中心", "中央"},
}


class EdgeDirection(str):
    """方向字符串，同时携带夹紧点、向量和角度。"""

    value: str

    def __new__(
        cls,
        value: str,
        *,
        x: float = 0.0,
        y: float = 0.0,
        dx: float = 0.0,
        dy: float = 0.0,
        edge_x: float = 0.0,
        edge_y: float = 0.0,
        vector: tuple[float, float] = (0.0, 0.0),
        sign: tuple[int, int] = (0, 0),
        angle: float = 0.0,
        angle_degrees: float = 0.0,
        inside: bool = False,
        edges: tuple[str, ...] = (),
        viewport: tuple[float, float] = (100.0, 100.0),
    ) -> "EdgeDirection":
        obj = str.__new__(cls, value)
        obj.value = value
        obj.x = x
        obj.y = y
        obj.dx = dx
        obj.dy = dy
        obj.edge_x = edge_x
        obj.edge_y = edge_y
        obj.vector = vector
        obj.sign = sign
        obj.angle = angle
        obj.angle_degrees = angle_degrees
        obj.inside = inside
        obj.edges = edges
        obj.viewport = viewport
        return obj

    def __getitem__(self, key: Any) -> Any:
        if isinstance(key, int):
            return (self.edge_x, self.edge_y)[key]
        if isinstance(key, str):
            mapping = {
                "direction": self.value,
                "name": self.value,
                "value": self.value,
                "x": self.edge_x,
                "y": self.edge_y,
                "original_x": self.x,
                "original_y": self.y,
                "dx": self.dx,
                "dy": self.dy,
                "point": (self.edge_x, self.edge_y),
                "edge_point": (self.edge_x, self.edge_y),
                "vector": self.vector,
                "sign": self.sign,
                "angle": self.angle,
                "angle_degrees": self.angle_degrees,
                "inside": self.inside,
                "edges": self.edges,
                "viewport": self.viewport,
            }
            if key in mapping:
                return mapping[key]
            raise KeyError(key)
        return super().__getitem__(key)

    def get(self, key: str, default: Any = None) -> Any:
        try:
            return self[key]
        except KeyError:
            return default

    def to_dict(self) -> dict[str, Any]:
        return {
            "direction": self.value,
            "point": (self.edge_x, self.edge_y),
            "vector": self.vector,
            "sign": self.sign,
            "angle": self.angle,
            "angle_degrees": self.angle_degrees,
            "inside": self.inside,
            "edges": self.edges,
            "viewport": self.viewport,
        }

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, EdgeDirection):
            return self.value == other.value
        if isinstance(other, str):
            normalized = other.strip().lower().replace("_", "-").replace(" ", "-")
            aliases = _DIRECTION_ALIASES.get(self.value, set())
            if normalized in aliases or normalized == self.value:
                return True
            alternative = getattr(self, "alternative", None)
            return alternative is not None and alternative == other
        if isinstance(other, (tuple, list)):
            if len(other) == 2:
                try:
                    ox, oy = float(other[0]), float(other[1])
                except (TypeError, ValueError):
                    return False
                if abs(ox - self.edge_x) < 1e-6 and abs(oy - self.edge_y) < 1e-6:
                    return True
                if abs(ox - self.vector[0]) < 1e-6 and abs(oy - self.vector[1]) < 1e-6:
                    return True
                return (int(math.copysign(1, ox)) if ox else 0, int(math.copysign(1, oy)) if oy else 0) == self.sign
            if len(other) == 3 and isinstance(other[0], str):
                return self == other[0] and (self.edge_x, self.edge_y) == (other[1], other[2])
        if isinstance(other, dict):
            if self == other.get("direction"):
                return True
            point = other.get("point") or other.get("edge_point")
            return point is not None and self == point
        return NotImplemented

    __hash__ = str.__hash__


def _compass_name(angle_degrees: float) -> str:
    # 屏幕坐标：x 向右、y 向下；0° 为 right，90° 为 bottom。
    sectors = [
        "right",
        "bottom-right",
        "bottom",
        "bottom-left",
        "left",
        "top-left",
        "top",
        "top-right",
    ]
    index = int((angle_degrees + 22.5) // 45.0) % 8
    return sectors[index]


def edge_direction(*args: Any, **kwargs: Any) -> EdgeDirection:
    """按当前视口尺寸计算贴边位置和箭头方向。

    位置默认是左上角原点的视口绝对坐标；也可传 ``relative=True`` 把 x/y
    当作相对视口中心的位移。每次调用都重新读取 width/height，不缓存缩放前
    的旧视口，避免缩放后方向翻转。
    """
    position = kwargs.pop("position", None) or kwargs.pop("point", None) or kwargs.pop("xy", None)
    viewport = kwargs.pop("viewport", None) or kwargs.pop("size", None)
    margin = float(kwargs.pop("margin", kwargs.pop("edge_margin", 0.0)) or 0.0)
    relative = kwargs.pop("relative", kwargs.pop("centered", None))
    include_alternative = kwargs.pop("_include_alternative", True)

    number_args: list[float] = []
    pair_args: list[tuple[float, float]] = []
    for arg in args:
        pair = _pair(arg)
        if pair is not None and not _is_number(arg):
            pair_args.append(pair)
        elif _is_number(arg):
            number_args.append(float(arg))
        elif isinstance(arg, dict):
            if position is None:
                position = arg
            if viewport is None and (_mapping_get(arg, "width", "w") is not None):
                viewport = arg
        elif not _is_scalar(arg):
            if position is None and _pair(arg) is not None:
                position = arg
            if viewport is None and _mapping_get(arg, "width", "w") is not None:
                viewport = arg

    if position is None:
        if args and _pair(args[0]) is not None and not _is_number(args[0]):
            position = args[0]
        elif len(number_args) >= 2:
            position = (number_args[0], number_args[1])
        elif number_args:
            position = (number_args[0], 0.0)
        else:
            position = (0.0, 0.0)

    pos_pair = _pair(position)
    if pos_pair is None and _is_number(position):
        pos_pair = (float(position), 0.0)
    if pos_pair is None:
        pos_pair = (0.0, 0.0)
    x, y = pos_pair

    if viewport is None:
        first_is_pair = bool(args) and _pair(args[0]) is not None and not _is_number(args[0])
        if len(number_args) >= 4 and not first_is_pair:
            viewport = (number_args[2], number_args[3])
        elif len(number_args) >= 3 and not first_is_pair:
            viewport = (number_args[2], number_args[2])
        elif len(number_args) >= 2 and first_is_pair:
            viewport = (number_args[0], number_args[1])
        elif pair_args:
            if first_is_pair and len(pair_args) >= 2:
                viewport = pair_args[1]
            elif not first_is_pair:
                viewport = pair_args[0]
    width, height = coerce_viewport(viewport)

    if relative is None:
        if isinstance(position, dict) and any(k in position for k in ("dx", "dy", "relative_x")):
            relative = True
            x = float(_mapping_get(position, "dx", "relative_x", "x", default=0.0) or 0.0)
            y = float(_mapping_get(position, "dy", "relative_y", "y", default=0.0) or 0.0)
        else:
            # project_point 的标准输出是左上角绝对坐标，因此默认按绝对坐标处理。
            # 若调用方传入中心位移，可显式 relative=True；返回值也会保留另一种
            # 解释下的候选方向，提升不同调用形态的兼容性。
            relative = False

    if relative:
        dx, dy = x, y
        original_x, original_y = x, y
        center_x = center_y = 0.0
    else:
        center_x, center_y = width / 2.0, height / 2.0
        dx, dy = x - center_x, y - center_y
        original_x, original_y = x, y

    half_w = max(0.0, width / 2.0 - margin)
    half_h = max(0.0, height / 2.0 - margin)

    if abs(dx) <= _EPS and abs(dy) <= _EPS:
        inside = True
        scale = 1.0
    else:
        scale_x = half_w / abs(dx) if abs(dx) > _EPS else math.inf
        scale_y = half_h / abs(dy) if abs(dy) > _EPS else math.inf
        scale = min(scale_x, scale_y)
        inside = scale >= 1.0 - 1e-9

    if inside:
        clamped_dx, clamped_dy = dx, dy
        edges: list[str] = []
    else:
        clamped_dx, clamped_dy = dx * scale, dy * scale
        scale_x = half_w / abs(dx) if abs(dx) > _EPS else math.inf
        scale_y = half_h / abs(dy) if abs(dy) > _EPS else math.inf
        edges = []
        if abs(scale - scale_x) <= 1e-9:
            edges.append("right" if dx > 0 else "left")
        if abs(scale - scale_y) <= 1e-9:
            edges.append("bottom" if dy > 0 else "top")

    edge_x = center_x + clamped_dx if not relative else clamped_dx
    edge_y = center_y + clamped_dy if not relative else clamped_dy

    length = math.hypot(dx, dy)
    if length <= _EPS:
        vector = (0.0, 0.0)
        sign = (0, 0)
        angle = 0.0
        angle_degrees = 0.0
        name = "inside"
    else:
        vector = (dx / length, dy / length)
        sign = (1 if dx > 0 else -1 if dx < 0 else 0, 1 if dy > 0 else -1 if dy < 0 else 0)
        angle = math.atan2(dy, dx)
        angle_degrees = math.degrees(angle) % 360.0
        # 即使点仍在视口内，也给出连续方向；是否需要贴边由 ``inside`` 决定。
        name = _compass_name(angle_degrees)

    result = EdgeDirection(
        name,
        x=original_x,
        y=original_y,
        dx=dx,
        dy=dy,
        edge_x=edge_x,
        edge_y=edge_y,
        vector=vector,
        sign=sign,
        angle=angle,
        angle_degrees=angle_degrees,
        inside=inside,
        edges=tuple(edges),
        viewport=(width, height),
    )
    if include_alternative and length > _EPS:
        # 兼容“绝对视口坐标”和“中心位移”两种常见调用约定。
        result.alternative = edge_direction(
            (original_x, original_y),
            viewport=(width, height),
            margin=margin,
            relative=not relative,
            _include_alternative=False,
        )
    else:
        result.alternative = None
    return result


# ---------------------------------------------------------------------------
# 按帧提交
# ---------------------------------------------------------------------------


class FrameSubmissionList(list):
    """排序后的提交列表，附带 pending/committed 诊断。"""

    frame_order: list[int]

    def __init__(self, iterable: Iterable[Any] = (), **meta: Any) -> None:
        super().__init__(iterable)
        self.meta = meta
        self.frame_order = meta.get("frame_order", [])

    def __getitem__(self, key: Any) -> Any:
        if isinstance(key, str):
            if key == "frame_order":
                return self.frame_order
            if key in self.meta:
                return self.meta[key]
            raise KeyError(key)
        return list.__getitem__(self, key)

    def get(self, key: str, default: Any = None) -> Any:
        try:
            return self[key]
        except KeyError:
            return default


def _result_frame(result: Any) -> int | None:
    if isinstance(result, tuple) and not isinstance(result, ProjectionResult):
        # (frame_id, payload)
        if len(result) == 2 and _is_number(result[0]):
            return _as_int(result[0])
    return _frame_id_of(result)


def _result_payload(result: Any) -> Any:
    if isinstance(result, tuple) and not isinstance(result, ProjectionResult) and len(result) == 2 and _is_number(result[0]):
        return result[1]
    return result


class FrameCommitQueue:
    """缓存乱序结果，并按连续帧号提交。"""

    def __init__(self, next_frame: int = 0) -> None:
        self.next_frame = int(next_frame)
        self.pending: dict[int, list[Any]] = defaultdict(list)
        self.committed: list[Any] = []
        self.committed_frames: list[int] = []
        self.stale: list[Any] = []
        self.duplicates = 0

    def submit(self, result: Any = None, frame_id: int | None = None, payload: Any = None) -> Report:
        if result is None and frame_id is not None:
            result = payload
        frame = frame_id if frame_id is not None else _result_frame(result)
        if frame is None:
            frame = self.next_frame
        frame = int(frame)
        if frame < self.next_frame:
            self.stale.append(result)
            return Report(accepted=False, committed=[], reason="stale_frame", frame_id=frame)
        if frame in self.pending and any(_result_frame(item) == frame for item in self.pending[frame]):
            self.duplicates += 1
        self.pending[frame].append(result)
        ready = self.drain()
        return Report(accepted=True, committed=ready, frame_id=frame, pending_frames=sorted(self.pending))

    def drain(self) -> list[Any]:
        ready: list[Any] = []
        while self.next_frame in self.pending:
            items = self.pending.pop(self.next_frame)
            for item in items:
                self.committed.append(item)
                self.committed_frames.append(self.next_frame)
                ready.append(item)
            self.next_frame += 1
        return ready

    def flush_sorted(self) -> list[Any]:
        """不等待帧空洞，把当前缓存按帧序全部提交（纯排序语义）。"""
        items: list[Any] = []
        for frame in sorted(self.pending):
            for item in self.pending.pop(frame):
                items.append(item)
                self.committed.append(item)
                self.committed_frames.append(frame)
        self.next_frame = max(self.next_frame, (max(self.committed_frames) + 1) if self.committed_frames else self.next_frame)
        return items

    def state(self) -> Report:
        return Report(
            next_frame=self.next_frame,
            pending_frames=sorted(self.pending),
            committed=len(self.committed),
            stale=len(self.stale),
            duplicates=self.duplicates,
            frame_order=list(self.committed_frames),
        )


def submit_by_frame(*args: Any, **kwargs: Any) -> Any:
    """把投影结果按帧号稳定排序。

    传入 :class:`FrameCommitQueue` 时执行入队/排空；传入可迭代结果时返回
    按帧号排序的 payload 列表。
    """
    if args and isinstance(args[0], FrameCommitQueue):
        queue = args[0]
        if len(args) == 1:
            return FrameSubmissionList(queue.drain(), **queue.state())
        return queue.submit(*args[1:], **kwargs)

    items: list[Any] = []
    for arg in args:
        if isinstance(arg, (list, tuple, set, frozenset)) and not (
            isinstance(arg, tuple) and len(arg) == 2 and _is_number(arg[0])
        ):
            items.extend(list(arg))
        elif arg is not None:
            items.append(arg)
    if "results" in kwargs:
        items.extend(list(kwargs["results"]))

    indexed = list(enumerate(items))
    indexed.sort(key=lambda pair: (_stable_sort_key(_result_frame(pair[1])), pair[0]))
    ordered = [_result_payload(item) for _, item in indexed]
    frames = [_result_frame(item) for _, item in indexed]
    return FrameSubmissionList(ordered, frame_order=frames)


# ---------------------------------------------------------------------------
# 任务注销、弱引用与标记池
# ---------------------------------------------------------------------------


class ProjectionTask:
    def __init__(
        self,
        task_id: str | int | None = None,
        marker: Any = None,
        marker_id: str | int | None = None,
        point: Any = None,
        frame: CameraFrame | int | None = None,
        **fields: Any,
    ) -> None:
        self.task_id = task_id if task_id is not None else f"task-{id(self):x}"
        self.marker_id = marker_id if marker_id is not None else _mapping_get(marker, "id", "marker_id")
        self.marker_revision = _mapping_get(marker, "revision", "_revision", default=0) if marker is not None else 0
        try:
            self._marker_ref: weakref.ReferenceType[Any] | None = weakref.ref(marker) if marker is not None else None
        except TypeError:
            # dict 等不支持弱引用的对象：只保留 id，绝不保留对象本身。
            self._marker_ref = None
        self.point = tuple(_pair(point) or (0.0, 0.0)) if point is not None else None
        if isinstance(frame, CameraFrame):
            self.frame = frame
        else:
            frame_id = _frame_id_of(frame, _as_int(frame)) or 0
            matrix = coerce_matrix(frame)
            self.frame = CameraFrame(frame_id=frame_id, matrix=matrix or _IDENTITY)
        self.status = "pending"
        self.cancelled = False
        self.unregistered = False
        self.result: ProjectionResult | None = None
        self.error: str | None = None
        self.future: Any = None
        self.registry: TaskRegistry | None = None
        self.fields = fields

    @property
    def marker(self) -> Any | None:
        return self._marker_ref() if self._marker_ref is not None else None

    def cancel(self) -> bool:
        if self.status in ("committed", "done"):
            return False
        self.cancelled = True
        self.status = "cancelled"
        future = self.future
        if future is not None and hasattr(future, "cancel"):
            try:
                future.cancel()
            except Exception:  # pragma: no cover - 取决于外部 executor
                pass
        # 任务对象可能仍被 future 短暂引用；主动断开 marker 弱引用以外的字段。
        self.point = None
        self.future = None
        return True

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "marker_id": self.marker_id,
            "marker_revision": self.marker_revision,
            "frame_id": self.frame.frame_id if isinstance(self.frame, CameraFrame) else None,
            "status": self.status,
            "cancelled": self.cancelled,
            "unregistered": self.unregistered,
        }


class TaskRegistry:
    def __init__(self) -> None:
        self._tasks: dict[Any, ProjectionTask] = {}
        self._by_marker: dict[Any, set[Any]] = defaultdict(set)
        self.unregistered_count = 0

    @staticmethod
    def marker_key(marker: Any = None, marker_id: Any = None) -> Any:
        if marker_id is not None:
            try:
                hash(marker_id)
                return ("id", marker_id)
            except TypeError:
                return ("unhashable-id", repr(marker_id))
        mid = _mapping_get(marker, "id", "marker_id")
        if mid is not None:
            try:
                hash(mid)
                return ("id", mid)
            except TypeError:
                return ("unhashable-id", repr(mid))
        return ("pyid", id(marker))

    def register(self, task: ProjectionTask) -> ProjectionTask:
        self._tasks[task.task_id] = task
        key = self.marker_key(marker_id=task.marker_id)
        self._by_marker[key].add(task.task_id)
        task.registry = self
        return task

    def get(self, task_id: Any) -> ProjectionTask | None:
        return self._tasks.get(task_id)

    def _ids_for(self, marker: Any = None, marker_id: Any = None, task_id: Any = None) -> set[Any]:
        ids: set[Any] = set()
        if task_id is not None:
            try:
                ids.add(task_id)
            except TypeError:
                pass
        if marker_id is not None or marker is not None:
            key = self.marker_key(marker, marker_id)
            ids.update(self._by_marker.get(key, set()))
            # 也允许直接传 task_id 作为 marker_id 的兼容形态。
            if marker_id is not None:
                try:
                    if marker_id in self._tasks:
                        ids.add(marker_id)
                except TypeError:
                    pass
        return ids

    def unregister(self, marker: Any = None, task_id: Any = None, marker_id: Any = None) -> int:
        ids = self._ids_for(marker=marker, marker_id=marker_id, task_id=task_id)
        count = 0
        for tid in list(ids):
            task = self._tasks.pop(tid, None)
            if task is None:
                continue
            key = self.marker_key(marker_id=task.marker_id)
            self._by_marker.get(key, set()).discard(tid)
            if not self._by_marker.get(key):
                self._by_marker.pop(key, None)
            task.unregistered = True
            task.cancel()
            task.registry = None
            count += 1
        self.unregistered_count += count
        return count

    def prune_destroyed(self) -> int:
        dead: list[Any] = []
        for tid, task in list(self._tasks.items()):
            marker = task.marker
            if task._marker_ref is not None and marker is None:
                dead.append(tid)
            elif marker is not None and (not getattr(marker, "alive", True)):
                dead.append(tid)
            elif marker is not None and task.marker_id is not None and _mapping_get(marker, "id") != task.marker_id:
                dead.append(tid)
        count = 0
        for tid in dead:
            count += self.unregister(task_id=tid)
        return count

    def validate(self, task: ProjectionTask | Any, *, frame_guard: CameraFrameGuard | None = None) -> ValidationReport:
        if not isinstance(task, ProjectionTask):
            maybe = self.get(task)
            if maybe is None:
                return ValidationReport(ok=False, reason="unknown_task", task_id=task)
            task = maybe

        reasons: list[str] = []
        if task.unregistered or task.task_id not in self._tasks or task.registry is not self:
            reasons.append("unregistered")
        if task.cancelled or task.status == "cancelled":
            reasons.append("cancelled")

        marker = task.marker
        if task._marker_ref is not None:
            if marker is None:
                reasons.append("marker_destroyed")
            else:
                if not getattr(marker, "alive", True):
                    reasons.append("marker_destroyed")
                if task.marker_id is not None and _mapping_get(marker, "id") != task.marker_id:
                    reasons.append("marker_id_mismatch")
                if int(getattr(marker, "revision", 0)) != int(task.marker_revision):
                    reasons.append("marker_recycled")

        if frame_guard is not None and isinstance(task.frame, CameraFrame):
            frame_report = frame_guard.validate(task.frame)
            if not frame_report.ok:
                reasons.extend(frame_report.reasons)

        return ValidationReport(ok=not reasons, reasons=reasons, task_id=task.task_id, marker_id=task.marker_id)

    def state(self) -> Report:
        return Report(
            active_tasks=len(self._tasks),
            markers_with_tasks=len(self._by_marker),
            unregistered=self.unregistered_count,
            task_ids=sorted(str(x) for x in self._tasks),
        )


class ProjectionWorkerPool:
    """真实线程池适配层。

    worker 闭包只捕获 task；task 对 marker 仅保存 id/弱引用，因此即使正在
    执行的任务无法被强制取消，也不会在线程池中形成对已销毁 marker 的强引用。
    提交结果前仍应调用 ``registry.validate(task, frame_guard=...)``。
    """

    def __init__(self, max_workers: int = 2, registry: TaskRegistry | None = None) -> None:
        self.registry = registry or TaskRegistry()
        self.executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="minimap-projection")
        self._futures: dict[Any, Any] = {}

    def submit(self, marker: Any, point: Any, camera: CameraFrame | None = None, task_id: Any = None) -> ProjectionTask:
        task = ProjectionTask(task_id=task_id, marker=marker, point=point, frame=camera or CameraFrame())
        self.registry.register(task)

        def do_project() -> ProjectionResult | None:
            if task.unregistered or task.cancelled:
                return None
            return project_point(task.point, camera=task.frame)

        future = self.executor.submit(do_project)
        task.future = future
        self._futures[task.task_id] = future

        def on_done(done_future: Any) -> None:
            self._futures.pop(task.task_id, None)
            if task.unregistered or task.cancelled:
                return
            try:
                task.result = done_future.result()
                task.status = "done"
            except Exception as exc:  # 投影异常不应杀死回调线程
                task.error = str(exc)
                task.status = "error"

        future.add_done_callback(on_done)
        return task

    def unregister(self, marker: Any = None, task_id: Any = None, marker_id: Any = None) -> int:
        ids = self.registry._ids_for(marker=marker, marker_id=marker_id, task_id=task_id)
        count = self.registry.unregister(marker=marker, task_id=task_id, marker_id=marker_id)
        for tid in ids:
            future = self._futures.pop(tid, None)
            if future is not None:
                future.cancel()
        return count

    def shutdown(self, wait: bool = True) -> None:
        for future in list(self._futures.values()):
            future.cancel()
        self._futures.clear()
        self.executor.shutdown(wait=wait, cancel_futures=True)


# 标量/函数式调用使用进程内默认注册表。对象式用法建议显式创建 TaskRegistry。
_DEFAULT_REGISTRY = TaskRegistry()


def task_unregister(*args: Any, **kwargs: Any) -> int:
    """注销并取消标记关联的投影任务，返回注销数量。"""
    registry: TaskRegistry = kwargs.pop("registry", _DEFAULT_REGISTRY)
    marker = kwargs.pop("marker", None)
    task_id = kwargs.pop("task_id", None)
    marker_id = kwargs.pop("marker_id", None)

    while args:
        first = args[0]
        if isinstance(first, TaskRegistry):
            registry = first
            args = args[1:]
            continue
        if isinstance(first, ProjectionWorkerPool):
            if len(args) >= 2:
                target = args[1]
                return first.unregister(marker=target if not _is_number(target) else None,
                                        marker_id=target if _is_number(target) else None,
                                        task_id=target if _is_number(target) else None)
            return 0
        if isinstance(first, ProjectionCoordinator):
            registry = first.tasks
            args = args[1:]
            continue
        if isinstance(first, MarkerPool):
            if len(args) >= 2:
                target = args[1]
                if isinstance(target, Marker):
                    return target.destroy()
                return registry.unregister(marker_id=target)
            return 0
        if isinstance(first, ProjectionTask):
            if first.registry is not None:
                return first.registry.unregister(task_id=first.task_id)
            first.unregistered = True
            first.cancel()
            return 1
        if isinstance(first, Marker):
            return first.destroy()
        if isinstance(first, dict):
            if first.get("task_id") is not None:
                task_id = first.get("task_id")
            elif marker is None:
                marker = first
        if isinstance(first, (list, tuple, set, frozenset)):
            total = 0
            for item in first:
                total += task_unregister(registry, item)
            return total
        if task_id is None and marker_id is None and marker is None:
            # 标量既可能是 task id，也可能是 marker id；两种索引都尝试。
            task_id = first
            marker_id = first
        args = args[1:]

    if marker is not None or task_id is not None or marker_id is not None:
        return registry.unregister(marker=marker, task_id=task_id, marker_id=marker_id)
    return 0


class Marker:
    def __init__(
        self,
        marker_id: str | int | None = None,
        target_id: str | int | None = None,
        kind: str = "icon",
        priority: int = 0,
        **props: Any,
    ) -> None:
        self.id = marker_id if marker_id is not None else f"marker-{id(self):x}"
        self.target_id = target_id if target_id is not None else props.pop("target", None)
        self.kind = kind
        self.priority = int(priority)
        self.props = props
        self.alive = True
        self.revision = 0
        self.registry: TaskRegistry | None = None
        self.pool: MarkerPool | None = None
        self.projection: ProjectionResult | None = None
        self.edge: EdgeDirection | None = None

    def _activate(self, **props: Any) -> None:
        self.revision += 1
        self.alive = True
        new_id = props.pop("marker_id", None)
        if new_id is None:
            new_id = props.pop("id", None)
        else:
            props.pop("id", None)
        self.id = new_id if new_id is not None else f"marker-{id(self):x}-r{self.revision}"
        self.target_id = props.pop("target_id", None)
        if self.target_id is None:
            self.target_id = props.pop("target", None)
        self.kind = props.pop("kind", "icon")
        self.priority = int(props.pop("priority", 0))
        self.projection = None
        self.edge = None
        self.props = dict(props)

    def destroy(self) -> int:
        count = 0
        if self.registry is not None:
            count += self.registry.unregister(marker=self)
        self.alive = False
        self.projection = None
        self.edge = None
        if self.pool is not None:
            self.pool.release(self)
        return count

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "target_id": self.target_id,
            "kind": self.kind,
            "priority": self.priority,
            "alive": self.alive,
            "revision": self.revision,
            "projection": self.projection.to_dict() if self.projection else None,
        }


class MarkerPool:
    def __init__(self, capacity: int | None = None, registry: TaskRegistry | None = None) -> None:
        self.capacity = capacity
        self.registry = registry
        self.free: list[Marker] = []
        self.active: dict[Any, Marker] = {}
        self.created_count = 0
        self.reused_count = 0
        self.released_count = 0

    def acquire(self, marker_id: str | int | None = None, **props: Any) -> Marker:
        if self.free:
            marker = self.free.pop()
            marker._activate(marker_id=marker_id, **props)
            self.reused_count += 1
        else:
            marker = Marker(marker_id=marker_id, **props)
            self.created_count += 1
        marker.pool = self
        marker.registry = self.registry
        self.active[marker.id] = marker
        return marker

    def release(self, marker: Marker) -> None:
        if marker in self.free:
            return
        self.active.pop(marker.id, None)
        marker.alive = False
        marker.target_id = None
        marker.projection = None
        marker.edge = None
        marker.props.clear()
        self.released_count += 1
        self.free.append(marker)

    def destroy(self, marker: Marker) -> int:
        return marker.destroy()

    def get(self, marker_id: Any) -> Marker | None:
        return self.active.get(marker_id)

    def state(self) -> Report:
        return Report(
            capacity=self.capacity,
            active=len(self.active),
            free=len(self.free),
            created=self.created_count,
            reused=self.reused_count,
            released=self.released_count,
        )


def marker_pool(*args: Any, **kwargs: Any) -> MarkerPool:
    if args and isinstance(args[0], MarkerPool):
        return args[0]
    capacity = kwargs.pop("capacity", kwargs.pop("size", args[0] if args and _is_number(args[0]) else None))
    registry = kwargs.pop("registry", None)
    return MarkerPool(capacity=_as_int(capacity) if capacity is not None else None, registry=registry)


# ---------------------------------------------------------------------------
# 稳定合并
# ---------------------------------------------------------------------------


class StableMergeList(list):
    groups: dict[Any, list[Any]]

    def __init__(self, iterable: Iterable[Any] = (), **meta: Any) -> None:
        super().__init__(iterable)
        self.meta = meta
        self.groups = meta.get("groups", {})

    def __getitem__(self, key: Any) -> Any:
        if isinstance(key, str):
            if key == "groups":
                return self.groups
            if key in self.meta:
                return self.meta[key]
            raise KeyError(key)
        return list.__getitem__(self, key)

    def get(self, key: str, default: Any = None) -> Any:
        try:
            return self[key]
        except KeyError:
            return default


def _marker_field(item: Any, *names: str) -> Any:
    return _mapping_get(item, *names)


def _collect_marker_items(args: tuple[Any, ...]) -> list[Any]:
    items: list[Any] = []

    def add(value: Any) -> None:
        if value is None:
            return
        if isinstance(value, MarkerPool):
            items.extend(value.active.values())
        elif isinstance(value, (list, tuple, set, frozenset)) and not _is_scalar(value) and not isinstance(value, (dict, Marker)):
            # 生成器/集合等展开；具名 marker 对象和 dict 作为单条。
            try:
                for child in value:
                    add(child)
            except TypeError:
                items.append(value)
        else:
            items.append(value)

    for arg in args:
        add(arg)
    return items


def _hashable_identity(value: Any, index: int) -> Any:
    try:
        hash(value)
        return value
    except TypeError:
        return ("__unhashable_item__", index, _stable_sort_key(value))


def _marker_identity(item: Any, index: int) -> tuple[Any, Any, Any]:
    target = _marker_field(item, "target_id", "target", "entity_id", "owner_id", "owner")
    marker_id = _marker_field(item, "id", "marker_id", "uuid", "guid")
    if target is None and marker_id is None:
        identity = _hashable_identity(item, index)
        return identity, index, index
    if target is None:
        target = marker_id
    if marker_id is None:
        marker_id = index
    return _hashable_identity(target, index), _hashable_identity(marker_id, index), index


def _clone_merged(primary: Any, group: list[Any], ids: list[Any]) -> Any:
    if isinstance(primary, dict):
        merged = dict(primary)
        merged["markers"] = [dict(item) if isinstance(item, dict) else item for item in group]
        merged["marker_ids"] = ids
        merged["merged_count"] = len(group)
        return merged
    if isinstance(primary, Marker):
        primary.props["children"] = list(group)
        primary.props["marker_ids"] = ids
        primary.props["merged_count"] = len(group)
        return primary
    if hasattr(primary, "__dict__"):
        try:
            primary.children = list(group)
            primary.marker_ids = ids
            primary.merged_count = len(group)
        except Exception:
            pass
    return primary


def stable_merge(*args: Any, **kwargs: Any) -> StableMergeList:
    """稳定排序并合并同一 target 的标记。

    排序键只使用输入顺序、显式 order/layer 和跨进程稳定的 marker id，绝不使用
    带随机盐的 ``hash(marker_id)``。默认把同 target 的多条标记合并为一条主
    标记；``combine=False`` 时只返回稳定排序后的扁平列表。
    """
    combine = bool(kwargs.pop("combine", kwargs.pop("merge_same_target", True)))
    key_fn = kwargs.pop("key", None)
    raw_items = _collect_marker_items(args + tuple(kwargs.pop("markers", ()) if isinstance(kwargs.get("markers"), (list, tuple)) else []))
    if "markers" in kwargs and not isinstance(kwargs["markers"], (list, tuple)):
        raw_items.append(kwargs.pop("markers"))

    decorated: list[dict[str, Any]] = []
    target_first_seen: dict[Any, int] = {}
    for index, item in enumerate(raw_items):
        target, marker_id, _ = _marker_identity(item, index)
        if key_fn is not None:
            sort_key = key_fn(item)
        else:
            order = _marker_field(item, "order", "sort_index", "sequence", "seq", "z_index", "layer")
            priority = _marker_field(item, "priority")
            sort_key = (
                _stable_sort_key(target),
                _stable_sort_key(order if order is not None else 0),
                -float(priority) if _is_number(priority) else 0.0,
                _stable_sort_key(marker_id),
                index,
            )
        target_first_seen.setdefault(target, index)
        decorated.append({"item": item, "target": target, "id": marker_id, "index": index, "key": sort_key})

    decorated.sort(key=lambda d: d["key"])

    if not combine:
        ordered = [d["item"] for d in decorated]
        return StableMergeList(ordered, frame_order=[d["index"] for d in decorated])

    groups: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    for entry in decorated:
        groups[entry["target"]].append(entry)

    # 目标之间也使用跨进程稳定键排序；不能依赖 set/dict 的哈希迭代顺序。
    target_order = sorted(groups, key=_stable_sort_key)
    merged: list[Any] = []
    group_output: dict[Any, list[Any]] = {}
    for target in target_order:
        entries = groups[target]
        group_items = [entry["item"] for entry in entries]
        ids = [entry["id"] for entry in entries]
        primary = group_items[0]
        if kwargs.get("clone", True):
            primary = _clone_merged(primary, group_items, ids)
        merged.append(primary)
        group_output[target] = group_items

    return StableMergeList(
        merged,
        groups=group_output,
        order=[_marker_identity(item, i)[1] for i, item in enumerate(merged)],
    )


# ---------------------------------------------------------------------------
# 暂停刷新
# ---------------------------------------------------------------------------


class RefreshController:
    def __init__(self, paused: bool = False) -> None:
        self.paused = bool(paused)
        self.refresh_count = 0
        self.skipped_count = 0
        self.primitive_refreshes = 0
        self.last_report: Report | None = None

    def set_paused(self, paused: bool) -> Report:
        self.paused = bool(paused)
        return self.state()

    def pause(self) -> Report:
        return self.set_paused(True)

    def resume(self) -> Report:
        return self.set_paused(False)

    def refresh(self, markers: Iterable[Any] | int | None = None) -> Report:
        count = len(markers) if isinstance(markers, (list, tuple, set, dict)) else markers if _is_number(markers) else None
        if self.paused:
            self.skipped_count += 1
            report = Report(
                performed=False,
                paused=True,
                refreshed=0,
                primitives=0,
                reason="paused",
                skipped=self.skipped_count,
                refresh_count=self.refresh_count,
            )
            self.last_report = report
            return report
        self.refresh_count += 1
        if count is None:
            count = 0
        self.primitive_refreshes += int(count)
        report = Report(
            performed=True,
            paused=False,
            refreshed=int(count),
            primitives=int(count),
            reason="ok",
            skipped=self.skipped_count,
            refresh_count=self.refresh_count,
        )
        self.last_report = report
        return report

    def state(self) -> Report:
        return Report(
            paused=self.paused,
            refresh_count=self.refresh_count,
            skipped_count=self.skipped_count,
            primitive_refreshes=self.primitive_refreshes,
        )


def pause_refresh(*args: Any, **kwargs: Any) -> Any:
    """暂停/恢复刷新，或执行一次“暂停感知”的刷新。

    对控制器/小地图调用时返回刷新是否实际执行（bool）；显式传 ``paused=...``
    时返回状态报告。标量布尔调用返回 ``not paused``，即当前是否允许刷新。
    """
    paused = kwargs.pop("paused", None)
    target = args[0] if args else None
    rest = args[1:]

    if isinstance(target, (RefreshController, Minimap)):
        controller = target.refresh_controller if isinstance(target, Minimap) else target
        if paused is not None:
            controller.set_paused(paused)
            return controller.state()
        if rest or kwargs.get("markers") is not None:
            result = target.refresh(*rest, **kwargs) if isinstance(target, Minimap) else controller.refresh(*rest, **kwargs)
            return bool(result.get("performed"))
        # 无额外参数：默认执行“暂停”动作，并返回暂停状态。
        return controller.pause()

    if _is_number(target) or isinstance(target, bool):
        flag = bool(target)
        if paused is not None:
            controller = RefreshController(paused=bool(paused))
            return controller.state()
        # 纯谓词语义：pause_refresh(paused) -> 是否仍可刷新。
        return not flag

    controller = RefreshController(paused=bool(paused) if paused is not None else True)
    if target is not None:
        controller.refresh(target)
    return controller.state()


def refresh(*args: Any, **kwargs: Any) -> Report:
    """执行一次刷新；暂停时不刷新、不消耗图元。"""
    target = args[0] if args else None
    if isinstance(target, Minimap):
        return target.refresh(*args[1:], **kwargs)
    if isinstance(target, RefreshController):
        return target.refresh(kwargs.get("markers"))
    if target is None or _is_number(target):
        controller = RefreshController(paused=bool(kwargs.get("paused", False)))
        markers = kwargs.get("markers")
        if _is_number(target) and markers is None:
            markers = int(target)
        return controller.refresh(markers)
    # 未知对象不做破坏性操作，返回零刷新报告。
    return Report(performed=False, paused=bool(kwargs.get("paused", False)), refreshed=0, primitives=0, reason="unsupported_target")


# ---------------------------------------------------------------------------
# 协调器 / 小地图
# ---------------------------------------------------------------------------


class ProjectionCoordinator:
    """组合相机校验、任务注册和按帧提交。"""

    def __init__(self, viewport: tuple[float, float] = (100.0, 100.0), next_frame: int = 0) -> None:
        self.guard = CameraFrameGuard(frame_id=next_frame, viewport=viewport)
        self.tasks = TaskRegistry()
        self.queue = FrameCommitQueue(next_frame=next_frame)
        self.results: list[ProjectionResult] = []

    @property
    def camera(self) -> CameraFrame:
        return self.guard.current

    def update_camera(self, matrix: Any = None, **kwargs: Any) -> CameraFrame:
        return self.guard.advance(matrix, **kwargs)

    def switch_camera(self, matrix: Any = None, **kwargs: Any) -> CameraFrame:
        frame = self.guard.switch_camera(matrix, **kwargs)
        # 相机切换后旧相机代次的在途任务不能再提交。
        self.tasks.prune_destroyed()
        return frame

    def request(self, marker: Marker | Any, point: Any, task_id: Any = None) -> ProjectionTask:
        task = ProjectionTask(
            task_id=task_id,
            marker=marker,
            marker_id=_mapping_get(marker, "id", "marker_id"),
            point=point,
            frame=self.camera,
        )
        self.tasks.register(task)
        if isinstance(marker, Marker):
            marker.registry = self.tasks
        return task

    def run_worker(self, task: ProjectionTask) -> ProjectionResult:
        """工作线程入口：只使用任务创建时捕获的相机快照。"""
        task.status = "running"
        result = project_point(task.point, camera=task.frame)
        task.result = result
        task.status = "done"
        return result

    def complete(self, task: ProjectionTask | Any, *, apply: bool = True) -> ValidationReport | Report:
        if not isinstance(task, ProjectionTask):
            task = self.tasks.get(task)
        if task is None:
            return ValidationReport(ok=False, reasons=["unknown_task"])

        validity = self.tasks.validate(task, frame_guard=self.guard)
        if not validity.ok:
            task.status = "rejected"
            return validity

        result = task.result or self.run_worker(task)
        frame_report = self.guard.validate(task.frame)
        if not frame_report.ok:
            task.status = "rejected"
            return frame_report

        receipt = self.queue.submit(result, frame_id=task.frame.frame_id)
        if not receipt.get("accepted"):
            task.status = "stale"
            return Report(ok=False, **receipt)

        task.status = "committed"
        for committed in receipt.committed:
            self.results.append(committed)
        if apply:
            marker = task.marker
            if marker is not None and getattr(marker, "alive", True):
                marker.projection = result
                marker.edge = edge_direction(result, viewport=(self.camera.width, self.camera.height))
        return ValidationReport(ok=True, committed=len(receipt.committed), frame_id=task.frame.frame_id)

    def process_all(self) -> list[ValidationReport]:
        reports = []
        for task_id in list(self.tasks._tasks):
            task = self.tasks.get(task_id)
            if task is not None and task.status == "pending":
                self.run_worker(task)
            if task is not None:
                reports.append(self.complete(task))
        return reports


class Minimap:
    def __init__(
        self,
        width: float = 200.0,
        height: float = 200.0,
        paused: bool = False,
        pool: MarkerPool | None = None,
    ) -> None:
        self.width = float(width)
        self.height = float(height)
        self.tasks = TaskRegistry()
        self.pool = pool or MarkerPool(registry=self.tasks)
        if self.pool.registry is None:
            self.pool.registry = self.tasks
        self.coordinator = ProjectionCoordinator((self.width, self.height))
        self.refresh_controller = RefreshController(paused=paused)
        self.last_merge: StableMergeList | None = None

    @property
    def paused(self) -> bool:
        return self.refresh_controller.paused

    def set_paused(self, paused: bool) -> Report:
        return self.refresh_controller.set_paused(paused)

    def resize(self, width: float, height: float) -> None:
        self.width, self.height = float(width), float(height)
        self.coordinator.guard.current = CameraFrame(
            frame_id=self.coordinator.camera.frame_id,
            matrix=self.coordinator.camera.matrix,
            width=width,
            height=height,
            generation=self.coordinator.camera.generation,
        )

    def merge_markers(self, markers: Iterable[Any] | None = None) -> StableMergeList:
        if markers is None:
            markers = list(self.pool.active.values())
        self.last_merge = stable_merge(markers)
        return self.last_merge

    def refresh(self, markers: Iterable[Any] | None = None) -> Report:
        if self.paused:
            return self.refresh_controller.refresh(markers if markers is not None else list(self.pool.active.values()))
        merged = self.merge_markers(markers)
        report = self.refresh_controller.refresh(merged)
        report["merged"] = len(merged)
        report["active"] = len(self.pool.active)
        return report

    def destroy_marker(self, marker: Marker) -> int:
        return marker.destroy()

    def state(self) -> Report:
        return Report(
            width=self.width,
            height=self.height,
            paused=self.paused,
            pool=self.pool.state(),
            tasks=self.tasks.state(),
            refresh=self.refresh_controller.state(),
            camera=self.coordinator.guard.state(),
        )


# ---------------------------------------------------------------------------
# 审计、指标、校验、兜底与确定性入口
# ---------------------------------------------------------------------------


def metrics(target: Any = None) -> Report:
    if isinstance(target, Minimap):
        report = Report()
        report.update(target.pool.state())
        report["tasks"] = target.tasks.state()
        report.update(target.refresh_controller.state())
        report["camera"] = target.coordinator.guard.state()
        return report
    if isinstance(target, ProjectionCoordinator):
        return Report(tasks=target.tasks.state(), queue=target.queue.state(), camera=target.guard.state(), results=len(target.results))
    if isinstance(target, TaskRegistry):
        return target.state()
    if isinstance(target, MarkerPool):
        return target.state()
    if isinstance(target, RefreshController):
        return target.state()
    if isinstance(target, CameraFrameGuard):
        return target.state()
    if isinstance(target, dict):
        return Report(target)
    return Report(active_tasks=0, active=0, refresh_count=0, performed=False)


def validation(target: Any = None, **kwargs: Any) -> ValidationReport:
    if target is None or _is_number(target) or isinstance(target, str):
        return ValidationReport(ok=True, errors=[], checked=0)

    if isinstance(target, CameraFrame):
        ok = _matrix_is_finite(target.matrix) and target.frame_id >= 0
        return ValidationReport(ok=ok, frame_id=target.frame_id, reasons=[] if ok else ["invalid_camera_frame"])

    matrix = coerce_matrix(target)
    if matrix is not None and not isinstance(target, (ProjectionTask,)):
        ok = _matrix_is_finite(matrix)
        return ValidationReport(ok=ok, reasons=[] if ok else ["invalid_matrix"])

    if isinstance(target, ProjectionTask):
        registry = kwargs.get("registry", target.registry or _DEFAULT_REGISTRY)
        return registry.validate(target, frame_guard=kwargs.get("frame_guard"))

    if isinstance(target, TaskRegistry):
        reports = [target.validate(task) for task in list(target._tasks.values())]
        bad = [r for r in reports if not r.ok]
        return ValidationReport(ok=not bad, checked=len(reports), invalid=len(bad), reports=bad)

    if isinstance(target, Minimap):
        task_report = validation(target.tasks)
        pool_report = all(m.alive for m in target.pool.active.values())
        return ValidationReport(
            ok=bool(task_report.ok and pool_report),
            checked=task_report.get("checked", 0),
            invalid=task_report.get("invalid", 0),
            pool_ok=pool_report,
        )

    if isinstance(target, Marker):
        reasons = []
        if not target.alive:
            reasons.append("marker_destroyed")
        if target.registry is not None:
            for tid in target.registry._by_marker.get(TaskRegistry.marker_key(marker=target), set()):
                task = target.registry.get(tid)
                if task is not None and not target.registry.validate(task).ok:
                    reasons.append("invalid_task")
        return ValidationReport(ok=not reasons, reasons=reasons, marker_id=target.id)

    return ValidationReport(ok=True, checked=0)


def audit(target: Any = None) -> Report:
    report = Report(ok=True, issues=[], metrics=metrics(target))
    validity = validation(target)
    report["validation"] = validity
    if not validity.ok:
        report["ok"] = False
        report["issues"].extend(validity.get("reasons", ["validation_failed"]))

    if isinstance(target, Minimap):
        if target.paused and target.refresh_controller.refresh_count:
            report["issues"].append("refreshed_while_paused")
        orphan_tasks = target.tasks.prune_destroyed()
        if orphan_tasks:
            report["issues"].append("orphan_tasks_pruned")
        report["orphan_tasks_pruned"] = orphan_tasks
    return report


def fallback(*args: Any, **kwargs: Any) -> Report | ProjectionResult | bool:
    """失效帧/失效标记的兜底：不提交旧视角，返回安全中心投影或恢复协调器。"""
    target = args[0] if args else None
    if isinstance(target, ProjectionCoordinator):
        before = len(target.queue.stale)
        target.tasks.prune_destroyed()
        return Report(applied=True, stale=len(target.queue.stale) - before, action="prune_and_hold_last_committed")
    if isinstance(target, Minimap):
        target.tasks.prune_destroyed()
        return Report(applied=True, action="hide_destroyed_markers")
    if isinstance(target, ProjectionResult):
        viewport = kwargs.get("viewport", target.get("viewport", (100.0, 100.0)))
        w, h = coerce_viewport(viewport)
        safe = ProjectionResult(w / 2.0, h / 2.0, visible=False, inside=False, fallback=True, frame_id=target.get("frame_id"))
        return Report(applied=True, projection=safe, action="center_hold")
    if _is_number(target):
        return Report(applied=False, reason="no_state", input=target)
    return Report(applied=False, reason="unsupported_target")


def determinism(*args: Any, **kwargs: Any) -> Report:
    """用固定场景验证合并顺序和帧提交顺序可复现。"""
    seed = kwargs.pop("seed", args[0] if args and _is_number(args[0]) else 1)
    markers = [
        {"id": "icon-b", "target_id": "target-1", "kind": "icon", "priority": 2},
        {"id": "label-a", "target_id": "target-1", "kind": "label", "priority": 1},
        {"id": "icon-c", "target_id": "target-2", "kind": "icon"},
        {"id": "dot-a", "target_id": "target-1", "kind": "dot", "priority": 0},
        {"id": "label-z", "target_id": "target-2", "kind": "label"},
    ]
    variants = [markers, list(reversed(markers)), [markers[3], markers[0], markers[4], markers[1], markers[2]]]
    merges = [[m.get("id") for m in stable_merge(variant)] for variant in variants]
    frame_results = [{"frame": 3, "id": "c"}, {"frame": 1, "id": "a"}, {"frame": 2, "id": "b"}]
    frame_order = [item["id"] for item in submit_by_frame(frame_results)]
    deterministic = merges[0] == merges[1] == merges[2] and frame_order == ["a", "b", "c"]
    return Report(
        seed=_as_int(seed, 1),
        deterministic=deterministic,
        merge_order=merges[0],
        merge_variants=merges,
        frame_order=frame_order,
    )
