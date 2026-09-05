"""Sparse off-plane audit. Geometry remains the measurement source."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

import numpy as np

from ai.calibration import CalibrationMode, CalibrationState
from ai.contracts import TrackResult

OFF_PLANE_WARN_M = 0.03
OFF_PLANE_RATIO = 0.04
AUDIT_STRIDE = 8


class DepthEstimator(Protocol):
    name: str

    def predict(self, rgb: np.ndarray) -> np.ndarray:
        """Metric depth in metres, shape (H, W)."""


@dataclass
class OffPlaneReading:
    frame: int
    z_geom: float | None = None
    z_model: float | None = None
    residual_m: float | None = None
    confidence: float = 0.0
    flag: str = "skipped"

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame": self.frame,
            "z_geom": self.z_geom,
            "z_model": self.z_model,
            "residual_m": self.residual_m,
            "confidence": self.confidence,
            "flag": self.flag,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "OffPlaneReading | None":
        if not data:
            return None
        return cls(
            frame=int(data.get("frame", 0)),
            z_geom=None if data.get("z_geom") is None else float(data["z_geom"]),
            z_model=None if data.get("z_model") is None else float(data["z_model"]),
            residual_m=None if data.get("residual_m") is None else float(data["residual_m"]),
            confidence=float(data.get("confidence") or 0.0),
            flag=str(data.get("flag") or "skipped"),
        )


@dataclass
class DepthAuditState:
    enabled: bool = False
    experimental_correction: bool = False
    model_name: str = ""
    available: bool = False
    message: str = ""
    threshold_m: float = OFF_PLANE_WARN_M
    readings: dict[int, OffPlaneReading] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "experimental_correction": self.experimental_correction,
            "model_name": self.model_name,
            "available": self.available,
            "message": self.message,
            "threshold_m": self.threshold_m,
            "readings": {str(k): v.to_dict() for k, v in self.readings.items()},
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "DepthAuditState":
        if not data:
            return cls()
        readings: dict[int, OffPlaneReading] = {}
        for key, raw in (data.get("readings") or {}).items():
            item = OffPlaneReading.from_dict(raw)
            if item is not None:
                readings[int(key)] = item
        return cls(
            enabled=bool(data.get("enabled", False)),
            experimental_correction=bool(data.get("experimental_correction", False)),
            model_name=str(data.get("model_name") or ""),
            available=bool(data.get("available", False)),
            message=str(data.get("message") or ""),
            threshold_m=float(data.get("threshold_m") or OFF_PLANE_WARN_M),
            readings=readings,
        )

    def off_plane_frames(self) -> list[int]:
        return sorted(frame for frame, item in self.readings.items() if item.flag == "off_plane")


FrameLoader = Callable[[int], np.ndarray]


def select_audit_frames(result: TrackResult, stride: int = AUDIT_STRIDE) -> list[int]:
    visible = [p for p in result.points if p.visible]
    if not visible:
        return []
    chosen: set[int] = {visible[0].frame, visible[-1].frame}
    for point in visible:
        if point.confidence < 0.6:
            chosen.add(point.frame)
        if point.frame % max(stride, 1) == 0:
            chosen.add(point.frame)
    return sorted(chosen)


def robust_center_depth(depth: np.ndarray, x: float, y: float, radius: int = 4) -> float | None:
    if depth.ndim != 2:
        return None
    h, w = depth.shape
    cx, cy = int(round(x)), int(round(y))
    if not (0 <= cx < w and 0 <= cy < h):
        return None
    x0, x1 = max(cx - radius, 0), min(cx + radius + 1, w)
    y0, y1 = max(cy - radius, 0), min(cy + radius + 1, h)
    patch = depth[y0:y1, x0:x1].reshape(-1)
    patch = patch[np.isfinite(patch) & (patch > 1e-4)]
    if patch.size < 3:
        return None
    # Shrink toward the centre: drop the outer 20% to avoid background bleed.
    lo, hi = np.quantile(patch, [0.2, 0.8])
    core = patch[(patch >= lo) & (patch <= hi)]
    if core.size == 0:
        core = patch
    return float(np.median(core))


def audit_track(
    result: TrackResult,
    calibration: CalibrationState,
    load_frame: FrameLoader,
    estimator: DepthEstimator,
    *,
    threshold_m: float = OFF_PLANE_WARN_M,
    stride: int = AUDIT_STRIDE,
) -> DepthAuditState:
    state = DepthAuditState(
        enabled=True,
        model_name=estimator.name,
        available=True,
        threshold_m=threshold_m,
    )
    if calibration.mode is not CalibrationMode.PLANAR or not calibration.active:
        state.available = False
        state.message = "离面抽检需要已应用的运动平面标定"
        return state
    plane_size = 1.0
    if calibration.plane is not None:
        plane_size = max(calibration.plane.width_m, calibration.plane.height_m, 1e-6)
    limit = max(threshold_m, OFF_PLANE_RATIO * plane_size)
    cache: dict[int, np.ndarray] = {}
    for frame in select_audit_frames(result, stride=stride):
        point = next((p for p in result.points if p.frame == frame and p.visible), None)
        if point is None:
            continue
        z_geom = calibration.expected_depth_m(point.x, point.y)
        if frame not in cache:
            try:
                cache[frame] = estimator.predict(load_frame(frame))
            except Exception as exc:  # noqa: BLE001
                state.readings[frame] = OffPlaneReading(
                    frame=frame, z_geom=z_geom, flag="skipped", confidence=0.0
                )
                state.message = str(exc)
                continue
        z_model = robust_center_depth(cache[frame], point.x, point.y)
        residual = None
        flag = "ok"
        conf = 0.0
        if z_geom is not None and z_model is not None:
            residual = abs(z_model - z_geom)
            conf = max(0.0, min(1.0, 1.0 - residual / max(limit * 3.0, 1e-6)))
            if residual > limit:
                flag = "off_plane"
        else:
            flag = "skipped"
        state.readings[frame] = OffPlaneReading(
            frame=frame,
            z_geom=z_geom,
            z_model=z_model,
            residual_m=residual,
            confidence=conf,
            flag=flag,
        )
    off = state.off_plane_frames()
    if off:
        state.message = f"{len(off)} 帧可能离开运动平面（阈值 {limit:.3f} m）"
    elif not state.message:
        state.message = "抽检未发现明显离面"
    return state


class FakeDepthEstimator:
    """Test double: geometric depth plus an optional residual bump."""

    name = "fake-depth"

    def __init__(
        self,
        calibration: CalibrationState,
        extra: dict[int, float] | None = None,
        *,
        bump_m: float = 0.0,
    ) -> None:
        self._calibration = calibration
        self._extra = extra or {}
        self.bump_m = bump_m
        self.calls = 0

    def predict(self, rgb: np.ndarray) -> np.ndarray:
        self.calls += 1
        h, w = rgb.shape[:2]
        depth = np.zeros((h, w), dtype=np.float64)
        for y in range(0, h, 8):
            for x in range(0, w, 8):
                z = self._calibration.expected_depth_m(float(x), float(y))
                if z is not None:
                    depth[y : y + 8, x : x + 8] = z
        bump = self.bump_m
        if self._extra and rgb.size:
            bump += float(self._extra.get(int(rgb[0, 0, 0]), 0.0))
        if bump:
            depth = np.where(depth > 0.0, depth + bump, depth)
        return depth


def try_load_moge() -> tuple[DepthEstimator | None, str]:
    """Load MoGe-2 ViT-S if the optional package is installed. Never import in fast track."""
    try:
        import torch
        from moge.model.v2 import MoGeModel
    except Exception:
        return None, "未安装 MoGe-2。可选：python -m pip install moge-2"
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = MoGeModel.from_pretrained("Ruicheng/moge-2-vits-normal").to(device).eval()
    except Exception as exc:  # noqa: BLE001
        return None, f"无法加载 MoGe-2：{exc}"
    return _MogeEstimator(model, device), ""


class _MogeEstimator:
    name = "moge-2-vits"

    def __init__(self, model, device: str) -> None:  # noqa: ANN001
        self._model = model
        self._device = device

    def predict(self, rgb: np.ndarray) -> np.ndarray:
        import torch

        image = torch.tensor(rgb / 255.0, dtype=torch.float32, device=self._device).permute(2, 0, 1)
        with torch.no_grad():
            output = self._model.infer(image)
        depth = output.get("depth")
        if depth is None:
            raise RuntimeError("MoGe-2 未返回深度")
        array = depth.detach().cpu().numpy()
        if array.ndim == 3:
            array = array[0]
        return array.astype(np.float64)
