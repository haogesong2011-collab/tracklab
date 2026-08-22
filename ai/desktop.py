"""Desktop acceptance helpers: AI work must not share the playback decode thread.

This module defines the async bridge contract that the UI will call. The worker
runs off the Qt main thread and reports progress / cancellation without touching
app.frame_pump.FramePump.
"""

from __future__ import annotations

import csv
import json
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from PySide6.QtCore import QObject, QThread, Qt, Signal

from ai.calibration import CalibrationState
from ai.contracts import (
    TRACK_COLORS,
    CancelToken,
    ProgressEvent,
    TrackLayer,
    TrackPoint,
    TrackPrompt,
    TrackResult,
)
from ai.kinematics import series_for_result
from ai.models import ColorBlobTracker, load_video
from ai.schema import Point2D
from ai.stabilize import ShakeCompensation, estimate_shake
from engine.video_index import VideoInfo


ProgressHandler = Callable[[ProgressEvent], None]


@dataclass
class DesktopAcceptanceCriteria:
    """Gates from the AI test plan for UI-integrated inference."""

    max_ui_block_ms: float = 200.0
    min_offline_fps_1080p: float = 10.0
    max_peak_memory_gb: float = 2.0
    require_progress: bool = True
    require_cancel: bool = True
    require_manual_override: bool = True


DEFAULT_CRITERIA = DesktopAcceptanceCriteria()


@dataclass
class AcceptanceResult:
    name: str
    passed: bool
    details: str = ""
    measurements: dict = field(default_factory=dict)


@dataclass
class ProjectDocument:
    video_path: Path
    tracks: list[TrackLayer]
    active_track_id: str | None = None
    schema: str = "tracklab.project.v3"
    model_name: str = ""
    model_version: str = ""
    model_license: str = ""
    show_contours: bool = True
    show_prompts: bool = True
    show_calibration: bool = True
    calibration: CalibrationState = field(default_factory=CalibrationState)


class TrackWorker(QObject):
    """Qt-friendly wrapper around a Tracker implementation."""

    progress = Signal(object)
    finished = Signal(object)
    failed = Signal(str)

    def __init__(
        self,
        video_path: Path,
        seed: tuple[float, float],
        parent=None,  # noqa: ANN001
        *,
        track_id: str = "",
        start_frame: int = 0,
        end_frame: int | None = None,
        prompts: list[TrackPrompt] | None = None,
        tracker=None,  # noqa: ANN001
    ) -> None:
        super().__init__(parent)
        self._path = Path(video_path)
        self._seed = seed
        self._token = CancelToken()
        self.track_id = track_id
        self._start_frame = start_frame
        self._end_frame = end_frame
        self._prompts = list(prompts or [])
        self._tracker = tracker

    def cancel(self) -> None:
        self._token.cancel()

    def run(self) -> None:
        try:
            info = load_video(self._path)
            tracker = self._tracker
            if tracker is None:
                from ai.sam2_tracker import Sam2Tracker

                tracker = Sam2Tracker()

            def on_progress(event: ProgressEvent) -> None:
                self.progress.emit(event)

            result = tracker.track(
                info,
                self._seed,
                cancel=self._token,
                progress=on_progress,
                start_frame=self._start_frame,
                end_frame=self._end_frame,
                prompts=self._prompts,
            )
            result.track_id = self.track_id  # type: ignore[attr-defined]
            self.finished.emit(result)
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(str(exc))


class ShakeWorker(QObject):
    """Four-corner shake estimate. Dedicated thread, never FramePump."""

    progress = Signal(object)
    finished = Signal(object)
    failed = Signal(str)

    def __init__(self, video_path: Path, parent=None) -> None:  # noqa: ANN001
        super().__init__(parent)
        self._path = Path(video_path)
        self._token = CancelToken()

    def cancel(self) -> None:
        self._token.cancel()

    def run(self) -> None:
        try:
            info = load_video(self._path)

            def on_progress(event: ProgressEvent) -> None:
                self.progress.emit(event)

            result = estimate_shake(info, cancel=self._token, progress=on_progress)
            self.finished.emit(result)
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(str(exc))


def run_track_in_thread(
    video_path: Path,
    seed: tuple[float, float],
    *,
    on_progress: ProgressHandler | None = None,
    on_finished: Callable[[TrackResult], None] | None = None,
    on_failed: Callable[[str], None] | None = None,
    track_id: str = "",
    start_frame: int = 0,
    end_frame: int | None = None,
    prompts: list[TrackPrompt] | None = None,
    tracker=None,  # noqa: ANN001
) -> tuple[QThread, TrackWorker]:
    """Spawn a dedicated QThread — never reuse FramePump's thread."""
    thread = QThread()
    worker = TrackWorker(
        video_path,
        seed,
        track_id=track_id,
        start_frame=start_frame,
        end_frame=end_frame,
        prompts=prompts,
        tracker=tracker,
    )
    worker.moveToThread(thread)
    thread.started.connect(worker.run)
    if on_progress:
        worker.progress.connect(on_progress, Qt.ConnectionType.QueuedConnection)
    if on_finished:
        worker.finished.connect(on_finished, Qt.ConnectionType.QueuedConnection)
    if on_failed:
        worker.failed.connect(on_failed, Qt.ConnectionType.QueuedConnection)
    worker.finished.connect(thread.quit)
    worker.failed.connect(thread.quit)
    return thread, worker


def run_shake_in_thread(
    video_path: Path,
    *,
    on_progress: ProgressHandler | None = None,
    on_finished: Callable[[ShakeCompensation], None] | None = None,
    on_failed: Callable[[str], None] | None = None,
) -> tuple[QThread, ShakeWorker]:
    """Spawn a dedicated QThread for corner-anchor shake compensation."""
    thread = QThread()
    worker = ShakeWorker(video_path)
    worker.moveToThread(thread)
    thread.started.connect(worker.run)
    if on_progress:
        worker.progress.connect(on_progress, Qt.ConnectionType.QueuedConnection)
    if on_finished:
        worker.finished.connect(on_finished, Qt.ConnectionType.QueuedConnection)
    if on_failed:
        worker.failed.connect(on_failed, Qt.ConnectionType.QueuedConnection)
    worker.finished.connect(thread.quit)
    worker.failed.connect(thread.quit)
    return thread, worker


def apply_manual_override(
    result: TrackResult, frame: int, point: Point2D
) -> TrackResult:
    """User correction after AI — required by acceptance checklist."""
    points = list(result.points)
    replacement = TrackPoint(
        frame=frame,
        x=point.x,
        y=point.y,
        visible=True,
        confidence=1.0,
        manual=True,
    )
    for i, existing in enumerate(points):
        if existing.frame == frame:
            points[i] = replacement
            break
    else:
        points.append(replacement)
        points.sort(key=lambda p: p.frame)
    return TrackResult(
        clip_id=result.clip_id,
        points=points,
        confidence=result.confidence,
        failure_reason=result.failure_reason,
        model_name=result.model_name,
        model_version=result.model_version,
        elapsed_s=result.elapsed_s,
    )


PROJECT_SCHEMA = "tracklab.project.v3"
LEGACY_SCHEMA = "tracklab.project.v1"
LEGACY_V2_SCHEMA = "tracklab.project.v2"


def new_track_id() -> str:
    return uuid.uuid4().hex[:10]


def layer_from_result(
    result: TrackResult,
    *,
    name: str = "轨迹 1",
    color: str = TRACK_COLORS[0],
    track_id: str | None = None,
) -> TrackLayer:
    return TrackLayer(
        track_id=track_id or new_track_id(),
        name=name,
        color=color,
        result=result,
        status="done",
    )


def write_track_project(
    path: Path,
    video_path: Path,
    result: TrackResult | None = None,
    *,
    tracks: list[TrackLayer] | None = None,
    active_track_id: str | None = None,
    model_name: str = "",
    model_version: str = "",
    model_license: str = "",
    show_contours: bool = True,
    show_prompts: bool = True,
    show_calibration: bool = True,
    calibration: CalibrationState | None = None,
) -> None:
    """Persist video path + tracks (including manual overrides) to JSON."""
    layers = list(tracks or [])
    if not layers and result is not None:
        layers = [layer_from_result(result)]
    payload = {
        "schema": PROJECT_SCHEMA,
        "video_path": str(video_path),
        "active_track_id": active_track_id or (layers[0].track_id if layers else None),
        "model": {
            "name": model_name,
            "version": model_version,
            "license": model_license,
        },
        "display": {
            "contours": show_contours,
            "prompts": show_prompts,
            "calibration": show_calibration,
        },
        "calibration": (calibration or CalibrationState()).to_dict(),
        "tracks": [layer.to_dict() for layer in layers],
        "track": None if result is None else result.to_dict(),
    }
    Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def read_track_project(path: Path) -> ProjectDocument:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    video = Path(raw["video_path"])
    schema = str(raw.get("schema", LEGACY_SCHEMA))
    model = raw.get("model") or {}
    display = raw.get("display") or {}
    tracks_raw = raw.get("tracks") or []
    tracks = [TrackLayer.from_dict(item) for item in tracks_raw]
    if not tracks and raw.get("track"):
        result = TrackResult.from_dict(raw["track"])
        tracks = [layer_from_result(result, name="轨迹 1")]
    active = raw.get("active_track_id")
    if active is None and tracks:
        active = tracks[0].track_id
    return ProjectDocument(
        video_path=video,
        tracks=tracks,
        active_track_id=active,
        schema=schema,
        model_name=str(model.get("name", "")),
        model_version=str(model.get("version", "")),
        model_license=str(model.get("license", "")),
        show_contours=bool(display.get("contours", True)),
        show_prompts=bool(display.get("prompts", True)),
        show_calibration=bool(display.get("calibration", True)),
        calibration=CalibrationState.from_dict(raw.get("calibration")),
    )


def export_track_csv(
    path: Path,
    result: TrackResult,
    info: VideoInfo,
    *,
    calibration: CalibrationState | None = None,
) -> None:
    samples = series_for_result(result, info, calibration=calibration)
    unit = samples[0].position_unit if samples else ("m" if calibration and calibration.active else "px")
    speed = samples[0].speed_unit if samples else ("m/s" if unit == "m" else "px/s")
    pos_key = "m" if unit == "m" else "px"
    spd_key = "m_s" if unit == "m" else "px_s"
    with Path(path).open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            [
                "frame",
                "time_s",
                f"x_{pos_key}",
                f"y_{pos_key}",
                f"vx_{spd_key}",
                f"vy_{spd_key}",
                f"v_{spd_key}",
                "visible",
                "confidence",
            ]
        )
        for sample in samples:
            writer.writerow(
                [
                    sample.frame,
                    f"{sample.time_s:.6f}",
                    "" if sample.x is None else f"{sample.x:.4f}",
                    "" if sample.y is None else f"{sample.y:.4f}",
                    "" if sample.vx is None else f"{sample.vx:.4f}",
                    "" if sample.vy is None else f"{sample.vy:.4f}",
                    "" if sample.speed is None else f"{sample.speed:.4f}",
                    int(sample.visible),
                    f"{sample.confidence:.4f}",
                ]
            )


def check_cancel_latency(
    video_path: Path,
    seed: tuple[float, float],
    *,
    criteria: DesktopAcceptanceCriteria = DEFAULT_CRITERIA,
) -> AcceptanceResult:
    """Cancel mid-run and measure how quickly the worker stops emitting work."""
    import os
    import sys

    from PySide6.QtWidgets import QApplication

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance() or QApplication(sys.argv)
    events: list[ProgressEvent] = []
    done: dict = {"ok": False, "result": None, "error": None}
    t0 = {"start": 0.0, "cancel": 0.0, "stop": 0.0}
    holders: dict = {}

    def on_progress(event: ProgressEvent) -> None:
        events.append(event)
        if event.current >= 2 and t0["cancel"] == 0.0:
            t0["cancel"] = time.perf_counter()
            holders["worker"].cancel()

    def on_finished(result: TrackResult) -> None:
        t0["stop"] = time.perf_counter()
        done["ok"] = True
        done["result"] = result

    def on_failed(message: str) -> None:
        t0["stop"] = time.perf_counter()
        done["error"] = message

    thread, worker = run_track_in_thread(
        video_path,
        seed,
        on_finished=on_finished,
        on_failed=on_failed,
        tracker=ColorBlobTracker(),
    )
    worker.progress.connect(on_progress, Qt.ConnectionType.DirectConnection)
    holders["thread"] = thread
    holders["worker"] = worker
    t0["start"] = time.perf_counter()
    thread.start()
    deadline = t0["start"] + 8.0
    while time.perf_counter() < deadline and t0["stop"] == 0.0:
        app.processEvents()
        if not thread.isRunning():
            app.processEvents()
            break
        time.sleep(0.005)
    thread.quit()
    thread.wait(5000)
    app.processEvents()

    if done["error"]:
        return AcceptanceResult("cancel_latency", False, details=str(done["error"]))
    if t0["cancel"] == 0.0 or t0["stop"] == 0.0:
        return AcceptanceResult(
            "cancel_latency",
            False,
            details="cancel or finish never observed",
            measurements=dict(t0),
        )
    latency_ms = (t0["stop"] - t0["cancel"]) * 1000.0
    result = done["result"]
    cancelled = result is not None and result.failure_reason.value == "cancelled"
    return AcceptanceResult(
        "cancel_latency",
        passed=cancelled and latency_ms < criteria.max_ui_block_ms,
        details=f"cancel_to_finish_ms={latency_ms:.1f}",
        measurements={"cancel_to_finish_ms": latency_ms, "cancelled": cancelled},
    )


ACCEPTANCE_CHECKLIST = """
Desktop AI acceptance checklist
================================
1. Inference runs in a dedicated QThread (ai.desktop.TrackWorker), never inside
   app.frame_pump.FramePump.
2. Progress events update a status / progress widget without blocking scrub.
3. Cancel returns control quickly; TrackResult.failure_reason == cancelled.
4. Predictions land on the correct frame index (VideoInfo frame numbers).
5. Manual override via apply_manual_override persists into the project file.
6. Peak RSS during 1080p offline analysis stays under 2 GB on the lab machine.
7. Offline throughput on the lab 1080p clip is at least 10 fps.
8. Shake compensation runs in ai.desktop.ShakeWorker (own QThread), never FramePump.
"""
