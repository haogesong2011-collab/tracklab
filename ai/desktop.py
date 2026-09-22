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

from PySide6.QtCore import QObject, QThread, Qt, Signal, Slot

from ai.assistant_report import render_report_markdown
from ai.calibration import CalibrationState
from ai.contracts import (
    TRACK_COLORS,
    AssistantState,
    CancelToken,
    ProgressEvent,
    TrackLayer,
    TrackMode,
    TrackPoint,
    TrackPrompt,
    TrackResult,
)
from ai.depth_audit import DepthAuditState, audit_track, try_load_moge
from ai.kinematics import DEFAULT_VELOCITY_MODE, DEFAULT_VELOCITY_STEP, quality_label, series_for_result
from ai.model_manager import DownloadCancelled, ModelNotAvailable, ModelSpec, ensure_checkpoint
from ai.models import ColorBlobTracker, load_video
from ai.schema import Point2D
from ai.stabilize import ShakeCompensation, estimate_shake
from engine.decoder import FrameDecoder
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
    schema: str = "tracklab.project.v5"
    model_name: str = ""
    model_version: str = ""
    model_license: str = ""
    show_contours: bool = True
    show_prompts: bool = True
    show_calibration: bool = True
    track_mode: TrackMode = TrackMode.PRECISE
    calibration: CalibrationState = field(default_factory=CalibrationState)
    assistant: AssistantState = field(default_factory=AssistantState)
    depth_audit: DepthAuditState = field(default_factory=DepthAuditState)


def create_tracker(mode: TrackMode | str = TrackMode.PRECISE):
    """SAM Tiny (FAST) / SAM Small (PRECISE). Autotracker only if SAM cannot import."""
    resolved = TrackMode(mode)
    try:
        from ai.model_manager import spec_for_mode
        from ai.sam2_tracker import Sam2Tracker

        return Sam2Tracker(spec=spec_for_mode(resolved))
    except Exception:
        from ai.autotracker import TrackerAutoTracker

        return TrackerAutoTracker()


class TrackWorker(QObject):
    """Qt-friendly wrapper around a Tracker implementation."""

    progress = Signal(object)
    finished = Signal(object)
    failed = Signal(str)
    start_requested = Signal()

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
        track_mode: TrackMode = TrackMode.PRECISE,
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
        self._track_mode = track_mode

    def cancel(self) -> None:
        self._token.cancel()

    @Slot()
    def run(self) -> None:
        try:
            info = load_video(self._path)
            tracker = self._tracker
            stride = 1
            image_size = None
            if tracker is None:
                from ai.sam2_tracker import Sam2Tracker
                from ai.sam_runtime import SamRuntime, settings_for_mode

                spec, stride, image_size = settings_for_mode(self._track_mode)
                predictor = SamRuntime.instance().predictor_for(spec)
                tracker = Sam2Tracker(predictor=predictor, spec=spec)

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
                stride=stride,
                image_size=image_size,
            )
            result.track_id = self.track_id  # type: ignore[attr-defined]
            self.finished.emit(result)
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(str(exc))


class ShakeWorker(QObject):
    """Conservative background-motion estimate. Dedicated thread, never FramePump."""

    progress = Signal(object)
    finished = Signal(object)
    failed = Signal(str)

    def __init__(
        self,
        video_path: Path,
        parent=None,  # noqa: ANN001
        *,
        start_frame: int = 0,
        end_frame: int | None = None,
        exclude_by_frame: dict[int, list[tuple[float, float]]] | None = None,
    ) -> None:
        super().__init__(parent)
        self._path = Path(video_path)
        self._start_frame = start_frame
        self._end_frame = end_frame
        self._exclude_by_frame = exclude_by_frame or {}
        self._token = CancelToken()

    def cancel(self) -> None:
        self._token.cancel()

    def run(self) -> None:
        try:
            info = load_video(self._path)

            def on_progress(event: ProgressEvent) -> None:
                self.progress.emit(event)

            result = estimate_shake(
                info,
                cancel=self._token,
                progress=on_progress,
                start_frame=self._start_frame,
                end_frame=self._end_frame,
                exclude_by_frame=self._exclude_by_frame,
            )
            self.finished.emit(result)
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(str(exc))


class CheckpointDownloadWorker(QObject):
    """Download SAM weights off the UI thread. Never reuse FramePump."""

    progress = Signal(int, int)
    stage = Signal(str)
    finished = Signal(str)
    failed = Signal(str)
    cancelled = Signal()

    def __init__(self, spec: ModelSpec, parent=None) -> None:  # noqa: ANN001
        super().__init__(parent)
        self._spec = spec
        self._token = CancelToken()

    def cancel(self) -> None:
        self._token.cancel()

    @Slot()
    def run(self) -> None:
        try:
            path = ensure_checkpoint(
                self._spec,
                download=True,
                progress=self.progress.emit,
                stage=self.stage.emit,
                cancel=self._token,
            )
            if self._token.cancelled:
                self.cancelled.emit()
                return
            self.finished.emit(str(path))
        except DownloadCancelled:
            self.cancelled.emit()
        except (ModelNotAvailable, OSError) as exc:
            self.failed.emit(str(exc))
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
    track_mode: TrackMode = TrackMode.PRECISE,
    thread: QThread | None = None,
) -> tuple[QThread, TrackWorker]:
    """Spawn or queue a track job. Never reuse FramePump's thread."""
    worker = TrackWorker(
        video_path,
        seed,
        track_id=track_id,
        start_frame=start_frame,
        end_frame=end_frame,
        prompts=prompts,
        tracker=tracker,
        track_mode=track_mode,
    )
    owned = thread is None
    if thread is None:
        thread = QThread()
    worker.moveToThread(thread)
    worker.start_requested.connect(worker.run, Qt.ConnectionType.QueuedConnection)
    if on_progress:
        worker.progress.connect(on_progress, Qt.ConnectionType.QueuedConnection)
    if on_finished:
        worker.finished.connect(on_finished, Qt.ConnectionType.QueuedConnection)
    if on_failed:
        worker.failed.connect(on_failed, Qt.ConnectionType.QueuedConnection)
    if owned:
        thread.started.connect(worker.run)
        worker.finished.connect(thread.quit)
        worker.failed.connect(thread.quit)
    else:
        worker.start_requested.emit()
    return thread, worker


def run_shake_in_thread(
    video_path: Path,
    *,
    on_progress: ProgressHandler | None = None,
    on_finished: Callable[[ShakeCompensation], None] | None = None,
    on_failed: Callable[[str], None] | None = None,
    start_frame: int = 0,
    end_frame: int | None = None,
    exclude_by_frame: dict[int, list[tuple[float, float]]] | None = None,
) -> tuple[QThread, ShakeWorker]:
    """Spawn a dedicated QThread for conservative background compensation."""
    thread = QThread()
    worker = ShakeWorker(
        video_path,
        start_frame=start_frame,
        end_frame=end_frame,
        exclude_by_frame=exclude_by_frame,
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


def run_checkpoint_download(
    spec: ModelSpec,
    *,
    on_progress: Callable[[int, int], None] | None = None,
    on_stage: Callable[[str], None] | None = None,
    on_finished: Callable[[str], None] | None = None,
    on_failed: Callable[[str], None] | None = None,
    on_cancelled: Callable[[], None] | None = None,
) -> tuple[QThread, CheckpointDownloadWorker]:
    """Download and verify a SAM checkpoint on a dedicated QThread."""
    thread = QThread()
    worker = CheckpointDownloadWorker(spec)
    worker.moveToThread(thread)
    thread.started.connect(worker.run)
    if on_progress is not None:
        worker.progress.connect(on_progress, Qt.ConnectionType.QueuedConnection)
    if on_stage is not None:
        worker.stage.connect(on_stage, Qt.ConnectionType.QueuedConnection)
    if on_finished is not None:
        worker.finished.connect(on_finished, Qt.ConnectionType.QueuedConnection)
    if on_failed is not None:
        worker.failed.connect(on_failed, Qt.ConnectionType.QueuedConnection)
    if on_cancelled is not None:
        worker.cancelled.connect(on_cancelled, Qt.ConnectionType.QueuedConnection)
    worker.finished.connect(thread.quit)
    worker.failed.connect(thread.quit)
    worker.cancelled.connect(thread.quit)
    return thread, worker


class DepthAuditWorker(QObject):
    """Sparse MoGe off-plane audit. Dedicated thread, never FramePump."""

    progress = Signal(object)
    finished = Signal(object)
    failed = Signal(str)

    def __init__(
        self,
        video_path: Path,
        result: TrackResult,
        calibration: CalibrationState,
        parent=None,  # noqa: ANN001
        *,
        experimental_correction: bool = False,
        estimator=None,  # noqa: ANN001
    ) -> None:
        super().__init__(parent)
        self._path = Path(video_path)
        self._result = result
        self._calibration = calibration
        self._experimental = experimental_correction
        self._estimator = estimator
        self._token = CancelToken()

    def cancel(self) -> None:
        self._token.cancel()

    def run(self) -> None:
        decoder: FrameDecoder | None = None
        try:
            if self._token.cancelled:
                self.failed.emit("已取消")
                return
            estimator = self._estimator
            message = ""
            if estimator is None:
                estimator, message = try_load_moge()
            if estimator is None:
                self.failed.emit(message or "离面抽检模型不可用")
                return
            info = load_video(self._path)
            decoder = FrameDecoder(info)

            def load_frame(index: int):
                if self._token.cancelled:
                    raise RuntimeError("已取消")
                return decoder.frame(index)

            state = audit_track(
                self._result,
                self._calibration,
                load_frame,
                estimator,
            )
            state.experimental_correction = self._experimental
            self.finished.emit(state)
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(str(exc))
        finally:
            if decoder is not None:
                decoder.close()


def run_audit_in_thread(
    video_path: Path,
    result: TrackResult,
    calibration: CalibrationState,
    *,
    experimental_correction: bool = False,
    estimator=None,  # noqa: ANN001
    on_finished: Callable | None = None,
    on_failed: Callable | None = None,
) -> tuple[QThread, DepthAuditWorker]:
    thread = QThread()
    worker = DepthAuditWorker(
        video_path,
        result,
        calibration,
        experimental_correction=experimental_correction,
        estimator=estimator,
    )
    worker.moveToThread(thread)
    thread.started.connect(worker.run)
    if on_finished is not None:
        worker.finished.connect(on_finished, Qt.ConnectionType.QueuedConnection)
    if on_failed is not None:
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


PROJECT_SCHEMA = "tracklab.project.v5"
LEGACY_SCHEMA = "tracklab.project.v1"
LEGACY_V2_SCHEMA = "tracklab.project.v2"
LEGACY_V3_SCHEMA = "tracklab.project.v3"
LEGACY_V4_SCHEMA = "tracklab.project.v4"


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
    track_mode: TrackMode = TrackMode.PRECISE,
    calibration: CalibrationState | None = None,
    assistant: AssistantState | None = None,
    depth_audit: DepthAuditState | None = None,
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
            "track_mode": track_mode.value,
        },
        "calibration": (calibration or CalibrationState()).to_dict(),
        "track_mode": TrackMode(track_mode).value,
        "tracks": [layer.to_dict() for layer in layers],
        "track": None if result is None else result.to_dict(),
        "assistant": (assistant or AssistantState()).to_dict(),
        "depth_audit": (depth_audit or DepthAuditState()).to_dict(),
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
        track_mode=_track_mode_from_payload(raw, display),
        calibration=CalibrationState.from_dict(raw.get("calibration")),
        assistant=AssistantState.from_dict(raw.get("assistant")),
        depth_audit=DepthAuditState.from_dict(raw.get("depth_audit")),
    )


def _track_mode_from_display(display: dict) -> TrackMode:
    return _track_mode_from_payload({}, display)


def _track_mode_from_payload(raw: dict, display: dict) -> TrackMode:
    value = raw.get("track_mode", display.get("track_mode", TrackMode.PRECISE.value))
    try:
        return TrackMode(str(value))
    except ValueError:
        return TrackMode.PRECISE


def export_assistant_report(path: Path, markdown: str) -> None:
    dest = Path(path)
    if dest.suffix.lower() != ".md":
        dest = dest.with_suffix(".md")
    dest.write_text(markdown, encoding="utf-8")


def build_assistant_report(assistant: AssistantState, *, stale: bool = False) -> str:
    return render_report_markdown(
        assistant.analysis,
        confirmed_type=assistant.confirmed_type,
        sections=assistant.report_sections,
        teaching_level=assistant.teaching_level,
        generated_at=assistant.generated_at,
        model_id=assistant.model_id,
        stale=stale or assistant.stale,
    )


def export_track_csv(
    path: Path,
    result: TrackResult,
    info: VideoInfo,
    *,
    calibration: CalibrationState | None = None,
    velocity_step: int | None = None,
    velocity_mode: str | None = None,
    depth_audit: DepthAuditState | None = None,
) -> None:
    samples = series_for_result(
        result,
        info,
        calibration=calibration,
        velocity_step=DEFAULT_VELOCITY_STEP if velocity_step is None else velocity_step,
        velocity_mode=DEFAULT_VELOCITY_MODE if velocity_mode is None else velocity_mode,
        depth_audit=depth_audit,
    )
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
                "sigma_x",
                "sigma_y",
                "quality",
                "off_plane_m",
                "source",
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
                    "" if sample.sigma_x is None else f"{sample.sigma_x:.6f}",
                    "" if sample.sigma_y is None else f"{sample.sigma_y:.6f}",
                    quality_label(sample),
                    "" if sample.off_plane_m is None else f"{sample.off_plane_m:.6f}",
                    sample.source,
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
