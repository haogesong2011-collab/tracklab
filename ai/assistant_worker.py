"""Async DeepSeek / local-physics jobs. Dedicated QThread, never FramePump."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from PySide6.QtCore import QObject, QThread, Qt, Signal

from ai.calibration import CalibrationState
from ai.contracts import CancelToken, ExperimentType, TrackResult
from ai.deepseek_client import (
    DEFAULT_CHAT_MODEL,
    DEFAULT_REPORT_MODEL,
    DeepSeekClient,
    DeepSeekError,
    DeepSeekResponse,
)
from ai.depth_audit import DepthAuditState
from ai.physics import analyze_experiment
from engine.video_index import VideoInfo


@dataclass
class AssistantJob:
    kind: str
    track: TrackResult | None = None
    info: VideoInfo | None = None
    calibration: CalibrationState = field(default_factory=CalibrationState)
    shake_enabled: bool = False
    shake_offsets: tuple[tuple[float, float], ...] = ()
    clip_id: str = ""
    pendulum_length_m: float | None = None
    period_hint: bool = False
    force_type: ExperimentType | None = None
    depth_audit: DepthAuditState | None = None
    api_key: str = ""
    chat_model: str = DEFAULT_CHAT_MODEL
    report_model: str = DEFAULT_REPORT_MODEL
    messages: list[dict[str, str]] = field(default_factory=list)
    json_mode: bool = False
    stream: bool = False
    transport: Any = None


@dataclass
class AssistantOutcome:
    kind: str
    analysis: Any = None
    text: str = ""
    json_data: dict[str, Any] | None = None
    usage: dict[str, int] = field(default_factory=dict)
    model: str = ""
    cancelled: bool = False
    reasoning: str = ""


class AssistantWorker(QObject):
    chunk = Signal(str)
    reasoning = Signal(str)
    finished = Signal(object)
    failed = Signal(str)
    cancelled = Signal()
    usage = Signal(object)

    def __init__(self, job: AssistantJob, parent=None) -> None:  # noqa: ANN001
        super().__init__(parent)
        self.job = job
        self._token = CancelToken()

    def cancel(self) -> None:
        self._token.cancel()

    def run(self) -> None:
        try:
            if self.job.kind == "analyze":
                self._run_analyze()
                return
            self._run_network()
        except DeepSeekError as exc:
            self.failed.emit(str(exc))
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(str(exc))

    def _run_analyze(self) -> None:
        if self._token.cancelled:
            self.cancelled.emit()
            return
        analysis = analyze_experiment(
            self.job.track,
            self.job.info,
            self.job.calibration,
            clip_id=self.job.clip_id,
            pendulum_length_m=self.job.pendulum_length_m,
            period_hint=self.job.period_hint,
            force_type=self.job.force_type,
            shake_enabled=self.job.shake_enabled,
            shake_offsets=self.job.shake_offsets,
            depth_audit=self.job.depth_audit,
        )
        if self._token.cancelled:
            self.cancelled.emit()
            return
        self.finished.emit(AssistantOutcome(kind="analyze", analysis=analysis))

    def _run_network(self) -> None:
        if not self.job.api_key:
            raise DeepSeekError("尚未配置 API Key，请先打开 DeepSeek 设置。", code="auth")
        client = DeepSeekClient(
            self.job.api_key,
            chat_model=self.job.chat_model,
            report_model=self.job.report_model,
            transport=self.job.transport,
        )
        stream = self.job.stream and not self.job.json_mode
        response: DeepSeekResponse = client.complete(
            self.job.messages,
            model=self.job.report_model if self.job.json_mode else self.job.chat_model,
            json_mode=self.job.json_mode,
            stream=stream,
            on_chunk=self.chunk.emit if stream else None,
            on_reasoning=self.reasoning.emit if stream else None,
            cancel=self._token,
        )
        if response.usage.total_tokens:
            self.usage.emit(response.usage.as_dict())
        if response.cancelled or self._token.cancelled:
            self.cancelled.emit()
            self.finished.emit(
                AssistantOutcome(
                    kind=self.job.kind,
                    text=response.text,
                    json_data=response.json_data,
                    usage=response.usage.as_dict(),
                    model=response.model,
                    cancelled=True,
                    reasoning=response.reasoning,
                )
            )
            return
        self.finished.emit(
            AssistantOutcome(
                kind=self.job.kind,
                text=response.text,
                json_data=response.json_data,
                usage=response.usage.as_dict(),
                model=response.model,
                reasoning=response.reasoning,
            )
        )


def run_assistant_in_thread(
    job: AssistantJob,
    *,
    on_chunk: Callable[[str], None] | None = None,
    on_reasoning: Callable[[str], None] | None = None,
    on_finished: Callable[[AssistantOutcome], None] | None = None,
    on_failed: Callable[[str], None] | None = None,
    on_cancelled: Callable[[], None] | None = None,
    on_usage: Callable[[dict], None] | None = None,
) -> tuple[QThread, AssistantWorker]:
    thread = QThread()
    worker = AssistantWorker(job)
    worker.moveToThread(thread)
    thread.started.connect(worker.run)
    if on_chunk is not None:
        worker.chunk.connect(on_chunk, Qt.ConnectionType.QueuedConnection)
    if on_reasoning is not None:
        worker.reasoning.connect(on_reasoning, Qt.ConnectionType.QueuedConnection)
    if on_finished is not None:
        worker.finished.connect(on_finished, Qt.ConnectionType.QueuedConnection)
    if on_failed is not None:
        worker.failed.connect(on_failed, Qt.ConnectionType.QueuedConnection)
    if on_cancelled is not None:
        worker.cancelled.connect(on_cancelled, Qt.ConnectionType.QueuedConnection)
    if on_usage is not None:
        worker.usage.connect(on_usage, Qt.ConnectionType.QueuedConnection)
    worker.finished.connect(thread.quit)
    worker.failed.connect(thread.quit)
    worker.cancelled.connect(thread.quit)
    return thread, worker
