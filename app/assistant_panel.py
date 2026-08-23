"""Independent TrackLab assistant window: chat plus experiment context."""

from __future__ import annotations

import time

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QCloseEvent, QHideEvent, QKeyEvent, QShowEvent, QTextCursor
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QTextBrowser,
    QTextEdit,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from ai.api_credentials import (
    clear_api_key,
    get_api_key,
    has_api_key,
    mask_key,
    normalize_api_key,
    set_api_key,
)
from ai.contracts import (
    EXPERIMENT_LABELS,
    TEACHING_LEVEL_LABELS,
    ExperimentAnalysis,
    ExperimentCandidate,
    ExperimentType,
    TeachingLevel,
)
from ai.deepseek_client import DEFAULT_CHAT_MODEL, DEFAULT_REPORT_MODEL

_ROLE_LABELS = {
    "user": "你",
    "assistant": "助手",
    "local": "实验识别",
    "system": "系统",
}

_THINKING_LABELS = (
    "思考中",
    "正在阅读实验数据",
    "正在对照公式",
    "正在组织回答",
)


class DeepSeekSettingsDialog(QDialog):
    def __init__(
        self,
        parent: QWidget | None = None,
        *,
        chat_model: str = DEFAULT_CHAT_MODEL,
        report_model: str = DEFAULT_REPORT_MODEL,
        teaching_level: TeachingLevel = TeachingLevel.HIGH,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("DeepSeek 设置")
        self.setModal(True)
        self.resize(420, 280)
        self._status = QLabel("")
        self._status.setWordWrap(True)
        self._key = QLineEdit()
        self._key.setEchoMode(QLineEdit.EchoMode.Password)
        self._key.setPlaceholderText("在此粘贴 sk- 开头的 Key")
        self._current = QLabel("")
        self._refresh_current_key()
        self._chat_model = QLineEdit(chat_model)
        self._report_model = QLineEdit(report_model)
        self._level = QComboBox()
        for level, label in TEACHING_LEVEL_LABELS.items():
            self._level.addItem(label, level.value)
            if level is teaching_level:
                self._level.setCurrentIndex(self._level.count() - 1)
        form = QFormLayout()
        form.addRow("API Key", self._key)
        form.addRow("当前 Key（仅后四位）", self._current)
        form.addRow("问答模型", self._chat_model)
        form.addRow("报告模型", self._report_model)
        form.addRow("教学级别", self._level)
        save = QPushButton("保存 Key")
        save.clicked.connect(self._save_key)
        test = QPushButton("测试连接")
        test.clicked.connect(self._test_clicked)
        clear = QPushButton("清除 Key")
        clear.clicked.connect(self._clear_key)
        keys = QHBoxLayout()
        keys.addWidget(save)
        keys.addWidget(test)
        keys.addWidget(clear)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        buttons.accepted.connect(self.accept)
        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addLayout(keys)
        layout.addWidget(self._status)
        layout.addWidget(buttons)
        self.test_requested = False

    def chat_model(self) -> str:
        return self._chat_model.text().strip() or DEFAULT_CHAT_MODEL

    def report_model(self) -> str:
        return self._report_model.text().strip() or DEFAULT_REPORT_MODEL

    def teaching_level(self) -> TeachingLevel:
        try:
            return TeachingLevel(str(self._level.currentData()))
        except ValueError:
            return TeachingLevel.HIGH

    def _refresh_current_key(self) -> None:
        stored = get_api_key()
        self._current.setText(mask_key(stored))

    def _save_key(self) -> None:
        typed = self._key.text().strip()
        if not typed:
            if get_api_key():
                self._refresh_current_key()
                self._status.setText("已记住当前 Key。要更换请重新粘贴后再保存。")
                return
            self._status.setText("请把 sk- 开头的 Key 粘贴到上面的输入框，再点保存。")
            return
        cleaned = normalize_api_key(typed)
        if not cleaned:
            self._status.setText("没有识别到有效 Key。请只粘贴 sk- 开头的那一串英文和数字。")
            return
        persisted = set_api_key(cleaned, persist=True)
        self._key.clear()
        self._key.setPlaceholderText(mask_key(cleaned))
        self._refresh_current_key()
        if get_api_key():
            if persisted:
                self._status.setText(
                    f"已保存 {mask_key(cleaned)}。完整 Key 不会显示，下次打开仍然有效。"
                )
            else:
                self._status.setText(
                    f"本次已记住 {mask_key(cleaned)}，但未能写入本机；关掉软件后需重填。"
                )
        else:
            self._status.setText("保存失败，请重新粘贴 Key。")

    def _clear_key(self) -> None:
        clear_api_key()
        self._key.clear()
        self._refresh_current_key()
        self._status.setText("已清除 API Key。")

    def _test_clicked(self) -> None:
        typed = self._key.text().strip()
        if typed:
            set_api_key(typed, persist=False)
        if not has_api_key():
            self._status.setText("请先填写 API Key。")
            return
        self.test_requested = True
        self.accept()


class ComposerEdit(QTextEdit):
    send_requested = Signal()

    def keyPressEvent(self, event: QKeyEvent) -> None:  # noqa: ANN001
        if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            if event.modifiers() & Qt.KeyboardModifier.ShiftModifier:
                super().keyPressEvent(event)
                return
            self.send_requested.emit()
            event.accept()
            return
        super().keyPressEvent(event)


class _ThinkingBlock(QFrame):
    """Cursor-style collapsible thinking row above the assistant answer."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("assistantThinking")
        self._plain = ""
        self._running = False
        self._started = 0.0
        self._phase = 0
        self._label_i = 0
        self._timer = QTimer(self)
        self._timer.setInterval(420)
        self._timer.timeout.connect(self._tick)

        self._toggle = QToolButton()
        self._toggle.setObjectName("assistantThinkingToggle")
        self._toggle.setCheckable(True)
        self._toggle.setChecked(False)
        self._toggle.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self._toggle.setAutoRaise(True)
        self._toggle.setArrowType(Qt.ArrowType.RightArrow)
        self._toggle.setText("思考中")
        self._toggle.setCursor(Qt.CursorShape.PointingHandCursor)
        self._toggle.toggled.connect(self._on_toggled)

        self._body = QTextBrowser()
        self._body.setObjectName("assistantThinkingBody")
        self._body.setFrameShape(QFrame.Shape.NoFrame)
        self._body.setOpenExternalLinks(False)
        self._body.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._body.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._body.setUndoRedoEnabled(False)
        self._body.setVisible(False)
        self._body.document().setDefaultStyleSheet(
            "body { color: #8a857c; background: transparent; font-size: 13px; }"
        )

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        layout.addWidget(self._toggle)
        layout.addWidget(self._body)

    def start(self) -> None:
        self._plain = ""
        self._running = True
        self._started = time.monotonic()
        self._phase = 0
        self._label_i = 0
        self._body.clear()
        self._body.setVisible(False)
        self._toggle.setChecked(False)
        self.show()
        self._toggle.setEnabled(True)
        self._set_header(_THINKING_LABELS[0])
        self._timer.start()

    def append(self, text: str) -> None:
        if not text:
            return
        if not self.isVisible():
            self.show()
        self._plain += text
        self._body.setPlainText(self._plain)
        self._fit()
        if not self._toggle.isChecked():
            self._toggle.setChecked(True)

    def finish(self, *, restored: bool = False) -> None:
        self._timer.stop()
        running = self._running
        self._running = False
        if not running:
            if self._plain.strip():
                self._toggle.setChecked(False)
                self._set_header("思考过程")
                self._toggle.setEnabled(True)
                self.show()
            elif restored:
                self.hide()
            return
        elapsed = max(0.0, time.monotonic() - self._started)
        seconds = max(1, int(round(elapsed)))
        self._toggle.setChecked(False)
        self._set_header(f"已思考 {seconds} 秒")
        self._toggle.setEnabled(bool(self._plain.strip()))
        self.show()
        if self._plain.strip():
            self._body.setPlainText(self._plain)
            self._fit()
        else:
            self._body.setVisible(False)

    def plain(self) -> str:
        return self._plain

    def resizeEvent(self, event) -> None:  # noqa: ANN001
        super().resizeEvent(event)
        if self._body.isVisible():
            self._fit()

    def _on_toggled(self, checked: bool) -> None:
        self._toggle.setArrowType(
            Qt.ArrowType.DownArrow if checked else Qt.ArrowType.RightArrow
        )
        has_text = bool(self._plain.strip())
        self._body.setVisible(checked and has_text)
        if checked and has_text:
            self._fit()

    def _tick(self) -> None:
        if not self._running:
            return
        self._phase = (self._phase + 1) % 4
        if self._phase == 0:
            self._label_i = (self._label_i + 1) % len(_THINKING_LABELS)
        dots = "." * (self._phase + 1)
        label = _THINKING_LABELS[self._label_i]
        if self._plain.strip():
            label = "思考中"
        self._set_header(f"{label}{dots}")

    def _set_header(self, title: str) -> None:
        self._toggle.setText(title)
        self._toggle.setToolTip("展开或收起思考过程" if self._plain.strip() else title)

    def _fit(self) -> None:
        if not self._body.isVisible():
            return
        doc = self._body.document()
        width = max(self.width() - 8, 160)
        doc.setTextWidth(width)
        height = int(doc.size().height()) + 8
        self._body.setFixedHeight(max(min(height, 240), 28))


class _MessageBlock(QFrame):
    def __init__(
        self,
        kind: str,
        text: str = "",
        *,
        markdown: bool = False,
        live: bool = False,
        reasoning: str = "",
    ) -> None:
        super().__init__()
        self.setObjectName("assistantMessage")
        self.setProperty("kind", kind)
        self._plain = text
        role = QLabel(_ROLE_LABELS.get(kind, kind))
        role.setObjectName("assistantRole")
        self._thinking: _ThinkingBlock | None = None
        self._body = QTextBrowser()
        self._body.setObjectName("assistantMessageBody")
        self._body.setFrameShape(QFrame.Shape.NoFrame)
        self._body.setOpenExternalLinks(False)
        self._body.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._body.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._body.setUndoRedoEnabled(False)
        self._body.document().setDefaultStyleSheet(
            "body { color: #f0ece4; background: transparent; font-size: 15px; }"
            "a { color: #d4c4a8; }"
            "code { background: #2a2722; color: #f0ece4; }"
        )
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        layout.addWidget(role)
        if kind == "assistant":
            self._thinking = _ThinkingBlock()
            layout.addWidget(self._thinking)
            if live:
                self._thinking.start()
            elif reasoning:
                self._thinking.append(reasoning)
                self._thinking.finish(restored=True)
            else:
                self._thinking.hide()
        layout.addWidget(self._body)
        if kind == "assistant" and live and not text:
            self._body.hide()
        if markdown:
            self.set_markdown(text)
        else:
            self._body.setPlainText(text)
            self._fit()

    def append_reasoning(self, text: str) -> None:
        if self._thinking is not None:
            self._thinking.append(text)

    def reasoning(self) -> str:
        if self._thinking is None:
            return ""
        return self._thinking.plain()

    def finish_thinking(self, *, restored: bool = False) -> None:
        if self._thinking is not None:
            self._thinking.finish(restored=restored)

    def append_plain(self, text: str) -> None:
        if self._thinking is not None and self._thinking._running:
            self._thinking.finish()
        self._plain += text
        self._body.setPlainText(self._plain)
        self._body.show()
        self._clear_selection()
        self._fit()

    def set_markdown(self, text: str) -> None:
        self._plain = text
        self._body.setMarkdown(text or "")
        if text:
            self._body.show()
        self._clear_selection()
        self._fit()

    def plain(self) -> str:
        return self._plain

    def resizeEvent(self, event) -> None:  # noqa: ANN001
        super().resizeEvent(event)
        self._fit()

    def _clear_selection(self) -> None:
        cursor = self._body.textCursor()
        cursor.clearSelection()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        self._body.setTextCursor(cursor)

    def _fit(self) -> None:
        doc = self._body.document()
        width = max(self.width() - 4, 160)
        doc.setTextWidth(width)
        height = int(doc.size().height()) + 10
        if self._body.height() != height:
            self._body.setFixedHeight(max(height, 28))


class _ChatTranscript(QScrollArea):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("assistantChat")
        self.setWidgetResizable(True)
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._host = QWidget()
        self._host.setObjectName("assistantChatHost")
        self._col = QVBoxLayout(self._host)
        self._col.setContentsMargins(32, 24, 40, 24)
        self._col.setSpacing(28)
        self._col.addStretch(1)
        self.setWidget(self._host)
        self._current: _MessageBlock | None = None

    def add_user(self, text: str) -> None:
        self._insert(_MessageBlock("user", text, markdown=False))

    def begin_assistant(self, *, live: bool = True, reasoning: str = "") -> None:
        block = _MessageBlock("assistant", "", markdown=False, live=live, reasoning=reasoning)
        self._current = block
        self._insert(block)

    def append_reasoning(self, text: str) -> None:
        if self._current is None:
            self.begin_assistant()
        assert self._current is not None
        self._current.append_reasoning(text)
        self._scroll_to_end()

    def append_chunk(self, text: str) -> None:
        if self._current is None:
            self.begin_assistant()
        assert self._current is not None
        self._current.append_plain(text)
        self._scroll_to_end()

    def finish_assistant(self, text: str, *, cancelled: bool = False) -> None:
        if self._current is None:
            return
        self._current.finish_thinking()
        if cancelled:
            body = (self._current.plain() + "\n\n生成已取消").strip()
            self._current.set_markdown(body)
        else:
            self._current.set_markdown(text or self._current.plain())
        self._current = None
        self._scroll_to_end()

    def add_local(self, markdown: str) -> None:
        self._current = None
        self._insert(_MessageBlock("local", markdown, markdown=True))

    def clear_messages(self) -> None:
        self._current = None
        while self._col.count() > 1:
            item = self._col.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

    def to_plain(self) -> str:
        parts: list[str] = []
        for index in range(self._col.count()):
            widget = self._col.itemAt(index).widget()
            if isinstance(widget, _MessageBlock):
                parts.append(widget.plain())
        return "\n".join(parts)

    def _insert(self, block: _MessageBlock) -> None:
        self._col.insertWidget(self._col.count() - 1, block)
        QTimer.singleShot(0, self._scroll_to_end)

    def _scroll_to_end(self) -> None:
        bar = self.verticalScrollBar()
        bar.setValue(bar.maximum())


class AssistantWindow(QMainWindow):
    """Non-modal chat window; closing hides it and does not quit TrackLab."""

    visibility_changed = Signal(bool)

    def __init__(self, panel: AssistantPanel, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("assistantWindow")
        self.setWindowTitle("TrackLab 助手")
        self.setAttribute(Qt.WidgetAttribute.WA_QuitOnClose, False)
        self.setMinimumSize(800, 560)
        self.resize(960, 720)
        self.setCentralWidget(panel)

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: ANN001
        event.ignore()
        self.hide()

    def showEvent(self, event: QShowEvent) -> None:  # noqa: ANN001
        super().showEvent(event)
        self.visibility_changed.emit(True)

    def hideEvent(self, event: QHideEvent) -> None:  # noqa: ANN001
        super().hideEvent(event)
        self.visibility_changed.emit(False)


class AssistantPanel(QWidget):
    analyze_requested = Signal()
    confirm_requested = Signal(str)
    send_requested = Signal(str)
    stop_requested = Signal()
    report_requested = Signal()
    export_requested = Signal()
    copy_report_requested = Signal()
    preview_requested = Signal()
    clear_chat_requested = Signal()
    level_changed = Signal(str)
    settings_requested = Signal()
    length_changed = Signal(float)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("assistantPanel")
        self._streaming = False
        self._analysis: ExperimentAnalysis | None = None
        self._confirmed: ExperimentType | None = None
        self._chat_model = DEFAULT_CHAT_MODEL
        self._report_model = DEFAULT_REPORT_MODEL

        header = self._build_header()
        context = self._build_context()
        chat = self._build_chat()

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setObjectName("assistantSplitter")
        splitter.setChildrenCollapsible(False)
        splitter.addWidget(context)
        splitter.addWidget(chat)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([360, 600])
        self._splitter = splitter

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(header)
        layout.addWidget(splitter, stretch=1)

    def _build_header(self) -> QWidget:
        header = QFrame()
        header.setObjectName("assistantHeader")
        row = QHBoxLayout(header)
        row.setContentsMargins(16, 10, 16, 10)
        row.setSpacing(8)

        self._chip_key = _chip("未配置 Key", "bad")
        self._chip_model = _chip(DEFAULT_CHAT_MODEL, "")
        self._chip_confirm = _chip("未确认实验", "")
        self._level = QComboBox()
        self._level.setObjectName("assistantLevel")
        for level, label in TEACHING_LEVEL_LABELS.items():
            self._level.addItem(label, level.value)
        self._level.setCurrentIndex(1)
        self._level.currentIndexChanged.connect(self._on_level)

        row.addWidget(self._chip_key)
        row.addWidget(self._chip_model)
        row.addWidget(self._chip_confirm)
        row.addStretch(1)
        row.addWidget(QLabel("教学级别"))
        row.addWidget(self._level)

        self._notice = QLabel("")
        self._notice.setObjectName("assistantNotice")
        self._notice.setWordWrap(True)
        self._notice.hide()
        wrap = QWidget()
        wrap.setObjectName("assistantHeaderWrap")
        col = QVBoxLayout(wrap)
        col.setContentsMargins(0, 0, 0, 0)
        col.setSpacing(0)
        col.addWidget(header)
        col.addWidget(self._notice)
        return wrap

    def _build_context(self) -> QWidget:
        side = QWidget()
        side.setObjectName("assistantContext")
        col = QVBoxLayout(side)
        col.setContentsMargins(12, 12, 12, 12)
        col.setSpacing(10)

        self._ready = QLabel("打开视频并完成轨迹后，即可识别实验。")
        self._ready.setObjectName("assistantReady")
        self._ready.setWordWrap(True)
        self._stale = QLabel("")
        self._stale.setObjectName("assistantStale")
        self._stale.setWordWrap(True)
        self._stale.hide()
        self._analyze_btn = QPushButton("分析实验")
        self._analyze_btn.setObjectName("assistantPrimaryButton")
        self._analyze_btn.clicked.connect(self.analyze_requested.emit)

        status, status_layout = _card("实验状态")
        status_layout.addWidget(self._ready)
        status_layout.addWidget(self._stale)
        status_layout.addWidget(self._analyze_btn)

        self._candidates = QTextBrowser()
        self._candidates.setObjectName("assistantCandidates")
        self._candidates.setMinimumHeight(110)
        self._type = QComboBox()
        for kind, label in EXPERIMENT_LABELS.items():
            if kind is ExperimentType.UNKNOWN:
                continue
            self._type.addItem(label, kind.value)
        self._confirm_btn = QPushButton("确认此实验")
        self._confirm_btn.setObjectName("assistantPrimaryButton")
        self._confirm_btn.clicked.connect(self._on_confirm)
        self._length = QDoubleSpinBox()
        self._length.setRange(0.0, 20.0)
        self._length.setDecimals(3)
        self._length.setSuffix(" m")
        self._length.setSpecialValueText("未设置")
        self._length.valueChanged.connect(self._on_length)
        self._length_hint = QLabel("单摆需要已知摆长，才能由周期反推重力加速度。")
        self._length_hint.setObjectName("assistantHint")
        self._length_hint.setWordWrap(True)
        self._length_box = QWidget()
        length_layout = QVBoxLayout(self._length_box)
        length_layout.setContentsMargins(0, 0, 0, 0)
        length_layout.setSpacing(6)
        length_row = QHBoxLayout()
        length_row.addWidget(QLabel("摆长"))
        length_row.addWidget(self._length, stretch=1)
        length_layout.addLayout(length_row)
        length_layout.addWidget(self._length_hint)
        self._length_box.hide()
        self._type.currentIndexChanged.connect(self._refresh_pendulum_row)

        identify, identify_layout = _card("实验识别")
        identify_layout.addWidget(self._candidates)
        type_row = QHBoxLayout()
        type_row.addWidget(QLabel("类型"))
        type_row.addWidget(self._type, stretch=1)
        identify_layout.addLayout(type_row)
        identify_layout.addWidget(self._length_box)
        identify_layout.addWidget(self._confirm_btn)

        self._measures = QTextBrowser()
        self._measures.setObjectName("assistantMeasures")
        self._measures.setMinimumHeight(120)
        self._cal = QLabel("标定：未读取")
        self._cal.setObjectName("assistantCal")
        self._cal.setWordWrap(True)
        measures, measures_layout = _card("测量与标定")
        measures_layout.addWidget(self._cal)
        measures_layout.addWidget(self._measures)

        self._report = QTextBrowser()
        self._report.setObjectName("assistantReport")
        self._report.setMinimumHeight(140)
        self._report_btn = QPushButton("生成报告")
        self._report_btn.clicked.connect(self.report_requested.emit)
        self._copy_btn = QPushButton("复制")
        self._copy_btn.clicked.connect(self.copy_report_requested.emit)
        self._export_btn = QPushButton("导出 Markdown")
        self._export_btn.clicked.connect(self.export_requested.emit)
        report, report_layout = _card("实验报告")
        report_layout.addWidget(self._report, stretch=1)
        report_btns = QHBoxLayout()
        report_btns.addWidget(self._report_btn)
        report_btns.addWidget(self._copy_btn)
        report_btns.addWidget(self._export_btn)
        report_layout.addLayout(report_btns)

        self._preview_btn = QPushButton("查看将发送的数据")
        self._preview_btn.clicked.connect(self.preview_requested.emit)
        settings_btn = QPushButton("DeepSeek 设置…")
        settings_btn.clicked.connect(self.settings_requested.emit)
        teach, teach_layout = _card("教学数据")
        teach_layout.addWidget(
            QLabel("发给模型的是本地拟合摘要，不含视频、路径或 API Key。")
        )
        teach_layout.addWidget(self._preview_btn)
        teach_layout.addWidget(settings_btn)

        col.addWidget(status)
        col.addWidget(identify)
        col.addWidget(measures)
        col.addWidget(report)
        col.addWidget(teach)
        col.addStretch(1)

        scroll = QScrollArea()
        scroll.setObjectName("assistantContextScroll")
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setWidget(side)
        scroll.setMinimumWidth(300)
        scroll.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)
        return scroll

    def _build_chat(self) -> QWidget:
        column = QWidget()
        column.setObjectName("assistantChatColumn")
        layout = QVBoxLayout(column)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        title_bar = QFrame()
        title_bar.setObjectName("assistantChatTitle")
        title_row = QHBoxLayout(title_bar)
        title_row.setContentsMargins(16, 10, 16, 8)
        title = QLabel("对话")
        title.setObjectName("assistantChatHeading")
        self._clear_btn = QPushButton("清空会话")
        self._clear_btn.clicked.connect(self.clear_chat_requested.emit)
        title_row.addWidget(title)
        title_row.addStretch(1)
        title_row.addWidget(self._clear_btn)

        self._chat = _ChatTranscript()

        composer = QFrame()
        composer.setObjectName("assistantComposer")
        composer_layout = QVBoxLayout(composer)
        composer_layout.setContentsMargins(16, 10, 16, 14)
        composer_layout.setSpacing(8)
        self._input = ComposerEdit()
        self._input.setObjectName("assistantInput")
        self._input.setPlaceholderText("向助手提问…（Enter 发送，Shift+Enter 换行）")
        self._input.setMinimumHeight(88)
        self._input.setMaximumHeight(160)
        self._input.send_requested.connect(self._on_send)
        self._send_btn = QPushButton("发送")
        self._send_btn.setObjectName("assistantPrimaryButton")
        self._send_btn.clicked.connect(self._on_send)
        self._stop_btn = QPushButton("停止")
        self._stop_btn.clicked.connect(self.stop_requested.emit)
        self._stop_btn.setEnabled(False)
        send_row = QHBoxLayout()
        send_row.addStretch(1)
        send_row.addWidget(self._stop_btn)
        send_row.addWidget(self._send_btn)
        composer_layout.addWidget(self._input)
        composer_layout.addLayout(send_row)

        layout.addWidget(title_bar)
        layout.addWidget(self._chat, stretch=1)
        layout.addWidget(composer)
        return column

    def teaching_level(self) -> TeachingLevel:
        try:
            return TeachingLevel(str(self._level.currentData()))
        except ValueError:
            return TeachingLevel.HIGH

    def set_teaching_level(self, level: TeachingLevel) -> None:
        for i in range(self._level.count()):
            if self._level.itemData(i) == level.value:
                self._level.blockSignals(True)
                self._level.setCurrentIndex(i)
                self._level.blockSignals(False)
                return

    def pendulum_length_m(self) -> float | None:
        value = float(self._length.value())
        return None if value <= 0 else value

    def set_pendulum_length(self, value: float | None) -> None:
        self._length.blockSignals(True)
        self._length.setValue(0.0 if value is None else float(value))
        self._length.blockSignals(False)

    def set_models(self, chat_model: str, report_model: str) -> None:
        self._chat_model = chat_model or DEFAULT_CHAT_MODEL
        self._report_model = report_model or DEFAULT_REPORT_MODEL
        self._chip_model.setText(self._chat_model)
        self._chip_model.setToolTip(f"问答 {self._chat_model}\n报告 {self._report_model}")

    def set_readiness(
        self,
        *,
        has_video: bool,
        visible_points: int,
        calibrated: bool,
        shake_on: bool,
        has_key: bool,
        sam_running: bool,
        confirmed: bool,
    ) -> None:
        parts = []
        parts.append("已打开视频" if has_video else "未打开视频")
        parts.append(f"可见点 {visible_points}")
        parts.append("已标定" if calibrated else "未标定")
        parts.append("背景补偿开" if shake_on else "背景补偿关")
        parts.append("已配置 Key" if has_key else "未配置 Key")
        if sam_running:
            parts.append("SAM 跟踪进行中")
        if confirmed:
            parts.append("实验已确认")
        self._ready.setText(" · ".join(parts))
        _set_chip(self._chip_key, "已配置 Key" if has_key else "未配置 Key", "ok" if has_key else "bad")
        can_analyze = has_video and visible_points >= 6 and not sam_running
        self._analyze_btn.setEnabled(not self._streaming)
        self._analyze_btn.setToolTip(
            "" if can_analyze else "需要先打开视频并完成至少 6 个可见轨迹点。"
        )
        self._confirm_btn.setEnabled(visible_points >= 6)
        self._send_btn.setEnabled(has_key and not sam_running and not self._streaming)
        self._report_btn.setEnabled(has_key and confirmed and not sam_running and not self._streaming)
        self._preview_btn.setEnabled(True)
        self._refresh_confirm_chip()

    def set_stale(self, stale: bool, message: str = "") -> None:
        self._stale.setVisible(stale)
        self._stale.setText(message or "数据已变化，请重新分析。")
        self._refresh_confirm_chip()

    def set_analysis(self, analysis: ExperimentAnalysis | None, confirmed: ExperimentType | None) -> None:
        self._analysis = analysis
        self._confirmed = confirmed
        if analysis is None:
            self._candidates.setPlainText("尚未分析。点击「分析实验」用本地轨迹做识别。")
            self._measures.setPlainText("尚无拟合数值。")
            self._cal.setText("标定：未读取")
            self._refresh_confirm_chip()
            self._refresh_pendulum_row()
            return
        lines = []
        if analysis.warnings:
            lines.extend(analysis.warnings)
        if analysis.missing:
            lines.append("缺失：" + "、".join(analysis.missing))
        if not analysis.candidates:
            lines.append("无法可靠识别，请手动选择类型。")
        for item in analysis.candidates:
            mark = " ← 建议" if analysis.selected is item else ""
            lines.append(
                f"{item.label}  置信度 {item.confidence:.2f}  R² {item.fit.r2:.3f}  "
                f"nRMSE {item.fit.nrmse:.3f}{mark}"
            )
            lines.extend(f"  · {bit}" for bit in item.evidence)
        self._candidates.setPlainText("\n".join(lines) or "无候选。")
        target = confirmed or (analysis.selected.experiment_type if analysis.selected else None)
        if target is not None:
            for i in range(self._type.count()):
                if self._type.itemData(i) == target.value:
                    self._type.setCurrentIndex(i)
                    break
        cal = "已标定" if analysis.calibration_active else "未标定"
        self._cal.setText(
            f"标定：{cal} · 位置 {analysis.position_unit} · 速度 {analysis.speed_unit}\n"
            f"覆盖率 {analysis.coverage:.3f} · 跟踪置信度 {analysis.mean_track_confidence:.3f}"
        )
        self._measures.setPlainText(_format_measures(analysis, confirmed))
        self._refresh_confirm_chip()
        self._refresh_pendulum_row()

    def set_busy(self, busy: bool) -> None:
        self._streaming = busy
        self._stop_btn.setEnabled(busy)
        self._send_btn.setEnabled(not busy)
        self._analyze_btn.setEnabled(not busy)
        self._report_btn.setEnabled(not busy)

    def set_notice(self, text: str) -> None:
        self._notice.setText(text)
        self._notice.setVisible(bool(text.strip()))

    def add_analysis_result(self, analysis: ExperimentAnalysis | None) -> None:
        self._chat.add_local(_format_analysis_note(analysis))

    def add_user_message(self, text: str) -> None:
        self._chat.add_user(text)

    def begin_assistant_message(self, *, live: bool = True, reasoning: str = "") -> None:
        self._chat.begin_assistant(live=live, reasoning=reasoning)

    def append_reasoning(self, text: str) -> None:
        self._chat.append_reasoning(text)

    def append_chunk(self, text: str) -> None:
        self._chat.append_chunk(text)

    def current_reasoning(self) -> str:
        if self._chat._current is None:
            return ""
        return self._chat._current.reasoning()

    def finish_assistant_message(self, text: str, *, cancelled: bool = False) -> None:
        self._chat.finish_assistant(text, cancelled=cancelled)

    def clear_chat(self) -> None:
        self._chat.clear_messages()

    def chat_text(self) -> str:
        return self._chat.to_plain()

    def take_input(self) -> str:
        text = self._input.toPlainText().strip()
        self._input.clear()
        return text

    def set_report(self, markdown: str) -> None:
        self._report.setMarkdown(markdown or "")

    def report_text(self) -> str:
        return self._report.toMarkdown()

    def _refresh_confirm_chip(self) -> None:
        if self._stale.isVisible():
            _set_chip(self._chip_confirm, "数据已过期", "warn")
            return
        if self._confirmed is not None:
            label = EXPERIMENT_LABELS.get(self._confirmed, self._confirmed.value)
            _set_chip(self._chip_confirm, f"已确认：{label}", "ok")
            return
        _set_chip(self._chip_confirm, "未确认实验", "")

    def _refresh_pendulum_row(self) -> None:
        selected = str(self._type.currentData() or "")
        needed = selected == ExperimentType.PENDULUM.value
        if self._confirmed is ExperimentType.PENDULUM:
            needed = True
        if self._analysis is not None:
            needed = needed or any(
                item.experiment_type is ExperimentType.PENDULUM
                for item in self._analysis.candidates
            )
        self._length_box.setVisible(needed)

    def _on_confirm(self) -> None:
        value = str(self._type.currentData() or "")
        if value:
            self.confirm_requested.emit(value)

    def _on_send(self) -> None:
        if self._streaming:
            return
        text = self.take_input()
        if text:
            self.send_requested.emit(text)

    def _on_level(self) -> None:
        self.level_changed.emit(str(self._level.currentData() or TeachingLevel.HIGH.value))

    def _on_length(self, value: float) -> None:
        self.length_changed.emit(float(value))


def _card(title: str) -> tuple[QGroupBox, QVBoxLayout]:
    box = QGroupBox(title)
    box.setObjectName("assistantCard")
    layout = QVBoxLayout(box)
    layout.setContentsMargins(10, 16, 10, 10)
    layout.setSpacing(8)
    return box, layout


def _chip(text: str, tone: str) -> QLabel:
    label = QLabel(text)
    label.setObjectName("assistantChip")
    _set_chip(label, text, tone)
    return label


def _set_chip(label: QLabel, text: str, tone: str) -> None:
    label.setText(text)
    label.setProperty("tone", tone)
    style = label.style()
    if style is not None:
        style.unpolish(label)
        style.polish(label)


def _pick_candidate(
    analysis: ExperimentAnalysis | None, confirmed: ExperimentType | None
) -> ExperimentCandidate | None:
    if analysis is None:
        return None
    if confirmed is not None:
        for item in analysis.candidates:
            if item.experiment_type is confirmed:
                return item
    return analysis.selected


def _format_measures(analysis: ExperimentAnalysis, confirmed: ExperimentType | None) -> str:
    candidate = _pick_candidate(analysis, confirmed)
    if candidate is None:
        return "尚无拟合数值。确认实验类型后会显示本地运动学结果。"
    rows = [
        f"{candidate.label}",
        f"置信度 {candidate.confidence:.3f}  ·  R² {candidate.fit.r2:.4f}  ·  nRMSE {candidate.fit.nrmse:.4f}",
        (
            f"拟合区间 第 {candidate.fit.frame_start}–{candidate.fit.frame_end} 帧，"
            f"{candidate.fit.time_start_s:.3f}–{candidate.fit.time_end_s:.3f} s，"
            f"{candidate.fit.n_samples} 点"
        ),
    ]
    for name, value in candidate.fit.parameters.items():
        unit = candidate.fit.units.get(name, "")
        if value is None:
            rows.append(f"{name}：无法计算")
        else:
            suffix = f" {unit}" if unit else ""
            rows.append(f"{name} = {value:.6g}{suffix}")
    if candidate.warnings:
        rows.append("警告：" + "；".join(candidate.warnings))
    return "\n".join(rows)


def _format_analysis_note(analysis: ExperimentAnalysis | None) -> str:
    if analysis is None:
        return "本地识别没有返回结果。请确认已打开视频并完成轨迹后再试。"
    lines = ["这是本地轨迹拟合的结果，还没有调用 DeepSeek。"]
    if analysis.warnings:
        lines.extend(f"- {item}" for item in analysis.warnings)
    if analysis.missing:
        missing = "、".join(analysis.missing)
        lines.append(f"- 缺失条件：{missing}")
    if not analysis.candidates:
        lines.append("无法可靠识别类型，请在左侧手动选择后点「确认此实验」。")
        return "\n".join(lines)
    lines.append("")
    for item in analysis.candidates:
        mark = "（建议）" if analysis.selected is item else ""
        lines.append(
            f"- **{item.label}**{mark}：置信度 {item.confidence:.2f}，"
            f"R² {item.fit.r2:.3f}，nRMSE {item.fit.nrmse:.3f}"
        )
    lines.append("")
    lines.append("请在左侧确认实验类型，然后在下方提问。")
    return "\n".join(lines)


def show_payload_preview(parent: QWidget | None, payload: str) -> None:
    dialog = QDialog(parent)
    dialog.setWindowTitle("将发送给 DeepSeek 的数据")
    dialog.resize(520, 420)
    view = QTextEdit()
    view.setReadOnly(True)
    view.setPlainText(payload)
    buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
    buttons.rejected.connect(dialog.reject)
    layout = QVBoxLayout(dialog)
    layout.addWidget(QLabel("不含视频、文件路径或 API Key。"))
    layout.addWidget(view)
    layout.addWidget(buttons)
    dialog.exec()


def warn_missing_key(parent: QWidget | None) -> None:
    QMessageBox.information(parent, "需要 API Key", "请先在 DeepSeek 设置中填写 API Key。离线仍可做本地实验识别。")
