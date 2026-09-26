"""Floating card that walks the user through a predicted (fit) point.

Shown next to the pulsing yellow ring on frames where the tracker was not
sure and the position came from the trajectory fit. Three states:

* predict — "this point was predicted": 保留 (F) / 手动打点 (Shift+左键)
* manual  — how to place the point by hand
* done    — short confirmation, with a jump to the next pending point

The texts are built by plain functions so they can be tested without a
display.
"""

from __future__ import annotations

from dataclasses import dataclass

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from app.marks import PENDING_TEXT_ON_YELLOW, PENDING_YELLOW

FIT_ACCENT = PENDING_YELLOW  # same yellow as the pending points on video and charts
CARD_WIDTH = 216
DONE_VISIBLE_MS = 2600

STATE_HIDDEN = "hidden"
STATE_PREDICT = "predict"
STATE_MANUAL = "manual"
STATE_DONE = "done"


@dataclass(frozen=True)
class CardText:
    title: str
    body: str
    footer: str
    primary: str  # empty = no button
    secondary: str  # empty = no button


def predict_text(frame: int, sigma_px: float, points: int, method: str, remaining: int) -> CardText:
    body = f"轨迹拟合的位置 · ±{sigma_px:.1f} px"
    if method != "interpolate":
        body += " · 单侧外推"
    others = max(0, remaining - 1)
    footer = "⇧F 舍弃" + (f" · ⌘F 下一个（剩 {others}）" if others else "")
    return CardText(
        title=f"预测点 · 第 {frame + 1} 帧",
        body=body,
        footer=footer,
        primary="保留  F",
        secondary="手动打点",
    )


def manual_text(frame: int) -> CardText:
    return CardText(
        title=f"手动打点 · 第 {frame + 1} 帧",
        body="按住 Shift，左键点物体中心",
        footer="",
        primary="",
        secondary="取消  Esc",
    )


def done_text(action: str, remaining: int) -> CardText:
    if remaining > 0:
        return CardText(title=f"✓ {action}", body=f"还剩 {remaining} 个", footer="", primary="下一个  ⌘F", secondary="")
    return CardText(title=f"✓ {action}", body="全部确认完了", footer="", primary="", secondary="")


class FitPromptCard(QFrame):
    """Non-modal card; never takes keyboard focus so F / Shift+F keep working."""

    keep_requested = Signal()
    manual_requested = Signal()
    cancel_requested = Signal()
    next_requested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("fitPromptCard")
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setFixedWidth(CARD_WIDTH)
        self.setStyleSheet(
            f"""
            QFrame#fitPromptCard {{
                background: rgba(18, 24, 30, 236);
                border: 1px solid rgba(255, 214, 10, 150);
                border-left: 4px solid {FIT_ACCENT};
                border-radius: 10px;
            }}
            QLabel#fitPromptTitle {{ color: {FIT_ACCENT}; font-weight: 600; font-size: 13px; }}
            QLabel#fitPromptBody {{ color: #e8eef2; font-size: 12px; }}
            QLabel#fitPromptFooter {{ color: #93a4ae; font-size: 11px; }}
            QPushButton#fitPromptPrimary {{
                background: {FIT_ACCENT}; color: {PENDING_TEXT_ON_YELLOW}; border: none;
                border-radius: 6px; padding: 5px 10px; font-weight: 600;
            }}
            QPushButton#fitPromptPrimary:hover {{ background: #ffe45c; }}
            QPushButton#fitPromptSecondary {{
                background: transparent; color: {FIT_ACCENT};
                border: 1px solid {FIT_ACCENT}; border-radius: 6px; padding: 4px 10px;
            }}
            QPushButton#fitPromptSecondary:hover {{ background: rgba(255, 214, 10, 40); }}
            """
        )
        self._state = STATE_HIDDEN
        self._frame: int | None = None
        self._title = QLabel(self)
        self._title.setObjectName("fitPromptTitle")
        self._body = QLabel(self)
        self._body.setObjectName("fitPromptBody")
        self._body.setWordWrap(True)
        self._footer = QLabel(self)
        self._footer.setObjectName("fitPromptFooter")
        self._footer.setWordWrap(True)
        self._primary = QPushButton(self)
        self._primary.setObjectName("fitPromptPrimary")
        self._secondary = QPushButton(self)
        self._secondary.setObjectName("fitPromptSecondary")
        for button in (self._primary, self._secondary):
            button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            button.setCursor(Qt.CursorShape.PointingHandCursor)
        self._primary.clicked.connect(self._on_primary)
        self._secondary.clicked.connect(self._on_secondary)
        buttons = QHBoxLayout()
        buttons.setContentsMargins(0, 2, 0, 0)
        buttons.setSpacing(8)
        buttons.addWidget(self._primary)
        buttons.addWidget(self._secondary)
        buttons.addStretch()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(4)
        layout.addWidget(self._title)
        layout.addWidget(self._body)
        layout.addLayout(buttons)
        layout.addWidget(self._footer)
        self._hide_timer = QTimer(self)
        self._hide_timer.setSingleShot(True)
        self._hide_timer.timeout.connect(self.dismiss)
        self.hide()

    @property
    def state(self) -> str:
        return self._state

    @property
    def frame(self) -> int | None:
        return self._frame

    def show_prediction(self, frame: int, sigma_px: float, points: int, method: str, remaining: int) -> None:
        self._apply(STATE_PREDICT, frame, predict_text(frame, sigma_px, points, method, remaining))

    def show_manual(self, frame: int) -> None:
        self._apply(STATE_MANUAL, frame, manual_text(frame))

    def show_done(self, frame: int, action: str, remaining: int) -> None:
        self._apply(STATE_DONE, frame, done_text(action, remaining))
        self._hide_timer.start(DONE_VISIBLE_MS)

    def dismiss(self) -> None:
        self._hide_timer.stop()
        self._state = STATE_HIDDEN
        self._frame = None
        self.hide()

    def _apply(self, state: str, frame: int, text: CardText) -> None:
        if state != STATE_DONE:
            self._hide_timer.stop()
        self._state = state
        self._frame = frame
        self._title.setText(text.title)
        self._body.setText(text.body)
        self._footer.setText(text.footer)
        self._footer.setVisible(bool(text.footer))
        self._primary.setText(text.primary)
        self._primary.setVisible(bool(text.primary))
        self._secondary.setText(text.secondary)
        self._secondary.setVisible(bool(text.secondary))
        self._fit_height()
        self.show()
        self.raise_()

    def _fit_height(self) -> None:
        """Wrapped labels need height-for-width, not the plain size hint."""
        layout = self.layout()
        height = 0
        if layout is not None:
            layout.activate()
            if layout.hasHeightForWidth():
                height = int(layout.heightForWidth(CARD_WIDTH))
        if height <= 0:
            height = int(self.sizeHint().height())
        self.resize(CARD_WIDTH, max(height, 48))

    def _on_primary(self) -> None:
        if self._state == STATE_PREDICT:
            self.keep_requested.emit()
        elif self._state == STATE_DONE:
            self.next_requested.emit()

    def _on_secondary(self) -> None:
        if self._state == STATE_PREDICT:
            self.manual_requested.emit()
        elif self._state == STATE_MANUAL:
            self.cancel_requested.emit()
