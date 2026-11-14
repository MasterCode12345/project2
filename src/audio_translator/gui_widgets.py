"""Widgets of the GUI: the light, the translation pane, collapsible sections, the compact window.

They only display what ``gui.MainWindow`` hands them; none talks to the orchestrator.
"""

from __future__ import annotations

import statistics
from typing import Callable

from PySide6.QtCore import QSize, Qt, Signal
from PySide6.QtGui import QFontMetrics, QGuiApplication
from PySide6.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QSizePolicy,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from .gui_strings import VI, fmt_seconds

# Speaker indicator: colour + icon + words (never colour alone). "off" = no session.
LIGHT = {
    "green": ("#2e7d32", "✓", VI["level_green"]),
    "yellow": ("#f9a825", "⚠", VI["level_yellow"]),
    "red": ("#c62828", "✋", VI["level_red"]),
    "off": ("#9e9e9e", "–", ""),
}
MAX_SENTENCES = 200


class LightWidget(QWidget):
    """Coloured disc with an icon, the main reason, and (optionally) the other active reasons."""

    def __init__(self, size: int = 72, *, show_others: bool = True, text_pt: int = 16) -> None:
        super().__init__()
        self.size_px = size
        self.disc = QLabel()
        self.disc.setFixedSize(size, size)
        self.disc.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.text = QLabel()
        self.text.setWordWrap(True)
        self.text.setStyleSheet(f"font-size: {text_pt}pt; font-weight: bold;")
        self.others = QLabel()
        self.others.setWordWrap(True)
        self.others.setStyleSheet("color: #757575;")
        self.others.setVisible(show_others)
        self.show_others = show_others
        texts = QVBoxLayout()
        texts.setContentsMargins(0, 0, 0, 0)
        texts.addWidget(self.text)
        texts.addWidget(self.others)
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(self.disc, 0, Qt.AlignmentFlag.AlignVCenter)
        row.addLayout(texts, 1)
        row.setAlignment(texts, Qt.AlignmentFlag.AlignVCenter)
        self.level = "off"
        self.set_light("off")

    def set_light(self, level: str, reason: str = "", others: list[str] | None = None) -> None:
        color, icon, word = LIGHT.get(level, LIGHT["off"])
        self.level = level if level in LIGHT else "off"
        self.disc.setText(icon)
        self.disc.setStyleSheet(f"background-color: {color}; color: white; border-radius: {self.size_px // 2}px; "
                                f"font-size: {max(12, self.size_px * 2 // 5)}pt; font-weight: bold;")
        self.disc.setToolTip(word)
        self.text.setText(f"{word}: {reason}" if word and reason else (reason or VI["light_off"]))
        others = others or []
        self.others.setText(VI["light_others"].format("; ".join(others)) if others and self.show_others else "")
        tip = "\n".join([reason, *others]) if reason else ""
        self.text.setToolTip(tip)
        self.others.setToolTip(tip)


class _SentenceBlock(QWidget):
    def __init__(self, vi: str, en: str, latency: float | None) -> None:
        super().__init__()
        self.vi = QLabel(vi)
        self.vi.setWordWrap(True)
        self.vi.setStyleSheet("color: #8a8a8a; font-size: 10pt;")
        self.en = QLabel(en)
        self.en.setWordWrap(True)
        self.en.setStyleSheet("font-size: 14pt;")
        self.latency = QLabel(fmt_seconds(latency))
        self.latency.setStyleSheet("color: #8a8a8a;")
        self.latency.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignBottom)
        texts = QVBoxLayout()
        texts.setContentsMargins(0, 0, 0, 0)
        texts.setSpacing(1)
        texts.addWidget(self.vi)
        texts.addWidget(self.en)
        row = QHBoxLayout(self)
        row.setContentsMargins(6, 4, 6, 6)
        row.addLayout(texts, 1)
        row.addWidget(self.latency)


class TranscriptView(QWidget):
    """Sentence blocks (VI small grey, EN larger, latency right), newest last, at most
    ``MAX_SENTENCES``; follows the newest sentence unless the user scrolled up."""

    def __init__(self, clipboard: Callable[[], object] | None = None) -> None:
        super().__init__()
        self.title = QLabel(VI["transcript_title"])
        self.title.setStyleSheet("font-weight: bold;")
        self.copy_button = QPushButton(VI["copy_all"])
        self.clear_button = QPushButton(VI["clear"])
        self.list = QListWidget()
        self.list.setVerticalScrollMode(QListWidget.ScrollMode.ScrollPerPixel)
        self.list.setSelectionMode(QListWidget.SelectionMode.NoSelection)
        self.stats = QLabel()
        self.stats.setStyleSheet("color: #8a8a8a;")
        head = QHBoxLayout()
        head.addWidget(self.title, 1)
        head.addWidget(self.copy_button)
        head.addWidget(self.clear_button)
        col = QVBoxLayout(self)
        col.setContentsMargins(0, 0, 0, 0)
        col.addLayout(head)
        col.addWidget(self.list, 1)
        col.addWidget(self.stats)
        self.sentences: list[tuple[str, str, float | None]] = []
        self.session_latencies: list[float] = []
        self._clipboard = clipboard or QApplication.clipboard
        self.copy_button.clicked.connect(self.copy_all)
        self.clear_button.clicked.connect(self.clear)
        self._update_stats()

    def at_bottom(self) -> bool:
        bar = self.list.verticalScrollBar()
        return bar.value() >= bar.maximum() - 4

    def add(self, vi: str, en: str, latency: float | None) -> None:
        follow = self.at_bottom()
        block = _SentenceBlock(vi, en, latency)
        item = QListWidgetItem()
        self.list.addItem(item)
        self.list.setItemWidget(item, block)
        self._fit(item, block)
        self.sentences.append((vi, en, latency))
        while self.list.count() > MAX_SENTENCES:
            old = self.list.takeItem(0)
            del old
            self.sentences.pop(0)
        if latency is not None:
            self.session_latencies.append(latency)
        self._update_stats()
        if follow:
            self.list.scrollToBottom()

    def _fit(self, item: QListWidgetItem, block: QWidget) -> None:
        """Item height for the current width (word-wrapped labels need height-for-width)."""
        width = max(self.list.viewport().width(), 200)
        h = block.layout().heightForWidth(width)
        item.setSizeHint(QSize(width, h if h > 0 else block.sizeHint().height()))

    def resizeEvent(self, event) -> None:  # noqa: N802 (Qt API)
        super().resizeEvent(event)
        for i in range(self.list.count()):
            item = self.list.item(i)
            self._fit(item, self.list.itemWidget(item))

    def new_session(self) -> None:
        """Statistics are per session; the sentences stay on screen."""
        self.session_latencies = []
        self._update_stats()

    def clear(self) -> None:
        self.list.clear()
        self.sentences = []
        self.session_latencies = []
        self._update_stats()

    def text_all(self) -> str:
        return "".join(f"{vi}\n{en}\n\n" for vi, en, _lat in self.sentences)

    def copy_all(self) -> None:
        self._clipboard().setText(self.text_all())

    def _update_stats(self) -> None:
        lat = self.session_latencies
        self.stats.setText(VI["stats"].format(
            last=fmt_seconds(lat[-1] if lat else None),
            median=fmt_seconds(statistics.median(lat) if lat else None),
            n=len(lat)))

    def block(self, row: int) -> _SentenceBlock:
        return self.list.itemWidget(self.list.item(row))


class CollapsibleSection(QWidget):
    """A header button (▸/▾ title) that shows or hides ``content``; collapsed by default."""

    def __init__(self, title: str, content: QWidget, expanded: bool = False) -> None:
        super().__init__()
        self.title = title
        self.toggle_button = QToolButton()
        self.toggle_button.setCheckable(True)
        self.toggle_button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.toggle_button.setStyleSheet("QToolButton { border: none; font-weight: bold; }")
        self.toggle_button.toggled.connect(self.set_expanded)
        self.content = content
        col = QVBoxLayout(self)
        col.setContentsMargins(0, 0, 0, 0)
        col.addWidget(self.toggle_button)
        col.addWidget(content)
        self.set_expanded(expanded)

    @property
    def expanded(self) -> bool:
        return not self.content.isHidden()

    def set_expanded(self, on: bool) -> None:
        self.toggle_button.blockSignals(True)
        self.toggle_button.setChecked(on)
        self.toggle_button.blockSignals(False)
        self.toggle_button.setText(("▾ " if on else "▸ ") + self.title)
        self.content.setVisible(on)


class Banner(QWidget):
    """A coloured message bar (hidden when empty) with optional 'Chi tiết' and '×' buttons."""

    details_clicked = Signal()

    def __init__(self, background: str, color: str = "white", *, details: bool = False) -> None:
        super().__init__()
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setStyleSheet(f"Banner {{ background-color: {background}; border-radius: 4px; }} "
                           f"QLabel {{ color: {color}; font-weight: bold; }}")
        self.label = QLabel()
        self.label.setWordWrap(True)
        self.details_button = QPushButton(VI["details"])
        self.details_button.setVisible(details)
        self.details_button.clicked.connect(self.details_clicked.emit)
        self.close_button = QToolButton()
        self.close_button.setText(VI["close_banner"])
        self.close_button.setToolTip(VI["close_banner_tip"])
        self.close_button.clicked.connect(self.clear)
        row = QHBoxLayout(self)
        row.setContentsMargins(8, 6, 8, 6)
        row.addWidget(self.label, 1)
        row.addWidget(self.details_button)
        row.addWidget(self.close_button)
        self.clear()

    def show_message(self, text: str, tooltip: str = "") -> None:
        self.label.setText(text)
        self.label.setToolTip(tooltip)
        self.setVisible(True)

    def text(self) -> str:
        return self.label.text() if not self.isHidden() else ""

    def clear(self) -> None:
        self.label.setText("")
        self.label.setToolTip("")
        self.setVisible(False)


class CompactWindow(QWidget):
    """Small always-on-top window: light + main reason, newest EN sentence, Start/Stop, Expand.

    Not a ``Qt.Tool`` window: on macOS a tool window hides as soon as the app is inactive,
    i.e. when the user clicks into Meet/Zoom. Closing it expands back to the main window.
    """

    EN_WIDTH = 180

    def __init__(self, on_button: Callable[[], None], on_expand: Callable[[], None]) -> None:
        super().__init__(None, Qt.WindowType.Window | Qt.WindowType.WindowStaysOnTopHint)
        self.setWindowTitle(VI["compact_title"])
        self.light = LightWidget(40, show_others=False, text_pt=12)
        self.en = QLabel(VI["compact_no_sentence"])
        self.en.setFixedWidth(self.EN_WIDTH)
        self.en.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Preferred)
        self.full_en = ""
        self.button = QPushButton(VI["start"])
        self.button.clicked.connect(on_button)
        self.expand_button = QPushButton(VI["expand"])
        self.expand_button.clicked.connect(on_expand)
        self._on_expand = on_expand
        bottom = QHBoxLayout()
        bottom.addWidget(self.en, 1)
        bottom.addWidget(self.button)
        bottom.addWidget(self.expand_button)
        col = QVBoxLayout(self)
        col.setContentsMargins(8, 6, 8, 6)
        col.addWidget(self.light)
        col.addLayout(bottom)
        self.resize(360, 120)
        self._placed = False

    def set_sentence(self, en: str) -> None:
        self.full_en = en
        metrics = QFontMetrics(self.en.font())
        self.en.setText(metrics.elidedText(f"“{en}”", Qt.TextElideMode.ElideRight, self.EN_WIDTH))
        self.en.setToolTip(en)

    def place_top_right(self) -> None:
        """First show: top-right of the screen, away from the call controls at the bottom."""
        if self._placed:
            return
        self._placed = True
        screen = self.screen() or QGuiApplication.primaryScreen()
        if screen is None:
            return
        g = screen.availableGeometry()
        self.move(g.x() + g.width() - self.width() - 16, g.y() + 16)

    def closeEvent(self, event) -> None:  # noqa: N802 (Qt API)
        # Never lose the app: closing the small window shows the main one again. The close is
        # accepted (not ignored), so a Cmd+Q that closes every window is never blocked here.
        event.accept()
        self._on_expand()
