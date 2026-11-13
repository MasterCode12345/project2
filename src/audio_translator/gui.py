"""One simple PySide6 window. Talks only to the orchestrator API and its events.

Orchestrator events arrive on worker threads; they are marshalled to the Qt
thread through a ``Signal`` (queued connection) before touching any widget.
"""

from __future__ import annotations

import argparse
import sys
import threading

from PySide6.QtCore import QObject, Qt, Signal, Slot
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from .asr import LOCAL_MODELS
from .config import Settings
from .models import (ERROR, HEALTH, IDLE, INFO, MODEL, RESULT, RUNNING, STARTING, STATUS, STOPPED, STOPPING,
                     WARNING, Event)

DEFAULT_INPUT = "(System default)"
NO_OUTPUT = "(choose output device)"
FIXED_MODEL = "(cố định)"

# Vietnamese status labels for the model dropdown (keys: model_manager statuses).
_MODEL_STATUS = {
    "downloaded": "đã tải",
    "loading": "đang tải…",
    "loaded": "trong RAM",
    "failed": "lỗi",
}


def model_item_text(info) -> str:
    """Dropdown text for a ``ModelInfo``: name, status and ``✓`` when selected."""
    if info.status == "not-downloaded" and info.name in LOCAL_MODELS:
        status = "chưa convert (xem --list-models)"  # converted once locally, never downloaded
    elif info.status == "not-downloaded":
        status = f"chưa tải (~{info.size_label})"
    else:
        status = _MODEL_STATUS.get(info.status, info.status)
    return f"{info.name} — {status}" + (" ✓" if info.selected else "")


# Speaker indicator: colour + icon + words (never colour alone). "off" = no session.
LIGHT = {
    "green": ("#2e7d32", "✓", "XANH"),
    "yellow": ("#f9a825", "⚠", "VÀNG"),
    "red": ("#c62828", "✋", "ĐỎ"),
    "off": ("#9e9e9e", "–", ""),
}


class EventBridge(QObject):
    event = Signal(object)


def _device_names(kind: str) -> list[str]:
    try:
        from .devices import list_devices

        return [d.name for d in list_devices(kind)]
    except Exception:
        return []


class MainWindow(QMainWindow):
    def __init__(self, orchestrator=None, settings: Settings | None = None,
                 input_devices: list[str] | None = None,
                 output_devices: list[str] | None = None) -> None:
        super().__init__()
        self.setWindowTitle("Audio Translator")
        self.settings = settings or Settings()
        self.bridge = EventBridge()
        self.bridge.event.connect(self.on_event, Qt.ConnectionType.QueuedConnection)

        if orchestrator is None:
            from .orchestrator import Orchestrator

            orchestrator = Orchestrator(self.settings)
        self.orch = orchestrator
        self.orch.on_event = self.bridge.event.emit  # thread-safe signal emit

        self.input_combo = QComboBox()
        self.input_combo.addItem(DEFAULT_INPUT)
        self.input_combo.addItems(input_devices if input_devices is not None else _device_names("input"))
        self.output_combo = QComboBox()
        outs = output_devices if output_devices is not None else _device_names("output")
        blackhole = [n for n in outs if "blackhole" in n.lower()]
        if not blackhole:
            # Never default to real speakers: the user must choose explicitly.
            self.output_combo.addItem(NO_OUTPUT, "")
        for name in outs:
            self.output_combo.addItem(name, name)
        if blackhole:
            self.output_combo.setCurrentIndex(outs.index(blackhole[0]))

        self.lang_label = QLabel("Tiếng Việt → English")
        self.model_combo = QComboBox()
        models = self._list_models()
        self._has_models = bool(models)
        if self._has_models:
            for info in models:
                self.model_combo.addItem(model_item_text(info), info.name)
            self._reset_model_combo()
            # ``activated`` fires only on user interaction, never on programmatic changes.
            self.model_combo.activated.connect(self.on_model_activated)
        else:
            self.model_combo.addItem(FIXED_MODEL, "")
        self.button = QPushButton("Start")
        self.button.clicked.connect(self.on_button)
        self.light = QLabel()
        self.light.setFixedSize(56, 56)
        self.light.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.light_text = QLabel()
        self.light_text.setWordWrap(True)
        self.light_text.setStyleSheet("font-size: 16pt; font-weight: bold;")
        self.light_others = QLabel()
        self.light_others.setWordWrap(True)
        self.light_others.setStyleSheet("color: #616161;")
        self.light_level = "off"
        self.status_label = QLabel("Stopped")
        self.error_label = QLabel("")
        self.error_label.setStyleSheet("color: #c62828;")
        self.error_label.setWordWrap(True)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(2000)

        form = QFormLayout()
        form.addRow("Microphone:", self.input_combo)
        form.addRow("Output (virtual mic):", self.output_combo)
        form.addRow("Language:", self.lang_label)
        form.addRow("Model:", self.model_combo)
        layout = QVBoxLayout()
        layout.addLayout(form)
        layout.addWidget(self.button)
        light_row = QHBoxLayout()
        light_row.addWidget(self.light)
        light_texts = QVBoxLayout()
        light_texts.addWidget(self.light_text)
        light_texts.addWidget(self.light_others)
        light_row.addLayout(light_texts, 1)
        layout.addLayout(light_row)
        layout.addWidget(self.status_label)
        layout.addWidget(self.error_label)
        layout.addWidget(self.log, 1)
        central = QWidget()
        central.setLayout(layout)
        self.setCentralWidget(central)
        self.resize(560, 480)
        if not blackhole:
            self.error_label.setText("BlackHole not found. Install it with: brew install blackhole-2ch")
        self._apply_state(STOPPED)

    # ------------------------------------------------------------- actions
    @Slot()
    def on_button(self) -> None:
        if self._state in (STOPPED, IDLE):
            out = self.output_combo.currentData()
            if not out:
                self.error_label.setText("Choose an output device (e.g. BlackHole 2ch).")
                return
            inp = self.input_combo.currentText()
            self.error_label.setText("")
            self._apply_state(STARTING)
            # start() is asynchronous: model loading happens off the UI thread.
            if not self.orch.start(None if inp == DEFAULT_INPUT else inp, out):
                self._apply_state(self.orch.state)
        else:
            self.button.setEnabled(False)
            self.status_label.setText("Stopping…")
            # stop() closes PortAudio streams (may block); keep the Qt thread free.
            threading.Thread(target=self.orch.stop, name="at-gui-stop", daemon=True).start()

    @Slot(int)
    def on_model_activated(self, index: int) -> None:
        name = self.model_combo.itemData(index)
        if not name:
            return
        self.error_label.setText("")
        if not self.orch.select_model(name):
            self._reset_model_combo()  # refused: the orchestrator already logged why

    # ------------------------------------------------------------- models
    def _list_models(self) -> list:
        lister = getattr(self.orch, "list_models", None)
        if lister is None:
            return []
        try:
            return list(lister())
        except Exception:
            return []

    def _selected_model_name(self) -> str:
        return getattr(self.orch, "selected_model", "") or ""

    def _reset_model_combo(self) -> None:
        """Point the dropdown back at the orchestrator's selected model."""
        idx = self.model_combo.findData(self._selected_model_name())
        if idx >= 0:
            self.model_combo.setCurrentIndex(idx)

    def _refresh_model_items(self) -> None:
        for info in self._list_models():
            idx = self.model_combo.findData(info.name)
            if idx >= 0:
                self.model_combo.setItemText(idx, model_item_text(info))

    @Slot(object)
    def on_event(self, ev: Event) -> None:
        if ev.kind == STATUS:
            self.status_label.setText(ev.message)
            state = ev.data.get("state")
            if state:
                self._apply_state(state, keep_message=True)
        elif ev.kind == HEALTH:
            if ev.session_id is not None and ev.session_id == getattr(self.orch, "current_session", ev.session_id):
                self.set_light(ev.data.get("level", "off"), ev.data.get("reason", ""), ev.data.get("others") or [])
        elif ev.kind == RESULT:
            lat = ev.data.get("latency")
            lat_s = f"{lat:.2f} s" if lat is not None else "n/a"
            self.log.appendPlainText(
                f"#{ev.segment_id}  VI: {ev.data.get('source_text', '')}\n"
                f"     EN: {ev.data.get('text', '')}\n"
                f"     latency: {lat_s}"
            )
        elif ev.kind == MODEL:
            if not self._has_models:
                return
            self._refresh_model_items()
            if ev.data.get("detail") == "selected":
                self._reset_model_combo()
            if self._state in (STOPPED, IDLE) and (
                    ev.data.get("status") == "loading" or ev.data.get("detail") == "selected"):
                self.status_label.setText(ev.message)
            self.log.appendPlainText(f"{ev.kind}: {ev.message}")
        elif ev.kind == ERROR:
            self.error_label.setText(ev.message)
            self.log.appendPlainText(f"ERROR: {ev.message}")
            if ev.data.get("model") and self._has_models:
                self._refresh_model_items()
                self._reset_model_combo()
        elif ev.kind in (WARNING, INFO):
            self.log.appendPlainText(f"{ev.kind}: {ev.message}")
            if ev.data.get("model_fallback"):
                # The selected model is missing: say so where errors show, not only in the log.
                self.error_label.setText(
                    f"Model '{ev.data.get('model')}' chưa được convert, đang dùng tạm "
                    f"'{ev.data.get('fallback')}'. Convert một lần bằng: {ev.data.get('command')}")

    def set_light(self, level: str, reason: str = "", others: list[str] | None = None) -> None:
        color, icon, word = LIGHT.get(level, LIGHT["off"])
        self.light_level = level if level in LIGHT else "off"
        self.light.setText(icon)
        self.light.setStyleSheet(f"background-color: {color}; color: white; border-radius: 28px; "
                                 "font-size: 22pt; font-weight: bold;")
        self.light.setToolTip(word)
        self.light_text.setText(f"{word}: {reason}" if word and reason else (reason or "Chưa chạy"))
        others = others or []
        self.light_others.setText(("Cũng: " + "; ".join(others)) if others else "")
        tip = "\n".join([reason, *others]) if reason else ""
        self.light_text.setToolTip(tip)
        self.light_others.setToolTip(tip)

    def _apply_state(self, state: str, keep_message: bool = False) -> None:
        if state != RUNNING:
            self.set_light("off")
        self._state = state
        running_like = state in (STARTING, RUNNING)
        self.button.setText("Stop" if running_like else "Start")
        self.button.setEnabled(state != STOPPING)
        self.input_combo.setEnabled(not running_like and state != STOPPING)
        self.output_combo.setEnabled(not running_like and state != STOPPING)
        self.model_combo.setEnabled(self._has_models and state in (STOPPED, IDLE))
        if not keep_message:
            self.status_label.setText({STARTING: "Starting…", RUNNING: "Running",
                                       STOPPING: "Stopping…"}.get(state, "Stopped"))

    def closeEvent(self, event) -> None:  # noqa: N802 (Qt API)
        self.orch.on_event = None
        self.orch.shutdown()
        super().closeEvent(event)


def parse_args(argv: list[str] | None) -> tuple[argparse.Namespace, list[str]]:
    """GUI options (``--latency-log PATH``); anything else is left for Qt."""
    p = argparse.ArgumentParser(prog="audio-translator", add_help=True,
                                description="Audio Translator GUI (use --cli for the command line)")
    p.add_argument("--latency-log", default=None, metavar="PATH",
                   help="append one CSV row per segment to PATH (see eval/latency_report.py)")
    return p.parse_known_args(list(argv or []))


def main(argv: list[str] | None = None) -> int:
    args, qt_args = parse_args(argv)
    app = QApplication.instance() or QApplication(sys.argv[:1] + qt_args)
    win = MainWindow(settings=Settings(latency_log=args.latency_log))
    win.show()
    return app.exec()
