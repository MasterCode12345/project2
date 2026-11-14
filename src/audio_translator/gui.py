"""PySide6 GUI for real calls: a full window and an always-on-top compact window.

Talks only to the orchestrator API and its events. Orchestrator events arrive on
worker threads; they are marshalled to the Qt thread through a ``Signal``
(queued connection) before touching any widget. Every visible string comes from
``gui_strings``; the orchestrator's English texts go to the technical log.
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
from .gui_strings import (MODEL_NOT_CONVERTED, MODEL_NOT_DOWNLOADED, MODEL_STATUS, NOISE, TRANSLATOR_SHORT,
                          TRANSLATORS, VI, error_headline, is_google_fallback, model_event_text, status_text)
from .gui_widgets import Banner, CollapsibleSection, CompactWindow, LightWidget, TranscriptView
from .models import (ERROR, HEALTH, IDLE, INFO, MODEL, RESULT, RUNNING, STARTING, STATUS, STOPPED, STOPPING,
                     WARNING, Event)

DEFAULT_INPUT = VI["default_input"]
NO_OUTPUT = VI["no_output"]
FIXED_MODEL = VI["fixed_model"]


def model_item_text(info) -> str:
    """Dropdown text for a ``ModelInfo``: name, status and ``✓`` when selected."""
    if info.status == "not-downloaded" and info.name in LOCAL_MODELS:
        status = MODEL_NOT_CONVERTED  # converted once locally, never downloaded
    elif info.status == "not-downloaded":
        status = MODEL_NOT_DOWNLOADED.format(size=info.size_label)
    else:
        status = MODEL_STATUS.get(info.status, info.status)
    return f"{info.name} — {status}" + (" ✓" if info.selected else "")


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
        self.setWindowTitle(VI["window_title"])
        self.settings = settings or Settings()
        self.bridge = EventBridge()
        self.bridge.event.connect(self.on_event, Qt.ConnectionType.QueuedConnection)

        if orchestrator is None:
            from .orchestrator import Orchestrator

            orchestrator = Orchestrator(self.settings)
        self.orch = orchestrator
        self.orch.on_event = self.bridge.event.emit  # thread-safe signal emit
        self._shut = False
        app = QApplication.instance()
        if app is not None:  # Cmd+Q while compact (main window hidden) must still shut down
            app.aboutToQuit.connect(self.shutdown)

        # ---- banners
        self.error_label = Banner("#c62828", details=True)
        self.error_label.details_clicked.connect(self.show_details)
        self.warn_banner = Banner("#f9a825", color="#212121")

        # ---- control row
        self.lang_label = QLabel(VI["lang_pair"])  # later: a language-pair dropdown here
        self.lang_label.setStyleSheet("font-size: 13pt;")
        self.button = QPushButton(VI["start"])
        self.button.setMinimumHeight(44)
        self.button.setMinimumWidth(150)
        self.button.setStyleSheet("font-size: 15pt; font-weight: bold;")
        self.button.clicked.connect(self.on_button)
        self.status_label = QLabel(VI["light_off"])
        self.compact_button = QPushButton(VI["compact"])
        self.compact_button.clicked.connect(self.to_compact)

        # ---- light
        self.light_widget = LightWidget(72)
        self.light = self.light_widget.disc
        self.light_text = self.light_widget.text
        self.light_others = self.light_widget.others

        # ---- translation pane
        self.transcript = TranscriptView()

        # ---- settings (collapsed)
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
        self.output_help = QLabel(VI["output_help"])
        self.output_help.setStyleSheet("color: #8a8a8a;")
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
        self.translator_combo = QComboBox()
        for value, label in TRANSLATORS:
            self.translator_combo.addItem(label, value)
        self.noise_combo = QComboBox()
        for value, label in NOISE:
            self.noise_combo.addItem(label, value)
        self._init_from_settings()
        self.settings_note = QLabel(VI["settings_locked"])
        self.settings_note.setStyleSheet("color: #8a8a8a;")
        self.settings_note.setWordWrap(True)
        form = QFormLayout()
        form.addRow(VI["mic"], self.input_combo)
        form.addRow(VI["output"], self.output_combo)
        form.addRow("", self.output_help)
        form.addRow(VI["model"], self.model_combo)
        form.addRow(VI["translator"], self.translator_combo)
        form.addRow(VI["noise"], self.noise_combo)
        settings_box = QWidget()
        settings_col = QVBoxLayout(settings_box)
        settings_col.setContentsMargins(12, 0, 0, 0)
        settings_col.addLayout(form)
        settings_col.addWidget(self.settings_note)
        self.settings_section = CollapsibleSection(VI["settings"], settings_box)

        # ---- technical log (hidden)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(2000)
        self.log.setMinimumHeight(120)
        self.tech_section = CollapsibleSection(VI["tech_log"], self.log)

        # ---- layout
        controls = QHBoxLayout()
        controls.addWidget(self.lang_label)
        controls.addWidget(self.button)
        controls.addWidget(self.status_label, 1)
        controls.addWidget(self.compact_button)
        layout = QVBoxLayout()
        layout.addWidget(self.error_label)
        layout.addWidget(self.warn_banner)
        layout.addLayout(controls)
        layout.addWidget(self.light_widget)
        layout.addWidget(self.transcript, 1)
        layout.addWidget(self.settings_section)
        layout.addWidget(self.tech_section)
        central = QWidget()
        central.setLayout(layout)
        self.setCentralWidget(central)
        self.resize(640, 720)

        # ---- compact window (same state, one update path)
        self.compact = CompactWindow(self.on_button, self.to_full)

        if not blackhole:
            self.error_label.show_message(VI["no_blackhole"])
        self._apply_state(STOPPED)

    # ------------------------------------------------------------- settings
    def _orch_settings(self):
        return getattr(self.orch, "settings", None) or self.settings

    def _init_from_settings(self) -> None:
        """Dropdowns start from the orchestrator's settings (e.g. ``--translator nllb-600m``)."""
        s = self._orch_settings()
        for combo, value in ((self.translator_combo, getattr(s, "translator", None)),
                             (self.noise_combo, getattr(s, "ns_mode", None))):
            idx = combo.findData(value)
            if idx >= 0:
                combo.setCurrentIndex(idx)

    def _apply_settings(self) -> None:
        """Only while stopped, right before start(): the orchestrator reads both at session start."""
        s = getattr(self.orch, "settings", None)
        if s is None:
            return
        s.translator = self.translator_combo.currentData()
        s.ns_mode = self.noise_combo.currentData()

    # ------------------------------------------------------------- actions
    @Slot()
    def on_button(self) -> None:
        if self._state in (STOPPED, IDLE):
            out = self.output_combo.currentData()
            if not out:
                self.error_label.show_message(VI["choose_output"])
                self.settings_section.set_expanded(True)
                return
            inp = self.input_combo.currentText()
            self.error_label.clear()
            self.warn_banner.clear()
            self._apply_settings()
            self.transcript.new_session()
            self._apply_state(STARTING)
            # start() is asynchronous: model loading happens off the UI thread.
            if not self.orch.start(None if inp == DEFAULT_INPUT else inp, out):
                self._apply_state(self.orch.state)
        else:
            self.button.setEnabled(False)
            self.compact.button.setEnabled(False)
            self.status_label.setText(status_text(STOPPING, ""))
            # stop() closes PortAudio streams (may block); keep the Qt thread free.
            threading.Thread(target=self.orch.stop, name="at-gui-stop", daemon=True).start()

    @Slot()
    def show_details(self) -> None:
        if self.compact.isVisible():
            self.to_full()
        self.tech_section.set_expanded(True)
        self.log.verticalScrollBar().setValue(self.log.verticalScrollBar().maximum())

    @Slot()
    def to_compact(self) -> None:
        self.compact.show()
        self.compact.place_top_right()
        self.compact.raise_()
        self.hide()

    @Slot()
    def to_full(self) -> None:
        self.compact.hide()
        self.show()
        self.raise_()
        self.activateWindow()

    @Slot(int)
    def on_model_activated(self, index: int) -> None:
        name = self.model_combo.itemData(index)
        if not name:
            return
        self.error_label.clear()
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

    # ------------------------------------------------------------- events
    @Slot(object)
    def on_event(self, ev: Event) -> None:
        if ev.kind == STATUS:
            state = ev.data.get("state")
            self.log.appendPlainText(f"{ev.kind}: {ev.message}")
            if state:
                self._apply_state(state)
            self.status_label.setText(status_text(state or self._state, ev.message))
        elif ev.kind == HEALTH:
            if ev.session_id is not None and ev.session_id == getattr(self.orch, "current_session", ev.session_id):
                self.set_light(ev.data.get("level", "off"), ev.data.get("reason", ""), ev.data.get("others") or [])
        elif ev.kind == RESULT:
            vi, en = ev.data.get("source_text", ""), ev.data.get("text", "")
            self.transcript.add(vi, en, ev.data.get("latency"))
            self.compact.set_sentence(en)
        elif ev.kind == MODEL:
            self.log.appendPlainText(f"{ev.kind}: {ev.message}")
            if not self._has_models:
                return
            self._refresh_model_items()
            if ev.data.get("detail") == "selected":
                self._reset_model_combo()
            if self._state in (STOPPED, IDLE) and (
                    ev.data.get("status") == "loading" or ev.data.get("detail") == "selected"):
                text = model_event_text(ev.data.get("model", ""), ev.data.get("detail", ""))
                if text:
                    self.status_label.setText(text)
        elif ev.kind == ERROR:
            self.log.appendPlainText(f"ERROR: {ev.message}")
            # Vietnamese headline + hint; the English original stays in the tooltip and the log.
            self.error_label.show_message(error_headline(ev.message), ev.message)
            if ev.data.get("model") and self._has_models:
                self._refresh_model_items()
                self._reset_model_combo()
        elif ev.kind in (WARNING, INFO):
            self.log.appendPlainText(f"{ev.kind}: {ev.message}")
            if ev.data.get("model_fallback"):
                # The selected model is missing: say so where errors show, not only in the log.
                self.error_label.show_message(VI["model_fallback"].format(
                    model=ev.data.get("model"), fallback=ev.data.get("fallback"), command=ev.data.get("command")),
                    ev.message)
            elif ev.kind == WARNING and is_google_fallback(ev.message):
                value = self.translator_combo.currentData()
                label = TRANSLATOR_SHORT.get(value, self.translator_combo.currentText())
                self.warn_banner.show_message(VI["mt_google"].format(translator=label), ev.message)

    def set_light(self, level: str, reason: str = "", others: list[str] | None = None) -> None:
        """One update path for both windows: the light and its reason are always the same."""
        self.light_widget.set_light(level, reason, others)
        self.compact.light.set_light(level, reason, others)

    @property
    def light_level(self) -> str:
        return self.light_widget.level

    def _apply_state(self, state: str) -> None:
        if state != RUNNING:
            self.set_light("off")
        self._state = state
        running_like = state in (STARTING, RUNNING)
        label = VI["stop"] if running_like else VI["start"]
        for b in (self.button, self.compact.button):
            b.setText(label)
            b.setEnabled(state != STOPPING)
        stopped = state in (STOPPED, IDLE)
        for w in (self.input_combo, self.output_combo, self.translator_combo, self.noise_combo):
            w.setEnabled(stopped)
        self.model_combo.setEnabled(self._has_models and stopped)
        self.settings_note.setVisible(not stopped)
        self.status_label.setText(status_text(state, ""))

    # ------------------------------------------------------------- shutdown
    @Slot()
    def shutdown(self) -> None:
        """Idempotent: from closeEvent and from QApplication.aboutToQuit (Cmd+Q while compact)."""
        if self._shut:
            return
        self._shut = True
        self.orch.on_event = None
        self.orch.shutdown()
        self.compact.hide()

    def closeEvent(self, event) -> None:  # noqa: N802 (Qt API)
        self.shutdown()
        super().closeEvent(event)


def parse_args(argv: list[str] | None) -> tuple[argparse.Namespace, list[str]]:
    """GUI options; anything else is left for Qt (CLI-only options are refused)."""
    from .denoise import NS_MODES
    from .translate import MT_MODELS

    d = Settings()
    p = argparse.ArgumentParser(prog="audio-translator", add_help=True,
                                description="Audio Translator GUI (use --cli for the command line)")
    p.add_argument("--latency-log", default=None, metavar="PATH",
                   help="append one CSV row per segment to PATH (see eval/latency_report.py)")
    p.add_argument("--translator", default=d.translator, choices=["offline", "google", *MT_MODELS],
                   help="initial translation backend (can be changed in Cài đặt while stopped)")
    p.add_argument("--ns", default=d.ns_mode, choices=list(NS_MODES),
                   help="initial noise suppression (can be changed in Cài đặt while stopped)")
    p.add_argument("--asr-max-tokens-per-s", type=float, default=None, metavar="N",
                   help="experimental, default off: cap Whisper output tokens per second of audio")
    p.add_argument("--asr-temperatures", default=None, metavar="T,T,...",
                   type=lambda v: tuple(float(x) for x in v.split(",") if x.strip()),
                   help="experimental, default off: Whisper temperature fallback ladder, e.g. 0,0.2,0.4")
    p.add_argument("--asr-fallback-max-tokens-per-s", type=float, default=None, metavar="N",
                   help="experimental, default off: cap only the fallback (temperature > 0) attempts at "
                        "ceil(audio_s x N) + 32 tokens")
    p.add_argument("--asr-first-min-tokens", type=int, default=None, metavar="N",
                   help="experimental, default off: give the temperature-0 attempt a wide limit of at least N tokens")
    p.add_argument("--asr-decode-budget-s", type=float, default=None, metavar="S",
                   help="experimental, default off: no more fallback attempts once a segment took S seconds")
    p.add_argument("--asr-cap-tail-strip", action="store_true",
                   help="experimental, default off: strip a junk tail when a capped attempt reached its limit")
    args, rest = p.parse_known_args(list(argv or []))
    from .cli import build_parser

    cli_opts = {o for a in build_parser()._actions for o in a.option_strings}
    cli_only = [a for a in rest if a.split("=", 1)[0] in cli_opts]
    if cli_only:  # an app option must not silently go to Qt and be ignored
        p.error(f"{' '.join(cli_only)}: only with --cli (the GUI accepts --latency-log, --translator, --ns, "
                "--asr-max-tokens-per-s, --asr-temperatures and the other --asr-* decoding options)")
    return args, rest


def main(argv: list[str] | None = None) -> int:
    args, qt_args = parse_args(argv)
    app = QApplication.instance() or QApplication(sys.argv[:1] + qt_args)
    win = MainWindow(settings=Settings(latency_log=args.latency_log, translator=args.translator, ns_mode=args.ns,
                                       asr_max_tokens_per_s=args.asr_max_tokens_per_s,
                                       asr_temperatures=args.asr_temperatures,
                                       asr_fallback_max_tokens_per_s=args.asr_fallback_max_tokens_per_s,
                                       asr_first_min_tokens=args.asr_first_min_tokens,
                                       asr_decode_budget_s=args.asr_decode_budget_s,
                                       asr_cap_tail_strip=args.asr_cap_tail_strip))
    win.show()
    return app.exec()
