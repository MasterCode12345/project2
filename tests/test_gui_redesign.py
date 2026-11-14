"""GUI redesign (offscreen): compact/full modes, sync, translation pane, settings, banners, strings."""

import os
import threading

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtWidgets import QApplication, QComboBox, QLabel, QPushButton, QToolButton  # noqa: E402

from audio_translator.config import Settings  # noqa: E402
from audio_translator.models import (  # noqa: E402
    ERROR, HEALTH, INFO, MODEL, RESULT, RUNNING, STARTING, STATUS, STOPPED, STOPPING, WARNING, Event,
)


class Orch:
    """Orchestrator stand-in: records start/stop/shutdown, has settings and a session id."""

    def __init__(self, settings=None):
        self.settings = settings if settings is not None else Settings()
        self.on_event = None
        self.state = STOPPED
        self.started_with = None
        self.starts = []
        self.stopped = threading.Event()
        self.shutdowns = 0
        self.current_session = None

    def start(self, inp=None, out=None):
        self.started_with = (inp, out)
        self.starts.append((inp, out, self.settings.translator, self.settings.ns_mode))
        self.state = STARTING
        return True

    def stop(self):
        self.stopped.set()

    def shutdown(self):
        self.shutdowns += 1


@pytest.fixture
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def make(qapp):
    made = []

    def _make(settings=None, outs=("Speakers", "BlackHole 2ch")):
        from audio_translator.gui import MainWindow

        orch = Orch(settings)
        w = MainWindow(orch, input_devices=["Mic A"], output_devices=list(outs))
        made.append(w)
        return w, orch

    yield _make
    for w in made:  # explicit, ordered cleanup: Qt deletes the windows, not Python's GC
        w.shutdown()
        w.compact.hide()
        w.hide()
        w.compact.deleteLater()
        w.deleteLater()
    qapp.processEvents()


def run(w, orch, sid=1):
    w.button.click()
    orch.current_session = sid
    w.on_event(Event(STATUS, "Listening", sid, data={"state": RUNNING}))


def result(w, en, vi="câu", lat=1.0, sid=1, seg=1):
    w.on_event(Event(RESULT, "", sid, seg, data={"source_text": vi, "text": en, "latency": lat}))


# ------------------------------------------------------------ compact / full
def test_compact_and_full_modes_and_the_on_top_flag(make):
    w, orch = make()
    w.show()
    assert not (w.windowFlags() & Qt.WindowType.WindowStaysOnTopHint)
    assert w.compact.windowFlags() & Qt.WindowType.WindowStaysOnTopHint
    assert w.compact.windowType() == Qt.WindowType.Window  # not Tool: it would hide on macOS when inactive
    w.compact_button.click()
    assert w.compact.isVisible() and not w.isVisible()
    assert w.compact.width() <= 420 and w.compact.height() <= 200
    w.compact.expand_button.click()
    assert w.isVisible() and not w.compact.isVisible()
    assert not (w.windowFlags() & Qt.WindowType.WindowStaysOnTopHint)
    w.to_compact()
    w.compact.close()  # closing the small window expands back: the app is never lost
    assert w.isVisible() and not w.compact.isVisible()


def test_compact_window_first_opens_top_right(make):
    w, _ = make()
    w.to_compact()
    g = (w.compact.screen() or QApplication.primaryScreen()).availableGeometry()
    pos = w.compact.pos()
    assert w.compact.TOP_OFFSET == 80
    assert pos.x() + w.compact.width() == g.x() + g.width() - 16  # flush right (16 px margin)
    assert pos.y() == g.y() + 80  # below the browser toolbar and its "⋮" menu
    w.compact.move(g.x() + 10, g.y() + 200)  # the user moves it: kept next time
    w.to_full()
    w.to_compact()
    assert w.compact.pos().y() == g.y() + 200


def test_light_and_newest_sentence_stay_in_sync(make):
    w, orch = make()
    run(w, orch)
    w.on_event(Event(HEALTH, "", 1, data={"level": "red", "reason": "Dừng lại một chút (câu 12,4 s)",
                                          "others": ["Đang xử lý, nói chậm lại"]}))
    for lw in (w.light_widget, w.compact.light):
        assert lw.level == "red" and lw.disc.text() == "✋" and lw.text.text() == "ĐỎ: Dừng lại một chút (câu 12,4 s)"
    assert w.light_others.text() == "Cũng: Đang xử lý, nói chậm lại" and w.compact.light.others.isHidden()
    long_en = "I would like to talk today about the group's graduation project, speech recognition and translation."
    result(w, long_en)
    assert w.transcript.block(0).en.text() == long_en and w.compact.full_en == long_en
    assert w.compact.en.toolTip() == long_en and w.compact.en.text().endswith("…")
    assert len(w.compact.en.text()) < len(long_en)
    w.on_event(Event(HEALTH, "", 99, data={"level": "yellow", "reason": "old session"}))  # stale: ignored
    assert w.light_widget.level == "red" and w.compact.light.level == "red"
    w.on_event(Event(STATUS, "Stopped", 1, data={"state": STOPPED}))
    assert w.light_widget.level == "off" and w.compact.light.level == "off"


def test_compact_button_stops_and_starts_with_the_current_settings(make):
    w, orch = make()
    w.translator_combo.setCurrentIndex(w.translator_combo.findData("google"))
    w.to_compact()
    assert w.compact.button.text() == "▶ Bắt đầu"
    w.compact.button.click()  # start from the compact window
    assert orch.starts == [(None, "BlackHole 2ch", "google", "off")]
    assert w.compact.button.text() == "■ Dừng"
    orch.current_session = 1
    w.on_event(Event(STATUS, "Listening", 1, data={"state": RUNNING}))
    w.compact.button.click()
    assert orch.stopped.wait(2)  # off the Qt thread
    assert not w.compact.button.isEnabled() and not w.button.isEnabled()
    w.on_event(Event(STATUS, "Stopping…", 1, data={"state": STOPPING}))
    assert not w.compact.button.isEnabled()
    w.on_event(Event(STATUS, "Stopped", 1, data={"state": STOPPED}))
    assert w.compact.button.text() == "▶ Bắt đầu" and w.compact.button.isEnabled()


# ------------------------------------------------------------ translation pane
def test_sentence_blocks_and_statistics(make):
    w, orch = make()
    run(w, orch)
    t = w.transcript
    assert t.stats.text() == "Câu gần nhất: – · Trung vị phiên: – · Số câu: 0"
    result(w, "Hello.", "xin chào", 2.95, seg=1)
    result(w, "Thank you.", "cảm ơn", None, seg=2)
    result(w, "Fine.", "ổn", 4.0, seg=3)
    b = t.block(0)
    assert (b.vi.text(), b.en.text(), b.latency.text()) == ("xin chào", "Hello.", "3,0 s")
    assert t.block(1).latency.text() == "–"
    assert "#8a8a8a" in b.vi.styleSheet() and "14pt" in b.en.styleSheet()
    assert t.stats.text() == "Câu gần nhất: 4,0 s · Trung vị phiên: 3,5 s · Số câu: 2"
    w.on_event(Event(STATUS, "Stopped", 1, data={"state": STOPPED}))
    run(w, orch, sid=2)  # new session: statistics restart, sentences stay
    assert t.list.count() == 3 and t.stats.text().endswith("Số câu: 0")


def test_pane_keeps_200_sentences_and_copy_and_clear(make, qapp):
    from audio_translator.gui_widgets import MAX_SENTENCES

    w, orch = make()
    run(w, orch)
    for i in range(MAX_SENTENCES + 1):
        result(w, f"en {i}", f"vi {i}", 1.0, seg=i + 1)
    t = w.transcript
    assert MAX_SENTENCES == 200 and t.list.count() == 200
    assert t.block(0).en.text() == "en 1" and t.block(199).en.text() == "en 200"
    t.copy_button.click()
    text = QApplication.clipboard().text()
    assert text.startswith("vi 1\nen 1\n\n") and text.endswith("vi 200\nen 200\n\n")
    t.clear_button.click()
    assert t.list.count() == 0 and t.stats.text().endswith("Số câu: 0")


def test_pane_follows_new_sentences_unless_scrolled_up(make, qapp):
    w, orch = make()
    w.show()
    run(w, orch)
    t = w.transcript
    for i in range(40):
        result(w, f"sentence {i}", seg=i + 1)
    qapp.processEvents()
    bar = t.list.verticalScrollBar()
    assert bar.maximum() > 0 and bar.value() == bar.maximum()  # at the bottom: follows
    bar.setValue(0)  # the user scrolls up to read
    result(w, "new one", seg=41)
    qapp.processEvents()
    assert bar.value() == 0
    bar.setValue(bar.maximum())
    result(w, "newer", seg=42)
    qapp.processEvents()
    assert bar.value() == bar.maximum()


# ------------------------------------------------------------ settings / log / banners
def test_technical_log_hidden_by_default_and_receives_technical_lines(make):
    w, orch = make()
    assert not w.tech_section.expanded and w.log.isHidden()
    assert not w.settings_section.expanded
    w.on_event(Event(MODEL, "Model 'small': ready", data={"model": "small", "status": "loaded", "detail": "ready"}))
    w.on_event(Event(INFO, "Latency log: /tmp/x.csv"))
    w.on_event(Event(WARNING, "Overload: dropped oldest pending segment 3"))
    w.on_event(Event(ERROR, "Something odd happened"))
    log = w.log.toPlainText()
    for line in ("ready", "Latency log", "Overload", "Something odd"):
        assert line in log
    assert w.error_label.text() == "Đã xảy ra lỗi – bấm Chi tiết để xem"  # never the English
    assert w.error_label.label.toolTip() == "Something odd happened"
    w.error_label.details_button.click()
    assert w.tech_section.expanded and not w.log.isHidden()


@pytest.mark.parametrize("message,want", [
    ("Could not start: Output device 'BlackHole' not found",
     "Không tìm thấy thiết bị 'BlackHole' – kiểm tra đã cắm/cài và chọn lại trong Cài đặt"),
    ("Could not start: TTS voice does not support target language 'en'",
     "Không khởi động được – kiểm tra BlackHole đã cài và chọn lại Đầu ra trong Cài đặt"),
    ("Input device 'Mic A' stopped responding (removed?)",
     "Thiết bị 'Mic A' ngừng hoạt động – cắm lại rồi bấm Bắt đầu"),
    ("Cannot open output device 'BlackHole 2ch': busy", "Không mở được thiết bị 'BlackHole 2ch' – chọn lại trong Cài đặt"),
    ("Could not load model 'large-v3': boom; keeping 'medium'",
     "Không tải được model 'large-v3' – vẫn dùng model đang chọn"),
])
def test_error_banner_headlines(make, message, want):
    w, _ = make()
    w.on_event(Event(ERROR, message))
    assert w.error_label.text() == want and w.error_label.label.toolTip() == message


def test_settings_locked_while_running_and_applied_at_start(make):
    w, orch = make()
    combos = (w.input_combo, w.output_combo, w.translator_combo, w.noise_combo)
    assert all(c.isEnabled() for c in combos) and w.settings_note.isHidden()
    w.translator_combo.setCurrentIndex(w.translator_combo.findData("nllb-600m"))
    w.noise_combo.setCurrentIndex(w.noise_combo.findData("logmmse"))
    run(w, orch)
    assert orch.starts[-1][2:] == ("nllb-600m", "logmmse")
    for state in (STARTING, RUNNING, STOPPING):
        w.on_event(Event(STATUS, state, 1, data={"state": state}))
        assert not any(c.isEnabled() for c in combos) and not w.settings_note.isHidden()
    w.translator_combo.setCurrentIndex(w.translator_combo.findData("google"))  # only possible in code
    assert orch.settings.translator == "nllb-600m"  # nothing applied while running
    w.on_event(Event(STATUS, "Stopped", 1, data={"state": STOPPED}))
    assert all(c.isEnabled() for c in combos)
    assert Settings().translator == "offline" and Settings().ns_mode == "off"  # defaults unchanged


def test_dropdowns_start_from_the_orchestrator_settings(make):
    w, _ = make(Settings(translator="nllb-600m", ns_mode="logmmse"))
    assert w.translator_combo.currentData() == "nllb-600m" and w.noise_combo.currentData() == "logmmse"
    w2, _ = make()
    assert w2.translator_combo.currentData() == "offline" and w2.noise_combo.currentData() == "off"
    from audio_translator.gui import parse_args

    args, _ = parse_args(["--translator", "nllb-600m", "--ns", "logmmse"])
    assert (args.translator, args.ns) == ("nllb-600m", "logmmse")


@pytest.mark.parametrize("message", [
    "Offline translation model nllb-600m unavailable (not converted); using Google Translate",
    "Offline translation unavailable (download failed); using Google Translate",
    "Offline translation failed (boom); using Google Translate for this segment",
])
def test_google_fallback_shows_a_yellow_banner(make, message):
    w, orch = make(Settings(translator="nllb-600m"))
    run(w, orch)
    assert w.warn_banner.text() == ""
    w.on_event(Event(WARNING, message))
    assert w.warn_banner.text() == "Đang dịch bằng Google (cần mạng) vì NLLB-600M chưa sẵn sàng"
    assert w.warn_banner.label.toolTip() == message and w.error_label.text() == ""
    w.on_event(Event(STATUS, "Stopped", 1, data={"state": STOPPED}))
    w.button.click()  # the next Start clears it
    assert w.warn_banner.text() == ""


# ------------------------------------------------------------ shutdown
def test_shutdown_on_quit_while_compact_and_only_once(make, qapp):
    w, orch = make()
    w.to_compact()
    assert not w.isVisible()
    qapp.aboutToQuit.emit()  # Cmd+Q while the main window is hidden
    assert orch.shutdowns == 1 and orch.on_event is None
    w.close()  # closeEvent too: shutdown is idempotent
    qapp.aboutToQuit.emit()
    assert orch.shutdowns == 1


# ------------------------------------------------------------ strings
# "Microphone" alone stays allowed: the help line names Meet's own setting ("chọn Microphone = BlackHole 2ch").
ENGLISH = ("Start", "Stop", "Stopped", "Running", "Starting", "Stopping", "Microphone:", "Output", "Language",
           "Model:", "Listening", "Loading", "Error", "error", "Settings", "Clear", "Copy", "Expand",
           "Compact", "Translation", "Details", "System default", "choose output", "not found",
           "Could not", "selected", "downloading", "loading", "ready")


def visible_strings(*roots):
    out = []
    for root in roots:
        out.append(root.windowTitle())
        for wdg in [root, *root.findChildren(QLabel), *root.findChildren(QPushButton), *root.findChildren(QToolButton)]:
            if isinstance(wdg, (QLabel, QPushButton, QToolButton)):
                out.append(wdg.text())
        for combo in root.findChildren(QComboBox):
            if combo in (getattr(root, "input_combo", None), getattr(root, "output_combo", None),
                         getattr(root, "model_combo", None)):
                continue  # device and model names are names, not UI text
            out += [combo.itemText(i) for i in range(combo.count())]
    return [s for s in out if s]


def test_no_english_left_on_the_ui(make):
    w, orch = make(outs=("Speakers",))  # no BlackHole: banner shown
    states = []
    for step in ("stopped", "running", "error"):
        if step == "running":
            w.output_combo.setCurrentIndex(w.output_combo.findData("Speakers"))
            run(w, orch)
            w.on_event(Event(HEALTH, "", 1, data={"level": "yellow", "reason": "Câu dài, hãy ngắt nghỉ"}))
            result(w, "Hello.", "xin chào", 1.0)
            w.on_event(Event(WARNING, "Offline translation failed (x); using Google Translate for this segment"))
        if step == "error":
            w.on_event(Event(ERROR, "Something odd happened"))
            w.on_event(Event(STATUS, "Stopped", 1, data={"state": STOPPED}))
        w.settings_section.set_expanded(True)
        strings = visible_strings(w, w.compact)
        # the translated sentence itself is English on purpose
        strings = [s for s in strings if "Hello." not in s]
        states.append(step)
        bad = [s for s in strings for word in ENGLISH if word in s]
        assert not bad, f"{step}: {bad}"
    assert states == ["stopped", "running", "error"]
