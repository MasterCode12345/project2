import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from audio_translator import cli  # noqa: E402
from audio_translator.models import (  # noqa: E402
    ERROR, MODEL, RESULT, RUNNING, STARTING, STATUS, STOPPED, STOPPING, Event,
)
from audio_translator.model_manager import CATALOG, ModelInfo  # noqa: E402
from audio_translator.tts import EdgeTTS  # noqa: E402


class FakeOrch:
    def __init__(self, settings=None, on_event=None):
        self.settings = settings
        self.on_event = on_event
        self.state = STOPPED
        self.started_with = None
        self.stopped = False
        self.shut = False

    def start(self, inp=None, out=None):
        self.started_with = (inp, out)
        self.state = STARTING
        return True

    def stop(self):
        self.stopped = True

    def shutdown(self):
        self.shut = True


class ModelOrch(FakeOrch):
    """FakeOrch with the orchestrator's model API."""

    def __init__(self, settings=None, on_event=None, selected="medium", downloaded=("small", "medium")):
        super().__init__(settings, on_event)
        self.selected_model = selected
        self.status = {n: ("downloaded" if n in downloaded else "not-downloaded") for n in CATALOG}
        self.select_calls = []
        self.accept = True
        self.pending = None  # latest accepted selection still loading (latest wins)

    def list_models(self):
        return [ModelInfo(n, CATALOG[n], self.status[n], n == self.selected_model) for n in CATALOG]

    def select_model(self, name):
        self.select_calls.append(name)
        if not (self.accept and self.state == STOPPED):
            return False
        self.pending = name
        if self.status[name] == "loaded" or name == self.selected_model:
            self.pending = None
            self.selected_model = name
        return True

    # helpers that mimic orchestrator events
    def loading(self, name, detail="loading"):
        self.status[name] = "loading"
        return Event(MODEL, f"Model '{name}': {detail}", data={
            "model": name, "status": "loading", "detail": detail, "selected": False})

    def loaded(self, name):
        self.status[name] = "loaded"
        self.selected_model = name
        self.pending = None
        return Event(MODEL, f"Model '{name}' selected", data={
            "model": name, "status": "loaded", "detail": "selected", "selected": True})


def test_edge_tts_supports_only_voice_language():
    t = EdgeTTS("en-US-AriaNeural")
    assert t.supports("en") and not t.supports("vi")


def test_cli_format_result_shows_latency():
    ev = Event(RESULT, "", 1, 3, data={"source_text": "Xin chào", "text": "Hello", "latency": 1.234})
    s = cli.format_event(ev)
    assert "Xin chào" in s and "Hello" in s and "1.23 s" in s


def test_cli_main_exits_nonzero_on_error(capsys):
    class ErrOrch(FakeOrch):
        def start(self, inp=None, out=None):
            self.on_event(Event(ERROR, "Output device 'BlackHole' not found"))
            self.on_event(Event(STATUS, "Stopped", data={"state": STOPPED}))
            return True

    made = []

    def factory(settings, on_event):
        o = ErrOrch(settings, on_event)
        made.append(o)
        return o

    assert cli.main(["--model", "small"], orchestrator_factory=factory) == 1
    assert made[0].settings.model_size == "small" and made[0].settings.output_device == "BlackHole"
    assert made[0].settings.translator == "offline"
    assert made[0].shut
    assert "not found" in capsys.readouterr().out


@pytest.fixture
def qapp():
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


def _window(qapp):
    from audio_translator.gui import MainWindow

    orch = FakeOrch()
    w = MainWindow(orch, input_devices=["Mic A"], output_devices=["Speakers", "BlackHole 2ch"])
    return w, orch


def test_gui_preselects_blackhole_and_starts(qapp):
    w, orch = _window(qapp)
    assert w.output_combo.currentText() == "BlackHole 2ch"
    assert w.lang_label.text() == "Tiếng Việt → English"
    assert w.button.text() == "Start"
    w.button.click()
    assert orch.started_with == (None, "BlackHole 2ch")
    assert w.button.text() == "Stop"
    w.on_event(Event(STATUS, "Listening", data={"state": RUNNING}))
    assert w.status_label.text() == "Listening" and not w.output_combo.isEnabled()
    w.on_event(Event(RESULT, "", 1, 1, data={"source_text": "Xin chào", "text": "Hello", "latency": 0.9}))
    assert "Xin chào" in w.log.toPlainText() and "0.90 s" in w.log.toPlainText()
    w.button.click()
    assert not w.button.isEnabled()
    import time

    deadline = time.monotonic() + 2
    while not orch.stopped and time.monotonic() < deadline:
        time.sleep(0.01)
    assert orch.stopped  # called off the Qt thread
    w.on_event(Event(STATUS, "Stopped", data={"state": STOPPED}))
    assert w.button.text() == "Start" and w.button.isEnabled()
    w.close()
    assert orch.shut


def test_gui_shows_device_error_and_returns_to_stopped(qapp):
    w, orch = _window(qapp)
    w.button.click()
    w.on_event(Event(STATUS, "Running", data={"state": RUNNING}))
    w.on_event(Event(ERROR, "Output device 'BlackHole 2ch' stopped unexpectedly"))
    w.on_event(Event(STATUS, "Stopped", data={"state": STOPPED}))
    assert "BlackHole 2ch" in w.error_label.text()
    assert w.button.text() == "Start" and w.button.isEnabled()


def test_gui_events_from_worker_thread_are_queued_to_qt_thread(qapp):
    import threading

    w, orch = _window(qapp)
    t = threading.Thread(target=lambda: orch.on_event(Event(STATUS, "From worker", data={"state": RUNNING})))
    t.start()
    t.join()
    assert w.status_label.text() != "From worker"  # not touched from the worker thread
    qapp.processEvents()
    assert w.status_label.text() == "From worker"


def test_gui_without_blackhole_never_defaults_to_speakers(qapp):
    from audio_translator.gui import MainWindow

    orch = FakeOrch()
    w = MainWindow(orch, input_devices=["Mic A"], output_devices=["Speakers"])
    assert w.output_combo.currentData() == ""
    assert "brew install blackhole-2ch" in w.error_label.text()
    w.button.click()
    assert orch.started_with is None and w.button.text() == "Start"


def test_cli_list_models_prints_catalog_with_status(capsys):
    from audio_translator.model_manager import ModelManager

    mgr = ModelManager(lambda n: None, is_downloaded=lambda n: n in ("small", "medium"))
    assert cli.print_models("small", manager=mgr) == 0
    out = capsys.readouterr().out
    rows = [ln for ln in out.splitlines()[1:] if ln.startswith(" ")]  # model rows are indented
    lines = {ln.split()[1] if ln.startswith(" *") else ln.split()[0]: ln for ln in rows}
    assert list(lines) == list(CATALOG)
    assert "downloaded" in lines["small"] and lines["small"].startswith(" *")
    assert lines["medium"].rstrip().endswith(" downloaded")
    assert "not-downloaded" in lines["large-v3"] and "3.1 GB" in lines["large-v3"]
    # local (converted) models that are missing come with the one-time convert command
    assert "not-downloaded" in lines["phowhisper-small"]
    assert ("'phowhisper-small' must be converted once first: uv run --with transformers --with torch "
            "python scripts/convert_phowhisper.py small") in out
    assert "convert_phowhisper.py medium" in out


def test_cli_rejects_model_outside_catalog():
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(["--model", "large-v2"])


def test_cli_translator_option_defaults_to_offline():
    assert cli.build_parser().parse_args([]).translator == "offline"
    assert cli.build_parser().parse_args(["--translator", "google"]).translator == "google"
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(["--translator", "claude"])


CATALOG_INDEX = {n: i for i, n in enumerate(CATALOG)}


def _model_window(qapp, **kw):
    from audio_translator.gui import MainWindow

    orch = ModelOrch(**kw)
    w = MainWindow(orch, input_devices=["Mic A"], output_devices=["BlackHole 2ch"])
    return w, orch


def _texts(w):
    return [w.model_combo.itemText(i) for i in range(w.model_combo.count())]


def _pick(w, name):
    """Simulate a user choosing ``name`` (activated is the user-only signal)."""
    idx = w.model_combo.findData(name)
    w.model_combo.setCurrentIndex(idx)
    w.model_combo.activated.emit(idx)


def test_gui_model_dropdown_initial_view(qapp):
    w, orch = _model_window(qapp)
    texts = _texts(w)
    assert [w.model_combo.itemData(i) for i in range(w.model_combo.count())] == list(CATALOG)
    assert w.model_combo.currentData() == "medium" and w.model_combo.isEnabled()
    assert texts[CATALOG_INDEX["medium"]] == "medium — đã tải ✓"
    assert texts[CATALOG_INDEX["small"]] == "small — đã tải"
    assert texts[CATALOG_INDEX["large-v3"]] == "large-v3 — chưa tải (~3.1 GB)"
    assert texts[CATALOG_INDEX["tiny"]] == "tiny — chưa tải (~75 MB)"
    assert texts[CATALOG_INDEX["phowhisper-small"]] == "phowhisper-small — chưa convert (xem --list-models)"
    assert orch.select_calls == []  # programmatic setup never selects


def test_gui_pick_downloaded_model_loads_then_selects(qapp):
    w, orch = _model_window(qapp)
    _pick(w, "small")
    assert orch.select_calls == ["small"]
    w.on_event(orch.loading("small"))
    assert "loading" in w.status_label.text()
    assert _texts(w)[CATALOG_INDEX["small"]] == "small — đang tải…"
    w.on_event(orch.loaded("small"))
    # real order after "selected": "ready" for it, then maybe another model released
    w.on_event(Event(MODEL, "Model 'small': ready", data={
        "model": "small", "status": "loaded", "detail": "ready", "selected": True}))
    w.on_event(Event(MODEL, "Model 'tiny': released from memory", data={
        "model": "tiny", "status": "downloaded", "detail": "released from memory", "selected": False}))
    assert w.status_label.text() == "Model 'small' selected"
    assert "ready" in w.log.toPlainText() and "released" in w.log.toPlainText()
    assert w.model_combo.currentData() == "small"
    assert _texts(w)[CATALOG_INDEX["small"]] == "small — trong RAM ✓"
    assert _texts(w)[CATALOG_INDEX["medium"]] == "medium — đã tải"
    assert orch.select_calls == ["small"]  # event-driven updates never re-select


def test_gui_pick_not_downloaded_model_reports_download(qapp):
    w, orch = _model_window(qapp)
    _pick(w, "large-v3")
    assert orch.select_calls == ["large-v3"]
    w.on_event(orch.loading("large-v3", "downloading (~3.1 GB) and loading"))
    assert "downloading (~3.1 GB)" in w.status_label.text()
    assert w.model_combo.isEnabled()


def test_gui_model_load_failure_returns_to_selected(qapp):
    w, orch = _model_window(qapp)
    _pick(w, "large-v3")
    orch.status["large-v3"] = "failed"
    w.on_event(Event(ERROR, "Could not load model 'large-v3': boom; keeping 'medium'",
                     data={"model": "large-v3"}))
    assert "Could not load model" in w.error_label.text()
    assert w.model_combo.currentData() == "medium"
    assert _texts(w)[CATALOG_INDEX["large-v3"]] == "large-v3 — lỗi"


def test_gui_refused_model_selection_returns_to_selected(qapp):
    w, orch = _model_window(qapp)
    orch.accept = False
    _pick(w, "small")
    assert orch.select_calls == ["small"]
    assert w.model_combo.currentData() == "medium"


def test_gui_model_dropdown_locked_while_session_runs(qapp):
    w, orch = _model_window(qapp)
    w.button.click()
    assert not w.model_combo.isEnabled()
    for state in (RUNNING, STOPPING):
        w.on_event(Event(STATUS, state, data={"state": state}))
        assert not w.model_combo.isEnabled()
    w.on_event(Event(STATUS, "Stopped", data={"state": STOPPED}))
    assert w.model_combo.isEnabled()


def test_gui_without_model_api_shows_fixed_placeholder(qapp):
    w, orch = _window(qapp)
    assert w.model_combo.count() == 1 and w.model_combo.currentText() == "(cố định)"
    assert not w.model_combo.isEnabled()
    w.on_event(Event(STATUS, "Stopped", data={"state": STOPPED}))
    assert not w.model_combo.isEnabled()


def test_gui_repick_current_model_while_other_load_pending_reaches_orchestrator(qapp):
    w, orch = _model_window(qapp)
    _pick(w, "large-v3")
    assert orch.pending == "large-v3"
    _pick(w, "medium")
    assert orch.select_calls == ["large-v3", "medium"]  # latest wins in the orchestrator
    assert orch.pending is None and orch.selected_model == "medium"
    assert w.model_combo.currentData() == "medium"


def test_gui_empty_model_list_shows_fixed_placeholder(qapp):
    from audio_translator.gui import MainWindow

    class EmptyModelsOrch(FakeOrch):
        selected_model = "medium"

        def list_models(self):
            return []

        def select_model(self, name):
            raise AssertionError("must not be called")

    w = MainWindow(EmptyModelsOrch(), input_devices=["Mic A"], output_devices=["BlackHole 2ch"])
    assert w.model_combo.count() == 1 and w.model_combo.currentText() == "(cố định)"
    assert not w.model_combo.isEnabled()
    w.on_event(Event(STATUS, "Stopped", data={"state": STOPPED}))
    assert not w.model_combo.isEnabled()
