"""One test per row of the plan's I/O & edge-case matrix (plus lifecycle)."""

import threading
import time

import numpy as np

from audio_translator.config import Settings
from audio_translator.models import ERROR, INFO, RESULT, STOPPED, WARNING
from conftest import FakeASR, FakeTranslator, FakeTTS, silence_frame, utterance


def test_happy_path(harness):
    h = harness()
    h.start()
    h.say([silence_frame()] * 5, utterance(30))
    ev = h.log.wait_for(lambda e: e.kind == RESULT)
    assert ev.data["source_text"] == "Xin chào, bạn khỏe không?"
    assert ev.data["text"] == "EN<Xin chào, bạn khỏe không?>"
    assert ev.data["latency"] is not None and ev.data["latency"] >= 0
    ts = ev.data["timestamps"]
    for k in ("speech_end", "asr_end", "mt_end", "tts_end", "first_audio_out"):
        assert k in ts
    assert ts["speech_end"] <= ts["asr_end"] <= ts["mt_end"] <= ts["tts_end"] <= ts["first_audio_out"]
    assert ev.session_id == h.orch.current_session and ev.segment_id == 1
    time.sleep(0.2)
    out = h.outputs[-1]
    assert out.kw["samplerate"] == 48000 and out.kw["channels"] == 2
    assert out.nonzero_samples >= 4800  # 0.1 s @ 24 kHz resampled to 48 kHz
    h.orch.stop()
    h.log.wait_state(STOPPED)
    assert h.log.of(ERROR) == []


def test_long_utterance_is_split_and_all_played_in_order(harness):
    s = Settings(max_segment_s=0.3)  # 10 frames
    h = harness(settings=s)
    h.start()
    h.say(utterance(35))
    # 4 forced cuts cover pre-roll + 35 speech frames; the closing 300 ms of
    # silence has no voiced frame, so it is discarded and never reaches ASR.
    h.log.wait_for(lambda e: e.kind == RESULT and e.segment_id == 4)
    deadline = time.monotonic() + 5
    while h.script and time.monotonic() < deadline:
        time.sleep(0.01)
    time.sleep(0.2)
    results = [e.segment_id for e in h.log.of(RESULT)]
    assert results == sorted(results) == [1, 2, 3, 4]
    assert [a.size for a in h.asr.calls] == [10 * 480] * 4
    joined = np.concatenate(h.asr.calls)
    speech = np.concatenate([f for f in utterance(35, 0)])
    start = int(np.flatnonzero(joined)[0]) - 1  # sine starts at an exact zero sample
    assert np.array_equal(joined[start:start + speech.size], speech)  # no audio lost at the cuts


def test_network_failure_skips_segment_and_session_continues(harness):
    asr = FakeASR(responses=["một", "hai"])
    h = harness(asr=asr, translator=FakeTranslator(fail_on={"một"}))
    h.start()
    h.say(utterance(15), utterance(15))
    warn = h.log.wait_for(lambda e: e.kind == WARNING and "translation failed" in e.message)
    assert warn.segment_id == 1
    res = h.log.wait_for(lambda e: e.kind == RESULT)
    assert res.segment_id == 2 and res.data["source_text"] == "hai"
    assert h.orch.state == "running"


def test_tts_failure_skips_segment(harness):
    asr = FakeASR(responses=["một", "hai"])
    h = harness(asr=asr, tts=FakeTTS(fail_on={"EN<một>"}))
    h.start()
    h.say(utterance(15), utterance(15))
    h.log.wait_for(lambda e: e.kind == WARNING and "synthesis failed" in e.message)
    res = h.log.wait_for(lambda e: e.kind == RESULT)
    assert res.segment_id == 2  # no 5 s gap wait: the failed id was skipped explicitly
    assert res.data["timestamps"]["first_audio_out"] - res.data["timestamps"]["tts_end"] < 1.0


def test_empty_transcript_makes_no_mt_or_tts_call(harness):
    h = harness(asr=FakeASR(default="   "))
    h.start()
    h.say(utterance(15))
    h.log.wait_for(lambda e: e.kind == INFO and "no speech" in e.message)
    time.sleep(0.1)
    assert h.translator.calls == [] and h.tts.calls == []
    assert h.log.of(RESULT) == []


def test_stop_mid_flight_discards_old_session(harness):
    gate = threading.Event()
    h = harness(tts=FakeTTS(gate=gate))
    h.start()
    h.say(utterance(15), utterance(15))
    assert h.tts.entered.wait(5)
    deadline = time.monotonic() + 5
    while h.orch._tl_q.qsize() == 0 and time.monotonic() < deadline:
        time.sleep(0.01)  # let segment 2 queue up behind the blocked TTS
    out = h.outputs[-1]
    h.orch.stop()
    h.log.wait_state(STOPPED)
    gate.set()  # in-flight synthesis now completes for a dead session
    time.sleep(0.3)
    assert h.log.of(RESULT) == []
    assert out.nonzero_samples == 0 and not out.active
    assert h.orch.stats()["dropped_stale"] >= 1
    stopped = [e for e in h.log.events if e.data.get("state") == STOPPED][-1]
    assert stopped.data["discarded"] >= 1


def test_restart_after_stop_uses_new_session(harness):
    h = harness()
    h.start()
    first = h.orch.current_session
    h.orch.stop()
    n = len(h.log.events)
    h.log.wait_state(STOPPED)
    assert h.orch.start(None, "BlackHole")
    h.log.wait_state("running", after=n)
    assert h.orch.current_session == first + 1
    h.say(utterance(15))
    ev = h.log.wait_for(lambda e: e.kind == RESULT)
    assert ev.session_id == first + 1 and ev.segment_id == 1


def test_overload_drops_oldest_pending_and_warns(harness):
    gate = threading.Event()
    s = Settings(translation_queue_size=1, playout_max_pending=8)
    h = harness(settings=s, asr=FakeASR(responses=[f"câu {i}" for i in range(1, 7)]),
                tts=FakeTTS(gate=gate))
    h.start()
    for _ in range(5):
        h.say(utterance(12, 12))
    h.log.wait_for(lambda e: e.kind == WARNING and "Overload" in e.message)
    gate.set()
    h.log.wait_for(lambda e: e.kind == RESULT and e.segment_id == 5)
    played = [e.segment_id for e in h.log.of(RESULT)]
    dropped = [e.segment_id for e in h.log.of(WARNING) if "Overload" in e.message]
    assert played == sorted(played)
    assert set(played).isdisjoint(dropped) and set(played) | set(dropped) == {1, 2, 3, 4, 5}
    assert h.orch.stats()["dropped_overload"] == len(dropped) >= 1
    assert h.orch.state == "running"


def test_output_device_removed_reports_and_stops(harness):
    h = harness()
    h.start()
    h.outputs[-1].remove_device(notify=True)
    err = h.log.wait_for(lambda e: e.kind == ERROR)
    assert "BlackHole 2ch" in err.message
    h.log.wait_state(STOPPED)
    assert h.orch.state == STOPPED
    assert h.inputs[-1].closed and h.outputs[-1].closed


def test_output_device_hang_detected_by_watchdog(harness):
    h = harness()
    h.start()
    h.outputs[-1].hang()
    err = h.log.wait_for(lambda e: e.kind == ERROR, timeout=3)
    assert "Output device 'BlackHole 2ch'" in err.message
    h.log.wait_state(STOPPED)


def test_input_device_removed_reports_and_stops(harness):
    h = harness()
    h.start()
    h.inputs[-1].remove_device(notify=False)
    err = h.log.wait_for(lambda e: e.kind == ERROR, timeout=3)
    assert "Input device 'Fake Mic'" in err.message
    h.log.wait_state(STOPPED)


def test_missing_output_device_fails_start_cleanly(harness):
    h = harness(missing_output=True)
    assert h.orch.start(None, "BlackHole")
    err = h.log.wait_for(lambda e: e.kind == ERROR)
    assert "BlackHole" in err.message
    h.log.wait_state(STOPPED)
    assert h.outputs == [] and h.inputs == []
    assert not h.asr.loaded  # devices are resolved before the (large) model load


def test_failure_after_stop_is_not_reported(harness):
    gate = threading.Event()

    class SlowFailingMT(FakeTranslator):
        def translate(self, text):
            self.calls.append(text)
            gate.wait(5)
            raise ConnectionError("network unreachable")

    h = harness(translator=SlowFailingMT())
    h.start()
    h.say(utterance(15))
    deadline = time.monotonic() + 5
    while not h.translator.calls and time.monotonic() < deadline:
        time.sleep(0.01)
    h.orch.stop()
    h.log.wait_state(STOPPED)
    gate.set()
    time.sleep(0.3)
    assert h.log.of(WARNING) == []
    assert h.orch.stats()["segments_skipped"] == 0


def test_unsupported_tts_language_fails_start():
    from conftest import Harness

    class FrenchOnly(FakeTTS):
        def supports(self, lang):
            return lang == "fr"

    h = Harness(tts=FrenchOnly())
    try:
        h.orch.start(None, "BlackHole")
        err = h.log.wait_for(lambda e: e.kind == ERROR)
        assert "target language 'en'" in err.message
    finally:
        h.orch.shutdown()


def test_stop_while_starting_aborts(harness):
    load_gate = threading.Event()

    class SlowASR(FakeASR):
        def load(self):
            load_gate.wait(5)

    h = harness(asr=SlowASR())
    assert h.orch.start(None, "BlackHole")
    h.orch.stop()
    load_gate.set()
    h.log.wait_state(STOPPED)
    assert h.inputs == [] and h.outputs == []


# ------------------------------------------------- model selection (plan-model-selection)
from audio_translator.model_manager import ModelManager  # noqa: E402
from audio_translator.models import MODEL, RUNNING, STARTING  # noqa: E402
from conftest import FakeModelFactory  # noqa: E402


def model_harness(harness, gated=(), fail=(), downloaded=("small", "medium"), model="medium"):
    f = FakeModelFactory(gated=gated, fail=fail)
    mgr = ModelManager(f, max_loaded=2, is_downloaded=lambda n: n in downloaded)
    h = harness(settings=Settings(model_size=model), model_manager=mgr)
    return h, f, mgr


def selected_event(h, name):
    return h.log.wait_for(lambda e: e.kind == MODEL and e.data.get("model") == name
                          and e.data.get("detail") == "selected")


def test_list_models_marks_selected_and_download_state(harness):
    h, _, _ = model_harness(harness)
    info = {m.name: m for m in h.orch.list_models()}
    assert info["medium"].selected and not info["small"].selected
    assert info["small"].status == "downloaded" and info["large-v3"].status == "not-downloaded"


def test_cached_switch_is_immediate_without_reload(harness):
    h, f, _ = model_harness(harness)
    for name in ("small", "medium"):
        assert h.orch.select_model(name)
        selected_event(h, name)
    assert sorted(f.load_calls) == ["medium", "small"]
    for name in ("small", "medium", "small"):
        t0 = time.monotonic()
        assert h.orch.select_model(name)
        assert h.orch.selected_model == name and time.monotonic() - t0 < 1.0
    assert sorted(f.load_calls) == ["medium", "small"]  # no reload


def test_background_preload_returns_at_once(harness):
    h, f, mgr = model_harness(harness, gated=["large-v3-turbo"])
    t0 = time.monotonic()
    assert h.orch.select_model("large-v3-turbo")
    assert time.monotonic() - t0 < 0.2
    loading = h.log.wait_for(lambda e: e.kind == MODEL and e.data.get("status") == "loading")
    assert "downloading" in loading.message  # not on disk -> download first
    assert h.orch.selected_model == "medium"  # not applied until loaded
    f.release("large-v3-turbo")
    selected_event(h, "large-v3-turbo")
    assert h.orch.selected_model == "large-v3-turbo"
    statuses = [e.data["status"] for e in h.log.of(MODEL) if e.data.get("model") == "large-v3-turbo"]
    assert statuses.index("loading") < statuses.index("loaded")


def test_start_during_preload_waits_for_same_load(harness):
    h, f, _ = model_harness(harness, gated=["small"])
    h.orch.select_model("small")
    assert f.started["small"].wait(2)
    assert h.orch.start(None, "BlackHole")
    h.log.wait_for(lambda e: e.kind == "status" and "Loading ASR model 'small'" in e.message)
    assert h.orch.state == STARTING
    f.release("small")
    h.log.wait_state(RUNNING)
    assert f.load_calls == ["small"]
    assert h.orch.selected_model == "small"
    h.say(utterance(15))
    res = h.log.wait_for(lambda e: e.kind == RESULT)
    assert res.data["source_text"] == "text from small"


def test_stop_during_model_wait_aborts_and_unpins(harness):
    h, f, mgr = model_harness(harness, gated=["medium"])
    assert h.orch.start(None, "BlackHole")
    assert f.started["medium"].wait(2)
    h.orch.stop()
    f.release("medium")
    h.log.wait_state(STOPPED)
    assert not [e for e in h.log.of("status") if e.data.get("state") == RUNNING]
    assert not mgr._pins and mgr.is_loaded("medium")


def test_session_model_is_pinned_and_never_evicted(harness):
    h, f, mgr = model_harness(harness)
    h.start()
    assert mgr._pins["medium"] == 1
    mgr.get("small")
    mgr.get("tiny")  # cap 2: "medium" is LRU but in use
    assert "medium" in mgr.loaded_names()
    h.orch.stop()
    h.log.wait_state(STOPPED)
    assert not mgr._pins


def test_rapid_reselection_latest_wins(harness):
    h, f, mgr = model_harness(harness, gated=["tiny", "base"])
    h.orch.select_model("tiny")
    h.orch.select_model("base")
    f.release("base")
    selected_event(h, "base")
    f.release("tiny")
    h.log.wait_for(lambda e: e.kind == MODEL and e.data.get("model") == "tiny"
                   and e.data.get("status") == "loaded")
    time.sleep(0.05)
    assert h.orch.selected_model == "base"
    assert mgr.is_loaded("tiny")  # earlier load lands in cache, never applied


def test_rapid_reselection_to_cached_model_beats_slow_load(harness):
    h, f, mgr = model_harness(harness, gated=["large-v3"])
    h.orch.select_model("small")
    selected_event(h, "small")
    h.orch.select_model("large-v3")
    h.orch.select_model("small")
    assert h.orch.selected_model == "small"
    f.release("large-v3")
    h.log.wait_for(lambda e: e.kind == MODEL and e.data.get("model") == "large-v3"
                   and e.data.get("status") == "loaded")
    time.sleep(0.05)
    assert h.orch.selected_model == "small"


def test_select_while_running_is_refused(harness):
    h, f, _ = model_harness(harness)
    h.start()
    assert not h.orch.select_model("small")
    h.log.wait_for(lambda e: e.kind == WARNING and "Cannot change the model" in e.message)
    assert h.orch.selected_model == "medium" and h.orch.state == "running"
    h.say(utterance(15))
    assert h.log.wait_for(lambda e: e.kind == RESULT).data["source_text"] == "text from medium"


def test_load_failure_keeps_previous_model(harness):
    h, f, _ = model_harness(harness, fail=["small"])
    assert h.orch.select_model("small")
    err = h.log.wait_for(lambda e: e.kind == ERROR)
    assert "small" in err.message and "keeping 'medium'" in err.message
    assert h.orch.selected_model == "medium"
    h.start()  # previous model still usable
    assert f.load_calls[-1] == "medium"


def test_unknown_model_is_rejected(harness):
    h, _, _ = model_harness(harness)
    assert not h.orch.select_model("large-v2")
    assert "Unknown model" in h.log.wait_for(lambda e: e.kind == ERROR).message
    assert h.orch.selected_model == "medium"


def test_injected_asr_has_no_model_selection(harness):
    h = harness()
    assert h.orch.list_models() == []
    assert not h.orch.select_model("small")


def test_start_during_failing_preload_runs_on_previous_model(harness):
    h, f, _ = model_harness(harness, gated=["small"], fail=["small"])
    h.orch.select_model("small")
    assert f.started["small"].wait(2)
    assert h.orch.start(None, "BlackHole")
    f.release("small")
    h.log.wait_state(RUNNING)
    assert h.orch.selected_model == "medium"
    h.say(utterance(15))
    assert h.log.wait_for(lambda e: e.kind == RESULT).data["source_text"] == "text from medium"
    assert len(h.log.of(ERROR)) == 1 and "keeping 'medium'" in h.log.of(ERROR)[0].message


def test_loaded_event_for_selected_model_is_marked_selected(harness):
    h, f, _ = model_harness(harness)
    h.orch.select_model("small")
    ev = h.log.wait_for(lambda e: e.kind == MODEL and e.data.get("model") == "small"
                        and e.data.get("status") == "loaded" and e.data.get("detail") == "ready")
    assert ev.data["selected"] is True


# ------------------------------------------------------------ MT backend selection
class Backends:
    """Fake package preparer + offline/Google factories for the orchestrator's MT wiring."""

    def __init__(self, cached=True, prepare_error=None, offline_fail_on=()):
        self.cached = cached
        self.prepare_error = prepare_error
        self.prepare_calls = 0
        self.offline = FakeTranslator(fail_on=set(offline_fail_on))
        self.google = FakeTranslator()
        self.google.translate = lambda text, _g=self.google: (_g.calls.append(text), f"G<{text}>")[1]
        self.google_made = 0

    def prepare(self, cache_dir, url, on_download=None, source="vi", target="en"):
        self.prepare_calls += 1
        self.langs = (source, target)
        if not self.cached and on_download:
            on_download()
        if self.prepare_error:
            raise self.prepare_error
        self.cached = True
        return "/fake/pkg"

    def make_google(self, source, target):
        self.google_made += 1
        return self.google

    def kwargs(self):
        return dict(package_preparer=self.prepare, offline_translator_factory=lambda pkg: self.offline,
                    google_translator_factory=self.make_google)


def test_offline_happy_path_makes_no_google_call(harness):
    b = Backends()
    h = harness(mt=b.kwargs())
    h.start()
    h.say(utterance(15))
    ev = h.log.wait_for(lambda e: e.kind == RESULT)
    assert ev.data["text"] == "EN<Xin chào, bạn khỏe không?>"
    assert b.google_made == 0 and b.google.calls == []
    assert h.log.of(WARNING) == []


def test_offline_first_run_reports_download(harness):
    b = Backends(cached=False)
    h = harness(mt=b.kwargs())
    h.start()
    assert any("Downloading translation model" in e.message for e in h.log.of("status"))
    h.say(utterance(15))
    assert h.log.wait_for(lambda e: e.kind == RESULT).data["text"].startswith("EN<")


def test_offline_package_failure_starts_with_google_and_retries_next_session(harness):
    b = Backends(cached=False, prepare_error=RuntimeError("download failed: Network is unreachable"))
    h = harness(mt=b.kwargs())
    h.start()
    warn = h.log.wait_for(lambda e: e.kind == WARNING)
    assert "Network is unreachable" in warn.message and "Google" in warn.message
    h.say(utterance(15))
    assert h.log.wait_for(lambda e: e.kind == RESULT).data["text"] == "G<Xin chào, bạn khỏe không?>"
    h.orch.stop()
    h.log.wait_state(STOPPED)
    b.prepare_error = None
    n = len(h.log.events)
    assert h.orch.start(None, "BlackHole")
    h.log.wait_state(RUNNING, after=n)
    assert b.prepare_calls == 2
    h.say(utterance(15))
    assert h.log.wait_for(lambda e: e.kind == RESULT and h.log.events.index(e) > n).data["text"].startswith("EN<")


def test_offline_segment_error_falls_back_to_google_for_that_segment(harness):
    b = Backends(offline_fail_on={"một"})
    asr = FakeASR(responses=["một", "hai"])
    h = harness(mt=b.kwargs(), asr=asr)
    h.start()
    h.say(utterance(15), utterance(15))
    h.log.wait_for(lambda e: e.kind == RESULT and e.segment_id == 2)
    texts = [e.data["text"] for e in h.log.of(RESULT)]
    assert texts == ["G<một>", "EN<hai>"]
    assert b.offline.calls == ["một", "hai"] and b.google.calls == ["một"]
    assert any("Offline translation failed" in e.message for e in h.log.of(WARNING))


def test_offline_and_google_both_fail_skips_segment(harness):
    b = Backends(offline_fail_on={"một"})
    b.google.translate = lambda text: (_ for _ in ()).throw(ConnectionError("too many requests"))
    h = harness(mt=b.kwargs(), asr=FakeASR(responses=["một", "hai"]))
    h.start()
    h.say(utterance(15), utterance(15))
    ev = h.log.wait_for(lambda e: e.kind == RESULT)
    assert ev.segment_id == 2 and ev.data["text"] == "EN<hai>"
    assert any("translation failed" in e.message and "too many requests" in e.message
               for e in h.log.of(WARNING))
    assert h.orch.stats()["segments_skipped"] == 1


def test_google_mode_never_prepares_package(harness):
    b = Backends()
    h = harness(settings=Settings(translator="google"), mt=b.kwargs())
    h.start()
    h.say(utterance(15))
    assert h.log.wait_for(lambda e: e.kind == RESULT).data["text"] == "G<Xin chào, bạn khỏe không?>"
    assert b.prepare_calls == 0 and b.offline.calls == []


def test_empty_offline_result_skips_segment(harness):
    b = Backends()
    b.offline.translate = lambda text: ""
    h = harness(mt=b.kwargs())
    h.start()
    h.say(utterance(15))
    h.log.wait_for(lambda e: e.kind == INFO and "empty translation" in e.message)
    assert h.tts.calls == [] and b.google.calls == []


def test_stop_during_translation_download_aborts_before_asr_load(harness):
    gate, entered = threading.Event(), threading.Event()
    b = Backends(cached=False)
    real = b.prepare

    def gated(*a, **kw):
        entered.set()
        assert gate.wait(5)
        return real(*a, **kw)

    asr = FakeASR()
    kw = b.kwargs()
    kw["package_preparer"] = gated
    h = harness(mt=kw, asr=asr)
    assert h.orch.start(None, "BlackHole")
    assert entered.wait(2)
    h.orch.stop()
    gate.set()
    h.log.wait_state(STOPPED)
    assert not asr.loaded and h.inputs == [] and h.outputs == []


def test_package_preparer_gets_configured_languages(harness):
    b = Backends()
    h = harness(mt=b.kwargs())
    h.start()
    assert b.langs == ("vi", "en")


def test_offline_load_failure_uses_google_and_retries_next_session(harness):
    b = Backends()
    fails = [RuntimeError("cannot load model.bin")]

    def load():
        if fails:
            raise fails.pop()

    b.offline.load = load
    h = harness(mt=b.kwargs())
    h.start()
    warn = h.log.wait_for(lambda e: e.kind == WARNING)
    assert "cannot load model.bin" in warn.message and "Google" in warn.message
    h.say(utterance(15))
    assert h.log.wait_for(lambda e: e.kind == RESULT).data["text"] == "G<Xin chào, bạn khỏe không?>"
    h.orch.stop()
    h.log.wait_state(STOPPED)
    n = len(h.log.events)
    assert h.orch.start(None, "BlackHole")
    h.log.wait_state(RUNNING, after=n)
    assert b.prepare_calls == 2
    h.say(utterance(15))
    assert h.log.wait_for(lambda e: e.kind == RESULT and h.log.events.index(e) > n).data["text"].startswith("EN<")
