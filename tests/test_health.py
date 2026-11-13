"""Speaker indicator: signals and thresholds, smoothing, orchestrator HEALTH events, CLI/GUI, log."""

import csv
import os
import threading
import time

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from audio_translator import cli  # noqa: E402
from audio_translator.config import Settings  # noqa: E402
from audio_translator.health import (  # noqa: E402
    BACKLOG, GREEN, LENGTH, QUALITY, RED, YELLOW, HealthTracker, Metrics, assess, unclear_reason,
)
from audio_translator.models import HEALTH, RESULT, STATUS, STOPPED, Event  # noqa: E402
from conftest import FakeASR, FakeInputStream, utterance  # noqa: E402

S = Settings()


def top(m, s=S):
    r = assess(m, s)
    return (r[0].level, r[0].signal) if r else (GREEN, "")


# ------------------------------------------------------------ pure signals
@pytest.mark.parametrize("open_s,want", [(0.0, (GREEN, "")), (7.99, (GREEN, "")), (8.0, (YELLOW, LENGTH)),
                                         (11.99, (YELLOW, LENGTH)), (12.0, (RED, LENGTH)), (14.9, (RED, LENGTH))])
def test_length_thresholds(open_s, want):
    assert top(Metrics(open_s=open_s)) == want


@pytest.mark.parametrize("waiting,busy,want", [
    (0, 0.0, GREEN), (0, 3.0, GREEN), (0, 3.01, YELLOW), (1, 0.0, YELLOW), (1, 6.0, YELLOW),
    (0, 6.01, RED), (2, 0.0, RED), (5, 1.0, RED)])
def test_backlog_thresholds(waiting, busy, want):
    assert top(Metrics(asr_waiting=waiting, asr_busy_s=busy))[0] == want


def test_messages_and_details_are_in_vietnamese():
    (r,) = assess(Metrics(open_s=12.4), S)
    assert r.message == "Dừng lại một chút" and r.text == "Dừng lại một chút (câu 12,4 s)"
    (r,) = assess(Metrics(asr_waiting=1), S)
    assert r.message == "Đang xử lý, nói chậm lại" and "1 câu đang chờ" in r.text
    (r,) = assess(Metrics(asr_busy_s=7), S)
    assert r.message == "Dừng nói, chờ hệ thống bắt kịp" and "nhận dạng 7,0 s" in r.text
    (r,) = assess(Metrics(unclear="bộ lọc chặn"), S)
    assert (r.level, r.message) == (YELLOW, "Câu vừa rồi nghe không rõ, hãy nói lại")


def test_highest_level_wins_then_priority_backlog_length_quality():
    r = assess(Metrics(open_s=12.5, asr_waiting=1, unclear="x"), S)
    assert [(x.signal, x.level) for x in r] == [(LENGTH, RED), (BACKLOG, YELLOW), (QUALITY, YELLOW)]
    r = assess(Metrics(open_s=9, asr_waiting=1, unclear="x"), S)
    assert [x.signal for x in r] == [BACKLOG, LENGTH, QUALITY]
    r = assess(Metrics(open_s=13, asr_waiting=2), S)
    assert [x.signal for x in r] == [BACKLOG, LENGTH]


def test_unclear_reason_rules():
    u = lambda **k: unclear_reason(S, **{"filtered": False, "asr_s": 1.0, "audio_s": 3.0, "fallback": False, **k})
    assert u() == ""
    assert u(filtered=True) == "bộ lọc chặn"
    assert u(fallback=True) == "phải giải mã lại"
    assert unclear_reason(Settings(indicator_fallback=False), filtered=False, asr_s=1, audio_s=3, fallback=True) == ""
    assert "nhận dạng 6,6 s cho 1,6 s audio" in u(asr_s=6.56, audio_s=1.56)  # the real stall
    assert u(asr_s=1.69, audio_s=0.93) != ""  # the 0.93 s / 18-word hallucination
    assert u(asr_s=1.26, audio_s=1.05) == ""  # below the 1.5 s floor
    assert u(asr_s=2.08, audio_s=2.97) == ""  # normal: ratio 0.7
    assert u(asr_s=5.0, audio_s=0.0) == ""


# ------------------------------------------------------------ tracker
def test_up_is_immediate_down_waits_for_the_hold():
    t = HealthTracker(Settings(indicator_down_hold_s=1.0))
    assert t.update(0.0, Metrics()).level == GREEN
    assert t.update(0.1, Metrics(asr_waiting=2)).level == RED
    assert t.update(0.2, Metrics()).level == RED  # held
    h = t.update(0.9, Metrics(open_s=3))
    assert h.level == RED and h.metrics.open_s == 3  # numbers refresh while held
    assert t.update(1.25, Metrics()).level == GREEN


def test_a_flicker_back_up_restarts_the_hold():
    t = HealthTracker(Settings(indicator_down_hold_s=1.0))
    t.update(0.0, Metrics(asr_waiting=1))
    t.update(0.5, Metrics())
    t.update(0.8, Metrics(asr_waiting=1))  # back to yellow: hold restarts
    assert t.update(1.6, Metrics()).level == YELLOW
    assert t.update(2.7, Metrics()).level == GREEN


def test_quality_yellow_holds_then_expires():
    t = HealthTracker(Settings(indicator_unclear_hold_s=4.0, indicator_down_hold_s=0.0))
    t.note_unclear(10.0, "bộ lọc chặn")
    h = t.update(10.1, Metrics())
    assert (h.level, h.main.signal) == (YELLOW, QUALITY) and h.event_data()["reason"].endswith("(bộ lọc chặn)")
    assert t.update(13.9, Metrics()).level == YELLOW
    assert t.update(14.0, Metrics()).level == GREEN
    t.note_unclear(20.0, "")  # nothing unclear: ignored
    assert t.update(20.1, Metrics()).level == GREEN


def test_event_data_lists_other_active_reasons():
    t = HealthTracker(S)
    t.note_unclear(0.0, "phải giải mã lại")
    d = t.update(0.1, Metrics(open_s=12.0, asr_waiting=1)).event_data()
    assert d["level"] == RED and d["signal"] == LENGTH and d["reason"] == "Dừng lại một chút (câu 12,0 s)"
    assert d["others"] == ["Đang xử lý, nói chậm lại (1 câu đang chờ)",
                           "Câu vừa rồi nghe không rõ, hãy nói lại (phải giải mã lại)"]
    d = HealthTracker(S).update(0.0, Metrics()).event_data()
    assert (d["level"], d["others"]) == (GREEN, []) and d["reason"]


def test_history_gives_the_highest_level_in_a_window():
    t = HealthTracker(Settings(indicator_down_hold_s=0.0))
    t.update(1.0, Metrics())
    t.update(2.0, Metrics(open_s=9))
    t.update(3.0, Metrics(open_s=13))
    t.update(4.0, Metrics())
    t.update(5.0, Metrics(asr_waiting=1))
    assert t.max_between(0.0, 1.5) == (GREEN, "")
    assert t.max_between(1.5, 2.5) == (YELLOW, "Câu dài, hãy ngắt nghỉ")
    assert t.max_between(2.5, 6.0) == (RED, "Dừng lại một chút")
    assert t.max_between(4.2, 4.8) == (GREEN, "")
    assert t.max_between(5.5, 9.0) == (YELLOW, "Đang xử lý, nói chậm lại")
    t.reset(10.0)  # a new session forgets the old one
    assert t.max_between(2.5, 6.0) == ("", "") and t.max_between(10.0, 11.0) == (GREEN, "")


# ------------------------------------------------------------ orchestrator
def idx(h, ev):
    """Position of this very event (Event compares by value: equal greens are not the same)."""
    return next(i for i, e in enumerate(h.log.events) if e is ev)


def green_after(h, ev):
    return h.log.wait_for(lambda e: e.kind == HEALTH and e.data["level"] == GREEN and e.data["changed"]
                          and idx(h, e) > idx(h, ev))


def lights(h, sid=None):
    return [e for e in h.log.of(HEALTH) if sid is None or e.session_id == sid]


def fast(**kw):
    base = dict(indicator_rate_hz=100.0, indicator_down_hold_s=0.05)
    return Settings(**{**base, **kw})


def test_session_start_shows_green(harness):
    h = harness(settings=fast())
    h.start()
    ev = h.log.wait_for(lambda e: e.kind == HEALTH)
    assert ev.data["level"] == GREEN and ev.session_id == h.orch.current_session and ev.data["changed"]


def test_long_utterance_goes_yellow_then_red_and_back(harness, monkeypatch):
    monkeypatch.setattr(FakeInputStream, "period", 0.01)  # 30 ms audio per 10 ms: readable timings
    h = harness(settings=fast(indicator_long_warn_s=1.0, indicator_long_stop_s=2.0, max_segment_s=4.0))
    h.start()
    h.say(utterance(100))  # 3 s of speech
    h.log.wait_for(lambda e: e.kind == HEALTH and e.data["level"] == RED and e.data["signal"] == LENGTH)
    h.log.wait_for(lambda e: e.kind == RESULT)
    green_after(h, lights(h)[1])
    seq = [e.data["level"] for e in lights(h) if e.data["changed"]]
    assert seq[:3] == [GREEN, YELLOW, RED] and seq[-1] == GREEN


def test_backlog_turns_red_and_is_logged(harness, tmp_path):
    gate = threading.Event()

    class SlowASR(FakeASR):
        def transcribe(self, audio, sample_rate):
            if not self.calls:
                gate.wait(5)
            return super().transcribe(audio, sample_rate)

    p = tmp_path / "lat.csv"
    h = harness(settings=fast(indicator_asr_warn_s=0.1, indicator_asr_stop_s=10.0, latency_log=str(p)),
                asr=SlowASR(responses=["một", "hai", "ba"]))
    h.start()
    h.say(utterance(10))
    ev = h.log.wait_for(lambda e: e.kind == HEALTH and e.data["level"] == YELLOW and "nhận dạng" in e.data["reason"])
    assert ev.data["signal"] == BACKLOG and ev.data["reason"].startswith("Đang xử lý, nói chậm lại")
    h.say(utterance(10), utterance(10))
    ev = h.log.wait_for(lambda e: e.kind == HEALTH and e.data["level"] == RED)
    assert ev.data["reason"].startswith("Dừng nói, chờ hệ thống bắt kịp") and "2 câu đang chờ" in ev.data["reason"]
    gate.set()
    h.log.wait_for(lambda e: e.kind == RESULT and e.data["source_text"] == "ba")
    green_after(h, ev)
    end = time.monotonic() + 5
    while time.monotonic() < end and (not p.exists() or len(list(csv.DictReader(open(p)))) < 3):
        time.sleep(0.02)
    rows = {r["segment_id"]: r for r in csv.DictReader(open(p, encoding="utf-8"))}
    assert rows["1"]["health_max"] in (YELLOW, RED)
    assert (rows["3"]["health_max"], rows["3"]["health_reason"]) == (RED, "Dừng nói, chờ hệ thống bắt kịp")


def test_filtered_or_fallback_segment_shows_quality_yellow(harness):
    class FilteredASR(FakeASR):
        def transcribe(self, audio, sample_rate):
            n = len(self.calls)
            super().transcribe(audio, sample_rate)
            self.dropped = [("blacklist", "Hãy subscribe cho kênh")] if n == 0 else []
            self.last_temperature = 0.4 if n == 1 else 0.0
            return "" if n == 0 else "xin chào"

    h = harness(settings=fast(indicator_unclear_hold_s=0.3), asr=FilteredASR())
    h.start()
    h.say(utterance(10))
    ev = h.log.wait_for(lambda e: e.kind == HEALTH and e.data["signal"] == QUALITY)
    assert ev.data["level"] == YELLOW and "bộ lọc chặn" in ev.data["reason"]
    green_after(h, ev)
    h.say(utterance(10))
    h.log.wait_for(lambda e: e.kind == HEALTH and "phải giải mã lại" in e.data["reason"])


def test_clean_segment_stays_green(harness):
    h = harness(settings=fast())
    h.start()
    h.say(utterance(10))
    h.log.wait_for(lambda e: e.kind == RESULT)
    time.sleep(0.2)
    assert {e.data["level"] for e in lights(h)} == {GREEN}


def test_stop_start_resets_and_no_event_from_an_old_session(harness):
    gate = threading.Event()

    class StuckASR(FakeASR):
        def transcribe(self, audio, sample_rate):
            gate.wait(5)
            return super().transcribe(audio, sample_rate)

    h = harness(settings=fast(indicator_asr_warn_s=0.05, indicator_asr_stop_s=0.1), asr=StuckASR())
    h.start()
    s1 = h.orch.current_session
    h.say(utterance(10))
    h.log.wait_for(lambda e: e.kind == HEALTH and e.data["level"] == RED)
    n = len(h.log.events)
    h.orch.stop()
    stopped = h.log.wait_state(STOPPED, after=n)
    gate.set()
    time.sleep(0.3)
    i = idx(h, stopped)
    assert not [e for e in h.log.events[i:] if e.kind == HEALTH and e.session_id == s1]
    assert h.orch.start(None, "BlackHole")
    ev = h.log.wait_for(lambda e: e.kind == HEALTH and idx(h, e) > i)
    assert ev.session_id not in (None, s1) and ev.data["level"] == GREEN and ev.data["changed"]


def test_evaluation_rate_is_limited(harness, monkeypatch):
    h = harness(settings=Settings(indicator_rate_hz=10.0))
    calls = []
    real = h.orch._health.update
    monkeypatch.setattr(h.orch._health, "update", lambda now, m: calls.append(now) or real(now, m))
    h.start()
    h.say(*[utterance(5, 5) for _ in range(40)])  # keeps the monitor queue busy
    time.sleep(1.0)
    assert 3 <= len(calls) <= 12
    gaps = [b - a for a, b in zip(calls, calls[1:])]
    assert min(gaps) >= 0.099
    assert len([e for e in lights(h) if not e.data["changed"]]) == 0  # green: no refresh events


# ------------------------------------------------------------ CLI / GUI
def test_cli_formats_the_light():
    ev = Event(HEALTH, "", 1, data={"level": RED, "reason": "Dừng lại một chút (câu 12,4 s)",
                                    "others": ["Đang xử lý, nói chậm lại"], "changed": True})
    assert cli.format_event(ev) == "[đèn] ĐỎ: Dừng lại một chút (câu 12,4 s)  (cũng: Đang xử lý, nói chậm lại)"


def test_cli_prints_the_light_only_when_it_changes(capsys):
    from test_frontends import FakeOrch

    class LightOrch(FakeOrch):
        def start(self, inp=None, out=None):
            for lvl, changed in ((GREEN, True), (YELLOW, True), (YELLOW, False), (YELLOW, False), (GREEN, True)):
                self.on_event(Event(HEALTH, "", 1, data={"level": lvl, "reason": f"r-{lvl}", "changed": changed}))
            self.on_event(Event(STATUS, "Stopped", data={"state": STOPPED}))
            return True

    assert cli.main([], orchestrator_factory=lambda s, on_event: LightOrch(s, on_event)) == 0
    out = [line for line in capsys.readouterr().out.splitlines() if line.startswith("[đèn]")]
    assert out == ["[đèn] XANH: r-green", "[đèn] VÀNG: r-yellow", "[đèn] XANH: r-green"]


@pytest.fixture
def qapp():
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


def test_gui_light_shows_colour_icon_and_words(qapp):
    from test_frontends import _window

    from audio_translator.models import RUNNING

    w, orch = _window(qapp)
    assert w.light_level == "off" and w.light_text.text() == "Chưa chạy"
    w.on_event(Event(STATUS, "Running", 1, data={"state": RUNNING}))
    w.on_event(Event(HEALTH, "", 1, data={"level": RED, "reason": "Dừng lại một chút (câu 12,4 s)",
                                          "others": ["Đang xử lý, nói chậm lại"]}))
    assert w.light_level == RED and w.light.text() == "✋" and "#c62828" in w.light.styleSheet()
    assert w.light_text.text() == "ĐỎ: Dừng lại một chút (câu 12,4 s)"
    assert w.light_others.text() == "Cũng: Đang xử lý, nói chậm lại"
    assert "Đang xử lý" in w.light_text.toolTip()
    w.on_event(Event(HEALTH, "", 1, data={"level": YELLOW, "reason": "Câu dài, hãy ngắt nghỉ"}))
    assert (w.light.text(), w.light_others.text()) == ("⚠", "") and "#f9a825" in w.light.styleSheet()
    w.on_event(Event(HEALTH, "", 1, data={"level": GREEN, "reason": "Đang nghe, nói bình thường"}))
    assert w.light.text() == "✓" and w.light_text.text().startswith("XANH")
    w.on_event(Event(STATUS, "Stopped", 1, data={"state": STOPPED}))
    assert w.light_level == "off" and w.light.text() == "–"


def test_gui_ignores_the_light_of_a_stopped_session(qapp):
    from test_frontends import _window

    w, orch = _window(qapp)
    orch.current_session = None  # the session has stopped; a queued event arrives late
    w.on_event(Event(HEALTH, "", 1, data={"level": RED, "reason": "x"}))
    assert w.light_level == "off"
