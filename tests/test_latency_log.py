"""--latency-log: CSV rows for played/dropped segments, default model + fallback, flags."""

import csv
import time

import pytest

from audio_translator import cli
from audio_translator.config import Settings
from audio_translator.latency_log import COLUMNS, LatencyLog, durations
from audio_translator.model_manager import ModelManager
from audio_translator.models import RESULT, STOPPED, WARNING
from conftest import FakeASR, FakeModelFactory, FakeTranslator, utterance


def rows(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def wait_rows(path, n, timeout=5.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if path.exists() and len(rows(path)) >= n:
            return rows(path)
        time.sleep(0.02)
    raise AssertionError(f"{path}: fewer than {n} rows: {rows(path) if path.exists() else 'missing'}")


def test_durations_use_the_restored_stamp_when_present():
    ts = {"speech_end": 1.0, "segmented": 1.1, "asr_start": 1.2, "asr_end": 1.5, "mt_start": 1.5,
          "restored": 1.52, "mt_end": 1.6, "tts_start": 1.6, "tts_end": 2.0, "first_audio_out": 2.1}
    d = durations(ts)
    assert d["restore_ms"] == pytest.approx(20) and d["mt_ms"] == pytest.approx(80)
    assert d["e2e_ms"] == pytest.approx(1100) and d["playout_wait_ms"] == pytest.approx(100)
    del ts["restored"]
    assert "restore_ms" not in durations(ts) and durations(ts)["mt_ms"] == pytest.approx(100)


def test_log_writes_the_header_once_and_appends(tmp_path):
    p = tmp_path / "sub" / "lat.csv"
    for sid in (1, 2):
        log = LatencyLog(p)
        log.write(session_id=sid, segment_id=1, status="played", timestamps={"speech_end": 1.0},
                  session={"asr_model": "m"}, text_vi="xin chào", text_en="hello")
        log.close()
    log.write(session_id=3, segment_id=1, status="played", timestamps={}, session={})  # closed: ignored
    got = rows(p)
    assert list(got[0]) == COLUMNS and [r["session_id"] for r in got] == ["1", "2"]
    assert got[0]["n_words_vi"] == "2" and got[0]["asr_model"] == "m" and got[0]["e2e_ms"] == ""


def test_played_segment_row_matches_its_timestamps(harness, tmp_path):
    p = tmp_path / "lat.csv"
    h = harness(settings=Settings(latency_log=str(p)))
    h.start()
    h.say(utterance(30))
    ev = h.log.wait_for(lambda e: e.kind == RESULT)
    (r,) = wait_rows(p, 1)
    ts = ev.data["timestamps"]
    assert (r["status"], r["session_id"], r["segment_id"]) == ("played", str(ev.session_id), "1")
    assert float(r["e2e_ms"]) == pytest.approx((ts["first_audio_out"] - ts["speech_end"]) * 1000, abs=0.1)
    assert float(r["asr_ms"]) == pytest.approx((ts["asr_end"] - ts["asr_start"]) * 1000, abs=0.1)
    assert r["text_vi"] == "Xin chào, bạn khỏe không?" and r["text_en"] == "EN<Xin chào, bạn khỏe không?>"
    assert r["n_words_vi"] == "5" and float(r["audio_s"]) > 0 and r["forced_cut"] == "0"
    assert r["asr_model"] == "FakeASR" and r["vad"] == "injected" and r["restore"] == "off"
    h.orch.stop()
    h.log.wait_state(STOPPED)


def test_dropped_segments_are_logged_with_their_reason(harness, tmp_path):
    p = tmp_path / "lat.csv"
    asr = FakeASR(responses=["", "một", "hai"])
    h = harness(settings=Settings(latency_log=str(p)), asr=asr, translator=FakeTranslator(fail_on={"một"}))
    h.start()
    h.say(utterance(15), utterance(15), utterance(15))
    got = {r["segment_id"]: r for r in wait_rows(p, 3)}
    assert (got["1"]["status"], got["1"]["reason"]) == ("dropped", "no_speech")
    assert got["1"]["asr_ms"] != "" and got["1"]["mt_start"] == ""
    assert (got["2"]["status"], got["2"]["reason"], got["2"]["text_vi"]) == ("dropped", "mt_error", "một")
    assert got["2"]["detail"] != ""  # the translator's error message
    assert got["3"]["status"] == "played" and got["3"]["text_vi"] == "hai"


def test_ghost_sentence_reasons_from_the_asr_filter_reach_the_log(harness, tmp_path):
    p = tmp_path / "lat.csv"

    class FilteredASR(FakeASR):
        def transcribe(self, audio, sample_rate):
            self.dropped = [("blacklist", "Hãy subscribe cho kênh"), ("repetition", "x x x x")]
            return ""

    h = harness(settings=Settings(latency_log=str(p)), asr=FilteredASR())
    h.start()
    h.say(utterance(15))
    (r,) = wait_rows(p, 1)
    assert r["reason"] == "no_speech" and r["detail"] == "blacklist: Hãy subscribe cho kênh"


def test_log_off_writes_nothing(harness, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    h = harness()
    h.start()
    h.say(utterance(15))
    h.log.wait_for(lambda e: e.kind == RESULT)
    h.orch.stop()
    h.log.wait_state(STOPPED)
    assert list(tmp_path.iterdir()) == [] and h.orch._latlog is None


def test_unwritable_log_only_warns(harness, tmp_path):
    bad = tmp_path / "is-a-dir"
    bad.mkdir()
    h = harness(settings=Settings(latency_log=str(bad)))
    h.start()
    h.log.wait_for(lambda e: e.kind == WARNING and "Latency log disabled" in e.message)
    h.say(utterance(15))
    h.log.wait_for(lambda e: e.kind == RESULT)  # the session goes on


def test_default_model_is_phowhisper_small():
    assert Settings().model_size == "phowhisper-small" and Settings().asr_fallback_model == "small"
    assert cli.build_parser().parse_args([]).model == "phowhisper-small"


def test_unconverted_phowhisper_falls_back_to_small_and_the_log_names_small(harness, tmp_path):
    p = tmp_path / "lat.csv"
    f = FakeModelFactory()
    mgr = ModelManager(f, is_downloaded=lambda n: n == "small")
    h = harness(settings=Settings(latency_log=str(p)), model_manager=mgr)
    h.start()
    warn = h.log.wait_for(lambda e: e.kind == WARNING and e.data.get("model_fallback"))
    assert warn.data["model"] == "phowhisper-small" and warn.data["fallback"] == "small"
    assert "scripts/convert_phowhisper.py small" in warn.message and warn.data["command"] in warn.message
    h.say(utterance(15))
    res = h.log.wait_for(lambda e: e.kind == RESULT)
    assert res.data["source_text"] == "text from small" and f.load_calls == ["small"]
    (r,) = wait_rows(p, 1)
    assert r["asr_model"] == "small"  # what really ran, not the selection
    assert h.orch.selected_model == "phowhisper-small"  # converting later picks it up again


def test_converted_phowhisper_is_used_without_warning(harness):
    f = FakeModelFactory()
    mgr = ModelManager(f, is_downloaded=lambda n: n in ("small", "phowhisper-small"))
    h = harness(settings=Settings(), model_manager=mgr)
    h.start()
    h.say(utterance(15))
    h.log.wait_for(lambda e: e.kind == RESULT)
    assert f.load_calls == ["phowhisper-small"]
    assert not [e for e in h.log.of(WARNING) if e.data.get("model_fallback")]


def test_no_fallback_configured_fails_the_start_cleanly(harness):
    f = FakeModelFactory(fail=("phowhisper-small",))
    mgr = ModelManager(f, is_downloaded=lambda n: n == "small")
    h = harness(settings=Settings(asr_fallback_model=None), model_manager=mgr)
    assert h.orch.start(None, "BlackHole")
    h.log.wait_state(STOPPED)
    assert h.orch.state == STOPPED


def test_cli_and_gui_accept_latency_log():
    assert cli.build_parser().parse_args(["--latency-log", "x.csv"]).latency_log == "x.csv"
    assert cli.build_parser().parse_args([]).latency_log is None
    pytest.importorskip("PySide6")
    from audio_translator.gui import parse_args

    args, rest = parse_args(["--latency-log", "results/x.csv", "-style", "fusion"])
    assert args.latency_log == "results/x.csv" and rest == ["-style", "fusion"]
