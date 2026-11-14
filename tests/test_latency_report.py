"""eval/latency_report.py on a log written by a fake-stream session."""

import importlib.util
import sys
from pathlib import Path

from audio_translator.config import Settings
from audio_translator.models import STOPPED
from conftest import FakeASR, utterance

_ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("latency_report", _ROOT / "eval" / "latency_report.py")
lr = importlib.util.module_from_spec(_spec)
sys.modules["latency_report"] = lr
_spec.loader.exec_module(lr)


def test_report_on_a_real_fake_session(harness, tmp_path, capsys):
    from test_latency_log import wait_rows

    p = tmp_path / "lat.csv"
    h = harness(settings=Settings(latency_log=str(p)), asr=FakeASR(responses=["một hai", "", "ba"]))
    h.start()
    h.say(utterance(15), utterance(15), utterance(15))
    wait_rows(p, 3)
    h.orch.stop()
    h.log.wait_state(STOPPED)
    assert lr.main([str(p)]) == 0
    out = capsys.readouterr().out
    assert "3 segment(s) in 1 session(s): 2 played, 1 dropped" in out
    assert "e2e_ms" in out and "asr_ms" in out and "no_speech: 1" in out
    assert lr.main([str(p), "--markdown"]) == 0
    md = capsys.readouterr().out
    assert "| `e2e_ms` | 2 |" in md and "| `no_speech` | – | 1 |" in md
    stats = {s[0]: s for s in lr.stage_stats(lr.load([p]))}
    assert stats["e2e_ms"][1] == 2 and stats["e2e_ms"][4] <= stats["e2e_ms"][5]  # p95 <= max


def test_report_errors(tmp_path, capsys):
    assert lr.main([str(tmp_path / "missing.csv")]) == 2
    (tmp_path / "empty.csv").write_text("session_id,status\n")
    assert lr.main([str(tmp_path / "empty.csv")]) == 1
    assert lr.main([str(tmp_path / "empty.csv"), "--session", "9"]) == 1


def test_report_counts_the_highest_light_per_segment(tmp_path, capsys):
    from audio_translator.latency_log import LatencyLog

    p = tmp_path / "lat.csv"
    log = LatencyLog(p)
    for i, (status, lvl, why) in enumerate([("played", "green", ""), ("played", "red", "Dừng lại một chút"),
                                            ("dropped", "red", "Dừng lại một chút"),
                                            ("played", "yellow", "Câu dài, hãy ngắt nghỉ")], 1):
        log.write(session_id=1, segment_id=i, status=status, timestamps={"speech_end": 1.0}, session={},
                  reason="" if status == "played" else "no_speech", health_max=lvl, health_reason=why)
    log.close()
    assert lr.health_counts(lr.load([p])) == [("green", 1, 0, ""), ("yellow", 1, 0, "Câu dài, hãy ngắt nghỉ"),
                                              ("red", 1, 1, "Dừng lại một chút")]
    assert lr.main([str(p)]) == 0
    assert "speaker indicator" in capsys.readouterr().out
    old = tmp_path / "old.csv"  # a log written before the health columns existed
    old.write_text("session_id,segment_id,status,e2e_ms\n1,1,played,3000\n")
    assert lr.health_counts(lr.load([old])) == [] and lr.main([str(old)]) == 0
