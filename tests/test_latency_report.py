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
