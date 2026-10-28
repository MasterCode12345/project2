"""eval/denoise_bench.py: offline smoke run, data checks, text normalisation, metrics."""

import csv
import importlib.util
import sys
from pathlib import Path

import pytest

_PATH = Path(__file__).resolve().parents[1] / "eval" / "denoise_bench.py"
_spec = importlib.util.spec_from_file_location("denoise_bench", _PATH)
bench = importlib.util.module_from_spec(_spec)
sys.modules["denoise_bench"] = bench  # dataclasses resolve their module by name
_spec.loader.exec_module(bench)


def test_smoke_writes_every_arm_x_snr_row(tmp_path):
    out = tmp_path / "ns.csv"
    assert bench.main(["--smoke", "--out", str(out)]) == 0
    rows = list(csv.DictReader(out.open(encoding="utf-8")))
    assert list(rows[0]) == bench.COLUMNS
    assert len(rows) == len(bench.ARMS) * len(bench.SNRS) == 25
    got = {(r["arm"], r["snr_db"]) for r in rows}
    want = {(a, "clean" if s is None else f"{s:g}") for a, _ in bench.ARMS for s in bench.SNRS}
    assert got == want
    for r in rows:
        assert int(r["n_utts"]) > 0
        assert 0.0 <= float(r["wer"]) and 0.0 <= float(r["cer"])
        assert int(r["vad_false_triggers"]) >= 0
        if r["arm"] == "none":
            assert float(r["added_latency_ms"]) == 0.0 and float(r["rtf"]) == 0.0
        else:
            assert float(r["added_latency_ms"]) == 15.0 and 0.0 < float(r["rtf"]) < 0.1
    clean = [r for r in rows if r["snr_db"] == "clean"]
    assert all(int(r["vad_false_triggers"]) == 0 for r in clean)  # digital silence never triggers


@pytest.mark.parametrize("argv", [
    [],
    ["--vivos-dir", "/nonexistent/vivos", "--noise-dir", "/nonexistent/musan"],
])
def test_missing_data_dirs_fail_with_clear_message(argv, tmp_path, capsys):
    rc = bench.main(argv + ["--out", str(tmp_path / "x.csv")])
    assert rc != 0
    err = capsys.readouterr().err
    assert "--vivos-dir" in err and "--noise-dir" in err
    if argv:
        assert "/nonexistent/vivos" in err and "/nonexistent/musan" in err
    assert not (tmp_path / "x.csv").exists()


def test_vivos_dir_without_prompts_is_reported(tmp_path, capsys):
    (tmp_path / "vivos").mkdir()
    (tmp_path / "noise").mkdir()
    rc = bench.main(["--vivos-dir", str(tmp_path / "vivos"), "--noise-dir", str(tmp_path / "noise"),
                     "--out", str(tmp_path / "x.csv")])
    assert rc != 0
    assert "prompts.txt" in capsys.readouterr().err


def test_normalize_vi_keeps_tone_marks_and_drops_punctuation():
    assert bench.normalize_vi("Xin CHÀO, bạn khỏe không?!") == "xin chào bạn khỏe không"
    assert bench.normalize_vi("  Một   hai… “ba” ") == "một hai ba"


def test_error_rate_fallback_matches_definition(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def no_jiwer(name, *a, **kw):
        if name == "jiwer":
            raise ImportError
        return real_import(name, *a, **kw)

    monkeypatch.setattr(builtins, "__import__", no_jiwer)
    wer, cer = bench.error_rates(["một hai ba", "bốn"], ["một ba", ""])
    assert wer == pytest.approx(2 / 4)
    assert cer == pytest.approx(7 / 13)


def test_rtf_denominator_counts_all_processed_audio():
    import numpy as np

    clip = np.zeros(1000, np.int16)  # not a frame multiple: pad + flush are added
    r = bench.run_pipeline(clip, bench.make_stage(0.4, None), bench.energy_vad, None)
    frames = -(-clip.size // bench.FRAME) + 600 // 30 + 2 + (240 // bench.FRAME + 1)
    assert r.processed_s == pytest.approx(frames * bench.FRAME / bench.RATE)
    assert r.processed_s > clip.size / bench.RATE
