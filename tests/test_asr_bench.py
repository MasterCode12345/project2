"""eval/asr_bench.py: smoke run, FLEURS loader, model skipping, latency stats, data checks."""

import csv
import importlib.util
import sys
import wave
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("asr_bench", _ROOT / "eval" / "asr_bench.py")
ab = importlib.util.module_from_spec(_spec)
sys.modules["asr_bench"] = ab
_spec.loader.exec_module(ab)
db = ab.db


def test_smoke_writes_every_model_x_prompt_x_snr_row(tmp_path):
    out = tmp_path / "asr.csv"
    assert ab.main(["--smoke", "--out", str(out)]) == 0
    rows = list(csv.DictReader(out.open(encoding="utf-8")))
    assert list(rows[0]) == ab.COLUMNS
    got = {(r["model"], r["prompt"], r["snr_db"]) for r in rows}
    want = {(m, p, "clean" if s is None else f"{s:g}")
            for m in ("toy-a", "toy-b") for p in ("off", "on") for s in db.SNRS}
    assert got == want and len(rows) == len(want) == 20
    for r in rows:
        assert int(r["n_utts"]) == 3 and float(r["wer"]) >= 0 and float(r["asr_rtf"]) >= 0
        assert r["noise"] == ("-" if r["snr_db"] == "clean" else "smoke")


def test_snr_and_prompt_mode_options(tmp_path):
    assert ab.build_parser().parse_args(["--snrs", "20,0"]).snrs == [20.0, 0.0]
    assert ab.build_parser().parse_args(["--snrs", "clean"]).snrs == [None]
    out = tmp_path / "a.csv"
    assert ab.main(["--smoke", "--snrs", "clean", "--prompt-modes", "on", "--out", str(out)]) == 0
    assert {r["prompt"] for r in csv.DictReader(out.open(encoding="utf-8"))} == {"on"}
    assert ab.main(["--smoke", "--prompt-modes", "maybe", "--out", str(out)]) == 2
    assert ab.main(["--smoke", "--prompt", " ", "--out", str(out)]) == 2  # "on" without a prompt
    assert ab.main(["--smoke", "--prompt", "", "--prompt-modes", "off", "--snrs", "clean", "--out", str(out)]) == 0


def test_prompt_is_set_on_the_recognizer_per_mode():
    seen = []

    class Rec(ab._PromptToy):
        def transcribe(self, audio, sr):
            seen.append(self.initial_prompt)
            return ""

    utts, noises = db.smoke_data(0)
    ab.run(utts[:1], noises, ["m"], lambda n: Rec(), lambda: db.energy_vad, corpus="c", noise_name="n",
           prompts=[None, "Xin chào."], snrs=[None], seed=0, emit=lambda r: None)
    assert seen and set(seen) == {None, "Xin chào."} and seen[0] is None and seen[-1] == "Xin chào."


def test_unloadable_model_is_skipped_with_a_message():
    logs, rows = [], []

    def make(name):
        if name == "phowhisper-medium":
            raise RuntimeError("Model 'phowhisper-medium' is not converted yet")
        return ab._PromptToy()

    utts, noises = db.smoke_data(0)
    done = ab.run(utts[:1], noises, ["phowhisper-medium", "toy"], make, lambda: db.energy_vad,
                  corpus="c", noise_name="n", prompts=[None], snrs=[None], seed=0, emit=rows.append,
                  log=logs.append)
    assert done == 1 and [r["model"] for r in rows] == ["toy"]
    assert any("skipping model 'phowhisper-medium'" in m and "not converted" in m for m in logs)


def test_no_loadable_model_fails(tmp_path, monkeypatch, capsys):
    vivos, noise = tmp_path / "vivos", tmp_path / "noise"
    _write_vivos(vivos)
    noise.mkdir()
    _write_wav(noise / "n.wav", np.random.default_rng(0).normal(0, 500, 16000))
    import audio_translator.asr as asr_mod
    monkeypatch.setattr(asr_mod, "LOCAL_MODELS_DIR", tmp_path / "models")  # nothing converted
    rc = ab.main(["--vivos-dir", str(vivos), "--noise-dir", str(noise), "--models", "phowhisper-small",
                  "--snrs", "clean", "--out", str(tmp_path / "o.csv")])
    out = capsys.readouterr()
    assert rc == 1 and "skipping model 'phowhisper-small'" in out.out and "no model could be loaded" in out.err


@pytest.mark.parametrize("argv", [
    ["--corpus", "fleurs", "--noise-dir", "/nonexistent/noise"],
    ["--corpus", "vivos", "--vivos-dir", "/nonexistent/vivos"],
])
def test_missing_data_dirs_fail_with_clear_message(argv, tmp_path, capsys):
    assert ab.main(argv + ["--out", str(tmp_path / "o.csv")]) == 2
    err = capsys.readouterr().err
    assert "missing data directory" in err
    assert ("--fleurs-dir (not given)" in err) if "fleurs" in argv else ("--vivos-dir /nonexistent/vivos" in err)


def _write_wav(path, x, rate=16000):
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(np.clip(x, -32768, 32767).astype("<i2").tobytes())


def _write_vivos(root):
    waves = root / "test" / "waves" / "SPK01"
    waves.mkdir(parents=True)
    (root / "test" / "prompts.txt").write_text("SPK01_001 XIN CHÀO\n", encoding="utf-8")
    _write_wav(waves / "SPK01_001.wav", np.zeros(8000))


def test_fleurs_one_recording_per_sentence_rotating_speakers(tmp_path):
    (tmp_path / "test").mkdir()
    rows = []
    for sid, n_rec in (("10", 3), ("11", 2), ("12", 1)):
        for k in range(n_rec):
            f = f"{sid}_{k}.wav"
            _write_wav(tmp_path / "test" / f, np.full(1600, int(sid) * 10 + k))
            rows.append(f"{sid}\t{f}\tCâu {sid}, có dấu.\tcâu {sid} có dấu\tc â u\t1600\tMALE")
    (tmp_path / "test.tsv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    utts = ab.load_fleurs(tmp_path, None)
    assert [t for _, t in utts] == ["câu 10 có dấu", "câu 11 có dấu", "câu 12 có dấu"]  # normalised column
    # sentence j uses recording j % n: 10_0, 11_1, 12_0
    assert [int(a[0]) for a, _ in utts] == [100, 111, 120]
    assert len(ab.load_fleurs(tmp_path, 2)) == 2


def test_fleurs_missing_files_are_reported(tmp_path):
    with pytest.raises(FileNotFoundError, match="test.tsv"):
        ab.load_fleurs(tmp_path, None)
    (tmp_path / "test.tsv").write_text("1\tgone.wav\tX.\tx\n", encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="gone.wav"):
        ab.load_fleurs(tmp_path, None)


def test_latency_stats():
    assert ab.latency_stats([]) == (0.0, 0.0, 0.0)
    mean, p95, rtf = ab.latency_stats([(0.5, 2.0), (1.0, 2.0), (1.5, 4.0)])
    assert mean == pytest.approx(1000.0) and p95 == pytest.approx(1450.0) and rtf == pytest.approx(3.0 / 8.0)


def test_only_scored_segments_count_for_latency(monkeypatch):
    """Segments outside the utterance: not scored, not timed, counted in extra_segs."""
    clock = {"t": 0.0}
    monkeypatch.setattr(ab, "time", type("T", (), {"perf_counter": staticmethod(lambda: clock["t"])}))

    class Rec(ab._PromptToy):
        def transcribe(self, audio, sr):
            noise = audio.size > sr * 5
            clock["t"] += audio.size / sr * (10.0 if noise else 1.0)  # noise segment: RTF 10, speech: 1
            return "subscribe" if noise else "xin chào"

    def fake_pipeline(clip, stage, vad, asr):
        a, b = clip_spans[-1]
        asr.transcribe(np.zeros(16000 * 6, np.int16), 16000)  # inside the warm-up: outside the utterance
        asr.transcribe(np.zeros(16000, np.int16), 16000)
        return db.ArmRun([(0.0, 1.5, "subscribe"), (a, b, "xin chào")], 0.0, 1.0)

    clip_spans = []
    real_build = db.build_clip

    def build(*args):
        c = real_build(*args)
        clip_spans.append(c.speech_s)
        return c

    monkeypatch.setattr(db, "run_pipeline", fake_pipeline)
    monkeypatch.setattr(db, "build_clip", build)
    rows = []
    utts, noises = db.smoke_data(0)
    ab.run([(utts[0][0], "Xin chào")], noises, ["m"], lambda n: Rec(), lambda: None, corpus="c",
           noise_name="n", prompts=[None], snrs=[None], seed=0, emit=rows.append)
    (r,) = rows
    assert r["wer"] == "0.0000" and r["extra_segs"] == 1
    assert r["asr_rtf"] == "1.000" and r["asr_ms_mean"] == "1000" and r["asr_ms_p95"] == "1000"
