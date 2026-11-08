"""scripts/collect_hallucinations.py: noise disjoint from the benches, aggregation (fake ASR)."""

import importlib.util
import sys
import types
import wave
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("collect_hallucinations", _ROOT / "scripts" / "collect_hallucinations.py")
ch = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ch)


def _wav(path, x):
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(np.clip(x, -32768, 32767).astype("<i2").tobytes())


def test_sources_exclude_exactly_the_bench_noise_sample(tmp_path):
    rng = np.random.default_rng(0)
    for i in range(12):
        _wav(tmp_path / "noise" / f"n{i:02d}.wav", rng.normal(0, 100, 1600))
    for i in range(3):
        _wav(tmp_path / "music" / f"m{i}.wav", rng.normal(0, 100, 1600))
    bench = set(ch.db.load_noises.__globals__["np"].random.default_rng(0).choice(12, 5, replace=False))
    excluded = {tmp_path / "noise" / f"n{i:02d}.wav" for i in bench}
    assert ch.bench_sample(tmp_path / "noise", 5, 0) == excluded
    files = ch.disjoint_sources(tmp_path, 5, 0, 0)
    assert not set(files) & excluded and len(files) == 12 - 5 + 3
    assert files[1].parent.name == "music"  # interleaved noise/music
    assert len(ch.disjoint_sources(tmp_path, 5, 0, 4)) == 4


def test_collect_counts_outputs_per_model(tmp_path):
    rng = np.random.default_rng(0)
    x = np.zeros(16000 * 4)
    x[16000:32000] = rng.normal(0, 3000, 16000)
    _wav(tmp_path / "a.wav", x)

    class FakeASR:
        def transcribe_segments(self, audio, sr):
            return [types.SimpleNamespace(text=" Hãy Subscribe cho kênh! ", no_speech_prob=0.3, avg_logprob=-0.9,
                                          compression_ratio=1.2)]

    def vad(frame, rate):
        a = np.frombuffer(frame, np.int16).astype(float)
        return float(np.sqrt(np.mean(a * a))) > 50

    rows = ch.collect([tmp_path / "a.wav"], {"m1": FakeASR()}, lambda: vad, log=lambda *_: None)
    assert len(rows) == 1
    r = rows[0]
    assert r["model"] == "m1" and r["normalised"] == "hãy subscribe cho kênh" and r["words"] == 4
    assert r["count"] >= len(ch.LEVELS_DBFS) and r["n_files"] == 1 and r["avg_logprob"] == "-0.90"


def test_missing_musan_dir(tmp_path, capsys):
    assert ch.main(["--musan-dir", str(tmp_path)]) == 2
    assert "MUSAN noise dir not found" in capsys.readouterr().err
