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


def test_utterance_ending_at_clip_end_survives_vad_hangover():
    """Noise up to the clip end + a VAD that keeps voicing into the flush must not drop the utterance."""
    import numpy as np

    hangover = {"left": 0}

    def sticky_vad(frame: bytes, rate: int) -> bool:  # like webrtcvad after loud noise
        if bench.energy_vad(frame, rate):
            hangover["left"] = 5
            return True
        hangover["left"] -= 1
        return hangover["left"] >= 0

    rng = np.random.default_rng(0)
    clip = (rng.standard_normal(bench.FRAME * 60) * 3000).astype(np.int16)  # voiced to the last sample
    # raw VAD input: log-MMSE would remove this stationary "speech" before the VAD
    for stage in (None, bench.NoiseSuppressor(bench.LogMmseEnhancer(), mix=0.4, vad_input="raw")):
        hangover["left"] = 0
        r = bench.run_pipeline(clip, stage, sticky_vad, None)
        assert len(r.texts) == 1


def test_round_robin_covers_every_speaker_before_repeating():
    entries = [(f"SPK{s}_{i:03d}", f"t{s}{i}") for s in (1, 2, 3) for i in range(4 if s != 3 else 1)]
    got = [uid for uid, _ in bench.pick_round_robin(entries, 5)]
    assert got == ["SPK1_000", "SPK2_000", "SPK3_000", "SPK1_001", "SPK2_001"]
    assert len(bench.pick_round_robin(entries, None)) == len(entries)  # no limit: everything, once


def test_clip_layout_snr_and_fades():
    import numpy as np

    rng = np.random.default_rng(0)
    speech = np.sin(np.arange(16000) * 0.05) * 5000.0
    noise = rng.standard_normal(16000 * 10) * 1000.0
    clip = bench.build_clip(speech, noise, 10.0, rng)
    a, b = clip.speech_s
    assert a == pytest.approx(bench.WARMUP_S + bench.LEAD_S) and b - a == pytest.approx(1.0)
    seg = clip.noise_only[int(a * bench.RATE):int(b * bench.RATE)].astype(float)
    assert 20 * np.log10(bench.rms(speech) / bench.rms(seg)) == pytest.approx(10.0, abs=0.05)
    assert clip.noise_only[0] == 0 and abs(int(clip.noise_only[-1])) <= 1  # faded: no step at the edges
    assert clip.noise_window_s == (bench.WARMUP_S, pytest.approx(clip.noisy.size / bench.RATE - bench.FADE_S))
    clean = bench.build_clip(speech, noise, None, rng)
    assert not clean.noise_only.any()


def test_only_segments_overlapping_the_utterance_are_scored():
    run = bench.ArmRun([(0.0, 1.0, "warmup"), (2.4, 3.6, "câu nói"), (4.0, 4.5, "subscribe")], 0.0, 5.0)
    assert run.text_in((2.5, 3.5)) == "câu nói"
    assert run.count_in((2.5, 3.5)) == 1
    assert run.texts == ["warmup", "câu nói", "subscribe"]


def test_segment_times_are_compensated_for_stage_latency():
    import numpy as np

    clip = np.zeros(bench.FRAME * 100, np.int16)
    clip[bench.FRAME * 40:bench.FRAME * 60] = 5000  # "speech" at 1.2 s .. 1.8 s
    plain = bench.run_pipeline(clip, None, bench.energy_vad, None).segments
    # mix=1 + raw VAD input: the stage only delays the signal by its latency (15 ms)
    stage = bench.NoiseSuppressor(bench.LogMmseEnhancer(), mix=1.0, vad_input="raw")
    delayed = bench.run_pipeline(clip, stage, bench.energy_vad, None).segments
    assert len(plain) == len(delayed) == 1
    a, b, _ = plain[0]
    assert a <= 1.2 < b and b == pytest.approx(1.8)
    # The delayed output lands one frame later (+30 ms); removing the 15 ms latency keeps
    # both edges within half a frame of the undelayed run (uncompensated: 30 ms off).
    assert abs(delayed[0][0] - a) <= 0.015 + 1e-9 and abs(delayed[0][1] - b) <= 0.015 + 1e-9


def _float_wav(path, x, rate=16000, channels=1):
    import struct

    import numpy as np

    data = np.asarray(x, "<f4").tobytes()
    fmt = struct.pack("<HHIIHH", 3, channels, rate, rate * 4 * channels, 4 * channels, 32)
    body = b"WAVE" + b"fmt " + struct.pack("<I", len(fmt)) + fmt + b"data" + struct.pack("<I", len(data)) + data
    path.write_bytes(b"RIFF" + struct.pack("<I", len(body)) + body)


def test_read_wav_accepts_ieee_float_files(tmp_path):
    """FLEURS ships 32-bit float WAVs, which the stdlib wave module cannot open."""
    import numpy as np

    x = np.sin(np.arange(1600) * 0.1) * 0.5
    _float_wav(tmp_path / "f.wav", x)
    got = bench.read_wav(tmp_path / "f.wav")
    assert got.shape == (1600,) and np.allclose(got, x * 32768.0, atol=0.01)
    _float_wav(tmp_path / "st.wav", np.repeat(x, 2), channels=2)  # interleaved stereo -> mono mean
    assert np.allclose(bench.read_wav(tmp_path / "st.wav"), x * 32768.0, atol=0.01)
    (tmp_path / "bad.wav").write_bytes(b"RIFF\x04\x00\x00\x00WAVE")
    with pytest.raises(Exception):
        bench.read_wav(tmp_path / "bad.wav")
