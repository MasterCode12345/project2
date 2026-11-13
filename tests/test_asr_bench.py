"""eval/asr_bench.py: smoke run, FLEURS loader, model skipping, latency stats, data checks."""

import csv
import importlib.util
import sys
import time
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
    ab.run(utts[:1], noises, ["m"], lambda n: Rec(), lambda v: db.energy_vad, corpus="c", noise_name="n",
           prompts=[None, "Xin chào."], snrs=[None], seed=0, emit=lambda r: None)
    assert seen and set(seen) == {None, "Xin chào."} and seen[0] is None and seen[-1] == "Xin chào."


def test_unloadable_model_is_skipped_with_a_message():
    logs, rows = [], []

    def make(name):
        if name == "phowhisper-medium":
            raise RuntimeError("Model 'phowhisper-medium' is not converted yet")
        return ab._PromptToy()

    utts, noises = db.smoke_data(0)
    done = ab.run(utts[:1], noises, ["phowhisper-medium", "toy"], make, lambda v: db.energy_vad,
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
    ab.run([(utts[0][0], "Xin chào")], noises, ["m"], lambda n: Rec(), lambda v: None, corpus="c",
           noise_name="n", prompts=[None], snrs=[None], seed=0, emit=rows.append)
    (r,) = rows
    assert r["wer"] == "0.0000" and r["vad_extra_segs"] == 1
    assert r["asr_rtf"] == "1.000" and r["asr_ms_mean"] == "1000" and r["asr_ms_p95"] == "1000"


def test_smoke_vad_x_filter_matrix_has_hallucination_columns(tmp_path):
    out = tmp_path / "g.csv"
    assert ab.main(["--smoke", "--vads", "webrtc,silero", "--filters", "off,on", "--prompt-modes", "off",
                    "--out", str(out)]) == 0
    rows = list(csv.DictReader(out.open(encoding="utf-8")))
    assert {"vad_false_triggers", "vad_extra_segs", "ghost_texts"} <= set(rows[0])
    got = {(r["model"], r["vad"], r["filter"], r["snr_db"]) for r in rows}
    assert len(rows) == len(got) == 2 * 2 * 2 * len(db.SNRS)
    for r in rows:
        assert 0 <= int(r["ghost_texts"]) <= int(r["vad_false_triggers"])
        if r["snr_db"] == "clean":
            assert r["vad_false_triggers"] == "0"
    for bad in (["--vads", "energy"], ["--filters", "maybe"]):
        assert ab.main(["--smoke", *bad, "--out", str(out)]) == 2


def test_ghost_texts_count_noise_segments_that_still_produce_text():
    """Noise-only clip: every VAD segment in the scored window is a false trigger; ghosts = those with text."""
    rows = []

    class Rec(ab._PromptToy):
        def transcribe(self, audio, sr):
            ghost = "Hãy subscribe cho kênh Ghiền Mì Gõ"
            if self.segment_filter is not None:
                return ""  # the filter drops it
            return ghost

    rng = np.random.default_rng(1)
    speech = np.sin(np.arange(16000) * 0.2) * 8000.0
    bursts = np.zeros(16000 * 20)
    for k in range(0, bursts.size, 16000):
        bursts[k:k + 4000] = rng.normal(0, 4000, 4000)  # loud noise bursts: the energy VAD fires on them
    from audio_translator.asr import SegmentFilter
    ab.run([(speech, "x")], [bursts], ["m"], lambda n: Rec(), lambda v: db.energy_vad, corpus="c",
           noise_name="n", prompts=[None], snrs=[0.0], seed=0, emit=rows.append,
           filters=(None, SegmentFilter(blacklist=())))
    off, on = rows
    assert (off["filter"], on["filter"]) == ("off", "on")
    assert int(off["vad_false_triggers"]) > 0
    assert off["ghost_texts"] == off["vad_false_triggers"]  # every false trigger speaks without the filter
    assert on["vad_false_triggers"] == off["vad_false_triggers"] and on["ghost_texts"] == 0


def test_filter_modes_share_one_whisper_call_per_segment_and_keep_its_time():
    """filter off/on rows come from the same recognition: half the calls, same latency, the filter alone differs."""
    import types

    from audio_translator.asr import SegmentFilter, WhisperASR

    calls = []

    class FakeModel:
        def transcribe(self, pcm, **kw):
            calls.append(kw["initial_prompt"])
            time.sleep(0.002)
            return [types.SimpleNamespace(text="Hãy subscribe cho kênh", no_speech_prob=0.9, avg_logprob=-0.2,
                                          compression_ratio=1.0)], None

    def make(name):
        asr = WhisperASR("small")
        asr._model = FakeModel()
        return asr

    rng = np.random.default_rng(1)
    bursts = np.zeros(16000 * 20)
    for k in range(0, bursts.size, 16000):
        bursts[k:k + 4000] = rng.normal(0, 4000, 4000)
    speech = np.sin(np.arange(16000) * 0.2) * 8000.0
    rows = []
    ab.run([(speech, "x")], [bursts], ["m"], make, lambda v: db.energy_vad, corpus="c", noise_name="n",
           prompts=[None, "câu mồi"], snrs=[0.0], seed=0, emit=rows.append,
           filters=(None, SegmentFilter(blacklist=())))
    assert [(r["prompt"], r["filter"]) for r in rows] == [("off", "off"), ("off", "on"), ("on", "off"), ("on", "on")]
    per_prompt = len(calls) // 2
    assert calls == [None] * per_prompt + ["câu mồi"] * per_prompt  # one recognition per segment and prompt
    for off, on in (rows[:2], rows[2:]):
        assert off["asr_ms_mean"] == on["asr_ms_mean"] and float(off["asr_ms_mean"]) >= 2
        assert int(off["ghost_texts"]) > 0 and on["ghost_texts"] == 0  # no_speech_prob 0.9 > 0.6


def test_decode_label():
    assert ab.decode_label(None, 16, None) == "base"
    assert ab.decode_label(12, 16, None) == "cap12+16"
    assert ab.decode_label(None, 16, (0.0, 0.2, 0.4)) == "t0/0.2/0.4"
    assert ab.decode_label(12.5, 8, (0.0, 0.4)) == "cap12.5+8,t0/0.4"


def test_decode_probe_records_each_attempt_and_its_limit():
    import types

    class CT2:
        def __init__(self):
            self.n = iter([224, 30])

        def generate(self, features, prompts, **kw):
            return [types.SimpleNamespace(sequences_ids=[list(range(next(self.n)))])]

        other = "kept"

    class Rec:
        def __init__(self):
            self._model = types.SimpleNamespace(model=CT2())

        def transcribe_segments(self, audio, sr):
            self._model.model.generate(None, [[1, 2, 3]], max_length=448)
            self._model.model.generate(None, [[1, 2, 3]], max_length=55, sampling_temperature=0.2)
            return ["seg"]

    r = Rec()
    probe = ab.install_decode_probe(r)
    assert r._model.model.other == "kept"  # everything else passes through
    assert r.transcribe_segments(None, 16000) == ["seg"]
    assert [(t, n, lim) for _s, t, n, lim in r.last_decode] == [(0.0, 224, 224), (0.2, 30, 52)]
    assert probe.calls == r.last_decode
    assert ab.install_decode_probe(object()) is None


def test_decode_stats_columns():
    speech = [(1.0, 2.0, [(1.0, 0.0, 20, 224)]), (3.0, 3.0, [(1.0, 0.0, 52, 52), (2.0, 0.2, 30, 52)])]
    noise = [(0.5, 1.0, [(0.5, 0.0, 5, 224)]), (6.0, 3.0, [(2, 0, 224, 224), (2, .2, 224, 224), (2, .4, 9, 224)])]
    d = ab.decode_stats(speech, noise)
    assert d["asr_ms_max"] == "3000" and d["noise_segs"] == 2 and d["noise_asr_ms_max"] == "6000"
    assert (d["fallback_segs"], d["noise_fallback_segs"], d["noise_redecodes"], d["cap_hits"]) == (1, 1, 2, 1)
    assert d["max_tok_per_s"] == "10.00"  # last attempt: 20 tok / 2 s, 30 tok / 3 s
    assert ab.decode_stats([(1.0, 1.0)], [])["fallback_segs"] == 0  # toy recognizer: no attempts recorded


def test_segments_out_writes_one_row_per_asr_segment(tmp_path):
    out, segs = tmp_path / "s.csv", tmp_path / "segs.tsv"
    assert ab.main(["--smoke", "--snrs", "clean,5", "--prompt-modes", "off", "--out", str(out),
                    "--segments-out", str(segs)]) == 0
    rows = list(csv.DictReader(open(segs, encoding="utf-8"), delimiter="\t"))
    assert list(rows[0]) == ab.SEG_COLUMNS and {r["kind"] for r in rows} >= {"speech"}
    assert {r["decode"] for r in rows} == {"base"} and {r["snr_db"] for r in rows} == {"clean", "5"}
