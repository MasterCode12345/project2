"""eval/preroll_bench.py: smoke run, stream layout and SNR, segment positions, ghosts, metrics."""

import csv
import importlib.util
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("preroll_bench", _ROOT / "eval" / "preroll_bench.py")
pb = importlib.util.module_from_spec(_spec)
sys.modules["preroll_bench"] = pb
_spec.loader.exec_module(pb)
db = pb.db
RATE = pb.RATE


def test_smoke_rows_and_a_longer_pre_roll_recovers_the_soft_first_word(tmp_path):
    out, samples = tmp_path / "p.csv", tmp_path / "s.tsv"
    assert pb.main(["--smoke", "--snrs", "clean,10", "--quiet-every", "3", "--out", str(out),
                    "--samples", str(samples)]) == 0
    rows = list(csv.DictReader(out.open(encoding="utf-8")))
    assert list(rows[0]) == pb.COLUMNS
    assert [(r["snr_db"], int(r["pre_roll_ms"])) for r in rows] == [
        (s, p) for s in ("clean", "10") for p in pb.PRE_ROLLS]
    for snr in ("clean", "10"):
        cell = {int(r["pre_roll_ms"]): r for r in rows if r["snr_db"] == snr}
        lost = [int(cell[p]["first_lost"]) for p in pb.PRE_ROLLS]
        assert lost[0] == 6 and lost[-1] == 0 and lost == sorted(lost, reverse=True)
        assert all(int(r["best_first_lost"]) == 0 and r["best_wer"] == "0.0000" for r in cell.values())
        assert all(int(r["segs"]) == 6 and int(r["ghosts"]) == 0 for r in cell.values())
        assert float(cell[900]["seg_audio_s"]) > float(cell[300]["seg_audio_s"])
    srows = list(csv.DictReader(samples.open(encoding="utf-8"), delimiter="\t"))
    assert len(srows) == 12 and all(r["best"] == r["ref"] == r["pre_roll_900"] for r in srows)
    assert all(r["pre_roll_300"] != r["ref"] for r in srows)


def test_first_word_lost():
    assert pb.first_word_lost("Âm thầm chắt chiu", "m thầm chắt chiu")
    assert pb.first_word_lost("màu son trông", "son trông")
    assert not pb.first_word_lost("màu son", "Màu son.")
    assert not pb.first_word_lost("đồ án", "ờ đồ án")  # within the first 2 words
    assert not pb.first_word_lost("", "anything")


def test_stream_layout_level_and_snr():
    rng = np.random.default_rng(0)
    utts = [(np.full(RATE, 1000.0) * np.sin(np.arange(RATE) / 5), "a") for _ in range(5)]
    noise = [rng.normal(0, 1, RATE * 30)]
    clean = pb.build_stream(utts, noise, None, np.random.default_rng(1), gap_s=1.0, quiet_every=2, quiet_s=3.0)
    assert len(clean.spans) == 5 and len(clean.quiet) == 2
    assert clean.audio.size % pb.FRAME == 0
    starts = [a for a, _ in clean.spans]
    assert starts[0] == int((pb.WARMUP_S + 1.0) * RATE)
    assert starts[1] - clean.spans[0][1] == RATE  # one gap
    assert starts[2] - clean.spans[1][1] == 4 * RATE  # quiet stretch + gap
    mask = np.zeros(clean.audio.size, bool)
    for a, b in clean.spans:
        mask[a:b] = True
        assert abs(db.rms(clean.audio[a:b]) - 3000) < 5  # every utterance at the same level
    assert not clean.audio[~mask].any()  # clean: digital silence elsewhere
    noisy = pb.build_stream(utts, noise, 5.0, np.random.default_rng(1), gap_s=1.0, quiet_every=2, quiet_s=3.0)
    n = noisy.audio.astype(np.float64) - clean.audio
    snr = 20 * np.log10(db.rms(clean.audio[mask]) / db.rms(n[mask]))
    assert abs(snr - 5.0) < 0.2
    assert db.rms(n[~mask]) > 0  # constant noise over the gaps and quiet stretches too
    a, b = clean.quiet[0]
    assert abs(db.rms(n[a:b]) / db.rms(n[mask]) - 1) < 0.2


def test_segments_are_positioned_on_the_stream_and_assigned_by_overlap():
    utts, _ = pb.smoke_data(0)
    s = pb.build_stream(utts, [], None, np.random.default_rng(0), level=None)
    segs = pb.segment_stream(s.audio, pb.SmokeLagVad(), 900)
    assert len(segs) == len(utts)
    for a, b, audio in segs:
        assert np.array_equal(audio, s.audio[a:b])
    assert pb.assign(segs, s.spans) == list(range(len(utts)))
    assert pb.assign([(0, 10, None)], s.spans) == [None]
    assert pb.assign([(0, 100, None)], [(50, 60), (60, 95)]) == [1]


def test_a_segment_outside_every_utterance_with_text_is_a_ghost():
    class Always:
        def transcribe(self, audio, rate):
            return "ma"

    utts, _ = pb.smoke_data(1)
    s = pb.build_stream(utts[:2], [], None, np.random.default_rng(0), quiet_every=1, quiet_s=4.0, level=None)
    a, _b = s.quiet[0]
    s.audio[a + RATE:a + 2 * RATE] = (8000 * np.sin(np.arange(RATE) / 3)).astype(np.int16)  # a bang, no utterance
    rows = []
    pb.bench_cell(utts[:2], s, Always(), lambda: pb.SmokeLagVad(lag=0), [300], corpus="t", noise="-",
                  snr_db=None, seed=0, emit=rows.append)
    assert rows[0]["segs_outside"] == 1 and rows[0]["ghosts"] == 1 and rows[0]["segs"] == 3
