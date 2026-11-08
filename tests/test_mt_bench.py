"""eval/mt_bench.py: FLEURS pairing, custom set, fixed subsets, ASR cache, failures, smoke run."""

import csv
import importlib.util
import sys
from pathlib import Path

import pytest

pytest.importorskip("sacrebleu")  # dependency group `eval`

_ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("mt_bench", _ROOT / "eval" / "mt_bench.py")
mt = importlib.util.module_from_spec(_spec)
sys.modules["mt_bench"] = mt
_spec.loader.exec_module(mt)


def _tsv(path, rows):
    """FLEURS layout: id, file, raw, normalised, chars, n_samples, gender."""
    path.write_text("".join(f"{i}\t{f}\t{raw}\t{raw.lower()}\tx\t1\tMALE\n" for i, f, raw in rows),
                    encoding="utf-8")
    return path


def test_fleurs_pairs_join_on_sentence_id(tmp_path, capsys):
    src = tmp_path / "vi"
    src.mkdir()
    _tsv(src / "test.tsv", [("20", "b.wav", "Câu hai."), ("3", "z.wav", "Câu ba."), ("3", "a.wav", "Câu ba."),
                            ("7", "c.wav", "Chỉ có tiếng Việt.")])
    tgt = _tsv(tmp_path / "en.tsv", [("3", "e1.wav", "Sentence three."), ("20", "e2.wav", "Sentence two."),
                                     ("20", "e3.wav", "Sentence two."), ("99", "e4.wav", "English only.")])
    pairs = mt.load_fleurs_pairs(src, tgt, log=print)
    assert [(p.id, p.src, p.ref) for p in pairs] == [("3", "Câu ba.", "Sentence three."),
                                                     ("20", "Câu hai.", "Sentence two.")]  # numeric id order
    assert pairs[0].audio == src / "test" / "a.wav"  # smallest file name of the id's recordings
    assert "2 pairs (1 source-only and 1 target-only dropped)" in capsys.readouterr().out


def test_fleurs_id_with_two_texts_is_rejected(tmp_path):
    src = tmp_path / "vi"
    src.mkdir()
    _tsv(src / "test.tsv", [("1", "a.wav", "Một."), ("1", "b.wav", "Khác.")])
    with pytest.raises(ValueError, match="sentence id 1 has two different texts"):
        mt.load_fleurs_pairs(src, _tsv(tmp_path / "en.tsv", [("1", "e.wav", "One.")]), log=lambda s: None)
    with pytest.raises(ValueError, match="no sentence id in common"):
        _tsv(src / "test.tsv", [("1", "a.wav", "Một.")])
        mt.load_fleurs_pairs(src, _tsv(tmp_path / "en.tsv", [("2", "e.wav", "Two.")]), log=lambda s: None)


def test_custom_set_is_unreviewed_until_every_row_says_yes(tmp_path):
    p = tmp_path / "c.tsv"
    head = "# comment\nid\tcategory\treviewed\tsrc\tref\n"
    p.write_text(head + "c1\tname\tyes\tBách Khoa\tHUST\nc2\ttime\tno\t3 giờ chiều\t3 p.m.\n", encoding="utf-8")
    pairs, reviewed = mt.load_custom(p)
    assert [x.id for x in pairs] == ["c1", "c2"] and pairs[1].category == "time" and not reviewed
    p.write_text(head + "c1\tname\tYes\ta\tb\n", encoding="utf-8")
    assert mt.load_custom(p)[1]
    p.write_text(head + "c1\tname\tyes\ta\tb\nc1\tname\tyes\ta\tb\n", encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        mt.load_custom(p)


def test_shipped_custom_set_loads_and_is_marked_for_review():
    pairs, reviewed = mt.load_custom(_ROOT / "eval" / "data" / "mt_custom_vi_en.tsv")
    assert 20 <= len(pairs) <= 30 and not reviewed
    assert {"name", "time", "number", "unit", "loanword"} <= {p.category for p in pairs}


def test_subset_is_fixed_by_seed_and_keeps_id_order():
    pairs = [mt.Pair(str(i), f"s{i}", f"r{i}") for i in range(100)]
    a, b = mt.subset(pairs, 10, 0), mt.subset(pairs, 10, 0)
    assert a == b and len(a) == 10
    assert [int(p.id) for p in a] == sorted(int(p.id) for p in a)
    assert mt.subset(pairs, 10, 1) != a
    assert mt.subset(pairs, None, 0) == pairs and mt.subset(pairs, 500, 0) == pairs


def test_asr_cache_reuses_transcripts_across_runs(tmp_path):
    pairs = [mt.Pair("1", "a", "b", Path("x/1.wav")), mt.Pair("2", "c", "d", Path("x/2.wav"))]
    calls = []

    def make(model):
        calls.append(model)
        return lambda p: f" chữ\tASR  {p.id} "

    path = tmp_path / "cache.tsv"
    got = mt.transcribe_pairs(pairs, "m", make, mt.AsrCache(path), "vi", 5, log=lambda s: None)
    assert got == {"1": "chữ ASR 1", "2": "chữ ASR 2"} and calls == ["m"]
    again = mt.transcribe_pairs(pairs, "m", make, mt.AsrCache(path), "vi", 5, log=lambda s: None)
    assert again == got and calls == ["m"]  # nothing recognised, no model built
    mt.transcribe_pairs(pairs, "m", make, mt.AsrCache(path), "vi", 1, log=lambda s: None)
    assert calls == ["m", "m"]  # another beam is another key


def test_failed_calls_are_counted_and_left_out():
    tr = mt.FlakyTranslator()
    ticks = iter(range(100))
    hyps, secs, failed = mt.translate_all(tr, ["tôi đi", "", "tôi", "đi", "học"], clock=lambda: next(ticks))
    assert hyps[1] == "" and failed == 1 and hyps.count(None) == 1
    assert len(secs) == 3 and all(s == 1 for s in secs)  # empty input is not a call
    scorer = mt.Scorer("en")
    assert scorer.score([None, None], ["a", "b"]) is None
    got = scorer.score(["I go to school every day", None], ["I go to school every day", "x"])
    assert got["n"] == 1 and got["bleu"] == pytest.approx(100.0) and got["chrf"] == pytest.approx(100.0)
    cased = scorer.score(["i went to hanoi last year"], ["I went to Hanoi last year"])
    assert cased["bleu_lc"] == pytest.approx(100.0) and cased["chrf_lc"] == pytest.approx(100.0)
    assert cased["bleu"] < 100.0 and cased["chrf"] < 100.0  # the cased scores still see the case


def test_segmented_input_is_one_call_per_segment_joined_per_sentence():
    tr = mt.FlakyTranslator()  # third call fails
    ticks = iter(range(100))
    segs = [["tôi đi", "học"], ["trời mưa", "to"], [], ["tôi"]]
    hyps, secs, failed = mt.translate_segmented(tr, segs, clock=lambda: next(ticks))
    assert hyps[0] == "I go to school" and hyps[1] is None and hyps[2] == "" and hyps[3] == "I"
    assert failed == 1 and len(secs) == 4  # per-call latency, the unit the app translates
    assert mt.split_segments(" a  |||  b c ||| ") == ["a", "b c"] and mt.split_segments("") == []
    assert mt.seg_model_key("phowhisper-small", 2, 600) == "phowhisper-small+seg(vad2,600ms)"


def test_pivot_route_is_named_in_the_row_label():
    legs = {("vi", "en"), ("en", "fr")}
    b = mt.Backend("toy", lambda a, c: (a, c) in legs, lambda a, c: mt.ToyTranslator())
    label, tr = mt.build_translator(b, "vi", "fr")
    assert label == "toy+pivot:en" and tr.pivot == "en"
    assert mt.build_translator(b, "vi", "en")[0] == "toy"
    assert mt.build_translator(b, "fr", "vi") is None


def test_smoke_writes_every_backend_condition_row_and_samples(tmp_path):
    out = tmp_path / "mt.csv"
    assert mt.main(["--smoke", "--out", str(out)]) == 0
    rows = list(csv.DictReader(out.open(encoding="utf-8")))
    assert list(rows[0]) == mt.COLUMNS
    keys = {(r["corpus"], r["backend"], r["condition"], r["asr_model"]) for r in rows}
    for backend in ("toy", "flaky", "down"):
        assert {("smoke", backend, "text", "-"), ("smoke", backend, "text_norm", "-"),
                ("smoke", backend, "asr", "toy-asr-good"), ("smoke", backend, "asr", "toy-asr-bad"),
                ("smoke", backend, "asr_seg", "toy-asr-good"), ("custom_unreviewed", backend, "text", "-")} <= keys
    assert not any(r["corpus"] == "custom_unreviewed" and r["condition"].startswith("asr") for r in rows)
    by = {(r["corpus"], r["backend"], r["condition"], r["asr_model"]): r for r in rows}
    good, bad = by["smoke", "toy", "asr", "toy-asr-good"], by["smoke", "toy", "asr", "toy-asr-bad"]
    assert float(good["src_wer"]) == 0 and float(bad["src_wer"]) > 0 and float(bad["bleu"]) < float(good["bleu"])
    seg = by["smoke", "toy", "asr_seg", "toy-asr-good"]
    assert int(seg["n_inputs"]) > int(good["n_inputs"]) == 6 and float(seg["mean_words"]) <= 2
    assert seg["bleu"] == good["bleu"]  # word-by-word toy: cutting into segments changes nothing
    down = by["smoke", "down", "text", "-"]
    assert down["n_failed"] == "6" and down["n_sents"] == "0" and down["bleu"] == ""
    assert int(by["smoke", "flaky", "text", "-"]["n_failed"]) > 0
    samples = list(csv.DictReader((tmp_path / "mt_samples.tsv").open(encoding="utf-8"), delimiter="\t"))
    assert list(samples[0]) == mt.SAMPLE_COLUMNS
    assert any(s["hypothesis"] == mt.FAILED for s in samples)
    assert {s["condition"] for s in samples} == set(mt.CONDITIONS)


def test_runs_are_reproducible_except_latency(tmp_path):
    for name in ("a", "b"):
        assert mt.main(["--smoke", "--out", str(tmp_path / f"{name}.csv")]) == 0
    strip = lambda p: [{k: v for k, v in r.items() if "latency" not in k}  # noqa: E731
                       for r in csv.DictReader(p.open(encoding="utf-8"))]
    assert strip(tmp_path / "a.csv") == strip(tmp_path / "b.csv")


@pytest.mark.parametrize("argv, msg", [
    ([], "--fleurs-src-dir (not given)"),
    (["--fleurs-src-dir", "/nonexistent/vi", "--fleurs-tgt-tsv", "/nonexistent/en.tsv"], "/nonexistent/vi"),
    (["--smoke", "--src", "xx"], "unknown language 'xx'"),
    (["--smoke", "--tgt", "vi"], "same"),
    (["--smoke", "--conditions", "text,audio"], "--conditions"),
])
def test_bad_arguments_fail_with_a_clear_message(argv, msg, tmp_path, capsys):
    assert mt.main(argv + ["--out", str(tmp_path / "x.csv")]) == 2
    assert msg in capsys.readouterr().err
    assert not (tmp_path / "x.csv").exists()


def test_missing_wav_or_unconverted_model_stops_before_running(tmp_path, capsys, monkeypatch):
    src = tmp_path / "vi"
    (src / "test").mkdir(parents=True)
    _tsv(src / "test.tsv", [("1", "a.wav", "Một.")])
    tgt = _tsv(tmp_path / "en.tsv", [("1", "e.wav", "One.")])
    argv = ["--fleurs-src-dir", str(src), "--fleurs-tgt-tsv", str(tgt), "--out", str(tmp_path / "x.csv")]
    assert mt.main(argv) == 2
    assert "wav(s) missing" in capsys.readouterr().err
    (src / "test" / "a.wav").write_bytes(b"")
    import audio_translator.asr as asr

    monkeypatch.setattr(asr, "LOCAL_MODELS_DIR", tmp_path / "models")
    assert mt.main(argv + ["--asr-models", "phowhisper-small"]) == 2
    assert "convert_phowhisper.py small" in capsys.readouterr().err


def test_restore_rows_cover_asr_like_inputs_only(tmp_path, capsys):
    out = tmp_path / "mt.csv"
    assert mt.main(["--smoke", "--restore", "off,sent,vibert-capu", "--out", str(out)]) == 0
    assert "restore vibert-capu: skipped in --smoke" in capsys.readouterr().out
    rows = list(csv.DictReader(out.open(encoding="utf-8")))
    sent = {(r["condition"], r["asr_model"]) for r in rows if r["backend"] == "toy" and r["restore"] == "sent"}
    assert sent == {("text_norm", "-"), ("asr", "toy-asr-good"), ("asr", "toy-asr-bad"),
                    ("asr_seg", "toy-asr-good"), ("asr_seg", "toy-asr-bad")}
    assert all(r["restore_mean_ms"] != "" for r in rows if r["restore"] == "sent" and r["n_inputs"] != "0")
    assert all(r["restore_mean_ms"] == "" for r in rows if r["restore"] == "off")
    assert not any(r["restore"] == "vibert-capu" for r in rows)


def test_restore_inputs_times_each_non_empty_segment():
    ticks = iter(range(100))
    out, secs = mt.restore_inputs(lambda t, lang: t.upper(), [["a", ""], ["b c"]], "vi", clock=lambda: next(ticks))
    assert out == [["A", ""], ["B C"]] and secs == [1, 1]


def test_unknown_restorer_is_rejected(tmp_path, capsys):
    assert mt.main(["--smoke", "--restore", "magic", "--out", str(tmp_path / "x.csv")]) == 2
    assert "unknown restorer" in capsys.readouterr().err
