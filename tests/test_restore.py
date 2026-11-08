"""restore.py: sentence case, WordPiece, GECToR edits, CapuRestorer with a fake encoder (offline)."""

import importlib.util
import types
from pathlib import Path

import numpy as np
import pytest

from audio_translator import restore as rs

LABELS = ["$KEEP", "$TRANSFORM_CASE_CAPITAL", "$APPEND_,", "$APPEND_.", "$TRANSFORM_VERB_VB_VBN",
          "$TRANSFORM_CASE_UPPER", "$APPEND_:", "$APPEND_?", "$TRANSFORM_VERB_VB_VBC", "$TRANSFORM_CASE_LOWER",
          "$TRANSFORM_CASE_CAPITAL_1", "$TRANSFORM_CASE_UPPER_-1", "$MERGE_SPACE", "@@UNKNOWN@@", "@@PADDING@@"]


def test_sentence_case():
    assert rs.sentence_case("  hôm nay   trời đẹp ") == "Hôm nay trời đẹp."
    assert rs.sentence_case("bạn đi đâu?") == "Bạn đi đâu?"
    assert rs.sentence_case("") == ""


def test_wordpiece_is_cased_greedy_and_splits_punctuation():
    wp = rs.WordPiece(["[UNK]", "Hà", "h", "##à", "nội", "5", "000", ".", "##i", "ph"])
    assert wp.word("hà") == ["h", "##à"]  # the vocabulary is cased: lowercase "hà" is two pieces
    assert wp.word("Hà") == ["Hà"]
    assert wp.word("5.000") == ["5", ".", "000"]
    assert wp.word("xyz") == ["[UNK]"]
    assert wp.word("$START") == ["$START"]  # added token, never split
    assert wp.word("phi") == ["ph", "##i"]


@pytest.mark.parametrize("words, labels, want", [
    (["hôm", "nay", "đẹp"], ["$KEEP", "$TRANSFORM_CASE_CAPITAL", "$KEEP", "$APPEND_."], ["Hôm", "nay", "đẹp", "."]),
    (["a", "b"], ["$TRANSFORM_CASE_CAPITAL", "$APPEND_,", "$APPEND_?"], ["a", ",", "b", "?"]),  # START transform: no-op
    (["a", ",", "b"], ["$KEEP", "$APPEND_,", "$KEEP", "$KEEP"], ["a", ",", "b"]),  # no second comma next to one
    (["usa", "vn"], ["$KEEP", "$TRANSFORM_CASE_UPPER", "$TRANSFORM_CASE_UPPER_-1"], ["USA", "Vn"]),
    (["iphone"], ["$KEEP", "$TRANSFORM_CASE_CAPITAL_1"], ["iPhone"]),
    (["đi", "học"], ["$KEEP", "$TRANSFORM_VERB_VB_VBN", "$KEEP"], ["đi", "học"]),  # English verb labels: kept
    (["face", "book", "hay"], ["$KEEP", "$MERGE_SPACE", "$KEEP", "$KEEP"], ["facebook", "hay"]),
])
def test_apply_labels(words, labels, want):
    assert rs.apply_labels(words, labels) == want


class FakeEncoder:
    """Hidden state = one-hot of a label chosen per token by ``rule``; the head is the identity."""

    def __init__(self, rule):
        self.rule, self.calls = rule, []

    def forward_batch(self, tokens):
        self.calls.append(tokens)
        n = max(len(t) for t in tokens)
        h = np.zeros((len(tokens), n, len(LABELS)), np.float32)
        for b, toks in enumerate(tokens):
            for i, t in enumerate(toks):
                h[b, i, LABELS.index(self.rule(t, i, toks))] = 1.0
        return types.SimpleNamespace(last_hidden_state=h)


def _model_dir(tmp_path, vocab):
    d = tmp_path / "vibert-capu-ct2"
    d.mkdir()
    (d / "model.bin").write_text("x")
    np.savez(d / rs.HEADS_FILE, classifier_weight=np.eye(len(LABELS), dtype=np.float32),
             classifier_bias=np.zeros(len(LABELS), np.float32))
    (d / rs.LABELS_FILE).write_text("\n".join(LABELS) + "\n", encoding="utf-8")
    (d / rs.VOCAB_FILE).write_text("\n".join(["[UNK]", *vocab]) + "\n", encoding="utf-8")
    return tmp_path


def _rule(tok, i, toks):
    """Capitalise the first word, comma after "nay", period at the end; once applied, keep."""
    if tok in (",", "."):
        return "$KEEP"
    if i == 1 and tok.islower():
        return "$TRANSFORM_CASE_CAPITAL"
    if tok == "nay" and (i + 1 >= len(toks) or toks[i + 1] != ","):
        return "$APPEND_,"
    if i == len(toks) - 1:
        return "$APPEND_."
    return "$KEEP"


def test_capu_restorer_runs_until_stable(tmp_path):
    root = _model_dir(tmp_path, ["hôm", "Hôm", "nay", "trời", "đẹp", ",", "."])
    enc = FakeEncoder(_rule)
    r = rs.CapuRestorer(root=root, encoder_factory=lambda d: enc)
    assert r.restore("HÔM NAY, trời đẹp") == "Hôm nay, trời đẹp."  # input normalised first
    assert enc.calls[0][0] == ["$START", "hôm", "nay", "trời", "đẹp"]
    assert 2 <= len(enc.calls) <= 3  # iterations stop when nothing changes
    assert r.supports("vi") and not r.supports("en")


def test_capu_restorer_short_input_empty_input_and_windows(tmp_path):
    root = _model_dir(tmp_path, ["a", "A", "b", "."])
    enc = FakeEncoder(lambda t, i, toks: "$KEEP")
    r = rs.CapuRestorer(root=root, encoder_factory=lambda d: enc, chunk_words=4)
    assert r.restore("a b") == "a b" and enc.calls == []  # fewer than min_len words: unchanged
    assert r.restore(" ... ") == ""
    r.restore(" ".join(["a"] * 10))
    assert sorted(len(c[0]) - 1 for c in enc.calls) == [3, 3, 4]  # 10 words in 3 near-equal windows


def test_unconverted_model_names_the_convert_command(tmp_path):
    r = rs.CapuRestorer(root=tmp_path)
    with pytest.raises(RuntimeError, match=r"convert_capu_model.py vibert-capu"):
        r.restore("tôi đi học hôm nay")
    with pytest.raises(ValueError):
        rs.CapuRestorer("nope")


def test_make_restorer():
    assert rs.make_restorer("off") is None
    assert rs.make_restorer("sent")("xin chào", "vi") == "Xin chào."
    assert callable(rs.make_restorer("vibert-capu"))
    with pytest.raises(ValueError):
        rs.make_restorer("nope")
    assert rs.RESTORE_MODELS["vibert-capu"][1] and len(rs.RESTORE_MODELS["vibert-capu"][1]) == 40


def test_convert_script_keeps_existing_output_and_explains_missing_torch(tmp_path, capsys, monkeypatch):
    spec = importlib.util.spec_from_file_location(
        "convert_capu_model", Path(__file__).resolve().parents[1] / "scripts" / "convert_capu_model.py")
    cv = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cv)
    out = tmp_path / "vibert-capu-ct2"
    out.mkdir()
    (out / "model.bin").write_text("x")
    msgs = []
    assert cv.convert("vibert-capu", tmp_path, log=msgs.append) == out and "already exists" in msgs[0]
    monkeypatch.setattr(cv, "MT_MODELS_DIR", tmp_path / "empty")
    try:
        import torch  # noqa: F401
    except ImportError:
        assert cv.main(["vibert-capu"]) == 2
        assert 'uv run --with "transformers<5" --with torch' in capsys.readouterr().err
