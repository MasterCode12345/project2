"""Offline (Argos via CTranslate2) translator, package install and fallback. No network."""

import io
import json
import zipfile
from types import SimpleNamespace

import pytest

from audio_translator.translate import (
    FallbackTranslator,
    OfflineTranslator,
    PackageError,
    ensure_argos_package,
    split_sentences,
)

URL = "https://example.invalid/v1/translate-vi_en-1_9.argosmodel"
PKG = "translate-vi_en-1_9"


def make_zip(meta=None, files=None, top=PKG) -> bytes:
    meta = {"from_code": "vi", "to_code": "en"} if meta is None else meta
    files = {
        "metadata.json": json.dumps(meta),
        "sentencepiece.model": "spm",
        "model/model.bin": "bin",
        "model/config.json": "{}",
        "model/shared_vocabulary.json": "[]",
    } if files is None else files
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, data in files.items():
            zf.writestr(f"{top}/{name}" if top else name, data)
    return buf.getvalue()


class FakeOpener:
    def __init__(self, payload: bytes | Exception):
        self.payload = payload
        self.calls: list[str] = []

    def __call__(self, url):
        self.calls.append(url)
        if isinstance(self.payload, Exception):
            raise self.payload
        return io.BytesIO(self.payload)


def leftovers(cache):
    return sorted(p.name for p in cache.iterdir()) if cache.exists() else []


# ------------------------------------------------------------ package install
def test_first_run_downloads_installs_and_reports(tmp_path):
    opener, notified = FakeOpener(make_zip()), []
    pkg = ensure_argos_package(tmp_path, URL, opener, on_download=lambda: notified.append(1))
    assert pkg == tmp_path / PKG
    assert (pkg / "model" / "model.bin").is_file() and (pkg / "sentencepiece.model").is_file()
    assert opener.calls == [URL] and notified == [1]
    assert leftovers(tmp_path) == [PKG]  # temp dir and archive cleaned up


def test_cached_package_is_reused_without_network(tmp_path):
    ensure_argos_package(tmp_path, URL, FakeOpener(make_zip()))
    opener, notified = FakeOpener(OSError("offline")), []
    pkg = ensure_argos_package(tmp_path, URL, opener, on_download=lambda: notified.append(1))
    assert pkg == tmp_path / PKG and opener.calls == [] and notified == []


@pytest.mark.parametrize("payload, cause", [
    (OSError("Network is unreachable"), "Network is unreachable"),
    (b"this is not a zip", "corrupt archive"),
    (make_zip(meta={"from_code": "en", "to_code": "vi"}), "expected vi→en"),
    (make_zip(files={"metadata.json": json.dumps({"from_code": "vi", "to_code": "en"})}),
     "missing sentencepiece.model"),
])
def test_failed_install_names_cause_and_leaves_nothing(tmp_path, payload, cause):
    with pytest.raises(PackageError, match=cause):
        ensure_argos_package(tmp_path, URL, FakeOpener(payload))
    assert leftovers(tmp_path) == []


def test_broken_cached_package_is_replaced(tmp_path):
    (tmp_path / PKG).mkdir()
    (tmp_path / PKG / "metadata.json").write_text("{}")
    pkg = ensure_argos_package(tmp_path, URL, FakeOpener(make_zip()))
    assert (pkg / "model" / "model.bin").is_file()


@pytest.mark.parametrize("missing", ["model/config.json", "model/shared_vocabulary.json"])
def test_cached_package_missing_model_file_is_refetched(tmp_path, missing):
    pkg = ensure_argos_package(tmp_path, URL, FakeOpener(make_zip()))
    (pkg / missing).unlink()
    opener = FakeOpener(make_zip())
    assert ensure_argos_package(tmp_path, URL, opener) == pkg
    assert opener.calls == [URL] and (pkg / missing).is_file()


def test_stale_download_temp_dirs_are_removed(tmp_path):
    ensure_argos_package(tmp_path, URL, FakeOpener(make_zip()))
    stale = tmp_path / ".download-killed"
    (stale / "extract").mkdir(parents=True)
    (stale / "package.zip").write_bytes(b"partial")
    opener = FakeOpener(OSError("offline"))
    ensure_argos_package(tmp_path, URL, opener)
    assert leftovers(tmp_path) == [PKG] and opener.calls == []


# ------------------------------------------------------------ offline translator
class FakeSPM:
    def encode(self, text, out_type=str):
        return ["▁" + w for w in text.split()]


class FakeCT2:
    """'Translates' by upper-casing SentencePiece pieces; records batches."""

    def __init__(self):
        self.batches = []

    def translate_batch(self, batch, beam_size=4):
        self.batches.append(batch)
        return [SimpleNamespace(hypotheses=[[t.upper() for t in toks]]) for toks in batch]


def offline(tmp_path):
    ct2 = FakeCT2()
    loads = []
    t = OfflineTranslator(tmp_path, translator_factory=lambda d: loads.append(d) or ct2,
                          tokenizer_factory=lambda f: FakeSPM())
    return t, ct2, loads


def test_offline_translates_and_strips_markers(tmp_path):
    t, ct2, loads = offline(tmp_path)
    assert t.translate("Xin chào, bạn khỏe không?") == "XIN CHÀO, BẠN KHỎE KHÔNG?"
    assert loads == [str(tmp_path / "model")]


def test_offline_multi_sentence_translated_separately_and_joined(tmp_path):
    t, ct2, _ = offline(tmp_path)
    out = t.translate("Câu một. Câu hai?")
    assert out == "CÂU MỘT. CÂU HAI?" and "▁" not in out
    assert len(ct2.batches[-1]) == 2


def test_offline_loads_lazily_once(tmp_path):
    t, _, loads = offline(tmp_path)
    assert loads == []
    t.translate("Một.")
    t.translate("Hai.")
    assert len(loads) == 1


def test_offline_empty_input_returns_empty(tmp_path):
    t, ct2, loads = offline(tmp_path)
    assert t.translate("   ") == "" and ct2.batches == [] and loads == []


def test_split_sentences():
    assert split_sentences("Câu một. Câu hai?  Ba!") == ["Câu một.", "Câu hai?", "Ba!"]
    assert split_sentences("không dấu chấm") == ["không dấu chấm"]


# ------------------------------------------------------------ fallback
class Fixed:
    def __init__(self, result="", fail=False):
        self.result, self.fail, self.calls = result, fail, []

    def translate(self, text):
        self.calls.append(text)
        if self.fail:
            raise RuntimeError("boom")
        return self.result


def test_fallback_used_only_for_failing_call():
    primary, fb, seen = Fixed("offline"), Fixed("google"), []
    t = FallbackTranslator(primary, fb, seen.append)
    assert t.translate("a") == "offline" and fb.calls == []
    primary.fail = True
    assert t.translate("b") == "google" and len(seen) == 1
    primary.fail = False
    assert t.translate("c") == "offline" and fb.calls == ["b"]


def test_fallback_factory_is_lazy_and_errors_propagate():
    made = []
    t = FallbackTranslator(Fixed(fail=True), lambda: made.append(1) or Fixed(fail=True))
    with pytest.raises(RuntimeError):
        t.translate("x")
    assert made == [1]
