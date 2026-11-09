"""languages.py: the central language table and per-language normalisers."""

import importlib.util
import sys
from pathlib import Path

import pytest

from audio_translator.config import Settings
from audio_translator.languages import LANGUAGES, NORMALIZERS, get_language, mt_code, normalize


def test_every_row_is_complete_and_consistent():
    for code, lang in LANGUAGES.items():
        assert lang.code == code
        assert lang.normalizer in NORMALIZERS
        assert lang.bleu_tokenize and lang.fleurs and lang.whisper and lang.asr_model
        assert lang.edge_voice.lower().startswith(code + "-")
        assert {"argos", "google"} <= set(lang.mt)


def test_vi_en_rows_match_the_current_app_settings():
    s = Settings()
    assert get_language(s.source_lang).mt["argos"] == "vi" and get_language(s.target_lang).mt["argos"] == "en"
    assert get_language(s.target_lang).edge_voice == s.tts_voice  # the table does not change the default voice
    assert get_language("vi").asr_model == "phowhisper-small" and get_language("en").asr_model == "small"
    assert (get_language("vi").fleurs, get_language("en").fleurs) == ("vi_vn", "en_us")
    assert mt_code("vi", "nllb") == "vie_Latn" and mt_code("vi", "no-such-backend") is None


def test_unknown_language_is_a_clear_error():
    with pytest.raises(ValueError, match="Unknown language 'xx'"):
        get_language("xx")


def test_vi_normaliser_is_the_bench_one():
    spec = importlib.util.spec_from_file_location(
        "denoise_bench", Path(__file__).resolve().parents[1] / "eval" / "denoise_bench.py")
    db = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("denoise_bench", db)
    spec.loader.exec_module(db)
    for text in ("Xin CHÀO, bạn khỏe không?!", "  Một   hai… “ba” ", "Đại học Bách Khoa Hà Nội."):
        assert normalize(text, "vi") == db.normalize_vi(text)


def test_en_normaliser_keeps_word_internal_apostrophes():
    assert normalize("He was moved to Addenbrooke's Hospital, in Cambridge.", "en") == \
        "he was moved to addenbrooke's hospital in cambridge"
    assert normalize("I DON’T know -- 'really'?", "en") == "i don't know really"
    assert normalize("one-hundred percent", "en") == "one hundred percent"
