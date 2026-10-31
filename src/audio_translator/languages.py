"""Central language table: one row per app language code.

Every per-language detail lives here: the code each MT backend expects, the
Whisper language, the preferred ASR model, the default Edge TTS voice, the
FLEURS config, the text normaliser and the sacrebleu BLEU tokenizer. Adding a
language means adding one ``Language`` row (and, for a script without spaces
between words, one normaliser in ``NORMALIZERS`` and a matching
``bleu_tokenize`` such as ``zh`` or ``ja-mecab``).

Nothing in the app reads this table yet (the session still uses
``Settings.source_lang``/``target_lang`` directly); the MT bench does.
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass, field
from typing import Callable, Mapping


@dataclass(frozen=True)
class Language:
    code: str  # the app's code (ISO 639-1)
    name: str
    whisper: str  # faster-whisper ``language=``
    asr_model: str  # preferred catalog model for this language
    edge_voice: str  # default Edge TTS voice when this is the target
    fleurs: str  # google/fleurs config
    normalizer: str  # key into NORMALIZERS
    bleu_tokenize: str  # sacrebleu ``tokenize=`` when this is the target
    mt: Mapping[str, str] = field(default_factory=dict)  # MT backend -> that backend's code


def _strip_punct(text: str, keep: str = "") -> str:
    """NFC, lowercase, every punctuation/symbol char not in ``keep`` -> space, collapse spaces."""
    text = unicodedata.normalize("NFC", text).lower()
    text = "".join(" " if unicodedata.category(c)[0] in "PS" and c not in keep else c for c in text)
    return " ".join(text.split())


def normalize_vi(text: str) -> str:
    """Lowercase, drop punctuation/symbols, keep tone marks (same as eval/denoise_bench.normalize_vi)."""
    return _strip_punct(text)


def normalize_en(text: str) -> str:
    """Lowercase, drop punctuation/symbols; apostrophes inside a word stay (``don't``, ``Addenbrooke's``)."""
    text = unicodedata.normalize("NFC", text).replace("’", "'")
    words = _strip_punct(text, keep="'").split()
    return " ".join(w for w in (w.strip("'") for w in words) if w)


NORMALIZERS: dict[str, Callable[[str], str]] = {"vi": normalize_vi, "en": normalize_en}

LANGUAGES: dict[str, Language] = {
    lang.code: lang
    for lang in (
        Language("vi", "Vietnamese", whisper="vi", asr_model="phowhisper-small",
                 edge_voice="vi-VN-HoaiMyNeural", fleurs="vi_vn", normalizer="vi", bleu_tokenize="13a",
                 mt={"argos": "vi", "google": "vi", "nllb": "vie_Latn", "m2m100": "vi", "madlad": "<2vi>"}),
        Language("en", "English", whisper="en", asr_model="small",
                 edge_voice="en-US-AriaNeural", fleurs="en_us", normalizer="en", bleu_tokenize="13a",
                 mt={"argos": "en", "google": "en", "nllb": "eng_Latn", "m2m100": "en", "madlad": "<2en>"}),
    )
}


def get_language(code: str) -> Language:
    try:
        return LANGUAGES[code]
    except KeyError:
        raise ValueError(f"Unknown language '{code}' (known: {', '.join(sorted(LANGUAGES))})") from None


def mt_code(code: str, backend: str) -> str | None:
    """The code ``backend`` uses for app language ``code``; None if the backend does not list it."""
    return get_language(code).mt.get(backend)


def normalize(text: str, code: str) -> str:
    return NORMALIZERS[get_language(code).normalizer](text)
