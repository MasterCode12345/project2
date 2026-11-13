"""M4: faster-whisper ASR, Vietnamese, loaded once."""

from __future__ import annotations

import threading
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import numpy as np

# Models that are not on the Hugging Face hub in CTranslate2 format: converted
# once with scripts/convert_phowhisper.py into LOCAL_MODELS_DIR/<name>-ct2/.
# name -> (source repo, pinned revision). PhoWhisper (VinAI, BSD-3-Clause) is
# Whisper fine-tuned on 844 h of Vietnamese, *including the VIVOS train set*.
LOCAL_MODELS: dict[str, tuple[str, str]] = {
    "phowhisper-small": ("vinai/PhoWhisper-small", "a86b604c346caf7148c37512eafe783a16420adb"),
    "phowhisper-medium": ("vinai/PhoWhisper-medium", "55a7e3eb6c906de891f8f06a107754427dd3be79"),
}
LOCAL_MODELS_DIR = Path("~/.cache/audio-translator/models")


def local_model_dir(name: str, root: Path | None = None) -> Path:
    return (root or LOCAL_MODELS_DIR).expanduser() / f"{name}-ct2"


def convert_command(name: str) -> str:
    size = name.removeprefix("phowhisper-")
    return f"uv run --with transformers --with torch python scripts/convert_phowhisper.py {size}"


def resolve_model(name: str, root: Path | None = None) -> str:
    """faster-whisper model argument: a hub name, or the local CT2 dir of a converted model."""
    if name not in LOCAL_MODELS:
        return name
    path = local_model_dir(name, root)
    if not (path / "model.bin").is_file():
        raise RuntimeError(f"Model '{name}' is not converted yet (expected {path}/model.bin). "
                           f"Run: {convert_command(name)}")
    return str(path)


# ------------------------------------------------------------- hallucination filter
BLACKLIST_PATH = Path(__file__).with_name("hallucinations_vi.txt")
MIN_BLACKLIST_WORDS = 4  # shorter phrases would also hit real sentences


def normalize_text(text: str) -> str:
    """Lowercase, NFC, punctuation/symbols to spaces, collapsed (tone marks kept)."""
    text = unicodedata.normalize("NFC", text).lower()
    text = "".join(ch if unicodedata.category(ch)[0] not in "PS" else " " for ch in text)
    return " ".join(text.split())


def _min_repeats(n: int) -> int:
    """Consecutive copies of an n-word unit that count as a decoding loop."""
    return 4 if n == 1 else 3 if n <= 4 else 2


def collapse_repeats(text: str, max_unit: int = 8) -> str:
    """Keep one copy of a word sequence Whisper repeated in a loop.

    A unit of n words (compared normalised) repeated consecutively at least
    ``_min_repeats(n)`` times (4 for one word, 3 for 2-4 words, 2 for 5 or
    more) is written once: "cạo a a a a a" -> "cạo a". Shorter repetitions
    ("rất rất rất", "bác sĩ phẫu thuật bác sĩ phẫu thuật") are speech and are kept. The longest
    span covered wins; text without a loop is returned unchanged.
    """
    words = text.split()
    keys = [normalize_text(w) for w in words]
    out, i = [], 0
    while i < len(words):
        best_n, best_len = 0, 0
        for n in range(1, max_unit + 1):
            unit = keys[i:i + n]
            if len(unit) < n or not all(unit):
                break
            r = 1
            while keys[i + r * n:i + (r + 1) * n] == unit:
                r += 1
            if r >= _min_repeats(n) and r * n > best_len:
                best_n, best_len = n, r * n
        if best_n:
            out.extend(words[i + best_len - best_n:i + best_len])  # last copy: keeps its punctuation
            i += best_len
        else:
            out.append(words[i])
            i += 1
    return " ".join(out) if len(out) < len(words) else text


def load_blacklist(path: Path = BLACKLIST_PATH) -> tuple[str, ...]:
    """Phrases, one per line (``phrase<TAB># source``); ``#`` lines are comments."""
    phrases = []
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        phrase = normalize_text(line.split("\t", 1)[0].split("#", 1)[0])
        if not phrase:
            continue
        if len(phrase.split()) < MIN_BLACKLIST_WORDS:
            raise ValueError(f"{path}:{n}: blacklist phrase needs >= {MIN_BLACKLIST_WORDS} words: {phrase!r}")
        phrases.append(phrase)
    return tuple(phrases)


@dataclass
class SegmentFilter:
    """Drops a Whisper segment that is probably not speech, or a known hallucination.

    A segment is dropped when ``no_speech_prob > max_no_speech_prob``, OR
    ``avg_logprob < min_avg_logprob``, OR ``compression_ratio >
    max_compression_ratio``, OR its normalised text contains a blacklisted
    phrase as whole words. Other segments of the same utterance are kept.
    A kept segment's text goes through ``collapse_repeats`` (``collapse_loops``):
    PhoWhisper sometimes loops on noisy audio ("cạo a a a a …").
    """

    max_no_speech_prob: float = 0.6
    min_avg_logprob: float = -1.0
    max_compression_ratio: float = 2.4
    blacklist: tuple[str, ...] = field(default_factory=load_blacklist)
    collapse_loops: bool = True

    def reason(self, seg) -> str | None:
        """Why ``seg`` would be dropped, or None to keep it."""
        if seg.no_speech_prob > self.max_no_speech_prob:
            return "no_speech_prob"
        if seg.avg_logprob < self.min_avg_logprob:
            return "avg_logprob"
        if seg.compression_ratio > self.max_compression_ratio:
            return "compression_ratio"
        padded = f" {normalize_text(seg.text)} "
        if any(f" {phrase} " in padded for phrase in self.blacklist):
            return "blacklist"
        return None

    def keep(self, seg) -> bool:
        return self.reason(seg) is None

    def clean(self, text: str) -> str:
        """Text of a kept segment, with decoding loops written once."""
        return collapse_repeats(text) if self.collapse_loops else text


def filter_from_settings(settings) -> SegmentFilter | None:
    s = settings
    if not s.asr_filter:
        return None
    return SegmentFilter(s.asr_max_no_speech_prob, s.asr_min_avg_logprob, s.asr_max_compression_ratio)


class Recognizer(Protocol):
    def load(self) -> None: ...

    def transcribe(self, audio: np.ndarray, sample_rate: int) -> str: ...


class WhisperASR:
    def __init__(
        self,
        model_size: str = "medium",
        *,
        language: str = "vi",
        device: str = "cpu",
        compute_type: str = "int8",
        beam_size: int = 5,
        initial_prompt: str | None = None,
        segment_filter: SegmentFilter | None = None,
    ) -> None:
        self.model_size = model_size
        self.language = language
        self.device = device
        self.compute_type = compute_type
        self.beam_size = beam_size
        self.initial_prompt = initial_prompt or None
        self.segment_filter = segment_filter  # None: keep every segment (previous behaviour)
        self.dropped: list[tuple[str, str]] = []  # (reason, text) of the last call; "repetition": shortened, kept
        self.last_temperature = 0.0  # highest decoding temperature of the last call (> 0: fallback used)
        self._model = None
        self._lock = threading.Lock()

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def load(self) -> None:
        with self._lock:
            if self._model is None:
                from faster_whisper import WhisperModel

                self._model = WhisperModel(
                    resolve_model(self.model_size), device=self.device, compute_type=self.compute_type
                )

    def transcribe_segments(self, audio: np.ndarray, sample_rate: int) -> list:
        """Raw faster-whisper segments (text, no_speech_prob, avg_logprob, compression_ratio, …)."""
        if sample_rate != 16000:
            raise ValueError("WhisperASR expects 16 kHz audio")
        self.load()
        pcm = audio.astype(np.float32) / 32768.0 if audio.dtype == np.int16 else audio.astype(np.float32)
        segments, _info = self._model.transcribe(
            pcm,
            language=self.language,
            beam_size=self.beam_size,
            vad_filter=False,  # M3 already segmented
            condition_on_previous_text=False,
            initial_prompt=self.initial_prompt,
        )
        segments = list(segments)
        self.last_temperature = max((getattr(x, "temperature", 0.0) or 0.0 for x in segments), default=0.0)
        return segments

    def transcribe(self, audio: np.ndarray, sample_rate: int) -> str:
        kept, self.dropped = [], []
        for seg in self.transcribe_segments(audio, sample_rate):
            why = self.segment_filter.reason(seg) if self.segment_filter is not None else None
            text = seg.text.strip()
            if why is None:
                if self.segment_filter is not None:
                    clean = self.segment_filter.clean(text)
                    if clean != text:
                        self.dropped.append(("repetition", text))
                    text = clean
                kept.append(text)
            else:
                self.dropped.append((why, text))
        return " ".join(kept).strip()
