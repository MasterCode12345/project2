"""M4: faster-whisper ASR, Vietnamese, loaded once."""

from __future__ import annotations

import math
import re
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


JUNK_TOKENS_PATH = Path(__file__).with_name("junk_tokens_vi.txt")


def load_junk_tokens(path: Path = JUNK_TOKENS_PATH) -> tuple[str, ...]:
    """Tokens, one per line (``token<TAB># evidence``); ``#`` lines are comments."""
    tokens = []
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        tok = line.split("\t", 1)[0].split("#", 1)[0].strip().lower()
        if not tok:
            continue
        if not re.fullmatch(r"\w+", tok):
            raise ValueError(f"{path}:{n}: a junk token is one word: {tok!r}")
        tokens.append(tok)
    return tuple(tokens)


JUNK_TOKENS = load_junk_tokens()


def strip_junk(text: str, tokens: tuple[str, ...] = JUNK_TOKENS) -> str:
    """Drop junk tokens (``junk_tokens_vi.txt``: "unk", "ofc", …) that stand alone as a word, in any
    case and position; "" when no word is left. Words containing one ("junk", "unknown") stay."""
    if not tokens:
        return text
    pat = re.compile(r"(?<!\w)(?:" + "|".join(map(re.escape, tokens)) + r")(?!\w)", re.IGNORECASE)
    out = pat.sub(" ", text)
    if out == text:
        return text
    out = re.sub(r"\s+", " ", out).strip()
    out = re.sub(r"\s+([.,!?;:])", r"\1", out)
    return out if re.search(r"\w", out) else ""


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
    drop_junk: bool = True  # strip_junk: junk_tokens_vi.txt, never in the references; WER never worse
    junk_tokens: tuple[str, ...] = field(default_factory=lambda: JUNK_TOKENS)

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
        return self.cleaned(text)[0]

    def cleaned(self, text: str) -> tuple[str, list[str]]:
        """``clean`` plus what changed: "junk" (junk token removed), "repetition" (loop written once)."""
        why = []
        if self.drop_junk:
            t = strip_junk(text, self.junk_tokens)
            if t != text:
                why.append("junk")
                text = t
        if self.collapse_loops:
            t = collapse_repeats(text)
            if t != text:
                why.append("repetition")
                text = t
        return text, why


def filter_from_settings(settings) -> SegmentFilter | None:
    s = settings
    if not s.asr_filter:
        return None
    return SegmentFilter(s.asr_max_no_speech_prob, s.asr_min_avg_logprob, s.asr_max_compression_ratio,
                         drop_junk=s.asr_drop_junk)


# Whisper's own decode limit per 30 s window (half the 448-token text context).
MAX_NEW_TOKENS = 224
_TAIL_DOTS = re.compile(r"\.{2,}\s*$")
_TAIL_A = re.compile(r"(?:\s+a\.?)+\s*$", re.IGNORECASE)


def strip_cap_tail(text: str) -> str:
    """Drop the junk tail a token-capped decode can leave: a trailing run of dots (-> one dot) and
    trailing stand-alone "a"/"a." tokens. Only used when the attempt reached its token limit."""
    out = _TAIL_DOTS.sub(".", text.rstrip())
    while True:
        new = _TAIL_A.sub("", out).rstrip()
        new = _TAIL_DOTS.sub(".", new)
        if new == out:
            break
        out = new
    return out if re.search(r"\w", out) else text


class DecodeLimiter:
    """Stands in for ``WhisperModel.model`` (CTranslate2) to bound each decoding attempt.

    faster-whisper decodes a segment with up to 6 attempts (temperature 0, then sampling at
    0.2 … 1.0) and has a single ``max_new_tokens`` for all of them; that single limit changes
    the temperature-0 beam result (plan-asr-anti-stall.md). Per call this wrapper can:

    * lower ``max_length`` of the temperature-0 attempt (``t0_tokens``) and of the fallback
      attempts (``fallback_tokens``) separately (None = untouched);
    * stop decoding once ``budget_s`` has passed since the segment's first attempt: a fallback
      call then returns the previous **real** result again, so faster-whisper sees it fail its
      checks, runs out of temperatures at no cost and picks the best real attempt. No synthetic
      or empty result is ever produced.

    Two optional rules for the temperature-0 attempt only (C3, plan-asr-anti-stall.md) make
    faster-whisper retry at the next temperature instead of accepting it:

    * ``reject_truncated_t0``: the attempt stopped at its token limit (no EOT);
    * ``reject_loop_t0``: its text has a run of >= 3 dots or a 4-word group repeated (a loop,
      also a partial one at the end).

    A rejected result keeps its tokens; only its score is set to -1e9, so its average log-prob
    fails the threshold (retry), and the "all attempts failed" choice (highest average log-prob)
    never picks it over a real retry. Nothing is deleted from the text.

    **Token limit.** Limits here are **real tokens** (what the model may generate). CTranslate2's
    Whisper does not generate ``max_length - prompt`` tokens: measured on phowhisper-small
    (plan-asr-anti-stall.md, C4), it generates at most ``max_length // 2`` with the plain 3-token
    prompt (the app's default) and ``max_length - prompt + 1`` with an initial-prompt context
    (28 tokens). ``max_length_for``/``real_token_limit`` apply that, so a limit of N gives N.
    ``calls`` records (seconds, temperature, generated tokens, real token limit) per real attempt.

    ``reject_truncated_all`` (C4): an attempt of any temperature that stopped at its limit (no
    EOT) is marked failed (see ``_mark_failed``); when every attempt was truncated,
    ``all_truncated`` is set and ``WhisperASR`` restores the chosen one (see there).
    """

    REJECT_OFFSET = 1000.0  # average log-prob drop of a marked attempt (C4)

    def __init__(self, inner, clock=None) -> None:
        import time

        self._inner = inner
        self._clock = clock or time.monotonic
        self.t0_tokens: int | None = None
        self.fallback_tokens: int | None = None
        self.budget_s: float | None = None
        self.calls: list[tuple[float, float, int, int]] = []
        self.budget_skips = 0
        self.reject_truncated_t0 = False
        self.reject_loop_t0 = False
        self.decode_text = None  # token ids -> text, for reject_loop_t0
        self.t0_rejected = ""  # why the temperature-0 attempt of this segment was rejected ("" = not)
        self.reject_truncated_all = False
        self.truncated: list[bool] = []  # per real attempt of this segment
        self.rejected = 0  # attempts of this segment marked failed
        self._start: float | None = None
        self._last = None

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def begin(self, t0_tokens: int | None, fallback_tokens: int | None, budget_s: float | None) -> None:
        """Before each segment."""
        self.t0_tokens, self.fallback_tokens, self.budget_s = t0_tokens, fallback_tokens, budget_s
        self.calls, self.budget_skips, self._start, self._last = [], 0, None, None
        self.t0_rejected, self.truncated, self.rejected = "", [], 0

    def generate(self, *args, **kw):
        temp = float(kw.get("sampling_temperature", 0.0) or 0.0)
        now = self._clock()
        if (temp > 0 and self.budget_s is not None and self._start is not None and self._last is not None
                and now - self._start > self.budget_s):
            self.budget_skips += 1
            return self._last
        if self._start is None:
            self._start = now
        prompts = args[1] if len(args) > 1 else kw.get("prompts", [[]])
        limit = self.fallback_tokens if temp > 0 else self.t0_tokens
        plen = len(prompts[0])
        if limit is not None:
            kw["max_length"] = min(kw.get("max_length", 448), max_length_for(limit, plen))
        out = self._inner.generate(*args, **kw)
        n = len(out[0].sequences_ids[0]) if out and out[0].sequences_ids else 0
        cap = real_token_limit(kw.get("max_length", 448), plen)
        self.calls.append((self._clock() - now, temp, n, cap))
        truncated = bool(out) and n >= cap
        self.truncated.append(truncated)
        why = ""
        if out and truncated and (self.reject_truncated_all or (temp == 0 and self.reject_truncated_t0)):
            why = "truncated"
        elif out and temp == 0 and self.reject_loop_t0 and self.decode_text is not None and has_loop_tail(
                self.decode_text(out[0].sequences_ids[0])):
            why = "loop"
        if why:
            if temp == 0:
                self.t0_rejected = why
            self.rejected += 1
            out = [_mark_failed(out[0], offset=self.REJECT_OFFSET if self.reject_truncated_all else None)] \
                + list(out[1:])
        self._last = out
        return out

    @property
    def all_truncated(self) -> bool:
        """Every real attempt of this segment stopped at its token limit."""
        return bool(self.truncated) and all(self.truncated)


class _Rejected:
    """A real CTranslate2 result marked as failed: same tokens, lowered score (see DecodeLimiter)."""

    def __init__(self, result, scores) -> None:
        self.sequences_ids = result.sequences_ids
        self.no_speech_prob = result.no_speech_prob
        self.scores = scores
        self.rejected = True


def _mark_failed(result, offset: float | None = None) -> _Rejected:
    """Same tokens; the score makes faster-whisper's average log-prob fail its -1 threshold so the
    next temperature runs. ``offset`` None: score -1e9 (C3, never chosen over a real retry);
    a number: average log-prob lowered by ``offset`` (C4), which keeps the order between marked
    attempts (faster-whisper's "all failed" choice = best real attempt) and can be undone."""
    n = len(result.sequences_ids[0]) if result.sequences_ids else 0
    scores = list(getattr(result, "scores", None) or [0.0])
    if offset is None or n == 0:
        return _Rejected(result, [-1e9 for _ in scores])
    # faster-whisper (length_penalty 1): avg = score * n / (n + 1)
    return _Rejected(result, [s - offset * (n + 1) / n for s in scores])


def max_length_for(tokens: int, prompt_len: int = 3) -> int:
    """``max_length`` that lets CTranslate2's Whisper generate ``tokens`` tokens (measured rule)."""
    tokens = min(tokens, MAX_NEW_TOKENS)
    return min(448, 2 * tokens + 1 if prompt_len <= 4 else prompt_len + tokens - 1)


def real_token_limit(max_length: int, prompt_len: int = 3) -> int:
    """Tokens CTranslate2's Whisper really generates at most for ``max_length`` (measured: ``//2``
    with the plain 3-token prompt, ``max_length - prompt + 1`` with an initial-prompt context)."""
    real = max_length // 2 if prompt_len <= 4 else max_length - prompt_len + 1
    return min(real, MAX_NEW_TOKENS)


_DOT_RUN = re.compile(r"\.{3,}")


def trim_truncated_tail(text: str, n: int = 4) -> str:
    """Text of an attempt cut by its token limit: drop from the second copy of a repeated
    ``n``-word group on (a loop), then a trailing dot run / stray "a" (``strip_cap_tail``)."""
    spans = [(m.start(), m.group(0).lower()) for m in re.finditer(r"\w+", text)]
    words = [w for _, w in spans]
    first: dict[tuple[str, ...], int] = {}
    for i in range(len(words) - n + 1):
        j = first.setdefault(tuple(words[i:i + n]), i)
        if i - j >= n:
            text = text[:spans[i][0]].rstrip()
            break
    return strip_cap_tail(text) if text else text


def has_loop_tail(text: str, n: int = 4) -> bool:
    """A run of >= 3 dots, or a group of ``n`` words occurring twice without overlap (a decoding
    loop, also when the second copy is cut off right after those ``n`` words)."""
    if _DOT_RUN.search(text):
        return True
    words = re.findall(r"\w+", text.lower())
    first: dict[tuple[str, ...], int] = {}
    for i in range(len(words) - n + 1):
        gram = tuple(words[i:i + n])
        j = first.setdefault(gram, i)
        if i - j >= n:
            return True
    return False


def decode_label(per_s: float | None, base: int, temps: tuple[float, ...] | None, *,
                 fallback_per_s: float | None = None, first_min: int | None = None,
                 budget_s: float | None = None, tail: bool = False, reject: str = "") -> str:
    """``base`` (faster-whisper defaults), else the parts in use joined by ``,``: ``cap15+32``
    (every attempt), ``t0/0.2/0.4``, ``fb15+32`` (fallback attempts only), ``t0min100``,
    ``budget4s``, ``tail``."""
    parts = []
    if per_s:
        parts.append(f"cap{per_s:g}+{base}")
    if temps:
        parts.append("t" + "/".join(f"{t:g}" for t in temps))
    if fallback_per_s:
        parts.append(f"fb{fallback_per_s:g}+{base}")
    if first_min:
        parts.append(f"t0min{first_min}")
    if budget_s:
        parts.append(f"budget{budget_s:g}s")
    if tail:
        parts.append("tail")
    if reject:
        parts.append("reject:" + reject.lstrip("+"))
    return ",".join(parts) or "base"


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
        max_tokens_per_s: float | None = None,
        max_tokens_base: int = 32,
        temperatures: tuple[float, ...] | None = None,
        fallback_max_tokens_per_s: float | None = None,
        first_min_tokens: int | None = None,
        decode_budget_s: float | None = None,
        cap_tail_strip: bool = False,
        reject_truncated_t0: bool = False,
        reject_loop_t0: bool = False,
        reject_truncated_all: bool = False,
    ) -> None:
        self.model_size = model_size
        self.language = language
        self.device = device
        self.compute_type = compute_type
        self.beam_size = beam_size
        self.initial_prompt = initial_prompt or None
        self.segment_filter = segment_filter  # None: keep every segment (previous behaviour)
        # Bounded decoding (None = faster-whisper defaults): output-token cap per second of audio,
        # and the temperature fallback ladder.
        self.max_tokens_per_s = max_tokens_per_s
        self.max_tokens_base = max_tokens_base
        self.temperatures = tuple(temperatures) if temperatures else None
        # Anti-stall per attempt (DecodeLimiter; None/False = off): cap only the fallback attempts
        # (C); also give the temperature-0 attempt a wide limit >= first_min_tokens (C+); stop
        # retrying after decode_budget_s per segment (C2); strip a junk tail when a capped
        # attempt reached its limit (B).
        self.fallback_max_tokens_per_s = fallback_max_tokens_per_s
        self.first_min_tokens = first_min_tokens
        self.decode_budget_s = decode_budget_s
        self.cap_tail_strip = cap_tail_strip
        self.reject_truncated_t0 = reject_truncated_t0
        self.reject_loop_t0 = reject_loop_t0
        self.reject_truncated_all = reject_truncated_all
        self.last_all_truncated = False  # C4: every attempt of the last call stopped at its limit
        self.last_hit_limit = False  # the last real attempt of the last call reached its token limit
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
                if self.uses_limiter:
                    lim = DecodeLimiter(self._model.model)
                    lim.reject_truncated_t0, lim.reject_loop_t0 = self.reject_truncated_t0, self.reject_loop_t0
                    lim.reject_truncated_all = self.reject_truncated_all
                    tok = getattr(self._model, "hf_tokenizer", None)
                    if tok is not None:
                        eot = tok.token_to_id("<|endoftext|>") or 50257
                        lim.decode_text = lambda ids, _t=tok, _e=eot: _t.decode([i for i in ids if i < _e])
                    self._model.model = lim

    @property
    def uses_limiter(self) -> bool:
        return bool(self.max_tokens_per_s or self.fallback_max_tokens_per_s or self.first_min_tokens
                    or self.decode_budget_s or self.cap_tail_strip or self.reject_truncated_t0
                    or self.reject_loop_t0 or self.reject_truncated_all)

    @property
    def limiter(self) -> "DecodeLimiter | None":
        m = getattr(self._model, "model", None) if self._model is not None else None
        return m if isinstance(m, DecodeLimiter) else None

    def attempt_limits(self, audio_s: float) -> tuple[int | None, int | None]:
        """(temperature-0 limit, fallback limit) in **real** tokens for ``audio_s`` of audio (None = none).

        ``max_tokens_per_s`` caps every attempt; ``fallback_max_tokens_per_s`` the retries only;
        ``first_min_tokens`` gives temperature 0 at least that many."""
        def cap(per_s):
            return min(MAX_NEW_TOKENS, math.ceil(audio_s * per_s) + self.max_tokens_base) if per_s else None

        fb = cap(self.fallback_max_tokens_per_s or self.max_tokens_per_s)
        t0 = cap(self.max_tokens_per_s)
        if self.first_min_tokens:
            per_s = self.fallback_max_tokens_per_s or self.max_tokens_per_s or 0
            t0 = min(MAX_NEW_TOKENS, max(self.first_min_tokens, cap(per_s) or 0))
        return t0, fb

    @property
    def decode(self) -> str:
        """Label of the decoding options in use (``base`` = faster-whisper defaults)."""
        return decode_label(self.max_tokens_per_s, self.max_tokens_base, self.temperatures,
                            fallback_per_s=self.fallback_max_tokens_per_s, first_min=self.first_min_tokens,
                            budget_s=self.decode_budget_s, tail=self.cap_tail_strip,
                            reject=("trunc" if self.reject_truncated_all else "t0trunc" if self.reject_truncated_t0 else "")
                            + ("+loop" if self.reject_loop_t0 else ""))

    def max_new_tokens(self, audio_s: float) -> int | None:
        """Real-token cap of every attempt for ``audio_s`` seconds of audio (None = no cap).
        Applied through ``DecodeLimiter`` (real tokens), not faster-whisper's ``max_new_tokens``,
        which CTranslate2 turns into about half as many tokens (plan-asr-anti-stall.md, C4)."""
        if not self.max_tokens_per_s:
            return None
        return min(MAX_NEW_TOKENS, math.ceil(audio_s * self.max_tokens_per_s) + self.max_tokens_base)

    def decode_options(self, audio_s: float) -> dict:
        """Extra ``WhisperModel.transcribe`` options; empty when bounded decoding is off."""
        kw: dict = {}
        if self.temperatures:
            kw["temperature"] = list(self.temperatures)
        return kw

    def transcribe_segments(self, audio: np.ndarray, sample_rate: int) -> list:
        """Raw faster-whisper segments (text, no_speech_prob, avg_logprob, compression_ratio, …)."""
        if sample_rate != 16000:
            raise ValueError("WhisperASR expects 16 kHz audio")
        self.load()
        pcm = audio.astype(np.float32) / 32768.0 if audio.dtype == np.int16 else audio.astype(np.float32)
        lim = self.limiter
        if lim is not None:
            lim.begin(*self.attempt_limits(pcm.size / sample_rate), self.decode_budget_s)
        segments, _info = self._model.transcribe(
            pcm,
            language=self.language,
            beam_size=self.beam_size,
            vad_filter=False,  # M3 already segmented
            condition_on_previous_text=False,
            initial_prompt=self.initial_prompt,
            **self.decode_options(pcm.size / sample_rate),
        )
        segments = list(segments)
        self.last_temperature = max((getattr(x, "temperature", 0.0) or 0.0 for x in segments), default=0.0)
        self.last_hit_limit = bool(lim is not None and lim.calls and lim.calls[-1][2] >= lim.calls[-1][3])
        self.last_all_truncated = bool(lim is not None and self.reject_truncated_all and lim.all_truncated)
        if self.last_all_truncated:
            # C4, every attempt stopped at its limit: faster-whisper chose the best of them (marking
            # keeps their order). Undo the marking so the ASR filter judges the real log-prob, and
            # cut the truncated tail (a repeated part, a dot run, stray "a").
            for seg in segments:
                try:
                    seg.avg_logprob = seg.avg_logprob + DecodeLimiter.REJECT_OFFSET
                    seg.text = trim_truncated_tail(seg.text)
                except AttributeError:
                    pass
        if self.cap_tail_strip and self.last_hit_limit:
            for seg in segments:  # B: only when the attempt really reached its limit
                try:
                    seg.text = strip_cap_tail(seg.text)
                except AttributeError:  # immutable segment tuple
                    pass
        return segments

    def transcribe(self, audio: np.ndarray, sample_rate: int) -> str:
        kept, self.dropped = [], []
        for seg in self.transcribe_segments(audio, sample_rate):
            why = self.segment_filter.reason(seg) if self.segment_filter is not None else None
            text = seg.text.strip()
            if why is None:
                if self.segment_filter is not None:
                    clean, changed = self.segment_filter.cleaned(text)
                    self.dropped += [(why_c, text) for why_c in changed]
                    text = clean
                if text:
                    kept.append(text)
            else:
                self.dropped.append((why, text))
        return " ".join(kept).strip()
