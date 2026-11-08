"""M4b: restore casing and punctuation in ASR text before translation (optional, off by default).

PhoWhisper writes lowercase text without punctuation; the MT models translate
that input markedly worse (``eval/mt_bench.py``: oracle punctuation is worth
more BLEU than casing). Restorers:

* ``sent``: capital first letter + final period. No model, any cased script.
* ``vibert-capu``: dragonSwing/vibert-capu (CC-BY-SA-4.0), a viBERT
  (FPTAI/vibert-base-cased) GECToR-style tagger for Vietnamese that labels
  each word with ``$KEEP``, ``$TRANSFORM_CASE_*``, ``$APPEND_{. , ? :}`` or
  ``$MERGE_SPACE``. ``scripts/convert_capu_model.py`` converts it once to a
  CTranslate2 encoder plus the label head as numpy arrays; at run time only
  ctranslate2 + numpy are needed (no torch, no transformers).

``CapuRestorer`` re-implements the model card's ``GecBERTModel`` inference
(defaults ``iterations=3, min_len=3, min_error_probability=0``, so the detect
head has no effect and is not exported) with two differences: English
``$TRANSFORM_VERB_*`` labels keep the word (they have no meaning for
Vietnamese), and input longer than ``chunk_words`` is cut into equal windows
without the card's overlap merging.
"""

from __future__ import annotations

import re
import threading
import unicodedata
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .languages import normalize
from .translate import MT_MODELS_DIR

RESTORE_MODELS = {
    # name: (repo, revision, base tokenizer repo, base revision, languages)
    "vibert-capu": ("dragonSwing/vibert-capu", "261c60f2c30b02455dfce21a43c3ef14fc26992c",
                    "FPTAI/vibert-base-cased", "728a91287e4517d9312066a6aa048fadf4e41e91", ("vi",)),
}
RESTORERS = ("off", "sent", *RESTORE_MODELS)
APP_RESTORERS = ("auto", *RESTORERS)  # auto: a converted model for the source language, else off
HEADS_FILE = "heads.npz"
LABELS_FILE = "labels.txt"
VOCAB_FILE = "wordpiece_vocab.txt"
START_TOKEN = "$START"
_MERGE_PUNCT = re.compile(r"\s+([:.,?])")
_REPEATED_PUNCT = re.compile(r"([\.\,\?\:]\s+)+")


def restore_model_dir(name: str, root: Path | None = None) -> Path:
    return (root or MT_MODELS_DIR).expanduser() / f"{name}-ct2"


def auto_restorer_name(lang: str, root: Path | None = None) -> str | None:
    """The first ``RESTORE_MODELS`` entry that supports ``lang`` and is converted, or None."""
    for name, entry in RESTORE_MODELS.items():
        if lang in entry[4] and (restore_model_dir(name, root) / "model.bin").is_file():
            return name
    return None


def restore_convert_command(name: str) -> str:
    return f'uv run --with "transformers<5" --with torch python scripts/convert_capu_model.py {name}'


def sentence_case(text: str) -> str:
    """Capital first letter and a final period (unless it already ends with punctuation)."""
    text = " ".join(text.split())
    if not text:
        return text
    text = text[0].upper() + text[1:]
    return text if unicodedata.category(text[-1])[0] == "P" else text + "."


# ------------------------------------------------------------------ WordPiece
def _is_punct(ch: str) -> bool:
    """BERT's rule: ASCII non-alphanumeric printables count as punctuation, plus Unicode P*."""
    cp = ord(ch)
    if 33 <= cp <= 47 or 58 <= cp <= 64 or 91 <= cp <= 96 or 123 <= cp <= 126:
        return True
    return unicodedata.category(ch).startswith("P")


class WordPiece:
    """Cased BERT WordPiece (no lowercasing, no accent stripping), as ``BertTokenizerFast(do_lower_case=False)``."""

    def __init__(self, vocab: list[str], *, unk: str = "[UNK]", prefix: str = "##", max_chars: int = 100,
                 added: tuple[str, ...] = (START_TOKEN,)) -> None:
        self.vocab = set(vocab)
        self.unk, self.prefix, self.max_chars = unk, prefix, max_chars
        self.added = set(added)

    def _pieces(self, chunk: str) -> list[str]:
        if len(chunk) > self.max_chars:
            return [self.unk]
        out, start = [], 0
        while start < len(chunk):
            end, piece = len(chunk), None
            while start < end:
                cand = chunk[start:end] if start == 0 else self.prefix + chunk[start:end]
                if cand in self.vocab:
                    piece = cand
                    break
                end -= 1
            if piece is None:
                return [self.unk]
            out.append(piece)
            start = end
        return out

    def word(self, word: str) -> list[str]:
        """Pieces of one input word: punctuation is split off first (BertPreTokenizer), then WordPiece."""
        if word in self.added:
            return [word]
        chunks, cur = [], ""
        for ch in unicodedata.normalize("NFC", word):
            if _is_punct(ch):
                if cur:
                    chunks.append(cur)
                chunks.append(ch)
                cur = ""
            elif not ch.isspace() and unicodedata.category(ch) not in ("Cc", "Cf"):
                cur += ch
        if cur:
            chunks.append(cur)
        return [p for c in chunks for p in self._pieces(c)] or [self.unk]


# ------------------------------------------------------------------ edits
def _case(token: str, label: str) -> str:
    if label.endswith("LOWER"):
        return token.lower()
    if label.endswith("UPPER"):
        return token.upper()
    if label.endswith("CAPITAL"):
        return token.capitalize()
    if label.endswith("CAPITAL_1"):
        return token[0] + token[1:].capitalize() if token else token
    if label.endswith("UPPER_-1"):
        return token[:-1].upper() + token[-1] if token else token
    return token


def apply_labels(words: list[str], labels: list[str]) -> list[str]:
    """One GECToR pass: ``labels[0]`` is the $START position, ``labels[i]`` belongs to ``words[i-1]``."""
    edits = []  # (start, end, label) in word positions
    for i, label in enumerate(labels[:len(words) + 1]):
        if label in ("$KEEP", "@@UNKNOWN@@", "@@PADDING@@"):
            continue
        if label.startswith("$TRANSFORM_"):
            edits.append((i - 1, i, label))
        elif label.startswith("$APPEND_"):
            edits.append((i, i, label[len("$APPEND_"):]))
        elif label.startswith("$MERGE_"):
            edits.append((i, i, label))
    out, shift = list(words), 0
    for start, end, label in edits:
        pos = start + shift
        if start < 0:
            continue
        if start == end:  # insert (punctuation or a $MERGE_ marker), never twice in a row
            if (pos < len(out) and out[pos] == label) or (pos > 0 and out[pos - 1] == label):
                continue
            out[pos:pos] = [label]
            shift += 1
        elif label.startswith("$TRANSFORM_CASE_") and pos < len(out):
            out[pos] = _case(out[pos], label)
        # $TRANSFORM_VERB_*: English verb forms, the word is kept
    if any(t.startswith("$MERGE_") for t in out):
        if out[0].startswith("$MERGE_"):
            out = out[1:]
        if out and out[-1].startswith("$MERGE_"):
            out = out[:-1]
        line = " ".join(out).replace(" $MERGE_HYPHEN ", "-").replace(" $MERGE_SPACE ", "")
        out = _REPEATED_PUNCT.sub(r"\1", line).split()
    return out


# ------------------------------------------------------------------ model
def _encoder_factory(model_dir: str):
    import ctranslate2

    return ctranslate2.Encoder(model_dir, device="cpu", compute_type="int8")


class CapuRestorer:
    """vibert-capu run with CTranslate2 (encoder) + numpy (label head)."""

    def __init__(self, name: str = "vibert-capu", *, root: Path | None = None, iterations: int = 3,
                 min_len: int = 3, chunk_words: int = 48,
                 encoder_factory: Callable[[str], Any] = _encoder_factory) -> None:
        if name not in RESTORE_MODELS:
            raise ValueError(f"Unknown restore model '{name}' (use one of {', '.join(RESTORE_MODELS)})")
        self.name = name
        self.languages = RESTORE_MODELS[name][4]
        self.model_dir = restore_model_dir(name, root)
        self.iterations, self.min_len, self.chunk_words = iterations, min_len, chunk_words
        self._encoder_factory = encoder_factory
        self._encoder = None
        self._wp: WordPiece | None = None
        self._w = self._b = None
        self._labels: list[str] = []
        self._lock = threading.Lock()

    def supports(self, lang: str) -> bool:
        return lang in self.languages

    def load(self) -> None:
        with self._lock:
            if self._encoder is not None:
                return
            files = ("model.bin", HEADS_FILE, LABELS_FILE, VOCAB_FILE)
            missing = [f for f in files if not (self.model_dir / f).is_file()]
            if missing:
                raise RuntimeError(f"Restore model '{self.name}' is not converted yet (missing "
                                   f"{', '.join(missing)} in {self.model_dir}). Run: "
                                   f"{restore_convert_command(self.name)}")
            heads = np.load(self.model_dir / HEADS_FILE)
            self._w, self._b = heads["classifier_weight"].astype(np.float32), heads["classifier_bias"].astype(np.float32)
            self._labels = (self.model_dir / LABELS_FILE).read_text(encoding="utf-8").split("\n")[:self._w.shape[0]]
            vocab = (self.model_dir / VOCAB_FILE).read_text(encoding="utf-8").split("\n")
            self._wp = WordPiece([v for v in vocab if v])
            self._encoder = self._encoder_factory(str(self.model_dir))

    def _label_batch(self, batch: list[list[str]]) -> list[list[str]]:
        """Label of every position ($START + each word) for each word list."""
        tokens, offsets = [], []
        for words in batch:
            toks, offs = [], []
            for w in [START_TOKEN, *words]:
                offs.append(len(toks))
                toks.extend(self._wp.word(w))
            tokens.append(toks)
            offsets.append(offs)
        out = self._encoder.forward_batch(tokens)
        hidden = np.array(out.last_hidden_state, dtype=np.float32)  # (batch, max tokens, hidden)
        labels = []
        for k, offs in enumerate(offsets):
            logits = hidden[k, offs] @ self._w.T + self._b
            labels.append([self._labels[i] for i in logits.argmax(-1)])
        return labels

    def _restore_words(self, words: list[str]) -> list[str]:
        if len(words) < self.min_len:
            return words
        cur, seen = words, [words]
        for _ in range(self.iterations):
            new = apply_labels(cur, self._label_batch([cur])[0])
            if new == cur:
                break
            cur = new
            if new in seen:
                break
            seen.append(new)
        return cur

    def restore(self, text: str, lang: str = "vi") -> str:
        """Lowercase/unpunctuated form of ``text`` with casing and punctuation predicted again."""
        words = normalize(text, lang).split()
        if not words:
            return ""
        self.load()
        n = -(-len(words) // self.chunk_words)  # windows of at most chunk_words, sizes within one word
        cuts = [round(k * len(words) / n) for k in range(n + 1)]
        out = [w for a, b in zip(cuts, cuts[1:]) for w in self._restore_words(words[a:b])]
        return _MERGE_PUNCT.sub(r"\1", " ".join(out))


def make_restorer(name: str) -> Callable[[str, str], str] | None:
    """``(text, lang) -> text`` for a ``RESTORERS`` name; None for ``off``."""
    if name == "off":
        return None
    if name == "sent":
        return lambda text, lang: sentence_case(text)
    if name in RESTORE_MODELS:
        return CapuRestorer(name).restore
    raise ValueError(f"Unknown restorer '{name}' (use one of {', '.join(RESTORERS)})")
