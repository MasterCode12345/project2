"""M5: machine translation behind an adapter protocol.

Backends:

* ``OfflineTranslator`` -- the Argos Translate ``vi→en`` model package run
  directly with CTranslate2 + SentencePiece (no ``argostranslate`` library).
* ``GoogleTranslator`` -- Google Translate via ``deep-translator`` (network).
* ``FallbackTranslator`` -- tries a primary backend, falls back per call.
* ``PivotTranslator`` -- two backends chained through a pivot language.

``supports(src, tgt)`` (app language codes, see ``languages.py``) says whether
a backend can translate a pair directly; ``plan_route`` picks a direct route
or, only when there is none, a pivot through English.
"""

from __future__ import annotations

import json
import re
import shutil
import tempfile
import threading
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Protocol
from urllib.parse import urlparse


class Translator(Protocol):
    def translate(self, text: str) -> str: ...


def _mt_code(code: str, backend: str) -> str | None:
    from .languages import LANGUAGES

    lang = LANGUAGES.get(code)
    return lang.mt.get(backend) if lang else None


class GoogleTranslator:
    """Google Translate via ``deep-translator`` (network)."""

    def __init__(self, source: str = "vi", target: str = "en") -> None:
        from deep_translator import GoogleTranslator as _DT

        self._impl = _DT(source=source, target=target)

    @staticmethod
    def supports(src: str, tgt: str) -> bool:
        """Both app languages have a Google code in the language table (and differ)."""
        return src != tgt and _mt_code(src, "google") is not None and _mt_code(tgt, "google") is not None

    def translate(self, text: str) -> str:
        result = self._impl.translate(text)
        return (result or "").strip()


# ------------------------------------------------------------ Argos package
class PackageError(RuntimeError):
    """The offline translation package cannot be obtained or is invalid."""


_REQUIRED = (
    "metadata.json",
    "sentencepiece.model",
    "model/model.bin",
    "model/config.json",
    "model/shared_vocabulary.json",
)


def _default_opener(url: str):
    import urllib.request

    # The package host rejects urllib's default "Python-urllib" User-Agent (HTTP 403).
    req = urllib.request.Request(url, headers={"User-Agent": "audio-translator/0.1"})
    return urllib.request.urlopen(req, timeout=60)


def _check_package(pkg: Path, source: str, target: str) -> None:
    for rel in _REQUIRED:
        if not (pkg / rel).is_file():
            raise PackageError(f"package is missing {rel}")
    try:
        meta = json.loads((pkg / "metadata.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise PackageError(f"unreadable metadata.json: {exc}") from exc
    if not isinstance(meta, dict) or meta.get("from_code") != source or meta.get("to_code") != target:
        got = f"{meta.get('from_code')}→{meta.get('to_code')}" if isinstance(meta, dict) else "?"
        raise PackageError(f"package is {got}, expected {source}→{target}")


def ensure_argos_package(
    cache_dir: str | Path,
    url: str,
    opener: Callable[[str], Any] | None = None,
    *,
    source: str = "vi",
    target: str = "en",
    on_download: Callable[[], None] | None = None,
) -> Path:
    """Return the directory of the installed Argos package, downloading it once.

    The archive is downloaded and extracted inside ``cache_dir`` in a temporary
    directory, verified (zip integrity, required files, ``from_code``/``to_code``)
    and only then renamed into place, so a failure never leaves a partial package.
    ``on_download`` is called once before a download starts (not when cached).
    Raises ``PackageError`` on any failure.
    """
    cache = Path(cache_dir).expanduser()
    name = Path(urlparse(url).path).stem or "package"  # translate-vi_en-1_9
    final = cache / name
    # Temp dirs left by an install killed mid-way (daemon thread, app quit).
    if cache.is_dir():
        for stale in cache.glob(".download-*"):
            shutil.rmtree(stale, ignore_errors=True)
    if final.is_dir():
        try:
            _check_package(final, source, target)
            return final
        except PackageError:
            shutil.rmtree(final, ignore_errors=True)  # broken install: fetch again

    try:
        cache.mkdir(parents=True, exist_ok=True)
        tmp = Path(tempfile.mkdtemp(prefix=".download-", dir=cache))
    except OSError as exc:
        raise PackageError(f"cannot create cache directory {cache}: {exc}") from exc
    try:
        archive = tmp / "package.zip"
        if on_download is not None:
            on_download()
        try:
            with (opener or _default_opener)(url) as resp, open(archive, "wb") as f:
                shutil.copyfileobj(resp, f, 1 << 20)
        except Exception as exc:
            raise PackageError(f"download of {url} failed: {exc}") from exc

        extract = tmp / "extract"
        try:
            with zipfile.ZipFile(archive) as zf:
                bad = zf.testzip()
                if bad is not None:
                    raise PackageError(f"corrupt archive (bad entry {bad})")
                for member in zf.namelist():
                    p = PurePosixPath(member)
                    if p.is_absolute() or ".." in p.parts:
                        raise PackageError(f"unsafe path in archive: {member}")
                zf.extractall(extract)
        except zipfile.BadZipFile as exc:
            raise PackageError(f"corrupt archive: {exc}") from exc

        # Argos packages hold one top-level directory; accept a flat layout too.
        entries = [p for p in extract.iterdir() if not p.name.startswith("__MACOSX")]
        root = entries[0] if len(entries) == 1 and entries[0].is_dir() else extract
        _check_package(root, source, target)

        try:
            root.rename(final)
        except OSError:
            if final.is_dir():  # a concurrent install won the race
                _check_package(final, source, target)
            else:
                raise
        return final
    except PackageError:
        raise
    except Exception as exc:
        raise PackageError(f"cannot install translation package: {exc}") from exc
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ------------------------------------------------------------ offline backend
_SENTENCE_END = re.compile(r"(?<=[.!?…])\s+")


def split_sentences(text: str) -> list[str]:
    return [s for s in (p.strip() for p in _SENTENCE_END.split(text.strip())) if s]


def _ct2_factory(model_dir: str):
    import ctranslate2

    return ctranslate2.Translator(model_dir, device="cpu", compute_type="int8")


def _spm_factory(model_file: str):
    import sentencepiece

    return sentencepiece.SentencePieceProcessor(model_file=model_file)


class OfflineTranslator:
    """Argos ``vi→en`` package run with CTranslate2 + SentencePiece, fully offline."""

    def __init__(
        self,
        package_dir: str | Path,
        *,
        beam_size: int = 4,
        translator_factory: Callable[[str], Any] = _ct2_factory,
        tokenizer_factory: Callable[[str], Any] = _spm_factory,
    ) -> None:
        self.package_dir = Path(package_dir)
        self.beam_size = beam_size
        self._ct2_factory = translator_factory
        self._spm_factory = tokenizer_factory
        self._model = None
        self._sp = None
        self._lock = threading.Lock()

    def supports(self, src: str, tgt: str) -> bool:
        """The installed package translates exactly ``src``→``tgt`` (its ``metadata.json``)."""
        try:
            meta = json.loads((self.package_dir / "metadata.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return False
        if not isinstance(meta, dict):
            return False
        a, b = _mt_code(src, "argos"), _mt_code(tgt, "argos")
        return a is not None and b is not None and meta.get("from_code") == a and meta.get("to_code") == b

    def load(self) -> None:
        with self._lock:
            if self._model is None:
                sp = self._spm_factory(str(self.package_dir / "sentencepiece.model"))
                self._model = self._ct2_factory(str(self.package_dir / "model"))
                self._sp = sp

    def translate(self, text: str) -> str:
        sentences = split_sentences(text)
        if not sentences:
            return ""
        self.load()
        batch = [self._sp.encode(s, out_type=str) for s in sentences]
        results = self._model.translate_batch(batch, beam_size=self.beam_size)
        out = []
        for r in results:
            tokens = r.hypotheses[0] if r.hypotheses else []
            piece = "".join(tokens).replace("▁", " ").strip()
            if piece:
                out.append(" ".join(piece.split()))
        return " ".join(out)


class FallbackTranslator:
    """Use ``primary``; on an exception use ``fallback`` for that call only."""

    def __init__(self, primary: Translator, fallback: Translator | Callable[[], Translator],
                 on_fallback: Callable[[BaseException], None] | None = None) -> None:
        self.primary = primary
        self._fallback = fallback
        self._on_fallback = on_fallback

    @property
    def fallback(self) -> Translator:
        # A zero-arg factory is resolved on first use (no Google client unless needed).
        if not hasattr(self._fallback, "translate"):
            self._fallback = self._fallback()
        return self._fallback

    def translate(self, text: str) -> str:
        try:
            return self.primary.translate(text)
        except Exception as exc:
            if self._on_fallback is not None:
                self._on_fallback(exc)
            return self.fallback.translate(text)


class PivotTranslator:
    """``first`` (src→pivot) then ``second`` (pivot→tgt): used only when no direct model exists."""

    def __init__(self, first: Translator, second: Translator, pivot: str = "en") -> None:
        self.first, self.second, self.pivot = first, second, pivot

    def translate(self, text: str) -> str:
        middle = self.first.translate(text)
        return self.second.translate(middle) if middle.strip() else ""


def plan_route(supports: Callable[[str, str], bool], src: str, tgt: str, pivot: str = "en") -> str | None:
    """``"direct"`` if ``supports(src, tgt)``; else ``"pivot:<pivot>"`` if both legs are supported; else None."""
    if src == tgt:
        raise ValueError(f"source and target are the same language ({src})")
    if supports(src, tgt):
        return "direct"
    if pivot not in (src, tgt) and supports(src, pivot) and supports(pivot, tgt):
        return f"pivot:{pivot}"
    return None
