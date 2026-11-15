"""Glossary of proper names for MT: protect terms with placeholders, fall back safely.

* ``glossary_<src>_<tgt>.tsv`` (shipped) + an optional user file (``--glossary PATH``): one entry per
  line ``source<TAB>target[<TAB>flags]``; ``#`` comments. A source term needs >= 3 syllables
  (Vietnamese is written syllable by syllable) unless flagged ``short_ok``. The user file adds to
  the shipped one and overrides an identical source term.
* Matching (``Glossary.find``): Unicode NFC, case-insensitive, tone marks kept, whole words,
  longest term first, never across punctuation.
* ``GlossaryTranslator`` wraps a translator: a sentence without a term goes to it unchanged; one
  with terms is translated with placeholders, which must each come back exactly once, and are
  then replaced by the target terms. Otherwise the sentence is translated again without the
  glossary (a "fallback"). No placeholder ever reaches the output.

plan-glossary.md.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

GLOSSARY_DIR = Path(__file__).parent
MIN_SYLLABLES = 3
SHORT_OK = "short_ok"


def glossary_path(source: str, target: str) -> Path:
    return GLOSSARY_DIR / f"glossary_{source}_{target}.tsv"


def nfc(text: str) -> str:
    return unicodedata.normalize("NFC", text)


@dataclass(frozen=True)
class Entry:
    source: str  # NFC
    target: str
    origin: str  # "file:line"


class GlossaryError(ValueError):
    pass


def parse(path: Path, *, strict: bool) -> tuple[list[Entry], list[str]]:
    """Entries of ``path`` and the problems found (``file:line: why``). ``strict`` raises on the
    first problem (the shipped file); otherwise bad lines are skipped and reported (a user file)."""
    entries, problems = [], []
    for n, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = nfc(raw).rstrip("\n")
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        cols = [c.strip() for c in line.split("\t")]
        where = f"{path}:{n}"
        why = None
        if len(cols) < 2 or not cols[0] or not cols[1]:
            why = "needs 'source<TAB>target'"
        else:
            flags = {f.strip().lower() for f in (cols[2] if len(cols) > 2 else "").split(",") if f.strip()}
            if len(cols[0].split()) < MIN_SYLLABLES and SHORT_OK not in flags:
                why = (f"'{cols[0]}' has fewer than {MIN_SYLLABLES} syllables "
                       f"(generic words are refused; add '{SHORT_OK}' in column 3 to keep it)")
            elif not re.search(r"\w", cols[0]) or re.search(r"[.,;:!?]", cols[0]):
                why = f"'{cols[0]}' must be words only (no punctuation)"
        if why:
            if strict:
                raise GlossaryError(f"{where}: {why}")
            problems.append(f"{where}: {why}")
            continue
        entries.append(Entry(" ".join(cols[0].split()), cols[1], where))
    return entries, problems


class Glossary:
    def __init__(self, entries: list[Entry]) -> None:
        by_key: dict[str, Entry] = {}
        for e in entries:  # later entries (the user file) override identical source terms
            by_key[e.source.casefold()] = e
        self.entries = list(by_key.values())
        # longest first (syllables, then characters): the alternation tries them in this order
        ordered = sorted(self.entries, key=lambda e: (-len(e.source.split()), -len(e.source)))
        self._target = {e.source.casefold(): e for e in ordered}
        if ordered:
            alts = "|".join(r"[ \t]+".join(map(re.escape, e.source.split())) for e in ordered)
            self._re = re.compile(rf"(?<!\w)(?:{alts})(?!\w)", re.IGNORECASE)
        else:
            self._re = None

    @classmethod
    def load(cls, source: str, target: str, user_path: str | Path | None = None) -> tuple["Glossary", list[str]]:
        """Shipped file (strict) + user file (lenient). A missing user path raises ``FileNotFoundError``."""
        shipped = glossary_path(source, target)
        entries = parse(shipped, strict=True)[0] if shipped.is_file() else []
        problems: list[str] = []
        if user_path:
            p = Path(user_path).expanduser()
            if not p.is_file():
                raise FileNotFoundError(f"Glossary file not found: {p}")
            user, problems = parse(p, strict=False)
            entries += user
        return cls(entries), problems

    def __len__(self) -> int:
        return len(self.entries)

    def find(self, text: str) -> list[tuple[int, int, Entry]]:
        """(start, end, entry) of each term in ``text`` (NFC), left to right, longest first, no overlap."""
        if self._re is None:
            return []
        out = []
        for m in self._re.finditer(nfc(text)):
            key = " ".join(m.group(0).split()).casefold()
            out.append((m.start(), m.end(), self._target[key]))
        return out


# Placeholder forms (plan-glossary.md); "inline" puts the target term itself in the source.
FORMS = {
    "x": lambda k: f"X{k}",
    "zqx": lambda k: f"ZQX{k}",
    "bracket": lambda k: f"[{k}]",
    "under": lambda k: f"__T{k}__",
}
INLINE = "inline"
# Anything that still looks like one of our placeholders after restoring: never let it out.
_RESIDUE = {
    "x": re.compile(r"(?<![\w-])X\d+(?!\w)"),
    "zqx": re.compile(r"ZQX|ZQ\s*X", re.IGNORECASE),
    "bracket": re.compile(r"\[\s*\d+\s*\]"),
    "under": re.compile(r"__|(?<![\w])T\d+__"),
    INLINE: None,
}


def protect(text: str, found: list[tuple[int, int, Entry]], form: str) -> tuple[str, list[tuple[str, Entry]]]:
    """``text`` with each found term replaced by its placeholder; (placeholder, entry) pairs."""
    text = nfc(text)
    out, pos, slots = [], 0, []
    for k, (a, b, e) in enumerate(found, 1):
        ph = e.target if form == INLINE else FORMS[form](k)
        out += [text[pos:a], ph]
        slots.append((ph, e))
        pos = b
    out.append(text[pos:])
    return "".join(out), slots


def restore(translated: str, slots: list[tuple[str, Entry]], form: str) -> str | None:
    """The translation with placeholders replaced by target terms, or None when any placeholder is
    missing, repeated or deformed (the caller then falls back)."""
    out = translated
    for ph, e in slots:
        pat = re.compile(rf"(?<!\w){re.escape(ph)}(?!\w)" if form != "bracket" else re.escape(ph),
                         re.IGNORECASE if form == INLINE else 0)
        hits = pat.findall(out)
        if len(hits) != 1:
            return None
        out = pat.sub(lambda _m, t=e.target: t, out, count=1)
    residue = _RESIDUE.get(form)
    if residue is not None and residue.search(out):
        return None
    return out


class GlossaryTranslator:
    """Wraps a translator with the glossary. ``last`` = what the last call did: "" (no term), the
    source terms used (``;``-joined), or ``"fallback"``. ``stats`` counts calls."""

    def __init__(self, inner, glossary: Glossary, form: str = "x") -> None:
        if form not in FORMS and form != INLINE:
            raise ValueError(f"unknown placeholder form {form!r}")
        self.inner = inner
        self.glossary = glossary
        self.form = form
        self.last = ""
        self.stats = {"calls": 0, "with_terms": 0, "fallbacks": 0}

    def __getattr__(self, name):  # load(), supports(), … of the wrapped translator
        return getattr(self.inner, name)

    def translate(self, text: str) -> str:
        self.stats["calls"] += 1
        found = self.glossary.find(text)
        if not found:
            self.last = ""
            return self.inner.translate(text)  # unchanged input: identical output
        self.stats["with_terms"] += 1
        protected, slots = protect(text, found, self.form)
        result = restore(self.inner.translate(protected) or "", slots, self.form)
        if result is None:
            self.stats["fallbacks"] += 1
            self.last = "fallback"
            return self.inner.translate(text)
        self.last = ";".join(dict.fromkeys(e.source for _ph, e in slots))
        return result
