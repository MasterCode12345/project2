"""MT bench: BLEU/chrF and translation latency per backend x input condition.

Two questions:

* (A) how good is translation on the *reference* source sentence
  (``condition=text``), and on that sentence formatted like ASR output:
  lowercased, no punctuation (``condition=text_norm``);
* (B) how much worse it gets on what ASR heard (``condition=asr``), for each
  ``--asr-models`` model: Whisper transcribes the whole clean FLEURS recording
  (no VAD/segmenter, no noise, no prompt, beam 5), so the gap text_norm -> asr
  is recognition errors only and text -> text_norm is lost casing/punctuation;
* (C) what the app really does (``condition=asr_seg``): the recording goes
  through the app's VAD + Segmenter and ASR filter (``Settings``: VAD backend,
  its threshold, silence, ``asr_filter``), each segment is transcribed and translated on its own, and the
  segment translations are joined for scoring. Latency is then per segment,
  the unit the app translates; ``n_inputs``/``mean_words`` say how many
  translator inputs a row had and how long they were.

``--restore`` adds rows where each translator input of ``text_norm``/``asr``/
``asr_seg`` first goes through a casing/punctuation restorer
(``audio_translator.restore``: ``sent`` = capital first letter + final
period, ``vibert-capu`` = the converted Vietnamese tagger); its own time per
input is ``restore_mean_ms``/``restore_p95_ms``. ``restore=off`` is the input
as it is.

Data (never downloaded by this script):

* FLEURS (``google/fleurs``, CC-BY-4.0): ``--fleurs-src-dir`` holds the source
  language's ``test.tsv`` + ``test/<file>.wav``; ``--fleurs-tgt-tsv`` is the
  target language's ``test.tsv``. Column 0 is the FLoRes sentence id, shared
  by every language; pairs are the ids present on both sides, sorted by id.
  Source and reference are the raw transcription column (cased, punctuated).
  One recording per id: the smallest file name.
* ``--custom-tsv``: hand-written sentences (names, numbers, dates, units,
  loanwords), text conditions only. Rows are labelled
  ``corpus=custom_unreviewed`` until every sentence has ``reviewed=yes``.

Scores are sacrebleu corpus BLEU (``tokenize`` from the target language's row
in ``languages.py``; 13a for English) and chrF2 (``nc=6, nw=0``), one
reference, mixed case; ``bleu_lc``/``chrf_lc`` are the same scores ignoring
case (the app's output is speech, where case is not heard); the signatures are
printed. A failed call (Google
blocked…) counts in ``n_failed`` and that sentence is left out of the scores of
that row only; ``n_sents`` is the number scored. ``src_wer`` is the WER of the
translator's input against the normalised source. Latency times each
``translate`` call (perf_counter) after one untimed warm-up call per backend.

    uv run --group eval python eval/mt_bench.py --src vi --tgt en \\
        --fleurs-src-dir ~/data/fleurs_vi --fleurs-tgt-tsv ~/data/fleurs_en/test.tsv \\
        --custom-tsv eval/data/mt_custom_vi_en.tsv --asr-cache results/mt_asr_cache_vi.tsv \\
        --out results/mt_vi_en.csv
    uv run --group eval python eval/mt_bench.py --smoke --out /tmp/mt_smoke.csv
"""

from __future__ import annotations

import argparse
import csv
import gc
import sys
import time
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import denoise_bench as db  # noqa: E402

from audio_translator.languages import LANGUAGES, get_language, mt_code, normalize  # noqa: E402
from audio_translator.translate import PivotTranslator, plan_route  # noqa: E402

COLUMNS = ["corpus", "src_lang", "tgt_lang", "backend", "condition", "asr_model", "restore", "n_sents",
           "src_wer", "bleu", "chrf", "bleu_lc", "chrf_lc", "n_inputs", "mean_words", "mean_latency_ms",
           "p95_latency_ms", "restore_mean_ms", "restore_p95_ms", "n_failed"]
SAMPLE_COLUMNS = ["corpus", "id", "backend", "condition", "asr_model", "restore", "source", "input", "hypothesis",
                  "reference"]
CONDITIONS = ("text", "text_norm", "asr", "asr_seg")
FAILED = "<FAILED>"
SEG_SEP = "|||"  # between the segments of one recording, in the ASR cache and the samples file


@dataclass(frozen=True)
class Pair:
    id: str
    src: str
    ref: str
    audio: Path | None = None
    category: str = ""


@dataclass(frozen=True)
class Corpus:
    name: str
    pairs: list[Pair]
    with_audio: bool


# ------------------------------------------------------------------ corpora
def _id_key(sid: str):
    return (0, int(sid), "") if sid.isdigit() else (1, 0, sid)


def _read_fleurs_tsv(path: Path) -> dict[str, tuple[str, list[str]]]:
    """id -> (raw text, wav files); an id whose recordings disagree on the text is an error."""
    out: dict[str, tuple[str, list[str]]] = {}
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        cols = line.split("\t")
        if len(cols) < 4 or not cols[1].endswith(".wav"):
            continue
        sid, wav, raw = cols[0], cols[1], cols[2].strip()
        if sid not in out:
            out[sid] = (raw, [wav])
        elif out[sid][0] != raw:
            raise ValueError(f"{path}:{n}: sentence id {sid} has two different texts")
        else:
            out[sid][1].append(wav)
    return out


def load_fleurs_pairs(src_dir: Path, tgt_tsv: Path, log: Callable[[str], None] = print) -> list[Pair]:
    src_tsv = src_dir / "test.tsv"
    for p in (src_tsv, tgt_tsv):
        if not p.is_file():
            raise FileNotFoundError(f"FLEURS transcript not found: {p}")
    src, tgt = _read_fleurs_tsv(src_tsv), _read_fleurs_tsv(tgt_tsv)
    common = sorted(src.keys() & tgt.keys(), key=_id_key)
    log(f"FLEURS: {len(src)} source ids, {len(tgt)} target ids -> {len(common)} pairs "
        f"({len(src.keys() - tgt.keys())} source-only and {len(tgt.keys() - src.keys())} target-only dropped)")
    if not common:
        raise ValueError(f"no sentence id in common between {src_tsv} and {tgt_tsv}")
    return [Pair(i, src[i][0], tgt[i][0], src_dir / "test" / min(src[i][1])) for i in common]


def load_custom(path: Path) -> tuple[list[Pair], bool]:
    """(pairs in file order, every row reviewed=yes). Lines starting with '#' are comments."""
    lines = [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip() and not ln.startswith("#")]
    reader = csv.DictReader(lines, delimiter="\t", quoting=csv.QUOTE_NONE)
    need = {"id", "src", "ref", "reviewed"}
    if not need <= set(reader.fieldnames or ()):
        raise ValueError(f"{path}: header must have columns {sorted(need)}")
    pairs, reviewed, seen = [], True, set()
    for row in reader:
        sid = (row["id"] or "").strip()
        if not sid or sid in seen:
            raise ValueError(f"{path}: missing or duplicate id {sid!r}")
        seen.add(sid)
        pairs.append(Pair(sid, row["src"].strip(), row["ref"].strip(), category=(row.get("category") or "").strip()))
        reviewed &= row["reviewed"].strip().lower() == "yes"
    if not pairs:
        raise ValueError(f"{path}: no sentences")
    return pairs, reviewed


def subset(pairs: list[Pair], limit: int | None, seed: int) -> list[Pair]:
    """``limit`` pairs drawn with ``seed``, kept in id order (all pairs when no limit)."""
    if not limit or limit >= len(pairs):
        return list(pairs)
    pick = np.random.default_rng(seed).choice(len(pairs), size=limit, replace=False)
    return [pairs[i] for i in sorted(pick)]


# ---------------------------------------------------------------------- ASR
class AsrCache:
    """Transcripts keyed by (model, language, beam, id, audio file); appended as they are made."""

    COLUMNS = ["asr_model", "language", "beam_size", "id", "audio", "text"]

    def __init__(self, path: Path | None) -> None:
        self.path = path
        self._rows: dict[tuple, str] = {}
        if path is not None and path.is_file():
            with open(path, encoding="utf-8", newline="") as f:
                for r in csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE, escapechar="\\"):
                    self._rows[tuple(r[c] for c in self.COLUMNS[:-1])] = r["text"]

    @staticmethod
    def key(model: str, language: str, beam: int, pair: Pair) -> tuple:
        return (model, language, str(beam), pair.id, pair.audio.name if pair.audio else "")

    def get(self, key: tuple) -> str | None:
        return self._rows.get(key)

    def put(self, key: tuple, text: str) -> None:
        self._rows[key] = text
        if self.path is None:
            return
        new = not self.path.is_file()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a", encoding="utf-8", newline="") as f:
            w = csv.writer(f, delimiter="\t", quoting=csv.QUOTE_NONE, escapechar="\\")
            if new:
                w.writerow(self.COLUMNS)
            w.writerow([*key, " ".join(text.split())])


def transcribe_pairs(pairs: list[Pair], model: str, make_recognizer: Callable[[str], Callable[[Pair], str]],
                     cache: AsrCache, language: str, beam: int, log: Callable[[str], None]) -> dict[str, str]:
    """id -> transcript; the recognizer is only built when something is not cached."""
    out, recognize = {}, None
    for k, pair in enumerate(pairs):
        key = AsrCache.key(model, language, beam, pair)
        text = cache.get(key)
        if text is None:
            if recognize is None:
                recognize = make_recognizer(model)
            text = " ".join(recognize(pair).split())
            cache.put(key, text)
            if (k + 1) % 25 == 0:
                log(f"  {model}: {k + 1}/{len(pairs)} transcribed")
        out[pair.id] = text
    return out


# ---------------------------------------------------------------- translation
@dataclass
class Backend:
    name: str
    supports: Callable[[str, str], bool]
    make: Callable[[str, str], object]  # (src, tgt) -> Translator


def build_translator(backend: Backend, src: str, tgt: str) -> tuple[str, object] | None:
    """(row label, translator): direct, or a pivot through English only when there is no direct model."""
    route = plan_route(backend.supports, src, tgt)
    if route is None:
        return None
    if route == "direct":
        return backend.name, backend.make(src, tgt)
    pivot = route.split(":", 1)[1]
    return f"{backend.name}+{route}", PivotTranslator(backend.make(src, pivot), backend.make(pivot, tgt), pivot)


def translate_all(translator, sources: list[str], clock: Callable[[], float] = time.perf_counter):
    """(hypotheses with None for a failed call, per-call seconds, n_failed). Empty input -> "" untimed."""
    hyps, secs, failed = [], [], 0
    for s in sources:
        if not s.strip():
            hyps.append("")
            continue
        t0 = clock()
        try:
            h = translator.translate(s)
        except Exception:
            failed += 1
            hyps.append(None)
            continue
        secs.append(clock() - t0)
        hyps.append(" ".join((h or "").split()))
    return hyps, secs, failed


def translate_segmented(translator, segments: list[list[str]], clock: Callable[[], float] = time.perf_counter):
    """Each sentence's segments translated one call each and joined; a sentence with a failed call is None."""
    flat = [x for segs in segments for x in segs]
    out, secs, failed = translate_all(translator, flat, clock)
    hyps, k = [], 0
    for segs in segments:
        part = out[k:k + len(segs)]
        k += len(segs)
        hyps.append(None if None in part else " ".join(h for h in part if h))
    return hyps, secs, failed


def split_segments(text: str) -> list[str]:
    return [x.strip() for x in text.split(SEG_SEP) if x.strip()]


class Scorer:
    def __init__(self, tgt: str) -> None:
        from sacrebleu.metrics import BLEU, CHRF

        tok = get_language(tgt).bleu_tokenize
        # cased = the standard report; *_lc ignore case, which a listener of the TTS output never hears
        self.metrics = {"bleu": BLEU(tokenize=tok), "chrf": CHRF(),
                        "bleu_lc": BLEU(tokenize=tok, lowercase=True), "chrf_lc": CHRF(lowercase=True)}
        self.scored = False  # sacrebleu signatures need one evaluation first

    def signatures(self) -> str:
        return " | ".join(f"{k}: {m.get_signature()}" for k, m in self.metrics.items())

    def score(self, hyps: list[str | None], refs: list[str]) -> dict[str, float] | None:
        """Corpus scores over the sentences whose call succeeded, plus ``n`` (their count)."""
        ok = [(h, r) for h, r in zip(hyps, refs) if h is not None]
        if not ok:
            return None
        h, r = [p[0] for p in ok], [[p[1] for p in ok]]
        self.scored = True
        return {**{k: m.corpus_score(h, r).score for k, m in self.metrics.items()}, "n": len(ok)}


def _latency(secs: list[float]) -> tuple[str, str]:
    if not secs:
        return "", ""
    a = np.array(secs) * 1000
    return f"{a.mean():.0f}", f"{np.percentile(a, 95):.0f}"


# ---------------------------------------------------------------------- bench
def restore_inputs(restorer: Callable[[str, str], str], segments: list[list[str]], lang: str,
                   clock: Callable[[], float] = time.perf_counter) -> tuple[list[list[str]], list[float]]:
    """Every non-empty segment restored (one call each, timed); empty ones stay empty."""
    out, secs = [], []
    for segs in segments:
        row = []
        for x in segs:
            if not x.strip():
                row.append(x)
                continue
            t0 = clock()
            row.append(restorer(x, lang))
            secs.append(clock() - t0)
        out.append(row)
    return out, secs


def run(corpora: list[Corpus], src: str, tgt: str, backends: list[Backend], conditions: list[str],
        asr_models: list[str], asr_texts: Callable[[str, list[Pair], bool], dict[str, str]],
        emit: Callable[[dict], None], emit_sample: Callable[[dict], None], sample_ids: dict[str, set],
        log: Callable[[str], None] = print,
        restorers: dict[str, Callable[[str, str], str] | None] | None = None) -> int:
    """Emit one row per backend x corpus x condition (x ASR model) x restorer. Returns the number of backends run.

    A restorer (``restore.py``) rewrites casing/punctuation of each translator input before it is
    translated; it runs once per input, shared by every backend, and only on inputs that look like
    ASR output (``text_norm``, ``asr``, ``asr_seg``). ``off`` is the input as it is.
    """
    restorers = restorers or {"off": None}
    scorer = Scorer(tgt)
    asr = {}  # (condition, corpus, model) -> id -> text; one model loaded at a time
    for cond in ("asr", "asr_seg"):
        if cond in conditions:
            for model in asr_models:
                for c in corpora:
                    if c.with_audio:
                        asr[cond, c.name, model] = asr_texts(model, c.pairs, cond == "asr_seg")
                gc.collect()

    runs = {}  # corpus -> [(condition, ASR model, restorer, segments per sentence, restore seconds)]
    for c in corpora:
        base = []
        for cond in conditions:
            if cond == "text":
                base.append((cond, "-", [[p.src] for p in c.pairs]))
            elif cond == "text_norm":
                base.append((cond, "-", [[normalize(p.src, src)] for p in c.pairs]))
            elif c.with_audio:
                split = split_segments if cond == "asr_seg" else lambda t: [t]
                base.extend((cond, m, [split(asr[cond, c.name, m][p.id]) for p in c.pairs]) for m in asr_models)
        runs[c.name] = []
        for cond, model, segments in base:
            for rname, restorer in restorers.items():
                if restorer is None:
                    runs[c.name].append((cond, model, rname, segments, []))
                elif cond != "text":
                    log(f"restore {rname}: {c.name} {cond} {model}")
                    runs[c.name].append((cond, model, rname, *restore_inputs(restorer, segments, src)))

    done = 0
    for backend in backends:
        try:
            built = build_translator(backend, src, tgt)
        except Exception as exc:
            log(f"{backend.name}: skipped, cannot be built ({exc})")
            continue
        if built is None:
            log(f"{backend.name}: skipped, no direct or pivot route for {src}->{tgt}")
            continue
        label, translator = built
        warm = next((p.src for c in corpora for p in c.pairs), "")
        try:
            translator.translate(warm)  # load the model / open the connection, untimed
        except Exception as exc:
            log(f"{label}: warm-up call failed ({exc})")
        done += 1
        for c in corpora:
            refs = [p.ref for p in c.pairs]
            norm_src = [normalize(p.src, src) for p in c.pairs]
            for cond, model, rname, segments, rsecs in runs[c.name]:
                hyps, secs, failed = translate_segmented(translator, segments)
                inputs = [x for segs in segments for x in segs if x.strip()]
                wer = db.error_rates(norm_src, [normalize(" ".join(segs), src) for segs in segments])[0]
                scores = scorer.score(hyps, refs)
                mean_ms, p95_ms = _latency(secs)
                r_mean, r_p95 = _latency(rsecs)
                words = np.mean([len(x.split()) for x in inputs]) if inputs else 0.0
                emit({"corpus": c.name, "src_lang": src, "tgt_lang": tgt, "backend": label, "condition": cond,
                      "asr_model": model, "restore": rname, "n_sents": scores["n"] if scores else 0,
                      "src_wer": f"{wer:.4f}",
                      **{k: f"{scores[k]:.2f}" if scores else "" for k in ("bleu", "chrf", "bleu_lc", "chrf_lc")},
                      "n_inputs": len(inputs), "mean_words": f"{words:.1f}",
                      "mean_latency_ms": mean_ms, "p95_latency_ms": p95_ms,
                      "restore_mean_ms": r_mean, "restore_p95_ms": r_p95, "n_failed": failed})
                for p, segs, h in zip(c.pairs, segments, hyps):
                    if p.id in sample_ids.get(c.name, ()):
                        emit_sample({"corpus": c.name, "id": p.id, "backend": label, "condition": cond,
                                     "asr_model": model, "restore": rname, "source": p.src,
                                     "input": f" {SEG_SEP} ".join(segs),
                                     "hypothesis": FAILED if h is None else h, "reference": p.ref})
    if scorer.scored:
        log(scorer.signatures())
    return done


# ---------------------------------------------------------------- real parts
def default_argos_package() -> Path:
    from urllib.parse import urlparse

    from audio_translator.config import Settings

    s = Settings()
    return Path(s.translation_cache_dir).expanduser() / Path(urlparse(s.argos_vi_en_url).path).stem


def argos_backend(packages: list[Path], beam_size: int = 4) -> Backend:
    """Argos packages already installed (the app's cache by default); never downloads."""
    from audio_translator.translate import OfflineTranslator

    trs = [OfflineTranslator(p, beam_size=beam_size) for p in packages if p.is_dir()]

    def find(a: str, b: str):
        return next((t for t in trs if t.supports(a, b)), None)

    def make(a: str, b: str):
        return find(a, b)

    return Backend("argos", lambda a, b: find(a, b) is not None, make)


def ct2_model_backend(name: str) -> Backend:
    """A converted NLLB/M2M100 model (``scripts/convert_mt_model.py``); one loaded model per pair."""
    from audio_translator.translate import MT_MODELS, CT2MultilingualTranslator

    family = MT_MODELS[name][2]
    return Backend(name, lambda a, b: CT2MultilingualTranslator.supports_family(family, a, b),
                   lambda a, b: CT2MultilingualTranslator(name, a, b))


def google_backend() -> Backend:
    from audio_translator.translate import GoogleTranslator

    return Backend("google", GoogleTranslator.supports,
                   lambda a, b: GoogleTranslator(mt_code(a, "google"), mt_code(b, "google")))


def seg_model_key(model: str, app) -> str:
    """ASR cache name of a segmented run (``app``: Settings): another segmentation is another transcript.

    webrtc without the ASR filter keeps the historical ``seg(vad2,600ms)`` name.
    """
    vad = f"vad{app.vad_aggressiveness}" if app.vad_backend == "webrtc" else f"{app.vad_backend}{app.silero_threshold:g}"
    return f"{model}+seg({vad},{app.silence_ms}ms{',filter' if app.asr_filter else ''})"


def whisper_recognizer(language: str, beam_size: int, segmented=None):
    """Whole-recording transcript (no VAD, no filter), or with ``segmented=Settings`` the app's VAD +
    Segmenter + ASR filter, one transcript per segment joined with SEG_SEP (empty segments dropped)."""
    def make(model: str) -> Callable[[Pair], str]:
        from audio_translator.asr import WhisperASR, filter_from_settings
        from audio_translator.segmenter import vad_from_settings

        asr = WhisperASR(model, language=language, beam_size=beam_size,
                         segment_filter=filter_from_settings(segmented) if segmented is not None else None)
        asr.load()

        def recognize(pair: Pair) -> str:
            import ctranslate2

            ctranslate2.set_random_seed(zlib.crc32(pair.id.encode()))  # sampled fallback: same text every run
            x = np.clip(np.rint(db.read_wav(pair.audio)), -32768, 32767).astype(np.int16)
            if segmented is None:
                return asr.transcribe(x, db.RATE)
            r = db.run_pipeline(x, None, vad_from_settings(segmented), asr, silence_ms=segmented.silence_ms)
            return f" {SEG_SEP} ".join(" ".join(t.split()) for t in r.texts if t.strip())

        return recognize

    return make


# ---------------------------------------------------------------------- smoke
_SMOKE = [  # (vi, en); the toy translator knows every word
    ("Tôi đi học.", "I go to school."),
    ("Hôm nay trời đẹp.", "Today the weather is nice."),
    ("Tôi thích Hà Nội.", "I like Hanoi."),
    ("Bạn đi đâu?", "Where are you going?"),
    ("Trời mưa to.", "It rains heavily."),
    ("Tôi đi Hà Nội.", "I go to Hanoi."),
]
_SMOKE_DICT = {"tôi": "I", "đi": "go", "học": "to school", "hôm": "", "nay": "today", "trời": "the weather",
               "đẹp": "is nice", "thích": "like", "hà": "", "nội": "Hanoi", "bạn": "you", "đâu": "where",
               "mưa": "rains", "to": "heavily"}


class ToyTranslator:
    """Word-by-word dictionary translation (offline stand-in)."""

    def translate(self, text: str) -> str:
        words = (_SMOKE_DICT.get(w, w) for w in normalize(text, "vi").split())
        return " ".join(w for w in words if w)


class FlakyTranslator(ToyTranslator):
    """Fails every third call, like a rate-limited web service."""

    def __init__(self) -> None:
        self.calls = 0

    def translate(self, text: str) -> str:
        self.calls += 1
        if self.calls % 3 == 0:
            raise ConnectionError("blocked")
        return super().translate(text)


class DownTranslator:
    def translate(self, text: str) -> str:
        raise ConnectionError("no network")


def smoke_parts():
    fleurs = [Pair(str(1000 + i), vi, en, Path(f"{1000 + i}.wav")) for i, (vi, en) in enumerate(_SMOKE)]
    custom = [Pair(f"c{i}", vi, en, category="toy") for i, (vi, en) in enumerate(_SMOKE[:3])]
    always = lambda a, b: (a, b) == ("vi", "en")  # noqa: E731
    backends = [Backend("toy", always, lambda a, b: ToyTranslator()),
                Backend("flaky", always, lambda a, b: FlakyTranslator()),
                Backend("down", always, lambda a, b: DownTranslator())]

    def make_recognizer(model: str, segmented: bool = False):
        drop = 1 if model == "toy-asr-bad" else 0

        def recognize(pair: Pair) -> str:  # lowercase, no punctuation; "bad" also loses the last word
            words = normalize(pair.src, "vi").split()
            words = words[:len(words) - drop]
            if not segmented:
                return " ".join(words)
            return f" {SEG_SEP} ".join(" ".join(words[i:i + 2]) for i in range(0, len(words), 2))  # 2-word segments

        return recognize

    return fleurs, custom, backends, ["toy-asr-good", "toy-asr-bad"], make_recognizer


# ----------------------------------------------------------------------- main
def _csv_list(text: str) -> list[str]:
    return [x.strip() for x in text.split(",") if x.strip()]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Translation quality (BLEU/chrF) and latency per backend x condition.")
    p.add_argument("--src", default="vi", help=f"source language ({', '.join(LANGUAGES)})")
    p.add_argument("--tgt", default="en", help="target language")
    p.add_argument("--fleurs-src-dir", type=Path, help="FLEURS source-language dir (test.tsv + test/*.wav)")
    p.add_argument("--fleurs-tgt-tsv", type=Path, help="FLEURS target-language test.tsv")
    p.add_argument("--custom-tsv", type=Path, help="hand-written sentences (id, category, reviewed, src, ref)")
    p.add_argument("--backends", default="argos,google",
                   help="comma list of argos, google, nllb-600m, m2m100-418m (converted models only)")
    p.add_argument("--conditions", default=",".join(CONDITIONS))
    p.add_argument("--seg-vad", choices=["webrtc", "silero"], default=None,
                   help="asr_seg: VAD of the segmentation (default: the app's Settings)")
    p.add_argument("--seg-filter", choices=["off", "on"], default=None,
                   help="asr_seg: the app's ASR filter (default: the app's Settings)")
    p.add_argument("--asr-models", default="small,phowhisper-small")
    p.add_argument("--restore", default="off",
                   help="comma list of off, sent, vibert-capu (converted models only): casing/punctuation "
                        "restoration of ASR-like inputs before translation")
    p.add_argument("--asr-cache", type=Path, help="TSV of transcripts reused across runs")
    p.add_argument("--beam-size", type=int, default=5, help="ASR beam (default 5, as in asr_bench)")
    p.add_argument("--argos-package", type=Path, action="append",
                   help="installed Argos package dir (repeatable; default: the app's vi->en package)")
    p.add_argument("--limit", type=int, default=0, help="FLEURS pairs to draw with --seed (0 = all)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-samples", type=int, default=20, help="FLEURS sentences in the samples file")
    p.add_argument("--out", type=Path, default=Path("results/mt_bench.csv"))
    p.add_argument("--samples-out", type=Path, help="default: <out stem>_samples.tsv")
    p.add_argument("--smoke", action="store_true", help="offline run on toy data and toy translators")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    for code in (args.src, args.tgt):
        if code not in LANGUAGES:
            print(f"error: unknown language '{code}' (known: {', '.join(LANGUAGES)})", file=sys.stderr)
            return 2
    if args.src == args.tgt:
        print("error: --src and --tgt are the same", file=sys.stderr)
        return 2
    conditions = _csv_list(args.conditions)
    if not conditions or any(c not in CONDITIONS for c in conditions):
        print(f"error: --conditions must be a subset of {','.join(CONDITIONS)}", file=sys.stderr)
        return 2
    log = lambda s: print(s, flush=True)  # noqa: E731
    src_lang = get_language(args.src)
    from audio_translator.restore import RESTORE_MODELS, RESTORERS, CapuRestorer, make_restorer, restore_model_dir

    wanted_restore = _csv_list(args.restore) or ["off"]
    bad = [r for r in wanted_restore if r not in RESTORERS]
    if bad:
        print(f"error: unknown restorer(s) {bad} (use {', '.join(RESTORERS)})", file=sys.stderr)
        return 2
    restorers = {}
    for r in wanted_restore:
        if r in RESTORE_MODELS:
            if not CapuRestorer(r).supports(args.src):
                log(f"restore {r}: skipped, no {args.src} support")
                continue
            if not args.smoke and not (restore_model_dir(r) / "model.bin").is_file():
                from audio_translator.restore import restore_convert_command

                log(f"restore {r}: skipped, not converted (run: {restore_convert_command(r)})")
                continue
            if args.smoke:
                log(f"restore {r}: skipped in --smoke (needs the converted model)")
                continue
        restorers[r] = make_restorer(r)

    if args.smoke:
        fleurs, custom, backends, asr_models, toy_recognizer = smoke_parts()
        corpora = [Corpus("smoke", fleurs, True), Corpus("custom_unreviewed", custom, False)]
        cache = AsrCache(None)

        def make_recognizer(model: str, segmented: bool):
            return toy_recognizer(model, segmented)

        def asr_key(model: str, segmented: bool) -> str:
            return f"{model}+seg" if segmented else model
    else:
        missing = [f"{flag} {path}" if path else f"{flag} (not given)"
                   for flag, path, ok in (("--fleurs-src-dir", args.fleurs_src_dir, Path.is_dir),
                                          ("--fleurs-tgt-tsv", args.fleurs_tgt_tsv, Path.is_file))
                   if path is None or not ok(path.expanduser())]
        if args.custom_tsv is not None and not args.custom_tsv.expanduser().is_file():
            missing.append(f"--custom-tsv {args.custom_tsv}")
        if missing:
            print("error: missing data: " + "; ".join(missing) + "\n(use --smoke for an offline run without data)",
                  file=sys.stderr)
            return 2
        asr_models = _csv_list(args.asr_models) if {"asr", "asr_seg"} & set(conditions) else []
        try:
            pairs = subset(load_fleurs_pairs(args.fleurs_src_dir.expanduser(), args.fleurs_tgt_tsv.expanduser(), log),
                           args.limit or None, args.seed)
            corpora = [Corpus(f"fleurs_{src_lang.fleurs}", pairs, True)]
            if args.custom_tsv is not None:
                cpairs, reviewed = load_custom(args.custom_tsv.expanduser())
                corpora.append(Corpus("custom" if reviewed else "custom_unreviewed", cpairs, False))
            if asr_models:
                gone = [p.audio for p in pairs if not p.audio.is_file()]
                if gone:
                    raise FileNotFoundError(f"{len(gone)} FLEURS wav(s) missing, e.g. {gone[0]}")
                from audio_translator.asr import resolve_model

                for m in asr_models:
                    resolve_model(m)  # a PhoWhisper model that is not converted fails here, with the command
        except (FileNotFoundError, ValueError, RuntimeError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        from audio_translator.translate import MT_MODELS, mt_convert_command, mt_model_dir

        wanted = _csv_list(args.backends)
        known = ("argos", "google", *MT_MODELS)
        unknown = [b for b in wanted if b not in known]
        if unknown:
            print(f"error: unknown backend(s) {unknown} (use {', '.join(known)})", file=sys.stderr)
            return 2
        backends = []
        for b in wanted:
            if b == "argos":
                pkgs = [p.expanduser() for p in (args.argos_package or [default_argos_package()])]
                log(f"argos packages: {[str(p) for p in pkgs if p.is_dir()] or 'none installed'}")
                backends.append(argos_backend(pkgs))
            elif b in MT_MODELS:
                if not (mt_model_dir(b) / "model.bin").is_file():
                    log(f"{b}: skipped, not converted (run: {mt_convert_command(b)})")
                    continue
                backends.append(ct2_model_backend(b))
            else:
                backends.append(google_backend())
        from audio_translator.config import Settings

        app = Settings(**({"vad_backend": args.seg_vad} if args.seg_vad else {}),
                       **({"asr_filter": args.seg_filter == "on"} if args.seg_filter else {}))
        recognizers = {False: whisper_recognizer(src_lang.whisper, args.beam_size),
                       True: whisper_recognizer(src_lang.whisper, args.beam_size, segmented=app)}
        cache = AsrCache(args.asr_cache)

        def make_recognizer(model: str, segmented: bool):
            return recognizers[segmented](model)

        def asr_key(model: str, segmented: bool) -> str:
            return seg_model_key(model, app) if segmented else model

    samples_out = args.samples_out or args.out.with_name(args.out.stem + "_samples.tsv")
    first = corpora[0]
    sample_ids = {c.name: {p.id for p in c.pairs} for c in corpora[1:]}  # hand-written sets: every sentence
    sample_ids[first.name] = {p.id for p in subset(first.pairs, args.n_samples, args.seed)}
    log(f"{args.src}->{args.tgt}: " + ", ".join(f"{c.name} {len(c.pairs)}" for c in corpora)
        + f"; backends {[b.name for b in backends]}; conditions {conditions}; ASR {asr_models}"
        + f"; restore {list(restorers)}")

    def asr_texts(model: str, pairs: list[Pair], segmented: bool) -> dict[str, str]:
        key = asr_key(model, segmented)
        log(f"ASR {key} on {len(pairs)} recordings")
        return transcribe_pairs(pairs, key, lambda _: make_recognizer(model, segmented), cache,
                                src_lang.whisper, args.beam_size, log)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    rows = 0
    with open(args.out, "w", newline="", encoding="utf-8") as f, \
            open(samples_out, "w", newline="", encoding="utf-8") as sf:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        sw = csv.DictWriter(sf, fieldnames=SAMPLE_COLUMNS, delimiter="\t")
        sw.writeheader()

        def emit(row: dict) -> None:
            nonlocal rows
            w.writerow(row)
            f.flush()
            rows += 1
            log("  ".join(f"{c}={row[c]}" for c in COLUMNS))

        def emit_sample(row: dict) -> None:
            sw.writerow(row)

        done = run(corpora, args.src, args.tgt, backends, conditions, asr_models, asr_texts, emit,
                   emit_sample, sample_ids, log, restorers)
    if done == 0:
        print("error: no backend could be run", file=sys.stderr)
        return 1
    print(f"wrote {rows} rows to {args.out} and samples to {samples_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
