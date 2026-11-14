"""ASR bench: WER/CER and ASR latency per model x initial prompt x SNR.

Same clip and scoring protocol as ``eval/denoise_bench.py`` (noise warm-up,
fades, SNR over the utterance span, a fresh VAD per clip, only the segments
that overlap the utterance are scored), without a noise-suppression stage:
real webrtcvad segmentation + faster-whisper, exactly as the app runs it.

Corpora (never downloaded by this script):

* VIVOS test: ``--vivos-dir`` (``test/prompts.txt`` + ``test/waves/``);
  utterances taken round-robin over speakers. PhoWhisper was fine-tuned on
  the VIVOS *train* set, so VIVOS test is in-domain for it.
* FLEURS Vietnamese test (``google/fleurs`` ``vi_vn``): ``--fleurs-dir`` with
  ``test.tsv`` + ``test/<file>.wav``; one recording per distinct sentence
  (rotating over the recordings of each sentence); reference = FLEURS's own
  normalised ``transcription`` column. Not among PhoWhisper's
  listed training sources: the out-of-domain check.

Latency columns time ``transcribe`` on the scored segments: mean and P95 per
segment, and RTF = total ASR time / total segment audio. ``load_s`` is the
one-off model load.

Hallucination columns (``--vads`` webrtc/silero, ``--filters`` off/on = the
app's ``--asr-filter``): ``vad_false_triggers`` = segments in the scored noise
window of the matching noise-only clip, ``ghost_texts`` = how many of those
still produce text after the filter (what would be translated and spoken),
``vad_extra_segs`` = segments of the speech clip outside the utterance. The
filter modes of one prompt x VAD share each Whisper call (``memoize_segments``),
so their rows differ only by the filter. One CSV row is appended as soon as it is measured, so a
long run that stops keeps what it finished.

    uv run --group eval python eval/asr_bench.py --corpus vivos --vivos-dir ~/data/vivos \\
        --noise-dir ~/data/demand_ch01 --out results/asr_vivos_demand.csv
    uv run --group eval python eval/asr_bench.py --smoke --out /tmp/asr_smoke.csv
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import sys
import time
import wave
from pathlib import Path
from typing import Callable

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import denoise_bench as db  # noqa: E402

DEFAULT_MODELS = ["small", "phowhisper-small", "phowhisper-medium", "large-v3-turbo"]
DEFAULT_PROMPT = "Xin chào, sau đây là đoạn hội thoại bằng tiếng Việt."
PROMPT_MODES = ("off", "on")
VAD_NAMES = ("webrtc", "silero")
FILTER_MODES = ("off", "on")
COLUMNS = ["corpus", "noise", "model", "prompt", "vad", "filter", "decode", "snr_db", "n_utts", "wer", "cer",
           "asr_ms_mean", "asr_ms_p95", "asr_rtf", "load_s", "vad_extra_segs", "vad_false_triggers",
           "ghost_texts", "asr_ms_max", "noise_segs", "noise_asr_ms_mean", "noise_asr_ms_p95", "noise_asr_ms_max",
           "fallback_segs", "noise_fallback_segs", "noise_redecodes", "cap_hits", "max_tok_per_s",
           "asr_ms_p50", "noise_asr_ms_p50", "redecodes", "budget_skips", "noise_budget_skips",
           "t0_rejected", "noise_t0_rejected"]
# Decoding columns (bounded decoding, plan-speaker-indicator.md), from DecodeProbe (one entry per
# CTranslate2 generate call = one decoding attempt): ``decode`` = the variant label;
# ``asr_ms_max`` over scored speech segments; ``noise_*`` = ASR time over every segment of the
# noise-only clips (where stalls happen); ``fallback_segs`` / ``noise_fallback_segs`` = segments
# that needed more than one attempt (temperature fallback); ``noise_redecodes`` = extra attempts on
# noise; ``cap_hits`` = scored speech segments with an attempt that reached its token limit (the cap,
# or Whisper's 224 without one: a cut-off sentence when it is real speech); ``max_tok_per_s`` =
# generated tokens of the last attempt per second of audio, maximum over scored speech segments.


# ------------------------------------------------------------------ corpora
def load_fleurs(root: Path, limit: int | None) -> list[tuple[np.ndarray, str]]:
    """FLEURS ``test.tsv`` (id, file, raw, normalised transcription, …): one recording per sentence id."""
    tsv = root / "test.tsv"
    if not tsv.is_file():
        raise FileNotFoundError(f"FLEURS transcript not found: {tsv}")
    by_sentence: dict[str, list[tuple[str, str]]] = {}
    for line in tsv.read_text(encoding="utf-8").splitlines():
        cols = line.split("\t")
        if len(cols) >= 4 and cols[1].endswith(".wav"):
            by_sentence.setdefault(cols[0], []).append((cols[1], cols[3]))
    utts = []
    for j, recs in enumerate(by_sentence.values()):
        if limit and len(utts) >= limit:
            break
        file, text = recs[j % len(recs)]  # rotate speakers across sentences
        wav = root / "test" / file
        if not wav.is_file():
            raise FileNotFoundError(f"FLEURS wav missing: {wav}")
        utts.append((db.read_wav(wav), text))
    if not utts:
        raise FileNotFoundError(f"no utterances listed in {tsv}")
    return utts


# --------------------------------------------------------------------- timing
class DecodeProbe:
    """Stands in for ``WhisperModel.model`` (CTranslate2) and records every ``generate`` call:
    (seconds, temperature, generated tokens, token limit). Observation only."""

    def __init__(self, inner) -> None:
        self._inner = inner
        self.calls: list[tuple[float, float, int, int]] = []

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def generate(self, *args, **kw):
        from audio_translator.asr import real_token_limit

        t0 = time.perf_counter()
        out = self._inner.generate(*args, **kw)
        # CTranslate2's Whisper generates at most max_length // 2 tokens (plan-asr-anti-stall.md step 0;
        # before 2026-11-14 this used max_length - prompt and so under-counted cap hits).
        limit = real_token_limit(kw.get("max_length", 448))
        n = len(out[0].sequences_ids[0]) if out and out[0].sequences_ids else 0
        self.calls.append((time.perf_counter() - t0, float(kw.get("sampling_temperature", 0.0) or 0.0), n, limit))
        return out


def install_decode_probe(asr) -> DecodeProbe | None:
    """After ``asr.load()``: record each decoding attempt; ``asr.last_decode`` = attempts of the
    last ``transcribe_segments`` call. None for recognizers without a CTranslate2 model (toys)."""
    model = getattr(getattr(asr, "_model", None), "model", None)
    if model is None or not hasattr(model, "generate"):
        return None
    from audio_translator.asr import DecodeLimiter

    if isinstance(model, DecodeLimiter):  # record the real decodes, after the limiter's changes
        probe = DecodeProbe(model._inner)
        model._inner = probe
    else:
        probe = DecodeProbe(model)
        asr._model.model = probe
    real = asr.transcribe_segments

    def probed(audio, sample_rate):
        probe.calls.clear()
        segs = real(audio, sample_rate)
        asr.last_decode = list(probe.calls)
        return segs

    asr.transcribe_segments = probed
    asr.last_decode = None
    return probe


class TimedASR:
    """Wraps a recognizer and records (seconds, audio seconds, decoding attempts) per transcribe call.

    When the recognizer's segments come from ``memoize_segments``, the time is
    that of the real Whisper call (a replay costs nothing).
    """

    def __init__(self, asr) -> None:
        self.asr = asr
        self.calls: list[tuple[float, float]] = []

    def transcribe(self, audio: np.ndarray, sample_rate: int) -> str:
        if hasattr(self.asr, "last_call_s"):
            self.asr.last_call_s = None
        if hasattr(self.asr, "last_decode"):
            self.asr.last_decode = None
        t0 = time.perf_counter()
        text = self.asr.transcribe(audio, sample_rate)
        secs = getattr(self.asr, "last_call_s", None)
        lim = getattr(self.asr, "limiter", None)
        self.calls.append((time.perf_counter() - t0 if secs is None else secs, audio.size / sample_rate,
                           getattr(self.asr, "last_decode", None) or [],
                           lim.budget_skips if lim is not None and secs is None else 0,
                           lim.t0_rejected if lim is not None and secs is None else ""))
        return text


def memoize_segments(asr) -> dict | None:
    """Make ``asr.transcribe_segments`` replay segments of audio it already recognised (same prompt).

    The ASR filter only drops segments, so filter off/on share one Whisper call
    per audio segment: identical text before filtering (no decoding
    nondeterminism between the two rows) and half the run time. The real
    call's duration is kept in ``asr.last_call_s``. Returns the memo (clear it to
    forget), or None when the recognizer has no ``transcribe_segments``.
    """
    real = getattr(asr, "transcribe_segments", None)
    if real is None:
        return None
    memo: dict = {}

    def cached(audio: np.ndarray, sample_rate: int):
        key = (hashlib.sha1(np.ascontiguousarray(audio).tobytes()).digest(), sample_rate, asr.initial_prompt)
        if key not in memo:
            t0 = time.perf_counter()
            segs = real(audio, sample_rate)
            memo[key] = (segs, time.perf_counter() - t0, getattr(asr, "last_decode", None))
        segs, asr.last_call_s, decode = memo[key]
        if decode is not None:
            asr.last_decode = decode
        return segs

    asr.transcribe_segments = cached
    asr.last_call_s = None
    return memo


def seed_decoder(seed: int) -> None:
    """Fix CTranslate2's RNG: faster-whisper falls back to sampling (temperature > 0) on hard
    audio, so without a seed a noisy row's WER changes from run to run."""
    try:
        import ctranslate2
    except ImportError:  # --smoke without the ASR stack
        return
    ctranslate2.set_random_seed(seed)


def latency_stats(calls: list[tuple[float, float]]) -> tuple[float, float, float]:
    """(mean ms, P95 ms, RTF) over transcribe calls; zeros when there are none."""
    if not calls:
        return 0.0, 0.0, 0.0
    secs = np.array([c[0] for c in calls])
    audio = sum(c[1] for c in calls)
    return float(secs.mean() * 1000), float(np.percentile(secs, 95) * 1000), float(secs.sum() / audio) if audio else 0.0


SEG_COLUMNS = ["decode", "snr_db", "clip", "kind", "audio_s", "asr_ms", "attempts", "text", "ref", "t0_rejected"]


def _seg_row(snr, k: int, kind: str, call: tuple, text: str, ref: str) -> dict:
    att = call[2] if len(call) > 2 else []
    return {"snr_db": "clean" if snr is None else f"{snr:g}", "clip": k, "kind": kind, "audio_s": f"{call[1]:.2f}",
            "asr_ms": f"{call[0] * 1000:.0f}",
            "attempts": " ".join(f"t{t:g}:{n}/{lim}:{s * 1000:.0f}ms" for s, t, n, lim in att),
            "text": text.strip(), "ref": ref, "t0_rejected": call[4] if len(call) > 4 else ""}


def decode_stats(speech: list[tuple], noise: list[tuple]) -> dict:
    """Bounded-decoding columns from TimedASR calls (seconds, audio seconds, attempts)."""
    def attempts(c):
        return c[2] if len(c) > 2 else []

    noise_ms = np.array([c[0] * 1000 for c in noise]) if noise else np.zeros(0)
    rates = [att[-1][2] / c[1] for c in speech if (att := attempts(c)) and c[1] > 0]
    return {
        "asr_ms_max": f"{max((c[0] for c in speech), default=0.0) * 1000:.0f}",
        "noise_segs": len(noise),
        "noise_asr_ms_mean": f"{noise_ms.mean() if noise_ms.size else 0.0:.0f}",
        "noise_asr_ms_p95": f"{np.percentile(noise_ms, 95) if noise_ms.size else 0.0:.0f}",
        "noise_asr_ms_max": f"{noise_ms.max() if noise_ms.size else 0.0:.0f}",
        "fallback_segs": sum(len(attempts(c)) > 1 for c in speech),
        "noise_fallback_segs": sum(len(attempts(c)) > 1 for c in noise),
        "noise_redecodes": sum(max(0, len(attempts(c)) - 1) for c in noise),
        "cap_hits": sum(any(n >= lim for _s, _t, n, lim in attempts(c)) for c in speech),
        "max_tok_per_s": f"{max(rates, default=0.0):.2f}",
        "asr_ms_p50": f"{np.percentile([c[0] * 1000 for c in speech], 50) if speech else 0.0:.0f}",
        "noise_asr_ms_p50": f"{np.percentile(noise_ms, 50) if noise_ms.size else 0.0:.0f}",
        "redecodes": sum(max(0, len(attempts(c)) - 1) for c in speech),
        "budget_skips": sum(c[3] if len(c) > 3 else 0 for c in speech),
        "noise_budget_skips": sum(c[3] if len(c) > 3 else 0 for c in noise),
        "t0_rejected": sum(bool(c[4]) if len(c) > 4 else 0 for c in speech),
        "noise_t0_rejected": sum(bool(c[4]) if len(c) > 4 else 0 for c in noise),
    }


# ---------------------------------------------------------------------- bench
def bench_model(utts, noises, asr, make_vad: Callable, *, corpus: str, noise_name: str, model: str,
                prompt: str | None, snrs, seed: int, load_s: float, emit: Callable[[dict], None],
                vad_name: str = "webrtc", segment_filter=None, decode: str = "base",
                seg_log: Callable[[dict], None] | None = None) -> None:
    asr.initial_prompt = prompt
    asr.segment_filter = segment_filter
    for snr in snrs:
        rng = np.random.default_rng(seed)  # identical clips for every model and prompt
        refs, hyps, scored, noise_calls = [], [], [], []
        extra = false_triggers = ghosts = 0
        for k, (speech, ref) in enumerate(utts):
            clip = db.build_clip(speech, noises[k % len(noises)], snr, rng)
            timed = TimedASR(asr)
            seed_decoder(seed + k)  # per clip, so a row does not depend on the rows before it
            r = db.run_pipeline(clip.noisy, None, make_vad(), timed)
            refs.append(db.normalize_vi(ref))
            hyps.append(db.normalize_vi(r.text_in(clip.speech_s)))
            for (a, b, text), call in zip(r.segments, timed.calls):
                if db.overlaps((a, b), clip.speech_s):
                    scored.append(call)
                else:
                    extra += 1
                if seg_log is not None:
                    seg_log(_seg_row(snr, k, "speech" if db.overlaps((a, b), clip.speech_s) else "extra",
                                     call, text, ref))
            seed_decoder(seed + k)
            timed_noise = TimedASR(asr)
            noise_run = db.run_pipeline(clip.noise_only, None, make_vad(), timed_noise)
            noise_calls += timed_noise.calls
            if seg_log is not None:
                for (_a, _b, text), call in zip(noise_run.segments, timed_noise.calls):
                    seg_log(_seg_row(snr, k, "noise", call, text, ""))
            for a, b, text in noise_run.segments:
                if db.overlaps((a, b), clip.noise_window_s):
                    false_triggers += 1
                    ghosts += bool(text.strip())
        wer, cer = db.error_rates(refs, hyps)
        mean_ms, p95_ms, rtf = latency_stats(scored)
        emit({"corpus": corpus, "noise": noise_name if snr is not None else "-", "model": model,
              "prompt": "on" if prompt else "off", "vad": vad_name,
              "filter": "on" if segment_filter is not None else "off", "decode": decode, "snr_db": "clean" if snr is None else f"{snr:g}",
              "n_utts": len(utts), "wer": f"{wer:.4f}", "cer": f"{cer:.4f}", "asr_ms_mean": f"{mean_ms:.0f}",
              "asr_ms_p95": f"{p95_ms:.0f}", "asr_rtf": f"{rtf:.3f}", "load_s": f"{load_s:.1f}",
              "vad_extra_segs": extra, "vad_false_triggers": false_triggers, "ghost_texts": ghosts,
              **decode_stats(scored, noise_calls)})


def run(utts, noises, models: list[str], make_asr: Callable[[str], object], make_vad: Callable, *,
        corpus: str, noise_name: str, prompts: list[str | None], snrs, seed: int,
        emit: Callable[[dict], None], log=print, vads: tuple[str, ...] = ("webrtc",),
        filters: tuple = (None,), decode: str = "base", seg_log: Callable[[dict], None] | None = None) -> int:
    """``make_vad(name)`` builds a fresh VAD; ``filters`` holds SegmentFilter or None (= off)."""
    """Benchmark each model in turn (one resident at a time); returns the number of models measured."""
    done = 0
    for model in models:
        t0 = time.perf_counter()
        try:
            asr = make_asr(model)
            asr.load()
        except Exception as exc:  # e.g. a PhoWhisper model that is not converted yet
            log(f"skipping model '{model}': {exc}")
            continue
        load_s = time.perf_counter() - t0
        install_decode_probe(asr)
        memo = memoize_segments(asr)
        for prompt in prompts:
            for vad_name in vads:
                if memo is not None:
                    memo.clear()  # shared by the filter modes of one prompt x VAD only
                for seg_filter in filters:
                    bench_model(utts, noises, asr, lambda v=vad_name: make_vad(v), corpus=corpus,
                                noise_name=noise_name, model=model, prompt=prompt, snrs=snrs, seed=seed,
                                load_s=load_s, emit=emit, vad_name=vad_name, segment_filter=seg_filter,
                                decode=decode, seg_log=seg_log)
        done += 1
        del asr
        gc.collect()  # free the model before loading the next one (8 GB machine)
    return done


# ----------------------------------------------------------------------- main
class _PromptToy(db.ToyASR):
    initial_prompt: str | None = None
    segment_filter = None

    def load(self) -> None:
        pass


def _snrs(text: str) -> list[float | None]:
    out: list[float | None] = []
    for part in text.split(","):
        part = part.strip().lower()
        out.append(None if part == "clean" else float(part))
    return out


def decode_label(per_s: float | None, base: int, temps: tuple[float, ...] | None, fallback_per_s=None,
                 first_min=None, budget_s=None, tail=False, reject="") -> str:
    from audio_translator.asr import decode_label as label

    return label(per_s, base, temps, fallback_per_s=fallback_per_s, first_min=first_min, budget_s=budget_s,
                 tail=tail, reject=reject)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="ASR bench: WER/CER and latency vs model, prompt and SNR.")
    p.add_argument("--smoke", action="store_true", help="synthetic signals + toy ASR, fully offline")
    p.add_argument("--corpus", choices=["vivos", "fleurs"], default="vivos")
    p.add_argument("--vivos-dir", type=Path, help="VIVOS root (contains test/prompts.txt, test/waves/)")
    p.add_argument("--fleurs-dir", type=Path, help="FLEURS vi_vn dir (contains test.tsv, test/*.wav)")
    p.add_argument("--noise-dir", type=Path, help="directory of noise .wav files")
    p.add_argument("--models", default=",".join(DEFAULT_MODELS),
                   help=f"comma-separated model names (default: {','.join(DEFAULT_MODELS)})")
    p.add_argument("--prompt", default=DEFAULT_PROMPT, help="initial prompt used for prompt=on")
    p.add_argument("--prompt-modes", default="off,on", help="comma-separated subset of off,on")
    p.add_argument("--vads", default="webrtc", help="comma-separated subset of webrtc,silero")
    p.add_argument("--silero-threshold", type=float, default=0.5)
    p.add_argument("--filters", default="off", help="comma-separated subset of off,on (ASR filter)")
    p.add_argument("--snrs", type=_snrs, default=list(db.SNRS),
                   help="comma-separated SNRs in dB and/or 'clean' (default: clean,20,10,5,0)")
    p.add_argument("--limit", type=int, default=100, help="max utterances (0 = all; default 100)")
    p.add_argument("--noise-files", type=int, default=20, help="noise files sampled (default 20)")
    p.add_argument("--beam-size", type=int, default=5)
    p.add_argument("--vad-aggressiveness", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-tokens-per-s", type=float, default=None,
                   help="bounded decoding: cap output at ceil(audio_s x N) + --max-tokens-base tokens (default: off)")
    p.add_argument("--max-tokens-base", type=int, default=32)
    p.add_argument("--temperatures", default=None,
                   help="bounded decoding: temperature ladder, e.g. 0,0.2,0.4 (default: faster-whisper's 6)")
    p.add_argument("--fallback-max-tokens-per-s", type=float, default=None,
                   help="anti-stall (C): cap only fallback attempts at ceil(audio_s x N) + base tokens")
    p.add_argument("--first-min-tokens", type=int, default=None,
                   help="anti-stall (C+): temperature-0 limit max(N, per-second cap), at most 224")
    p.add_argument("--decode-budget-s", type=float, default=None,
                   help="anti-stall (C2): no more fallback attempts once a segment took S seconds")
    p.add_argument("--cap-tail-strip", action="store_true",
                   help="anti-stall (B): strip a junk tail when a capped attempt reached its limit")
    p.add_argument("--reject-truncated-t0", action="store_true",
                   help="anti-stall (C3a): retry when the temperature-0 attempt stopped at its token limit")
    p.add_argument("--reject-loop-t0", action="store_true",
                   help="anti-stall (C3b): retry when the temperature-0 text has a dot run or a repeated 4-word group")
    p.add_argument("--decode-label", default=None, help="value of the decode column (default: derived)")
    p.add_argument("--segments-out", type=Path, default=None,
                   help="also write one TSV row per ASR segment: time, decoding attempts, text, reference")
    p.add_argument("--out", type=Path, default=Path("asr_bench.csv"), help="CSV output path")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    modes = [m.strip() for m in args.prompt_modes.split(",") if m.strip()]
    if not modes or any(m not in PROMPT_MODES for m in modes):
        print(f"error: --prompt-modes must be a subset of {','.join(PROMPT_MODES)}", file=sys.stderr)
        return 2
    if "on" in modes and not args.prompt.strip():
        print("error: --prompt is empty (prompt=on would run without a prompt)", file=sys.stderr)
        return 2
    prompts = [args.prompt if m == "on" else None for m in modes]
    vads = tuple(v.strip() for v in args.vads.split(",") if v.strip())
    filter_modes = [f.strip() for f in args.filters.split(",") if f.strip()]
    if not vads or any(v not in VAD_NAMES for v in vads):
        print(f"error: --vads must be a subset of {','.join(VAD_NAMES)}", file=sys.stderr)
        return 2
    if not filter_modes or any(f not in FILTER_MODES for f in filter_modes):
        print(f"error: --filters must be a subset of {','.join(FILTER_MODES)}", file=sys.stderr)
        return 2
    from audio_translator.asr import SegmentFilter
    from audio_translator.segmenter import silero_vad, webrtc_vad

    filters = tuple(SegmentFilter() if f == "on" else None for f in filter_modes)
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    temps = tuple(float(t) for t in args.temperatures.split(",") if t.strip()) if args.temperatures else None
    decode = args.decode_label or decode_label(args.max_tokens_per_s, args.max_tokens_base, temps,
                                               args.fallback_max_tokens_per_s, args.first_min_tokens,
                                               args.decode_budget_s, args.cap_tail_strip,
                                               ("t0trunc" if args.reject_truncated_t0 else "")
                                               + ("+loop" if args.reject_loop_t0 else ""))
    if args.smoke:
        utts, noises = db.smoke_data(args.seed)
        corpus, noise_name = "smoke", "smoke"
        models = ["toy-a", "toy-b"]

        def make_asr(name):
            return _PromptToy()

        def make_vad(name):
            return db.energy_vad if name == "webrtc" else silero_vad(args.silero_threshold)
    else:
        corpus_dir = args.vivos_dir if args.corpus == "vivos" else args.fleurs_dir
        corpus_flag = "--vivos-dir" if args.corpus == "vivos" else "--fleurs-dir"
        missing = [f"{flag} {path}" if path else f"{flag} (not given)"
                   for flag, path in ((corpus_flag, corpus_dir), ("--noise-dir", args.noise_dir))
                   if path is None or not Path(path).expanduser().is_dir()]
        if missing:
            print("error: missing data directory: " + "; ".join(missing)
                  + "\n(use --smoke for an offline run without data)", file=sys.stderr)
            return 2
        try:
            loader = db.load_vivos if args.corpus == "vivos" else load_fleurs
            utts = loader(corpus_dir.expanduser(), args.limit or None)
            noises = db.load_noises(args.noise_dir.expanduser(), args.noise_files, args.seed)
        except (FileNotFoundError, ValueError, wave.Error) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        corpus, noise_name = args.corpus, args.noise_dir.expanduser().name
        from audio_translator.asr import WhisperASR

        def make_asr(name):
            return WhisperASR(name, language="vi", beam_size=args.beam_size, max_tokens_per_s=args.max_tokens_per_s,
                              max_tokens_base=args.max_tokens_base, temperatures=temps,
                              fallback_max_tokens_per_s=args.fallback_max_tokens_per_s,
                              first_min_tokens=args.first_min_tokens, decode_budget_s=args.decode_budget_s,
                              cap_tail_strip=args.cap_tail_strip, reject_truncated_t0=args.reject_truncated_t0,
                              reject_loop_t0=args.reject_loop_t0)

        def make_vad(name):
            return webrtc_vad(args.vad_aggressiveness) if name == "webrtc" else silero_vad(args.silero_threshold)

    print(f"{corpus}: {len(utts)} utterance(s), {len(noises)} noise source(s), models {models}, "
          f"prompts {modes}, VADs {list(vads)}, filters {filter_modes}, SNRs {['clean' if s is None else s for s in args.snrs]}", flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    rows = 0
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        f.flush()

        def emit(row: dict) -> None:
            nonlocal rows
            w.writerow(row)
            f.flush()
            rows += 1
            print("  ".join(f"{c}={row[c]}" for c in COLUMNS), flush=True)

        seg_f = open(args.segments_out, "w", newline="", encoding="utf-8") if args.segments_out else None
        seg_w = csv.DictWriter(seg_f, fieldnames=SEG_COLUMNS, delimiter="\t") if seg_f else None
        if seg_w:
            seg_w.writeheader()

        def seg_log(row: dict) -> None:
            seg_w.writerow({"decode": decode, **row})
            seg_f.flush()

        try:
            done = run(utts, noises, models, make_asr, make_vad, corpus=corpus, noise_name=noise_name,
                       prompts=prompts, snrs=args.snrs, seed=args.seed, emit=emit,
                       log=lambda s: print(s, flush=True), vads=vads, filters=filters, decode=decode,
                       seg_log=seg_log if seg_w else None)
        finally:
            if seg_f:
                seg_f.close()
    if done == 0:
        print("error: no model could be loaded", file=sys.stderr)
        return 1
    print(f"wrote {rows} rows to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
