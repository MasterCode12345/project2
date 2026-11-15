"""Pre-roll bench (plan-first-word-preroll.md): does keeping more audio before the start vote stop
the first word of an utterance from being cut, and at what cost?

One **continuous stream** per condition, fed frame by frame (30 ms) through the app's real
``Segmenter`` + one ``SileroVad`` for the whole stream (as in a session), each segment recognised
by the app's default ASR (PhoWhisper-small, int8, beam 5, the default segment filter):

* 2 s warm-up, then each utterance preceded by ``--gap-s`` (1.5 s) of gap; after every
  ``--quiet-every`` (10) utterances a ``--quiet-s`` (10 s) stretch with no speech, to count ghosts.
* Every utterance is scaled to the same speech level (RMS ``--level``, peak-limited), like a speaker
  at a fixed distance; the noise (``--noise-dir``: files sampled with ``--seed``, 20 s pieces
  joined with 0.5 s cross-fades) runs at one constant level over the whole stream, set so that the
  SNR over the speech is ``--snrs``. ``clean`` = digital silence between utterances.
* For each ``--pre-rolls`` value (300 = today) the same stream is segmented again. The decoder is
  seeded per segment, so a row does not depend on the rows before it.

Columns per (corpus, noise, SNR, pre-roll):

* ``wer`` / ``cer`` over the utterances (each segment goes to the utterance it overlaps most);
* ``first_lost``: utterances whose first reference word is not among the first 2 words of the
  hypothesis (both normalised); ``also_lost_best``: of those, how many the best-possible run loses too;
* ``best_*``: the **best possible** of that cell: each whole utterance (with its noise, 0.5 s
  either side) transcribed alone, nothing cut by the segmenter;
* ``segs`` (all), ``segs_outside`` (overlapping no utterance), ``ghosts`` (of those, the ones with
  text after the filter: what would be translated and spoken);
* ``asr_ms_p50`` / ``asr_ms_p95`` per segment that overlaps an utterance, ``seg_audio_s`` mean.

``--samples`` writes every utterance whose first word is lost at any pre-roll: reference, best
possible, and the hypothesis per pre-roll. Never downloads anything.

    caffeinate -i uv run --group eval python eval/preroll_bench.py --corpus vivos --vivos-dir ~/data/vivos \\
        --noise-dir ~/data/demand_ch01 --snrs clean,10,5 --out results/preroll_vivos_demand.csv \\
        --samples results/preroll_vivos_demand_samples.tsv
    uv run --group eval python eval/preroll_bench.py --smoke --out /tmp/preroll_smoke.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import asr_bench as ab  # noqa: E402
import denoise_bench as db  # noqa: E402

from audio_translator.segmenter import Segmenter  # noqa: E402

RATE, FRAME = db.RATE, db.FRAME
PRE_ROLLS = (300, 450, 600, 900)
COLUMNS = ["corpus", "noise", "snr_db", "pre_roll_ms", "n_utts", "wer", "cer", "first_lost", "first_lost_rate",
           "also_lost_best", "best_wer", "best_cer", "best_first_lost", "best_first_lost_rate", "segs",
           "segs_outside", "ghosts", "asr_ms_p50", "asr_ms_p95", "seg_audio_s"]
WARMUP_S, PAD_S, PIECE_S, XFADE_S = 2.0, 0.5, 20.0, 0.5


def first_word_lost(ref: str, hyp: str) -> bool:
    r, h = db.normalize_vi(ref).split(), db.normalize_vi(hyp).split()
    return bool(r) and r[0] not in h[:2]


@dataclass
class Stream:
    audio: np.ndarray  # int16
    spans: list[tuple[int, int]]  # utterance sample spans
    quiet: list[tuple[int, int]]  # speech-free stretches


def level_speech(x: np.ndarray, level: float | None) -> np.ndarray:
    x = x.astype(np.float64)
    if level is None:  # the smoke run keeps its levels
        return x
    r, peak = db.rms(x), float(np.max(np.abs(x))) if x.size else 0.0
    if r <= 0 or peak <= 0:
        return x
    return x * min(level / r, 0.9 * 32767 / peak)


def continuous_noise(noises: list[np.ndarray], n: int, rng: np.random.Generator) -> np.ndarray:
    """``n`` samples of noise: 20 s pieces of rotating files joined with raised-cosine cross-fades."""
    piece, xf = int(PIECE_S * RATE), int(XFADE_S * RATE)
    ramp = 0.5 - 0.5 * np.cos(np.pi * np.arange(xf) / xf)
    out = np.zeros(0)
    k = 0
    while out.size < n:
        p = db.noise_for(noises[k % len(noises)].astype(np.float64), piece, rng)
        p = p / max(db.rms(p), 1e-9)
        if out.size:
            out[-xf:] = out[-xf:] * ramp[::-1] + p[:xf] * ramp
            p = p[xf:]
        out = np.concatenate([out, p])
        k += 1
    return out[:n]


def build_stream(utts, noises, snr_db: float | None, rng: np.random.Generator, *, gap_s: float = 1.5,
                 quiet_every: int = 10, quiet_s: float = 10.0, level: float | None = 3000.0) -> Stream:
    parts, spans, quiet = [np.zeros(int(WARMUP_S * RATE))], [], []
    pos = parts[0].size
    for k, (speech, _ref) in enumerate(utts):
        if k and quiet_every and k % quiet_every == 0:
            q = int(quiet_s * RATE)
            parts.append(np.zeros(q))
            quiet.append((pos, pos + q))
            pos += q
        g = int(gap_s * RATE)
        s = level_speech(speech, level)
        parts += [np.zeros(g), s]
        spans.append((pos + g, pos + g + s.size))
        pos += g + s.size
    tail = int(gap_s * RATE)
    parts.append(np.zeros(tail))
    clean = np.concatenate(parts)
    if snr_db is not None:
        noise = continuous_noise(noises, clean.size, rng)
        mask = np.zeros(clean.size, bool)
        for a, b in spans:
            mask[a:b] = True
        scale = db.rms(clean[mask]) / (max(db.rms(noise[mask]), 1e-9) * 10.0 ** (snr_db / 20.0))
        clean = clean + noise * scale
    pad = (-clean.size) % FRAME
    x = np.clip(np.rint(np.concatenate([clean, np.zeros(pad)])), -32768, 32767).astype(np.int16)
    return Stream(x, spans, quiet)


def segment_stream(x: np.ndarray, vad, pre_roll_ms: int, silence_ms: int = 600, max_segment_s: float = 15.0):
    """(start sample, end sample, audio) of every utterance the app's Segmenter cuts from ``x``."""
    seg = Segmenter(vad, sample_rate=RATE, pre_roll_ms=pre_roll_ms, silence_ms=silence_ms,
                    max_segment_s=max_segment_s)
    out = []
    frames = [x[i * FRAME:(i + 1) * FRAME] for i in range(x.size // FRAME)]
    silence = np.zeros(FRAME, np.int16)
    flush = silence_ms // 30 + 2
    i = 0
    while i < len(frames) + flush or (seg.triggered and i < len(frames) + flush + 200):
        f = frames[i] if i < len(frames) else silence
        for u in seg.feed(f, i * FRAME / RATE):  # t = frame start: speech_start = first kept sample
            a = int(round(u.speech_start * RATE))
            out.append((a, a + u.audio.size, u.audio))
        i += 1
    return out


def assign(segments, spans) -> list[int | None]:
    """Index of the utterance each segment overlaps most (None: no overlap)."""
    out = []
    for a, b, *_ in segments:
        best, best_ov = None, 0
        for k, (s, e) in enumerate(spans):
            ov = min(b, e) - max(a, s)
            if ov > best_ov:
                best, best_ov = k, ov
        out.append(best)
    return out


def _transcribe(asr, audio, seed):
    ab.seed_decoder(seed)
    t0 = time.perf_counter()
    text = asr.transcribe(audio, RATE)
    return text, time.perf_counter() - t0


def bench_cell(utts, stream: Stream, asr, make_vad: Callable, pre_rolls, *, corpus: str, noise: str,
               snr_db: float | None, seed: int, emit: Callable[[dict], None], samples: list | None = None,
               log=print):
    refs = [r for _, r in utts]
    nref = [db.normalize_vi(r) for r in refs]
    pad = int(PAD_S * RATE)
    best = []
    for k, (s, e) in enumerate(stream.spans):  # the best possible: whole utterance, nothing cut
        best.append(_transcribe(asr, stream.audio[max(0, s - pad):e + pad], seed + k)[0])
    best_lost = [first_word_lost(r, h) for r, h in zip(refs, best)]
    bw, bc = db.error_rates(nref, [db.normalize_vi(h) for h in best])
    hyps_by_pr = {}
    for pr in pre_rolls:
        segs = segment_stream(stream.audio, make_vad(), pr)
        owner = assign(segs, stream.spans)
        texts, secs = [], []
        for j, (_a, _b, audio) in enumerate(segs):
            text, sec = _transcribe(asr, audio, seed + j)
            texts.append(text)
            if owner[j] is not None:
                secs.append(sec)
        hyps = [" ".join(t for t, o in zip(texts, owner) if o == k) for k in range(len(utts))]
        hyps_by_pr[pr] = hyps
        lost = [first_word_lost(r, h) for r, h in zip(refs, hyps)]
        w, c = db.error_rates(nref, [db.normalize_vi(h) for h in hyps])
        outside = [j for j, o in enumerate(owner) if o is None]
        ms = np.array(secs) * 1000 if secs else np.zeros(1)
        row = {"corpus": corpus, "noise": noise if snr_db is not None else "-",
               "snr_db": "clean" if snr_db is None else f"{snr_db:g}", "pre_roll_ms": pr, "n_utts": len(utts),
               "wer": f"{w:.4f}", "cer": f"{c:.4f}", "first_lost": sum(lost),
               "first_lost_rate": f"{sum(lost) / len(utts):.4f}",
               "also_lost_best": sum(a and b for a, b in zip(lost, best_lost)),
               "best_wer": f"{bw:.4f}", "best_cer": f"{bc:.4f}", "best_first_lost": sum(best_lost),
               "best_first_lost_rate": f"{sum(best_lost) / len(utts):.4f}", "segs": len(segs),
               "segs_outside": len(outside), "ghosts": sum(bool(texts[j].strip()) for j in outside),
               "asr_ms_p50": f"{np.percentile(ms, 50):.0f}", "asr_ms_p95": f"{np.percentile(ms, 95):.0f}",
               "seg_audio_s": f"{np.mean([s[2].size for s in segs]) / RATE if segs else 0:.2f}"}
        emit(row)
    if samples is not None:
        for k, ref in enumerate(refs):
            if any(first_word_lost(ref, hyps_by_pr[pr][k]) for pr in pre_rolls):
                samples.append({"corpus": corpus, "noise": noise if snr_db is not None else "-",
                                "snr_db": "clean" if snr_db is None else f"{snr_db:g}", "utt": k,
                                "ref": nref[k], "best": db.normalize_vi(best[k]),
                                **{f"pre_roll_{pr}": db.normalize_vi(hyps_by_pr[pr][k]) for pr in pre_rolls}})


# ------------------------------------------------------------------------ smoke
class SmokeLagVad:
    """Energy VAD that, like Silero on a soft onset, only says "speech" once an utterance is well
    under way: after ``lag`` loud frames since the last 0.5 s of quiet (the first tone burst is
    never voiced). One instance per stream."""

    def __init__(self, lag: int = 12, threshold: float = 1500.0) -> None:
        self.lag, self.threshold = lag, threshold
        self.loud = self.quiet = 0

    def __call__(self, frame: bytes, rate: int) -> bool:
        x = np.frombuffer(frame, dtype=np.int16).astype(np.float32)
        if float(np.sqrt(np.mean(x * x))) > self.threshold:
            self.loud, self.quiet = self.loud + 1, 0
            return self.loud > self.lag
        self.quiet += 1
        if self.quiet >= 17:
            self.loud = 0
        return False


def smoke_data(seed: int):
    rng = np.random.default_rng(seed)
    names = list(db.SMOKE_VOCAB)
    utts = []
    for _ in range(6):
        words = [names[i] for i in rng.integers(0, len(names), 4)]
        utts.append((db.smoke_speech(words), " ".join(words)))
    return utts, [rng.normal(0, 1, RATE * 10)]


def _snrs(text: str) -> list[float | None]:
    return [None if s.strip() == "clean" else float(s) for s in text.split(",") if s.strip()]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--smoke", action="store_true", help="tone bursts + toy ASR + energy VAD, fully offline")
    p.add_argument("--corpus", choices=["vivos", "fleurs"], default="vivos")
    p.add_argument("--vivos-dir", type=Path)
    p.add_argument("--fleurs-dir", type=Path)
    p.add_argument("--noise-dir", type=Path, help="directory of noise .wav files (needed for a noisy SNR)")
    p.add_argument("--snrs", type=_snrs, default=[None, 10.0, 5.0], help="comma list, 'clean' = no noise")
    p.add_argument("--pre-rolls", default=",".join(map(str, PRE_ROLLS)))
    p.add_argument("--limit", type=int, default=200, help="max utterances (default 200)")
    p.add_argument("--noise-files", type=int, default=20)
    p.add_argument("--gap-s", type=float, default=1.5)
    p.add_argument("--quiet-every", type=int, default=10)
    p.add_argument("--quiet-s", type=float, default=10.0)
    p.add_argument("--level", type=float, default=3000.0, help="speech RMS of every utterance (int16 scale)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--samples", type=Path, default=None)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    pre_rolls = [int(v) for v in args.pre_rolls.split(",") if v.strip()]
    if args.smoke:
        utts, noises = smoke_data(args.seed)
        corpus, noise_name = "smoke", "smoke"
        asr = db.ToyASR()

        def make_vad():
            return SmokeLagVad()
    else:
        corpus_dir = args.vivos_dir if args.corpus == "vivos" else args.fleurs_dir
        need_noise = any(s is not None for s in args.snrs)
        missing = [] if corpus_dir and corpus_dir.expanduser().is_dir() else [f"--{args.corpus}-dir"]
        if need_noise and not (args.noise_dir and args.noise_dir.expanduser().is_dir()):
            missing.append("--noise-dir")
        if missing:
            print("error: missing data directory: " + ", ".join(missing) + " (use --smoke offline)", file=sys.stderr)
            return 2
        try:
            loader = db.load_vivos if args.corpus == "vivos" else ab.load_fleurs
            utts = loader(corpus_dir.expanduser(), args.limit or None)
            noises = db.load_noises(args.noise_dir.expanduser(), args.noise_files, args.seed) if need_noise else []
        except (FileNotFoundError, ValueError, wave.Error) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        corpus = args.corpus
        noise_name = args.noise_dir.expanduser().name if need_noise else "-"
        from audio_translator.asr import WhisperASR, filter_from_settings
        from audio_translator.config import Settings
        from audio_translator.segmenter import vad_from_settings

        s = Settings()
        asr = WhisperASR(s.model_size, language=s.source_lang, device=s.asr_device, compute_type=s.compute_type,
                         beam_size=s.beam_size, initial_prompt=s.asr_initial_prompt,
                         segment_filter=filter_from_settings(s))
        asr.load()

        def make_vad():
            return vad_from_settings(s)  # Silero 0.5, a fresh one per stream as per session

    print(f"{corpus}: {len(utts)} utterance(s), noise {noise_name}, SNRs "
          f"{['clean' if v is None else v for v in args.snrs]}, pre-rolls {pre_rolls}", flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    samples = [] if args.samples else None
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        f.flush()

        def emit(row):
            w.writerow(row)
            f.flush()
            print("  ".join(f"{k}={row[k]}" for k in COLUMNS), flush=True)

        for snr in args.snrs:
            stream = build_stream(utts, noises, snr, np.random.default_rng(args.seed), gap_s=args.gap_s,
                                  quiet_every=args.quiet_every, quiet_s=args.quiet_s,
                                  level=None if args.smoke else args.level)
            bench_cell(utts, stream, asr, make_vad, pre_rolls, corpus=corpus, noise=noise_name, snr_db=snr,
                       seed=args.seed, emit=emit, samples=samples)
    if samples is not None:
        args.samples.parent.mkdir(parents=True, exist_ok=True)
        with open(args.samples, "w", newline="", encoding="utf-8") as f:
            cols = ["corpus", "noise", "snr_db", "utt", "ref", "best", *[f"pre_roll_{pr}" for pr in pre_rolls]]
            w = csv.DictWriter(f, fieldnames=cols, delimiter="\t")
            w.writeheader()
            w.writerows(samples)
    return 0


if __name__ == "__main__":
    sys.exit(main())
