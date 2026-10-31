"""Noise-suppression bench: WER/CER, latency, RTF and VAD false triggers vs SNR.

Mixes clean Vietnamese speech with noise at controlled SNRs, runs each arm of
the M3a stage (NoiseSuppressor) + the real Segmenter + Whisper exactly as the
app does (minus threads), and writes one CSV row per arm x SNR.

Scoring protocol (each clip stands in for a slice of a live stream):

* clip = noise warm-up (``WARMUP_S``, not scored, so the VAD and the enhancer's
  noise tracker have adapted like in a running session) + ``LEAD_S`` noise +
  utterance + ``TAIL_S`` noise; the noise fades in/out over ``FADE_S`` so no
  digital-silence/noise step can trigger the VAD; SNR is measured over the
  utterance span. VAD and stage are fresh per clip (the app makes both per
  session).
* WER/CER use the text of the segments that overlap the utterance span
  (segment times compensated for the stage latency); segments that do not are
  counted in ``vad_extra_segs`` instead of being scored as hallucinated words.
* ``vad_false_triggers``: segments overlapping the scored noise window of a
  noise-only clip built the same way.
* ``--limit`` takes utterances round-robin over speakers, so a small run still
  covers every VIVOS test speaker.

Real run (data is never downloaded by this script)::

    uv run --group eval python eval/denoise_bench.py \\
        --vivos-dir ~/data/vivos --noise-dir ~/data/musan/noise --model small --limit 100

Offline smoke run (synthetic "speech" + a toy recognizer, no network, no data)::

    uv run --group eval python eval/denoise_bench.py --smoke --out /tmp/ns_smoke.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
import unicodedata
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from audio_translator.denoise import LogMmseEnhancer, NoiseSuppressor
from audio_translator.segmenter import Segmenter

RATE = 16000
FRAME = 480
# (arm name, OA weight ω; None = no stage at all)
ARMS: list[tuple[str, float | None]] = [
    ("none", None),
    ("logmmse", 0.0),
    ("logmmse+oa0.2", 0.2),
    ("logmmse+oa0.4", 0.4),
    ("logmmse+oa0.6", 0.6),
]
SNRS: list[float | None] = [None, 20.0, 10.0, 5.0, 0.0]  # None = clean
COLUMNS = ["arm", "snr_db", "n_utts", "wer", "cer", "added_latency_ms", "rtf", "vad_false_triggers",
           "vad_extra_segs"]
LEAD_S, TAIL_S = 0.5, 1.0  # noise-only padding around each utterance
WARMUP_S = 2.0  # unscored noise before the lead: VAD/enhancer adapt as in a running session
FADE_S = 0.3  # raised-cosine noise fade-in/out at the clip edges
MAX_EXTRA_FLUSH_FRAMES = 200  # 6 s of extra silence at most while an utterance is still open


# ------------------------------------------------------------------ text/metrics
def normalize_vi(text: str) -> str:
    """Lowercase, drop punctuation/symbols, keep Vietnamese tone marks (NFC)."""
    text = unicodedata.normalize("NFC", text).lower()
    text = "".join(" " if unicodedata.category(c)[0] in "PS" else c for c in text)
    return " ".join(text.split())


def _edits(ref: list, hyp: list) -> int:
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        cur = [i] + [0] * len(hyp)
        for j, h in enumerate(hyp, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (r != h))
        prev = cur
    return prev[-1]


def error_rates(refs: list[str], hyps: list[str]) -> tuple[float, float]:
    """Corpus WER and CER (jiwer when installed, identical Levenshtein fallback otherwise)."""
    pairs = [(r, h) for r, h in zip(refs, hyps) if r.strip()]
    if not pairs:
        return float("nan"), float("nan")
    refs, hyps = [p[0] for p in pairs], [p[1] for p in pairs]
    try:
        import jiwer

        return float(jiwer.wer(refs, hyps)), float(jiwer.cer(refs, hyps))
    except ImportError:
        w = sum(_edits(r.split(), h.split()) for r, h in pairs) / sum(len(r.split()) for r in refs)
        c = sum(_edits(list(r), list(h)) for r, h in pairs) / sum(len(r) for r in refs)
        return w, c


# ------------------------------------------------------------------------ audio
def _read_float_wav(path: Path) -> tuple[int, int, np.ndarray]:
    """(channels, rate, samples in [-1, 1]) of an IEEE-float WAV (format 3; e.g. FLEURS)."""
    data = path.read_bytes()
    if data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        raise ValueError(f"{path}: not a RIFF/WAVE file")
    pos, fmt, samples = 12, None, None
    while pos + 8 <= len(data):
        cid, size = data[pos:pos + 4], int.from_bytes(data[pos + 4:pos + 8], "little")
        body = data[pos + 8:pos + 8 + size]
        if cid == b"fmt ":
            tag, ch, rate = int.from_bytes(body[0:2], "little"), int.from_bytes(body[2:4], "little"), \
                int.from_bytes(body[4:8], "little")
            bits = int.from_bytes(body[14:16], "little")
            if tag == 0xFFFE and len(body) >= 26:  # WAVE_FORMAT_EXTENSIBLE: subformat GUID starts with the tag
                tag = int.from_bytes(body[24:26], "little")
            fmt = (tag, ch, rate, bits)
        elif cid == b"data":
            samples = body
        pos += 8 + size + (size & 1)
    if fmt is None or samples is None:
        raise ValueError(f"{path}: missing fmt or data chunk")
    tag, ch, rate, bits = fmt
    if tag != 3 or bits not in (32, 64):
        raise ValueError(f"{path}: unsupported WAV format {tag} with {bits}-bit samples")
    x = np.frombuffer(samples[: len(samples) - len(samples) % (bits // 8)], "<f4" if bits == 32 else "<f8")
    return ch, rate, x.astype(np.float64)


def read_wav(path: Path) -> np.ndarray:
    """Read a PCM or IEEE-float wav as 16 kHz mono float64 on the int16 scale."""
    try:
        with wave.open(str(path), "rb") as w:
            ch, width, rate = w.getnchannels(), w.getsampwidth(), w.getframerate()
            raw = w.readframes(w.getnframes())
    except wave.Error as exc:
        if "unknown format: 3" not in str(exc) and "unknown format: 65534" not in str(exc):
            raise
        ch, rate, x = _read_float_wav(Path(path))
        width, raw = 0, b""
        x = x * 32768.0
    if width == 0:
        pass  # float samples, already scaled
    elif width == 2:
        x = np.frombuffer(raw, "<i2").astype(np.float64)
    elif width == 4:
        x = np.frombuffer(raw, "<i4").astype(np.float64) / 65536.0
    elif width == 1:
        x = (np.frombuffer(raw, np.uint8).astype(np.float64) - 128.0) * 256.0
    else:
        raise ValueError(f"{path}: unsupported sample width {width}")
    x = x.reshape(-1, ch).mean(axis=1)
    if rate != RATE:
        import soxr

        x = soxr.resample(x, rate, RATE)
    return x


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x, dtype=np.float64)))) if x.size else 0.0


def noise_for(noise: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    if noise.size < n:
        noise = np.tile(noise, n // max(1, noise.size) + 1)
    start = int(rng.integers(0, noise.size - n + 1))
    return noise[start:start + n]


@dataclass
class Clip:
    noisy: np.ndarray  # int16: warm-up + lead + speech + tail, speech mixed with noise
    noise_only: np.ndarray  # int16: the same noise without speech
    speech_s: tuple[float, float]  # utterance span in the clip (seconds)
    noise_window_s: tuple[float, float]  # scored noise span (after warm-up, before the fade-out)


def fade_edges(x: np.ndarray, n: int) -> np.ndarray:
    """Raised-cosine fade-in over the first ``n`` samples and fade-out over the last ``n``."""
    n = min(n, x.size // 2)
    if n <= 0:
        return x
    ramp = 0.5 - 0.5 * np.cos(np.pi * np.arange(n) / n)
    y = x.copy()
    y[:n] *= ramp
    y[-n:] *= ramp[::-1]
    return y


def build_clip(speech: np.ndarray, noise: np.ndarray, snr_db: float | None,
               rng: np.random.Generator) -> Clip:
    """Warm-up + lead + utterance + tail, with noise at ``snr_db`` over the utterance span."""
    head = int((WARMUP_S + LEAD_S) * RATE)
    tail = int(TAIL_S * RATE)
    clean = np.concatenate([np.zeros(head), speech, np.zeros(tail)])
    if snr_db is None:
        n = np.zeros_like(clean)
    else:
        n = noise_for(noise, clean.size, rng)
        scale = rms(speech) / (max(rms(n[head:head + speech.size]), 1e-9) * 10.0 ** (snr_db / 20.0))
        n = fade_edges(n * scale, int(FADE_S * RATE))
    to16 = lambda x: np.clip(np.rint(x), -32768, 32767).astype(np.int16)  # noqa: E731
    total_s = clean.size / RATE
    return Clip(to16(clean + n), to16(n), (head / RATE, (head + speech.size) / RATE),
                (WARMUP_S, total_s - FADE_S))


def overlaps(seg: tuple[float, float], span: tuple[float, float]) -> bool:
    return seg[0] < span[1] and seg[1] > span[0]


# --------------------------------------------------------------------- pipeline
@dataclass
class ArmRun:
    segments: list[tuple[float, float, str]]  # (start_s, end_s, text) in clip time, latency removed
    stage_s: float
    processed_s: float  # audio actually run through the stage (clip + pad + flush)

    @property
    def texts(self) -> list[str]:
        return [text for _, _, text in self.segments]

    def text_in(self, span: tuple[float, float]) -> str:
        return " ".join(text for a, b, text in self.segments if overlaps((a, b), span))

    def count_in(self, span: tuple[float, float]) -> int:
        return sum(overlaps((a, b), span) for a, b, _ in self.segments)


def make_stage(mix: float | None, gate_db: float | None) -> NoiseSuppressor | None:
    if mix is None:
        return None
    return NoiseSuppressor(LogMmseEnhancer(), mix=mix, gate_snr_db=gate_db)


def run_pipeline(clip: np.ndarray, stage: NoiseSuppressor | None, vad: Callable[[bytes, int], bool],
                 asr, silence_ms: int = 600) -> ArmRun:
    """Stage + Segmenter (+ ASR when given) over one clip; flushed with digital silence.

    The flush runs until the segmenter is idle again: webrtcvad keeps voicing a
    few frames after loud noise stops (hangover), so a fixed ``silence_ms``
    flush would leave a noisy clip's utterance open and silently drop it.
    """
    seg = Segmenter(vad, sample_rate=RATE, silence_ms=silence_ms)
    pad = (-clip.size) % FRAME
    lat_frames = (stage.latency_samples // FRAME + 1) if stage else 0
    min_flush = silence_ms // 30 + 2 + lat_frames
    max_flush = min_flush + MAX_EXTRA_FLUSH_FRAMES
    x = np.concatenate([clip, np.zeros(pad, np.int16)])
    silence = np.zeros(FRAME, np.int16)
    n_clip = x.size // FRAME
    lat_s = stage.latency_s if stage else 0.0
    frame_s = FRAME / RATE
    segments, stage_s = [], 0.0
    i = 0
    while i < n_clip + min_flush or (seg.triggered and i < n_clip + max_flush):
        frame = x[i * FRAME:(i + 1) * FRAME] if i < n_clip else silence
        vad_frame = None
        if stage is not None:
            t0 = time.perf_counter()
            vad_frame, frame = stage.process(frame)
            stage_s += time.perf_counter() - t0
        i += 1
        for utt in seg.feed(frame, i * FRAME / RATE, vad_frame=vad_frame):
            # Segmenter times are frame *end* times of the (delayed) stage output.
            span = (utt.speech_start - frame_s - lat_s, utt.speech_end - lat_s)
            segments.append((*span, asr.transcribe(utt.audio, RATE) if asr is not None else ""))
    return ArmRun(segments, stage_s, i * FRAME / RATE)


def bench(utts: list[tuple[np.ndarray, str]], noises: list[np.ndarray], asr,
          make_vad: Callable[[], Callable[[bytes, int], bool]],
          gate_db: float | None, seed: int, log=print) -> list[dict]:
    rows = []
    for snr in SNRS:
        for arm, mix in ARMS:
            rng = np.random.default_rng(seed)  # same noise excerpts for every arm
            refs, hyps = [], []
            stage_s = audio_s = 0.0
            false_triggers = extra_segs = 0
            latency_ms = 0.0
            for k, (speech, ref) in enumerate(utts):
                clip = build_clip(speech, noises[k % len(noises)], snr, rng)
                stage = make_stage(mix, gate_db)
                latency_ms = stage.latency_samples * 1000.0 / RATE if stage else 0.0
                r = run_pipeline(clip.noisy, stage, make_vad(), asr)
                refs.append(normalize_vi(ref))
                hyps.append(normalize_vi(r.text_in(clip.speech_s)))
                extra_segs += len(r.segments) - r.count_in(clip.speech_s)
                stage_s += r.stage_s
                audio_s += r.processed_s
                noise_run = run_pipeline(clip.noise_only, make_stage(mix, gate_db), make_vad(), None)
                false_triggers += noise_run.count_in(clip.noise_window_s)
            wer, cer = error_rates(refs, hyps)
            row = {
                "arm": arm, "snr_db": "clean" if snr is None else f"{snr:g}", "n_utts": len(utts),
                "wer": f"{wer:.4f}", "cer": f"{cer:.4f}", "added_latency_ms": f"{latency_ms:.1f}",
                "rtf": f"{stage_s / audio_s if audio_s else 0.0:.4f}", "vad_false_triggers": false_triggers,
                "vad_extra_segs": extra_segs,
            }
            rows.append(row)
            log("  ".join(f"{c}={row[c]}" for c in COLUMNS))
    return rows


# ------------------------------------------------------------------------ smoke
SMOKE_VOCAB = {"một": 300.0, "hai": 450.0, "ba": 600.0, "bốn": 800.0,
               "năm": 1000.0, "sáu": 1250.0, "bảy": 1500.0, "tám": 1800.0}
_WORD_S, _GAP_S = 0.3, 0.12


def smoke_speech(words: list[str]) -> np.ndarray:
    """Each 'word' is a Hann-shaped tone burst at the word's frequency."""
    n, g = int(_WORD_S * RATE), int(_GAP_S * RATE)
    t = np.arange(n) / RATE
    env = np.hanning(n)
    parts = []
    for w in words:
        parts += [8000.0 * env * np.sin(2 * np.pi * SMOKE_VOCAB[w] * t), np.zeros(g)]
    return np.concatenate(parts)


class ToyASR:
    """Offline stand-in for Whisper: one word per detected tone burst."""

    def transcribe(self, audio: np.ndarray, sample_rate: int) -> str:
        x = audio.astype(np.float64)
        hop = sample_rate // 100  # 10 ms
        n = x.size // hop
        if n == 0:
            return ""
        env = np.sqrt(np.mean(x[: n * hop].reshape(n, hop) ** 2, axis=1))
        lo, hi = np.percentile(env, 20), env.max()
        if hi < 300.0 or hi < 2.0 * lo:
            return ""
        on = env > lo + 0.5 * (hi - lo)
        words, i = [], 0
        freqs, names = np.array(list(SMOKE_VOCAB.values())), list(SMOKE_VOCAB)
        while i < n:
            if not on[i]:
                i += 1
                continue
            j = i
            while j < n and on[j]:
                j += 1
            if j - i >= 6:  # >= 60 ms
                seg = x[i * hop:j * hop] * np.hanning((j - i) * hop)
                spec = np.abs(np.fft.rfft(seg))
                f = np.fft.rfftfreq(seg.size, 1 / sample_rate)
                band = (f > 200) & (f < 2200)
                peak = f[band][np.argmax(spec[band])]
                words.append(names[int(np.argmin(np.abs(freqs - peak)))])
            i = j
        return " ".join(words)


def energy_vad(frame: bytes, rate: int) -> bool:
    x = np.frombuffer(frame, dtype=np.int16).astype(np.float32)
    return float(np.sqrt(np.mean(x * x))) > 1500.0


def smoke_data(seed: int) -> tuple[list[tuple[np.ndarray, str]], list[np.ndarray]]:
    rng = np.random.default_rng(seed)
    names = list(SMOKE_VOCAB)
    utts = []
    for _ in range(3):
        words = [names[i] for i in rng.integers(0, len(names), 5)]
        utts.append((smoke_speech(words), " ".join(words).capitalize() + "."))
    white = rng.normal(0, 1, RATE * 10)
    brown = np.cumsum(rng.normal(0, 1, RATE * 10))
    brown -= np.convolve(brown, np.ones(400) / 400, mode="same")  # remove the drift
    return utts, [white, brown]


# ------------------------------------------------------------------------- real
def pick_round_robin(entries: list[tuple[str, str]], limit: int | None) -> list[tuple[str, str]]:
    """Take ``(uid, text)`` entries one speaker at a time (file order within a speaker)."""
    by_spk: dict[str, list[tuple[str, str]]] = {}
    for uid, text in entries:
        by_spk.setdefault(uid.split("_")[0], []).append((uid, text))
    queues = list(by_spk.values())
    out: list[tuple[str, str]] = []
    depth = 0
    while queues and (not limit or len(out) < limit):
        queues = [q for q in queues if depth < len(q)]
        for q in queues:
            if limit and len(out) >= limit:
                break
            out.append(q[depth])
        depth += 1
    return out


def load_vivos(root: Path, limit: int | None) -> list[tuple[np.ndarray, str]]:
    test = root / "test" if (root / "test" / "prompts.txt").is_file() else root
    prompts = test / "prompts.txt"
    if not prompts.is_file():
        raise FileNotFoundError(f"VIVOS prompts not found: {root}/test/prompts.txt (or {root}/prompts.txt)")
    entries = []
    for line in prompts.read_text(encoding="utf-8").splitlines():
        if line.strip():
            uid, _, text = line.strip().partition(" ")
            entries.append((uid, text))
    utts = []
    for uid, text in pick_round_robin(entries, limit):
        spk = uid.split("_")[0]
        wav = test / "waves" / spk / f"{uid}.wav"
        if not wav.is_file():
            found = list((test / "waves").glob(f"*/{uid}.wav"))
            if not found:
                raise FileNotFoundError(f"VIVOS wav missing for {uid}: {wav}")
            wav = found[0]
        utts.append((read_wav(wav), text))
    if not utts:
        raise FileNotFoundError(f"no utterances listed in {prompts}")
    return utts


def load_noises(root: Path, max_files: int, seed: int) -> list[np.ndarray]:
    files = sorted(root.rglob("*.wav"))
    if not files:
        raise FileNotFoundError(f"no .wav noise files under {root}")
    rng = np.random.default_rng(seed)
    pick = rng.choice(len(files), size=min(max_files, len(files)), replace=False)
    return [read_wav(files[i]) for i in sorted(pick)]


# ------------------------------------------------------------------------- main
def _gate(text: str) -> float | None:
    return None if text.lower() == "none" else float(text)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--smoke", action="store_true", help="synthetic signals + toy ASR, fully offline")
    p.add_argument("--vivos-dir", type=Path, help="VIVOS root (contains test/prompts.txt, test/waves/)")
    p.add_argument("--noise-dir", type=Path, help="directory of noise .wav files (e.g. MUSAN noise/)")
    p.add_argument("--model", default="small", help="Whisper model (default: small)")
    p.add_argument("--beam-size", type=int, default=5)
    p.add_argument("--vad-aggressiveness", type=int, default=2)
    p.add_argument("--limit", type=int, default=50, help="max utterances, taken round-robin over speakers (0 = all; default 50)")
    p.add_argument("--noise-files", type=int, default=20, help="noise files sampled (default 20)")
    p.add_argument("--gate-db", type=_gate, default=None,
                   help="SNR gate threshold for the stage, or 'none' (default: none = always enhance)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=Path("denoise_bench.csv"), help="CSV output path")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.smoke:
        utts, noises = smoke_data(args.seed)
        asr, make_vad = ToyASR(), (lambda: energy_vad)
    else:
        missing = [f"{flag} {path}" if path else f"{flag} (not given)"
                   for flag, path in (("--vivos-dir", args.vivos_dir), ("--noise-dir", args.noise_dir))
                   if path is None or not Path(path).expanduser().is_dir()]
        if missing:
            print("error: missing data directory: " + "; ".join(missing)
                  + "\n(use --smoke for an offline run without data)", file=sys.stderr)
            return 2
        try:
            utts = load_vivos(args.vivos_dir.expanduser(), args.limit or None)
            noises = load_noises(args.noise_dir.expanduser(), args.noise_files, args.seed)
        except (FileNotFoundError, ValueError, wave.Error) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        from audio_translator.asr import WhisperASR
        from audio_translator.segmenter import webrtc_vad

        asr = WhisperASR(args.model, language="vi", beam_size=args.beam_size)
        asr.load()
        def make_vad():
            return webrtc_vad(args.vad_aggressiveness)

    print(f"{len(utts)} utterance(s), {len(noises)} noise source(s), "
          f"{len(ARMS)} arms x {len(SNRS)} SNRs", flush=True)
    rows = bench(utts, noises, asr, make_vad, args.gate_db, args.seed, log=lambda s: print(s, flush=True))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} rows to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
