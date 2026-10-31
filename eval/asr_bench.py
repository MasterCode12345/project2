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
one-off model load. One CSV row is appended as soon as it is measured, so a
long run that stops keeps what it finished.

    uv run --group eval python eval/asr_bench.py --corpus vivos --vivos-dir ~/data/vivos \\
        --noise-dir ~/data/demand_ch01 --out results/asr_vivos_demand.csv
    uv run --group eval python eval/asr_bench.py --smoke --out /tmp/asr_smoke.csv
"""

from __future__ import annotations

import argparse
import csv
import gc
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
COLUMNS = ["corpus", "noise", "model", "prompt", "snr_db", "n_utts", "wer", "cer",
           "asr_ms_mean", "asr_ms_p95", "asr_rtf", "load_s", "extra_segs"]


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
class TimedASR:
    """Wraps a recognizer and records (seconds, audio seconds) per transcribe call."""

    def __init__(self, asr) -> None:
        self.asr = asr
        self.calls: list[tuple[float, float]] = []

    def transcribe(self, audio: np.ndarray, sample_rate: int) -> str:
        t0 = time.perf_counter()
        text = self.asr.transcribe(audio, sample_rate)
        self.calls.append((time.perf_counter() - t0, audio.size / sample_rate))
        return text


def latency_stats(calls: list[tuple[float, float]]) -> tuple[float, float, float]:
    """(mean ms, P95 ms, RTF) over transcribe calls; zeros when there are none."""
    if not calls:
        return 0.0, 0.0, 0.0
    secs = np.array([c[0] for c in calls])
    audio = sum(c[1] for c in calls)
    return float(secs.mean() * 1000), float(np.percentile(secs, 95) * 1000), float(secs.sum() / audio) if audio else 0.0


# ---------------------------------------------------------------------- bench
def bench_model(utts, noises, asr, make_vad: Callable, *, corpus: str, noise_name: str, model: str,
                prompt: str | None, snrs, seed: int, load_s: float, emit: Callable[[dict], None]) -> None:
    asr.initial_prompt = prompt
    for snr in snrs:
        rng = np.random.default_rng(seed)  # identical clips for every model and prompt
        refs, hyps, scored = [], [], []
        extra = 0
        for k, (speech, ref) in enumerate(utts):
            clip = db.build_clip(speech, noises[k % len(noises)], snr, rng)
            timed = TimedASR(asr)
            r = db.run_pipeline(clip.noisy, None, make_vad(), timed)
            refs.append(db.normalize_vi(ref))
            hyps.append(db.normalize_vi(r.text_in(clip.speech_s)))
            for (a, b, _), call in zip(r.segments, timed.calls):
                if db.overlaps((a, b), clip.speech_s):
                    scored.append(call)
                else:
                    extra += 1
        wer, cer = db.error_rates(refs, hyps)
        mean_ms, p95_ms, rtf = latency_stats(scored)
        emit({"corpus": corpus, "noise": noise_name if snr is not None else "-", "model": model,
              "prompt": "on" if prompt else "off", "snr_db": "clean" if snr is None else f"{snr:g}",
              "n_utts": len(utts), "wer": f"{wer:.4f}", "cer": f"{cer:.4f}", "asr_ms_mean": f"{mean_ms:.0f}",
              "asr_ms_p95": f"{p95_ms:.0f}", "asr_rtf": f"{rtf:.3f}", "load_s": f"{load_s:.1f}",
              "extra_segs": extra})


def run(utts, noises, models: list[str], make_asr: Callable[[str], object], make_vad: Callable, *,
        corpus: str, noise_name: str, prompts: list[str | None], snrs, seed: int,
        emit: Callable[[dict], None], log=print) -> int:
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
        for prompt in prompts:
            bench_model(utts, noises, asr, make_vad, corpus=corpus, noise_name=noise_name, model=model,
                        prompt=prompt, snrs=snrs, seed=seed, load_s=load_s, emit=emit)
        done += 1
        del asr
        gc.collect()  # free the model before loading the next one (8 GB machine)
    return done


# ----------------------------------------------------------------------- main
class _PromptToy(db.ToyASR):
    initial_prompt: str | None = None

    def load(self) -> None:
        pass


def _snrs(text: str) -> list[float | None]:
    out: list[float | None] = []
    for part in text.split(","):
        part = part.strip().lower()
        out.append(None if part == "clean" else float(part))
    return out


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
    p.add_argument("--snrs", type=_snrs, default=list(db.SNRS),
                   help="comma-separated SNRs in dB and/or 'clean' (default: clean,20,10,5,0)")
    p.add_argument("--limit", type=int, default=100, help="max utterances (0 = all; default 100)")
    p.add_argument("--noise-files", type=int, default=20, help="noise files sampled (default 20)")
    p.add_argument("--beam-size", type=int, default=5)
    p.add_argument("--vad-aggressiveness", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
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
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    if args.smoke:
        utts, noises = db.smoke_data(args.seed)
        corpus, noise_name = "smoke", "smoke"
        models = ["toy-a", "toy-b"]

        def make_asr(name):
            return _PromptToy()

        def make_vad():
            return db.energy_vad
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
        from audio_translator.segmenter import webrtc_vad

        def make_asr(name):
            return WhisperASR(name, language="vi", beam_size=args.beam_size)

        def make_vad():
            return webrtc_vad(args.vad_aggressiveness)

    print(f"{corpus}: {len(utts)} utterance(s), {len(noises)} noise source(s), models {models}, "
          f"prompts {modes}, SNRs {['clean' if s is None else s for s in args.snrs]}", flush=True)
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

        done = run(utts, noises, models, make_asr, make_vad, corpus=corpus, noise_name=noise_name,
                   prompts=prompts, snrs=args.snrs, seed=args.seed, emit=emit,
                   log=lambda s: print(s, flush=True))
    if done == 0:
        print("error: no model could be loaded", file=sys.stderr)
        return 1
    print(f"wrote {rows} rows to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
