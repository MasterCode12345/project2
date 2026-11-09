"""Collect what Whisper "hears" in noise-only audio, to curate hallucinations_vi.txt.

The noise must be *disjoint* from what the benches measure on, or the
blacklist would be tuned on the test set. Sources used here:

* MUSAN ``noise``: every file EXCEPT the ones ``eval/denoise_bench.py`` /
  ``eval/asr_bench.py`` sample with ``--noise-files N --seed S`` (defaults 20, 0);
* MUSAN ``music`` (never used by the benches);

never DEMAND (its ch01 files are a bench noise set and the other channels are
the same scenes). Each file becomes a noise-only clip built like the benches'
(warm-up, fades) at a few levels, segmented by webrtcvad + the app's Segmenter,
and every Whisper segment is printed with its count, model and confidence::

    uv run --group eval python scripts/collect_hallucinations.py --musan-dir ~/data/musan \\
        --models small,phowhisper-small --out results/hallucination_candidates.tsv

Phrases that recur and are clearly not speech (channel outros, greetings to
viewers…) are then copied by hand into ``src/audio_translator/hallucinations_vi.txt``
with their source.
"""

from __future__ import annotations

import argparse
import collections
import csv
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "eval"))
sys.path.insert(0, str(_ROOT / "src"))

import denoise_bench as db  # noqa: E402

from audio_translator.asr import normalize_text  # noqa: E402

LEVELS_DBFS = (-45.0, -35.0, -25.0)  # noise RMS levels (speech in VIVOS is around -25 dBFS)


def bench_sample(noise_dir: Path, n_files: int, seed: int) -> set[Path]:
    """The noise files the benches pick (same rule as ``denoise_bench.load_noises``)."""
    files = sorted(noise_dir.rglob("*.wav"))
    if not files:
        return set()
    rng = np.random.default_rng(seed)
    pick = rng.choice(len(files), size=min(n_files, len(files)), replace=False)
    return {files[i] for i in pick}


def disjoint_sources(musan_dir: Path, exclude_files: int, exclude_seed: int, max_files: int) -> list[Path]:
    noise_dir, music_dir = musan_dir / "noise", musan_dir / "music"
    excluded = bench_sample(noise_dir, exclude_files, exclude_seed)
    noise = [f for f in sorted(noise_dir.rglob("*.wav")) if f not in excluded]
    music = sorted(music_dir.rglob("*.wav")) if music_dir.is_dir() else []
    # interleave so a small --max-files still covers both kinds
    out: list[Path] = []
    for pair in zip(noise, music):
        out.extend(pair)
    out.extend(noise[len(music):] if len(noise) > len(music) else music[len(noise):])
    return out[:max_files] if max_files else out


def noise_clip(noise: np.ndarray, level_dbfs: float, rng: np.random.Generator, seconds: float = 8.0) -> np.ndarray:
    n = db.noise_for(noise, int(seconds * db.RATE), rng)
    n = n * (32768.0 * 10 ** (level_dbfs / 20.0) / max(db.rms(n), 1e-9))
    n = db.fade_edges(n, int(db.FADE_S * db.RATE))
    return np.clip(np.rint(n), -32768, 32767).astype(np.int16)


def collect(files: list[Path], asrs: dict, make_vad, *, seed: int = 0, log=print) -> list[dict]:
    """One row per distinct (model, normalised text): count, example raw text, mean confidences."""
    stats: dict[tuple[str, str], dict] = {}
    rng = np.random.default_rng(seed)
    for i, path in enumerate(files):
        noise = db.read_wav(path)
        for level in LEVELS_DBFS:
            clip = noise_clip(noise, level, rng)
            for model, asr in asrs.items():
                seg = db.Segmenter(make_vad(), sample_rate=db.RATE)
                for utt in _utterances(seg, clip):
                    for s in asr.transcribe_segments(utt, db.RATE):
                        key = normalize_text(s.text)
                        if not key:
                            continue
                        st = stats.setdefault((model, key), {"count": 0, "files": set(), "text": s.text.strip(),
                                                             "nsp": [], "lp": [], "cr": []})
                        st["count"] += 1
                        st["files"].add(path.name)
                        st["nsp"].append(s.no_speech_prob)
                        st["lp"].append(s.avg_logprob)
                        st["cr"].append(s.compression_ratio)
        log(f"[{i + 1}/{len(files)}] {path.parent.name}/{path.name}")
    rows = [{"model": m, "count": st["count"], "n_files": len(st["files"]), "words": len(k.split()),
             "normalised": k, "example": st["text"], "no_speech_prob": f"{np.mean(st['nsp']):.2f}",
             "avg_logprob": f"{np.mean(st['lp']):.2f}", "compression_ratio": f"{np.mean(st['cr']):.2f}"}
            for (m, k), st in stats.items()]
    rows.sort(key=lambda r: (-r["count"], r["model"], r["normalised"]))
    return rows


def _utterances(seg, clip: np.ndarray):
    pad = (-clip.size) % db.FRAME
    x = np.concatenate([clip, np.zeros(pad + db.FRAME * 60, np.int16)])
    for i in range(0, x.size, db.FRAME):
        for utt in seg.feed(x[i:i + db.FRAME], (i + db.FRAME) / db.RATE):
            yield utt.audio


COLUMNS = ["model", "count", "n_files", "words", "normalised", "example", "no_speech_prob", "avg_logprob",
           "compression_ratio"]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="List Whisper output on noise disjoint from the bench noise.")
    p.add_argument("--musan-dir", type=Path, required=True, help="MUSAN root (contains noise/ and music/)")
    p.add_argument("--models", default="small,phowhisper-small")
    p.add_argument("--exclude-files", type=int, default=20, help="bench --noise-files to exclude (default 20)")
    p.add_argument("--exclude-seed", type=int, default=0, help="bench --seed to exclude (default 0)")
    p.add_argument("--max-files", type=int, default=200, help="noise/music files to use (0 = all; default 200)")
    p.add_argument("--vad-aggressiveness", type=int, default=2)
    p.add_argument("--out", type=Path, default=Path("hallucination_candidates.tsv"))
    args = p.parse_args(argv)
    musan = args.musan_dir.expanduser()
    if not (musan / "noise").is_dir():
        print(f"error: MUSAN noise dir not found: {musan / 'noise'}", file=sys.stderr)
        return 2
    files = disjoint_sources(musan, args.exclude_files, args.exclude_seed, args.max_files)
    print(f"{len(files)} disjoint noise/music files "
          f"(excluded the bench sample: {args.exclude_files} files, seed {args.exclude_seed})", flush=True)
    from audio_translator.asr import WhisperASR
    from audio_translator.segmenter import webrtc_vad

    asrs = {}
    for m in (x.strip() for x in args.models.split(",") if x.strip()):
        asrs[m] = WhisperASR(m, language="vi")
        asrs[m].load()
    rows = collect(files, asrs, lambda: webrtc_vad(args.vad_aggressiveness),
                   log=lambda s: print(s, flush=True))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS, delimiter="\t")
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} distinct outputs to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
