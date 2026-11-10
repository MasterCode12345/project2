---
title: 'Hallucination guard on by default, PhoWhisper loop collapse, seeded benches'
type: 'feature'
created: '2026-11-09'
status: 'done'
baseline_revision: '5d254ff'
route: 'oneshot'
context:
  - '{project-root}/_bmad-output/initiative-audio-translator/plan-hallucination-guard.md'
---

## Intent

**Decision (human, 2026-11-09):** after the guard bench (`results/guard_vivos_{demand,musan}.csv`), turn the guard on by default (`vad_backend="silero"`, `asr_filter=True`) and finish the open items of step 2.

**Changes:**
- `config.py`: `vad_backend="silero"`, `asr_filter=True`. `cli.py`: `--asr-filter/--no-asr-filter`; `--vad webrtc --no-asr-filter` restores the old behaviour.
- `asr.collapse_repeats`, applied by `SegmentFilter.clean` to kept segments (`collapse_loops=True`). A unit of n words repeated consecutively (n=1: ≥ 4 times, n=2–4: ≥ 3, n≥5: ≥ 2), compared on normalised words, is written once, keeping the last copy so its punctuation survives. Recorded in `WhisperASR.dropped` as `("repetition", original)`.
  - Threshold check: 0 of 14,134 real sentences changed (VIVOS train+test prompts, FLEURS vi test raw + normalised). A first version (n=4 at ×2) changed one real VIVOS prompt ("bác sĩ phẫu thuật bác sĩ phẫu thuật…"), so n=4 needs ×3.
  - Post-hoc on the step-2 diagnosis (PhoWhisper, Silero): MUSAN 0 dB WER 17.3 → 14.7, DEMAND 5 dB 9.5 → 8.7.
- Bench reproducibility: faster-whisper falls back to sampling (temperature > 0) on hard audio, and CTranslate2's RNG was unseeded, so noisy webrtc rows varied between runs (13.2 vs 16.0). `asr_bench.seed_decoder(seed + k)` is called before every clip's Whisper calls; `mt_bench` seeds per recording (`crc32(pair.id)`). Checked: the same cell run twice gives identical WER/CER.
- `mt_bench` `asr_seg` follows the app's `Settings` (VAD backend + ASR filter), with `--seg-vad/--seg-filter`; cache key `seg(silero0.5,600ms,filter)`, and webrtc + filter off keeps the historical `seg(vad2,600ms)`.
- Small lint fixes (unused imports and variable).

## Verification

- `uv run --group eval pytest -q`: 273 passed.
- Confirmation bench (seeded): PhoWhisper-small × {webrtc, silero} × {off, on} on VIVOS × DEMAND/MUSAN → `results/guard2_vivos_*.csv`; MT `asr_seg` with the new defaults → `results/mt_vi_en_seg_silero.csv`.

## Results
Seeded confirmation run (2026-11-10), PhoWhisper-small, 100 VIVOS utts; WER webrtc+off → silero+on (%):

| Noise | clean | 20 dB | 10 dB | 5 dB | 0 dB |
|---|---|---|---|---|---|
| DEMAND | 6.85 → 5.62 | 6.03 → 6.75 | 6.24 → 7.16 | 7.36 → 8.59 | 13.09 → 12.07 |
| MUSAN | – | 10.43 → 5.93 | 6.85 → 6.75 | 13.70 → 7.87 | 14.83 → 14.83 |

- Loop collapse (silero off → on): DEMAND 5 dB 9.41 → 8.59, MUSAN 0 dB 17.38 → 14.83; elsewhere unchanged. One cell still exceeds "≤ 1 point worse": DEMAND 5 dB, +1.2 (~12 words / 100 utts); three cells improve clearly.
- ghost_texts unchanged from the guard bench: DEMAND 22–26 (real background voices), MUSAN 0–10, vs 96–141 for webrtc.
- MT `asr_seg` (FLEURS 347, Argos) with the new segmentation: 12.35 BLEU (off) / 15.15 (vibert-capu) vs 13.00 / 15.82 with webrtc; src WER unchanged (15.9%). Silero cuts more, shorter segments (843 × 13.3 words vs 761 × 14.8). Accepted trade-off for the noise robustness; follow-up: merge short segments before MT.
