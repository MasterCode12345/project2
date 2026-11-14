---
title: 'ASR anti-stall without a WER cost: compare (A) default, (B) cap + tail strip, (C) cap on fallback attempts only'
type: 'feature'
created: '2026-11-14'
status: 'measured — no configuration passes; all stay optional (default unchanged)'
baseline_revision: 'main @ fix/polish merge (junk tokens unk/ofc on)'
branch: 'feat/asr-anti-stall'
route: 'full'
risk: 'medium'
context:
  - '{project-root}/_bmad-output/initiative-audio-translator/plan-speaker-indicator.md'
  - '{project-root}/_bmad-output/initiative-audio-translator/plan-polish.md'
---

## Intent

**The problem: stalls.**
- With today's decoding, one segment can take 10–34 s of ASR. The bench showed this even on clean speech (`results/stall_*.csv`).
- Cause: a temperature-0 attempt loops to Whisper's 224-token limit (about 6 s of CPU). It is rejected by the compression or log-prob check and retried at a higher temperature, up to 6 times.

**The goal.**
- No more long stalls.
- **WER must not rise** compared with today's default (no cap). The supervisor: do not trade quality for latency.

## New evidence (2026-11-14): why the old cap cost WER

The earlier bench used cap 15 tok/s + 32 and reported `cap_hits = 0` on speech: no attempt reached the cap. Yet transcripts gained tails such as "a." or a run of dots. I re-decoded 5 clean VIVOS clips from that bench (same clip construction, Silero + filter, seed `0 + k`), with only the cap changed:

| Clip | No cap (today) | Cap 15+32 | `max_new_tokens = 224` (Whisper's own limit) |
|---|---|---|---|
| 8 | "m rộp biển mây … vất vờ." (18 tokens) | "làm rầm rộp … vất vờ…………" (41 / limit 79) | same as no cap |
| 30 | "đó chứng tỏ rằng chúng ta đã già." (16) | "… đã già. a……" (40 / 77) | same as no cap |
| 63 | "vì đó mới chính là thứ bạn cần." (13) | "… cần. a." (37 / 71) | same as no cap |
| 70 | "k sĩ la hối … tuổi trẻ." (T=0: 103 tokens, rejected; T=0.2: 21) | "nhạc sĩ la hối … a sĩ la hối …" (43 / 83, accepted at T=0) | "nhạc sĩ … tuổi trẻ………" (113, accepted at T=0) |
| 2 | "khi chẩm khỏi bệnh thì sẽ chém đầu." (19) | identical (19 / 77) | identical |

**Conclusions.**
1. Passing `max_new_tokens` changes the **beam-search result of the temperature-0 attempt**, even when the output stays far below the limit. The output gets *longer* (13 → 37 tokens), with a junk tail. CTranslate2's beam search evidently depends on `max_length`; I have not traced the exact rule in its C++ code.
2. The junk tail is **not** a cut-off: the attempt ended on its own, below the limit. So "strip the tail only when the segment hit the token limit" — config (B) as specified — would almost never trigger. The old run had 0 hits on speech. **(B) is therefore expected to equal the old cap's WER: +1.12 on clean VIVOS.** It will still be measured, as you asked.
3. Leaving the **temperature-0 attempt exactly as today** (no `max_new_tokens`) keeps its output identical. That is the basis of (C).

## The three configurations

**(A) Today's default.**
- `asr_max_tokens_per_s = None`, faster-whisper's 6 temperatures; Silero + filter with the junk tokens on.
- This is the reference.

**(B) Cap 15 tok/s + 32, plus a tail strip when the limit was hit.**
- `WhisperASR` knows the cap per call. When a kept segment's generated tokens reach it, a **junk tail** is removed:
  - a trailing run of ≥ 2 dots → one dot;
  - trailing stand-alone "a" / "a." tokens.
- Sentences that did not hit the limit are never touched.
- Per the evidence above, it is expected to change almost nothing.

**(C) No cap at temperature 0; the cap applies only to fallback attempts (temperature > 0).**
- faster-whisper has one `max_new_tokens` for all attempts. (C) therefore wraps the CTranslate2 `Whisper` object, the same proxy pattern as the bench's `DecodeProbe`.
- On a `generate` call with `sampling_temperature > 0`, `max_length` is lowered to `len(prompt) + cap(audio_s)`. Calls at temperature 0 are passed through **unchanged**.
- **Expected WER:** identical to (A) on every segment that needs no fallback (most of them). On the others, a capped fallback can only be cut shorter.
  - A cut output almost always fails the compression check and moves on to the next temperature.
  - Sampling draws the same random numbers token by token (seeded RNG), so the tokens before the cut are the same.
- **Expected time:** the temperature-0 loop still runs to 224 tokens (about 6 s), but each retry is bounded (about 2 s instead of about 6 s).
  - A 3-attempt stall: about 22 s → about 10 s. The worst case with 6 attempts: about 40 s → about 16 s.
  - So (C) bounds the stall but does not remove it; that is the price of not touching temperature 0.
- Option name: `asr_fallback_max_tokens_per_s` (+ the existing `asr_max_tokens_base`). CLI/GUI flag: `--asr-fallback-max-tokens-per-s`. Default `None` until decided.

## Measurement

The same protocol as `results/stall_*.csv`, with **all three configurations run fresh** on the same machine, sequentially, nothing else heavy running:
- VIVOS × DEMAND and VIVOS × MUSAN, 100 utterances, at clean, 20, 10, 5 and 0 dB;
- FLEURS clean, 100 sentences (244 segments);
- phowhisper-small, Silero + filter (junk tokens on), seed 0, `--segments-out` kept in the scratchpad.

**Stalls are reproduced the same way as before.** The bench already contains them: base had 10–34 s speech segments and 20–30 s noise-only segments. The noise-only clips (`noise_*` columns) are the stall reproduction. The earlier MUSAN-music probe (`stall_probe2.py`, the 3 s clip that took 22 s) is re-run for the three configurations as a spot check.

**Bench changes (`eval/asr_bench.py`):**
- `--fallback-max-tokens-per-s` for (C), and `--cap-tail-strip` for (B);
- new columns `asr_ms_p50`, `noise_asr_ms_p50` and `redecodes` (extra attempts on speech; `noise_redecodes` exists).

**Report table, per set and SNR:** WER, CER; ASR p50 / p95 / max on speech and on noise-only segments; segments that hit the token limit (`cap_hits`); segments that needed a fallback and the number of re-decodes.

**Estimated time:** about 35 min per configuration, about 1 h 45 min in total, in the background.

## Decision rule (yours)

- A configuration is turned **on by default** only if its **WER is not higher than (A) in any cell** (each set × SNR, no tolerance) **and** its max ASR time is the lowest among those that pass.
- If none passes, everything stays optional, and I report back.
- → **Question 1:** strict "not higher in any cell", or allow a tolerance for noise, e.g. ≤ +0.1 point (1 word in 1 000)? I propose strict, as you wrote.

## Files

| File | Change |
|---|---|
| `src/audio_translator/asr.py` | `WhisperASR`: `fallback_max_tokens_per_s` (proxy on the CT2 model, T > 0 only); `cap_tail_strip` and `strip_cap_tail()` for (B); `decode` label updated (`fb15+32`, `cap15+32+tail`) |
| `src/audio_translator/config.py` | `asr_fallback_max_tokens_per_s: float \| None = None`, `asr_cap_tail_strip: bool = False` (both off; defaults change only per the rule) |
| `src/audio_translator/orchestrator.py` | pass the two settings to `WhisperASR` (`_make_recognizer`), one line |
| `src/audio_translator/cli.py`, `gui.py` | `--asr-fallback-max-tokens-per-s`, `--asr-cap-tail-strip` (GUI: parse_args only, no widget) |
| `eval/asr_bench.py` | flags + columns above |
| `results/antistall_vivos_demand.csv`, `results/antistall_vivos_musan.csv`, `results/antistall_fleurs_clean.csv` | new |
| `README.md`, plan | the table, the decision |

## Tests

- **Proxy:** on a fake CT2 model, a `generate` call at T = 0 keeps `max_length`, a call at T > 0 gets the capped `max_length`, and other attributes pass through. Off by default, so no proxy is installed.
- **Tail strip:** only when the limit was hit; a sentence that did not hit it is untouched. Covers dots, a stray "a.", and a sentence that legitimately ends in "a" ("đi đâu a" — the rule keeps it unless the limit was hit).
- **Wiring:** Settings → recognizer, CLI/GUI flags, the `decode` label in the latency log.
- **Bench:** the new flags and columns (smoke + a stub model).
- The whole suite.

## Decisions (human, 2026-11-14)

1. **WER rule.**
   - No cell may be more than **0.1 point** worse than (A).
   - The **mean over all cells** must not be worse than (A).
   - The two main cells, **VIVOS clean and FLEURS clean**, must not be worse than (A).
   - Every cell that is worse, by any amount, is listed.
2. **(B).** Reuse the old cap15+32 rows, recomputed with today's filter. Run (B) fresh only on VIVOS clean and FLEURS clean, to check the prediction. Record "tried, failed, why".
3. **(C).** If it passes the WER rule, turn it on by default, even if max is still 8–16 s.
4. Two checks before the big bench: (a) and (b) below.
5. The big bench runs in the background. **The table is reported before any default changes.**

## 4a. Which T=0 limit leaves the output unchanged? (2026-11-14)

**Setup.** 12 clean VIVOS clips (DEMAND bench construction, Silero + filter, seed k):
- the 5 above, plus every clip whose base and cap transcripts differed at 0, 20 or 10 dB (11, 41, 44, 45, 67, 79, 96);
- each decoded with an **absolute** T=0 limit (`max_new_tokens`) of none, 100, 150, 200, 223 or 224.

**Results.**

| Limit | Identical to no limit |
|---|---|
| 100 | 11 / 12 |
| 150 | 11 / 12 |
| 200 | 11 / 12 |
| 223 | 11 / 12 |
| 224 | 11 / 12 |

- **The odd one out is clip 70 at every level, including 224.** It is the clip whose T=0 attempt loops without a limit (103 tokens, rejected, then T=0.2 gives "k sĩ la hối…"). With any limit, T=0 returns "nhạc sĩ la hối … tuổi trẻ" plus a run of dots (51–113 tokens). That text passes the checks: it is the *right* words, and the dots do not count in WER.
- The cap 15+32 used earlier (71–83 tokens for these clips) changed 3 non-looping clips. **From 100 tokens up, no non-looping clip changed.**

**(C+) is therefore added.**
- The T=0 attempt gets a wide limit: `max(100, ceil(audio_s × 15) + 32)`, at most 224.
- The per-second term keeps long utterances (≥ 4.5 s) from ever getting less than the old 15 tok/s + 32 headroom. A flat 100 would cut a real 12–15 s sentence; speech reaches about 10 tok/s.
- Fallback attempts are capped as in (C).
- Caveat: only 12 clips. The bench checks it on 100 VIVOS utterances × 5 SNRs and 244 FLEURS segments.

**Hypothesis: why does `max_length` change the beam result?** Not verified in CTranslate2's C++ code.
1. faster-whisper changes nothing but `max_length`: 448 without `max_new_tokens`, otherwise prompt + N.
2. CTranslate2's beam search with `patience = 1` stops once `beam_size` (5) hypotheses have finished. It scores finished hypotheses with length normalisation (`length_penalty = 1`, i.e. average log-prob per token).
3. Beams that drift into a high-probability tail ("a", dots, a repeat) have a *good* average score but never emit EOT by themselves.
4. With a small `max_length` they are **force-finished** at the limit and enter the finished pool, where length normalisation lets them win. With 448, they are still running when 5 natural endings have been collected, so the short correct sentence wins.

This explains a longer output with a tail when a limit exists. It does not fully explain outputs well under the limit (41 of 79), unless intermediate forced finishes of other beams change which beams survive pruning. It stays a hypothesis.

## 4b. Time budget in the model wrapper → (C2), judged feasible

**Mechanism.**
- The wrapper (the same proxy as (C)) notes the time of the first `generate` call of each segment.
- At a call with temperature > 0, if the elapsed time is over the budget (4 s), it **does not decode**. It returns the **result of the previous real attempt** again.
- faster-whisper finds it failing the same checks, runs through the remaining temperatures at no cost, and ends in its "all failed" branch. That branch selects the highest average log-prob among the **real** attempts: below the compression threshold if any, otherwise all of them.

**Why it is safe.**
- No empty or synthetic result is ever created, so a too-short or empty pick is impossible.
- The pick is exactly what faster-whisper would choose from the attempts that ran.
- No faster-whisper internals are patched; only the CTranslate2 object is wrapped, as in the bench's `DecodeProbe`.

**Risk, and its mitigation.**
- If the T=0 attempt alone exceeds 4 s (a 224-token loop takes about 6 s), every retry is skipped, and a sentence that (A) recovers by a fallback could be lost (filtered out as a loop).
- (C2) is therefore **(C+) + budget 4 s**: with the T=0 limit of about 100 tokens (about 2.5 s), a retry still fits.

## Configurations measured

| | T=0 attempt | Fallback attempts | Other |
|---|---|---|---|
| (A) | no limit | no limit | today's default |
| (B) | 15/s + 32 (all attempts, via `max_new_tokens`) | same | tail strip when the limit is hit; old rows recomputed + fresh VIVOS/FLEURS clean |
| (C) | no limit | 15/s + 32 | |
| (C+) | max(100, 15/s + 32) | 15/s + 32 | |
| (C2) | as (C+) | as (C+) | time budget 4 s per segment |

Fresh runs: (A), (C), (C+), (C2), at about 35 min each, plus (B) clean only.

## Results (2026-11-14)

**Data and runs.**
- Files: `results/antistall_vivos_demand.csv`, `results/antistall_vivos_musan.csv`, `results/antistall_fleurs_clean.csv`.
- phowhisper-small, Silero + filter (junk tokens on), 100 VIVOS utterances × 5 SNRs per noise; FLEURS 100 sentences (244 segments); seed 0.
- Sequential runs on the same machine. (A), (C), (C+) and (C2) were fresh.
- (B):
  - for every cell, the old cap15+32 rows (`results/stall_*.csv`), with WER/CER recomputed from their per-segment text under today's filter;
  - fresh on VIVOS clean and FLEURS clean.

**ΔWER against (A), in points; negative = better. Max ASR in ms (speech / noise-only).**

| Cell | (A) WER | (B) | (C) | (C+) | (C2) |
|---|---|---|---|---|---|
| VIVOS×DEMAND clean | 5.73 | +1.12 | +1.12 | 0.00 | 0.00 |
| VIVOS×DEMAND 20 | 6.85 | +0.92 | 0.00 | −0.20 | −0.20 |
| VIVOS×DEMAND 10 | 7.16 | +0.82 | −0.10 | −0.31 | −0.31 |
| VIVOS×DEMAND 5 | 8.69 | +0.61 | −0.10 | **+1.64** | **+1.64** |
| VIVOS×DEMAND 0 | 11.66 | +1.94 | +0.71 | **+1.43** | **+1.43** |
| VIVOS×MUSAN clean | 5.73 | +1.12 | +1.12 | 0.00 | 0.00 |
| VIVOS×MUSAN 20 | 5.93 | 0.00 | +0.10 | 0.00 | 0.00 |
| VIVOS×MUSAN 10 | 6.75 | −0.10 | +0.20 | −0.10 | −0.10 |
| VIVOS×MUSAN 5 | 7.87 | +0.21 | 0.00 | **+0.21** | **+0.21** |
| VIVOS×MUSAN 0 | 14.83 | −0.52 | −0.41 | −0.52 | −0.52 |
| FLEURS clean | 15.49 | −0.43 | +0.23 | −0.30 | −0.30 |
| **Mean ΔWER** | | +0.517 | +0.261 | +0.168 | +0.168 |
| **Max ASR, speech** | 35 875 | 2 874 | 8 503 | 2 876 | 2 869 |
| **Max ASR, noise-only** | 21 675 | 3 155 | 9 914 | 3 476 | 3 485 |

More detail per cell (CSV columns):
- **Speech P95** for (A) reaches 7 198 ms at DEMAND 0 dB and 3 836 ms on FLEURS; (C+)/(C2) are 1.4–2.1 s everywhere. P50 is about 1.15–1.24 s in every configuration.
- **Cap hits** (an attempt reached its limit on speech): (A) 0–8 (at 224); (B) 0; (C) 0–7, all on the uncapped T=0; (C+)/(C2) **0** everywhere.
- **Fallback segments / re-decodes on speech:** (A) up to 13 / 26 (FLEURS); (C) 13 / 13; (C+)/(C2) 0–2 / 0–2.
- **Budget skips in (C2): 0 in every cell.** With the T=0 limit of (C+), no segment ever reached 4 s, so (C2) ≡ (C+).

**Decision rule** (each cell ≤ +0.1, mean ≤ 0, VIVOS clean and FLEURS clean ≤ 0): **no configuration passes.** All stay optional; **no default was changed.**

**Why each one fails: "tried, failed, why".**
- **(B) — fails as predicted.**
  - The fresh runs equal the old cap exactly: VIVOS clean 6.85 (+1.12), FLEURS 15.06.
  - The tail strip **never triggered** (`cap_hits = 0`).
  - The cap changes the temperature-0 beam result **without reaching the limit**: tails of "a." and dots on otherwise correct sentences (clips 30, 63, 8). Stripping only on a hit cannot catch them.
- **(C) — fails on the clean cells (+1.12), and its max is still 8.5 s.**
  - The uncapped T=0 loop (about 6 s) remains.
  - The capped *sampling* attempts also change result, e.g. clip 70 "nhạc sĩ … a sĩ la hối với…". The cap acts on the sampling path too, not only by cutting.
- **(C+)/(C2) — pass both main cells (VIVOS clean 0.00, FLEURS −0.30), are better in 6 cells, and bring max from 35.9 s to 2.9 s.** They fail on three cells:
  - **DEMAND 5 dB (+1.64) is one clip.** In clip 70, the T=0 loop is now cut at about 51 tokens. Its compression ratio stays under 2.4, so faster-whisper accepts it without a retry: "nhạc sĩ la hối … tuổi trẻ. a sĩ la hối với ca khúc xuân và tuổi trẻ. a sĩ la hối với ca" (+11 words). In (A) the loop ran to 83 tokens, failed the compression check, and the retry gave the short line.
  - **DEMAND 0 dB (+1.43)** has the same pattern: partial-repeat tails in clips 45 and 79, and a changed first word in clips 41 and 44.
  - **MUSAN 5 dB (+0.21)** is one word in clip 82: "thông thốc thổi" → "không thốc khỏi".
  - A what-if on the cached text, stripping the "a."/dot tail only on outputs ending in a dot run, does not fix these cells: the damage is the **partial repeat**, not the dots.

**Limitations.** One model and one seed; 100 utterances per cell, so 1 point ≈ 7–10 words. The (C+) failures come from a handful of clips (one clip = 1.64 points at DEMAND 5 dB).

**Possible next steps** (not done; for the human to decide):
1. **(C3) = (C+) + a "truncated loop" guard.** When the T=0 attempt was limited and its tokens end in a repeat (the same n-gram ≥ 2 times in the tail) or a dot run, the wrapper runs the next temperature instead of accepting. Only possible by deciding in the wrapper, not by faking a result. Needs care and a re-bench.
2. A wider T=0 limit (150 or 200 tokens): fewer truncated loops get accepted, max about 4–6 s. One bench run each.
3. Keep (A) and rely on the speaker indicator. It already turns red at 6 s of ASR.

## Questions for you (answered above)

1. **Decision tolerance:** strict (proposed) or ≤ +0.1 point?
2. **(B):** the evidence says it will equal the old cap. Run it fresh anyway (proposed, about 35 min, so the table is complete), or reuse the old cap rows recomputed with today's filter?
3. **(C):** if it passes on WER but its max is still about 8–16 s (the temperature-0 loop remains), should it be turned on? It is better than today and costs no quality. The indicator already warns at 3 s / 6 s.
