---
title: 'Speaker indicator (green / yellow / red) + bounded ASR decoding against 20 s stalls'
type: 'feature'
created: '2026-11-13'
status: 'approved — in progress'
baseline_revision: 'eb9d36e'
branch: 'feat/speaker-indicator'
route: 'full'
risk: 'medium'
context:
  - '{project-root}/_bmad-output/initiative-audio-translator/plan-app-vi-en-defaults-latency.md'
  - '{project-root}/_bmad-output/initiative-audio-translator/plan-guard-defaults.md'
---

## Intent

**Problem:**
- In a real Meet test, one segment took `asr_ms` = 20 031 ms and returned nonsense ("vơ sơ đê a tha nói."). The segments after it queued behind it: `asr_wait` up to 12 s, e2e up to 22 s.
- The speaker gets no feedback. They cannot tell that a sentence is too long, that the system is falling behind, or that their last sentence was not understood.
- Supervisor feedback: show the speaker a light (green = safe, yellow = warning, red = stop) so that the conversation does not break or lose information. Today's latency is acceptable, so latency must not be reduced at the cost of translation quality.

**Approach:**
- **Part A (observation only):** a health indicator computed on the monitor thread, sent as a new `HEALTH` event, shown in the GUI and the CLI, and recorded in the latency log. Pipeline behaviour does not change.
- **Part B (opt-in, default off):** bound faster-whisper's decoding by capping output tokens relative to audio length and shortening the temperature-fallback ladder. Measure it with `eval/asr_bench.py`. The default changes only if the numbers pass and you approve.

## Evidence for the 20 s stall (Part B.1)

**The log.** The original file is lost: a later `mv` overwrote `~/gr2_manual/cu/demo_meet.csv`. These numbers were **copied by the user; the original file is lost**. Meet session, 2026-11-12 around 13:50, phowhisper-small / silero / asr_filter=on / vibert-capu / offline:
- 9 segments, all played.
- `asr_ms`: mean 3 316, P50 1 150, P95 12 743, max 20 031.
- `asr_wait_ms`: mean 1 529, P95 7 963, max 12 290.
- `playout_wait_ms`: max 3 228.
- `e2e_ms`: mean 7 057, P50 3 124, P95 19 761, max 21 973.
- The only slow segment was id 4: asr_wait 0.1 ms, **asr_ms 20 030.9**, e2e 21 973.2, "vơ sơ đê a tha nói." → "Sister Ahai said:".
- Its `audio_s` is not known.

The cause was reproduced independently, as follows.

**The probe.** `scratchpad/stall_probe2.py` runs phowhisper-small, CPU int8, beam 5, with the app's decode options and seed 0.
- Inputs: 25 MUSAN music chunks, 15 MUSAN noise chunks (1–15 s), and 15 VIVOS test utterances.
- It wraps `ctranslate2.Whisper.generate` to time each decoding attempt.

**Results:**

| Clip | Audio | ASR total | Attempts (temperature: ms, tokens) | Final text |
|---|---|---|---|---|
| music | 3.0 s | **22 235 ms** | 0.0: 5 919 ms, 224 tok → 0.2: 7 719 ms, 224 → 0.4: 6 938 ms, 224 → 0.6: 1 014 ms, 26 | "triển lãm triển khai tâm trạng thành công." |
| music | 15 s | 9 545 ms | 0.0: 6 761 ms, 224 → 0.2: 1 787 ms, 38 | nonsense |
| music | 15 s | 8 040 ms | 0.0: 5 794 ms, 224 → 0.2: 1 613 ms, 18 | nonsense |
| noise | 5.3 s | 5 009 ms | 4 attempts of 29–38 tokens | nonsense |
| VIVOS (15 clips) | 2.2–5.3 s | 1 012–1 497 ms | always 1 attempt, 13–34 tokens | correct |

**Mechanism** (faster-whisper 1.2.1, `generate_with_fallback`):
1. Defaults: `temperature = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]` (6 attempts), `compression_ratio_threshold = 2.4`, `log_prob_threshold = −1.0`, `max_new_tokens = None`.
2. On unclear audio, the decoder enters a loop and runs until the length limit: **224 tokens, about 6–7 s of CPU** per attempt.
3. That output fails the compression-ratio or log-prob check, so it is thrown away and decoding is retried at the next temperature. Each retry can again run to 224 tokens.
4. The time therefore goes on *rejected* attempts. The final text is short, which matches the 6-word output in the real log.
5. Worst case today is 6 × ~7 s ≈ 40 s for one segment. The ASR worker is single-threaded, so everything behind it waits.
6. `no_speech_prob` was 0.00 and `avg_logprob` about −0.7 on these clips. The ASR filter (thresholds 0.6 and −1.0) therefore does not drop them. They are not silence; they are unclear sound.

**Normal speech never needs this headroom.** The 15 VIVOS clips used at most 7.1 tokens per second of audio (34 tokens in 4.8 s).

**Real logs** (45 segments: `latency_cli.csv`, `latency_meet.csv`, `cu/demo_*.csv`):
- `audio_s`: P50 3.2 s, P90 6.6 s, max 7.4 s.
- ASR / audio ratio: < 0.71 on every segment except three, each with abnormal output or a stall:

| Audio | ASR | Ratio | Text |
|---|---|---|---|
| 1.56 s | 6.56 s | 4.2 | "bạn ăn tối chưa." (correct, but a stall) |
| 0.93 s | 1.69 s | 1.8 | 18 words: a hallucination |
| 1.05 s | 1.26 s | 1.2 | "thiên." |

## Boundaries & Constraints

**Always:**
- **Part A** only reads state. Nothing in segmentation, ASR, MT, TTS or playout changes.
  - It is computed on the monitor thread, at most 10 times per second.
  - The audio callbacks are untouched.
  - The only lock taken is a short lock around reading or writing a small history deque.
  - Workers only write a couple of floats or ints.
- **Part B** options default to *off*, which reproduces today's decoding exactly. A default changes only after the bench passes and you approve.
- Tests are offline with fake streams. All existing tests keep passing.
- Fixed seeds in the benches.

**Never:**
- Run the app with a mic or BlackHole.
- Add dependencies.
- Push without asking.
- Draw conclusions from small samples. The probe above is evidence of the mechanism, not a measurement of how often it happens.

## Part A — Speaker indicator

### A.1 Signals and thresholds (all in `Settings`)

The displayed level is the **highest** of the three signals, together with that signal's reason in Vietnamese. When several signals share the top level, the reason shown is the most urgent one, in the order backlog > length > quality.

**1. Utterance length** (the segment open in the Segmenter).

| Level | Condition | Message |
|---|---|---|
| green | < `indicator_long_warn_s = 8.0` | |
| yellow | 8–12 s | "Câu dài, hãy ngắt nghỉ" |
| red | ≥ `indicator_long_stop_s = 12.0` | "Dừng lại một chút" |

- How it is read: `Segmenter.open_s = len(_frames) × frame_ms`, which is audio time rather than wall time. Reading `len()` of a list from another thread is safe in CPython.
- Why these values:
  - The real logs have P90 6.6 s and max 7.4 s, so 8 s is above every sentence spoken so far and natural sentences stay green.
  - At 12 s the speaker still has 3 s before the forced cut at `max_segment_s = 15`. That cut can land mid-word and lose a word.
  - Long segments also mean a long ASR (cost grows with audio) and long TTS playback.
- After a forced cut, the counter restarts from 0 for the next segment, as the pipeline does.

**2. System backlog.** `asr_waiting = _seg_q.qsize()`; `asr_busy_s` = now − `asr_start` of the segment being transcribed. The ASR worker sets and clears one float for this.

| Level | Condition | Message |
|---|---|---|
| yellow | ≥ `indicator_queue_warn = 1` waiting, or `asr_busy_s` > `indicator_asr_warn_s = 3.0` | "Đang xử lý, nói chậm lại" |
| red | ≥ `indicator_queue_stop = 2` waiting, or `asr_busy_s` > `indicator_asr_stop_s = 6.0` | "Dừng nói, chờ hệ thống bắt kịp" |

Why these values:
- ASR P50 is about 1.1–1.3 s in the real logs. The Meet P95 is 1.8 s, and normal VIVOS clips took at most 1.5 s in the probe. So 3 s is clearly abnormal.
- Every probe decode over 5.7 s involved rejected 224-token attempts, so 6 s means "stuck".
- One waiting segment means the next translation is delayed by a whole ASR. Two waiting segments mean the delay grows with every sentence.

**3. Quality of the last finished segment** (yellow for `indicator_unclear_hold_s = 4.0` s): "Câu vừa rồi nghe không rõ, hãy nói lại". It triggers on either of:
- (a) the ASR filter dropped it: a `no_speech` drop with a filter reason (blacklist, no_speech_prob, logprob, compression);
- (b) abnormal ASR time: `asr_s ≥ indicator_unclear_min_asr_s = 1.5` **and** `asr_s / audio_s ≥ indicator_unclear_ratio = 1.5`.
  - In the real logs, every normal segment is under 0.71.
  - The rule flags the 1.56 s → 6.56 s stall and the 0.93 s / 18-word hallucination.
  - It does not flag the 1.05 s "thiên." case (1.26 s of ASR, below the 1.5 s floor). The floor avoids flagging very short clips, which always cost about 1 s.

**Proposal, needs your decision:**
- (c) Add a third trigger: faster-whisper needed a temperature fallback (`segment.temperature > 0`). `WhisperASR` would record this as observation only, with no decode change.
- In the probe, fallback was used on 0 of 15 VIVOS clips and on 11 of 40 music/noise clips. It is the most direct "the model was unsure" signal. The bench in B.3 will report how often it fires on clean and noisy speech before the threshold is trusted.

**Smoothing:**
- A level goes **up immediately**.
- It goes **down** only after the lower level has held for `indicator_down_hold_s = 1.0` s, so a queue that goes 1 → 0 → 1 does not flicker.

### A.2 Orchestrator

- New `health.py`, a pure function `assess(metrics, settings, now) → (level, reason, reasons)` with no threads, so it can be tested alone. A small `HealthTracker` adds the hold and smoothing and keeps a history of `(t, level, reason)` for the last 60 s.
- `_monitor_worker` calls `_tick_health()` after each queue poll, rate-limited to `indicator_rate_hz = 10`. The poll is already 50 ms.
- New event kind `HEALTH = "health"` in `models.py`:
  - fields: `data = {level: green|yellow|red, reason, reasons: [...], metrics: {open_s, asr_waiting, asr_busy_s, unclear}}` and `session_id`;
  - emitted when `(level, reason)` changes;
  - also emitted at most once a second while the level is not green, so the GUI can show the live sentence length.
- Session handling:
  - On Start the tracker resets and emits `green`.
  - On Stop or abort nothing more is emitted. A tick checks `sid == self._session`, and events from an old session are never sent.
  - The GUI greys the light on STOPPED.

### A.3 Display

- **GUI:** one large indicator above the status line:
  - a coloured disc (QLabel with a stylesheet) plus an icon character (✓ / ⚠ / ✋) plus the reason text, for example "✋ Dừng lại một chút (câu 12,4 s)";
  - it does not rely on colour alone;
  - grey "–" when stopped.
- **CLI:** one line, printed only when the level or reason changes, e.g. `[đèn] VÀNG: Câu dài, hãy ngắt nghỉ`. The once-a-second refreshes are not printed.

### A.4 Latency log

Two new columns, `health_max` and `health_reason`:
- the highest level seen in the tracker history between that segment's `speech_start` and its outcome, together with that level's reason;
- computed in the worker that writes the row (a read under the small lock).

Column order: the new columns go after `detail`. `eval/latency_report.py` adds the counts of `health_max` among played and dropped rows. Old CSVs without these columns still load.

## Part B — Bounded decoding (opt-in)

### B.2 Options (default `None` = today's decoding)

- `asr_max_tokens_per_s: float | None` and `asr_max_tokens_base: int = 16`, giving `max_new_tokens = ceil(audio_s × per_s) + base`, at most 224.
  - The proposed value is **12 tokens/s**. That is 1.7× the VIVOS maximum seen so far (7.1). It will be confirmed on the 100 VIVOS and FLEURS references before the bench: the measured maximum × 1.5, rounded up.
  - For 3 s of audio this caps an attempt at 52 tokens (about 1.4 s) instead of 224 (about 6 s).
- `asr_temperatures: tuple[float, ...] | None`. The proposed value is `(0.0, 0.2, 0.4)` instead of six values. The variant is only useful together with the cap.
- **A per-segment time budget is not feasible cleanly.** `ctranslate2.Whisper.generate` cannot be interrupted, and the fallback loop lives inside faster-whisper. The cap and the shorter ladder already bound the worst case to about 3 attempts × ~1.4 s for a 3 s clip, compared with about 40 s today.
- Both options are passed by `WhisperASR` into `transcribe` only when they are set. `eval/asr_bench.py` gets `--max-tokens-per-s` and `--temperatures`.

### B.3 Measurement

**Bench changes:**
- `asr_bench.py` adds the columns `asr_ms_max`, `noise_asr_ms_mean`, `noise_asr_ms_p95`, `noise_asr_ms_max` and `fallback_segs`.
  - The noise columns time ASR on the noise-only clip segments. These are where stalls happen; today's latency columns only time speech segments.
  - `fallback_segs` counts segments with temperature > 0.
- Rows written by an older version keep loading.

**Runs:**
- VIVOS × DEMAND and VIVOS × MUSAN, 100 utterances, phowhisper-small, `--vads silero --filters on`, all SNRs, seed 0.
- Three variants:
  1. baseline;
  2. cap 12 tok/s;
  3. cap + temperatures (0, 0.2, 0.4).
- Six runs in total. My estimate, from the guard benches, is roughly 30–60 min per run, executed in the background. Output files: `results/stall_vivos_{demand,musan}.csv`.

**Decision rule:**
- I will propose enabling a variant by default **only if** WER is no more than **+0.5 points** worse than baseline at **every** SNR of both noises, **and** max/P95 ASR time (speech and noise) drop clearly.
- I will ask you before changing any default.
- I will report anomalies as they are, for example a single loop driving a WER difference.

## Tests

- `test_health.py`:
  - each signal at its thresholds (just below and at the boundary);
  - the maximum across signals and the reason priority;
  - quality hold expiry;
  - down-hold smoothing.
- Orchestrator with fake streams:
  - a long utterance goes green → yellow → red, and back to green after the cut;
  - a blocked FakeASR produces backlog red;
  - a filtered segment produces quality yellow;
  - reset on Stop/Start;
  - no HEALTH from an old session;
  - the rate limit holds (≤ 10 per second);
  - the latency-log `health_max` and `health_reason` columns.
- CLI prints only on a change. The GUI shows the correct colour, icon and text, and grey on Stop.
- `WhisperASR` passes `max_new_tokens` and `temperature` only when they are set (stub model).
- Bench smoke run with the new columns.
- `uv run --group eval pytest -q`: all pass.

## Docs

- `docs/manual_test.md`, new scenario (d) "Đèn báo" ("Indicator light"):
  - one very long sentence (more than 15 s), with the expected light changes and timings;
  - several sentences back to back without a pause, expecting backlog yellow or red;
  - unclear sound (mumbling, a cough, music near the mic), expecting quality yellow.

  Record what the light showed against what was expected; the `health_max` column of the CSV records it as well.
- README: the light in §1, §2 and §4, the stall cause in §7 and §8, and the Part B numbers in §6 with their results file.

## Tasks

- [x] A: `health.py`, `Settings` thresholds, `Segmenter.open_s`, the ASR busy stamp, the monitor tick, `HEALTH` event, GUI light, CLI line, latency-log columns and report.
- [x] A: tests (35 new; 323 pass).
- [x] B: `WhisperASR` options (`asr_max_tokens_per_s`, `asr_max_tokens_base`, `asr_temperatures`; CLI `--asr-max-tokens-per-s`, `--asr-temperatures`; all off by default), `asr_bench.py` decode columns + `--segments-out`. [ ] the bench runs →, `asr_bench.py` flags and columns, the token-rate check on the references, the six bench runs, and `results/stall_*.csv`.
- [x] Docs: `manual_test.md` (d), timestamped logs, README (Part A). Part B numbers: pending.

## Decisions (human, 2026-11-13)

1. Quality trigger (c), temperature fallback, is **added** (observation only).
2. Reason priority: backlog > length > quality. The main reason is shown on the light. Other active reasons appear in a tooltip and a small second line; they are never dropped.
3. Part B uses **four** variants: baseline / cap only / temperatures only / both. It may run overnight. Also measure:
   - clean VIVOS and FLEURS sentences that hit the token cap (must be 0);
   - noise-only max and P95 of `asr_ms`, and the fallback count.

   The default stays off; report the numbers and the human decides.
4. The 20 s log was lost; use the numbers the user copied, with the source stated.
- Order of work: commit Part A first, then run the Part B benches.
- `docs/manual_test.md`: no more `mv` overwrite. Every run writes a timestamped log, e.g. `~/gr2_manual/demo_meet_$(date +%Y%m%d_%H%M).csv`.

## Part B: choice of the cap (2026-11-14, clean baselines, phowhisper-small, Silero + filter, seed 0)

Clean runs with today's decoding (scratch per-segment TSVs, `--segments-out`):
- **Loops happen on clean speech too.**
  - VIVOS: 2/100 speech segments needed a fallback; the worst took 13–14 s for 3 s of audio.
  - FLEURS: 13/244 segments needed a fallback; 8 of them had a first attempt at the 224-token limit. The worst was 38.2 s for 6.7 s of audio (6 attempts).
  - Typically the T=0 attempt loops to 224 tokens and a later attempt gives the right short sentence. So a cap should keep the text and remove the dead time.
- **Single-attempt token rates.** The highest on plausible output is **10.0 tok/s** (39 tokens / 3.9 s). The 5 rates above 10 (13.9–25 tok/s) are all garbage, e.g. "of. of. of…", or a real sentence followed by a long run of dots.
- **Cap:** measured max × 1.5 = **15 tok/s**, base **32**. Base 16 would hit the dots-tail case once (96 tokens for 4.4 s). With 15+32, no single-attempt segment of these two sets would hit the cap. This was tuned on the same sets, which is stated as a limitation.
  - For a 3 s segment an attempt stops at 77 tokens (about 2 s) instead of 224 (about 6 s).
- **Runs:** 4 variants (base / cap15+32 / t0,0.2,0.4 / both) × {VIVOS × DEMAND, VIVOS × MUSAN at clean, 20, 10, 5, 0 dB; FLEURS clean}, run sequentially. Output: `results/stall_*.csv`.

## Implementation notes (Part A)

- `health.py`: `assess` (pure), `unclear_reason`, `HealthTracker` (quality hold, down-hold, 60 s history used for `health_max`).
- `Segmenter.open_s`; `WhisperASR.last_temperature` (observation only, decoding unchanged); ASR worker stamps `_asr_busy` and posts `("unclear", sid, seg, why)` to the monitor queue.
- Monitor `_tick_health`: rate-limited; emits `HEALTH` when the light changes (`data.changed=True`) or once a second while not green (`changed=False`, numbers only); the emit re-checks the session under the lock, so no event follows STOPPED.
- The CLI prints only `changed` events; the GUI ignores HEALTH whose session is no longer current and greys the light on any state other than RUNNING.

## Open questions (answered above)

1. Add quality trigger (c), "temperature fallback used"? My proposal: yes, observation only.
2. Is the order of reasons when levels tie (backlog > length > quality) acceptable?
3. Part B variants: are the three variants above enough, or do you also want temperatures-only without the cap?
4. Can you send the path of the log with the 20 031 ms segment, so I can check its `audio_s`?
