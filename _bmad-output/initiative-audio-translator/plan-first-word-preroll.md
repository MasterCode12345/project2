---
title: 'Lost first word: separate the pre-roll length from the start-vote window'
type: 'fix'
created: '2026-11-15'
status: 'plan — waiting for approval (step 0)'
baseline_revision: '714f596'
branch: 'fix/first-word-preroll'
route: 'full'
risk: 'low-medium'
---

## Intent

The first word of an utterance is sometimes lost (reported: "đồ án" → "án").

**Hypothesis.** In `segmenter.py`, the idle ring of 10 frames (300 ms) is used for two things:
- the start vote: trigger when ≥ 5 of the last 10 frames are voiced;
- the pre-roll: on trigger, the ring becomes the start of the utterance.

Silero needs time to reach `p ≥ 0.5` on a soft first syllable. If more than about 150 ms of that syllable is spoken before the 5th voiced frame, it is already outside the ring, so it is cut.

**Verdict of step 0: confirmed**, with the measurement below. Doubling the pre-roll alone brings VIVOS back to the level of transcribing the whole utterance. It removes about half of the lost first words on FLEURS.

## Evidence (step 0)

### 1. Real logs (`~/gr2_manual/`, read only; nothing copied into the repo)

- 16 CSV files, 108 segments with text.
- Only **22** of them match one of the 28 sentences of `docs/manual_test.md` (a), (b) and (g) (fuzzy match ≥ 0.5). The rest is free speech without a reference.
- **Of the 22 matched segments, none** has a first word that was clearly cut. The 3 flagged by the rule "first reference word not among the first 2 words" are different sentences that matched by chance, not clipped ones.
- **Free speech (86 segments), heuristic only:** 7 segments start with a fragment that is not a Vietnamese word:
  - 4 begin with a bare consonant or a broken syllable (such as the reported "án");
  - 3 begin with `unk`, Whisper's junk token, which these logs predate the stripping of.

  None of the 7 follows a forced cut. The gaps to the previous segment are 0.6–13 s, so these are real utterance starts.

**Limit.** The real-log sample is small and mostly has no reference. It shows that the symptom exists, but not how often. The rate comes from the simulation below.

### 2. Simulated stream, Whisper as in the app

Script: scratchpad, not in the repo.
- **Stream.** 2 s of silence, then each utterance followed by 1.5 s of silence, as one continuous stream. One `SileroVad(0.5)` for the whole stream, as in a session.
- **Segmentation and ASR.** The real `Segmenter` frame by frame (30 ms), with `silence_ms` 600 and `max_segment_s` 15. ASR is PhoWhisper-small, int8, beam 5, with the app's default filter. The decoder is seeded per segment.
- **Pre-roll variants.** `Segmenter` subclassed so the vote still uses the last 10 frames while the pre-roll keeps the last N frames.
- **"Whole utterance".** Each utterance transcribed alone, padded with 0.5 s of silence on each side. This is what the ASR can do with nothing cut.
- **"First word lost".** The first reference word is not among the first 2 words of the hypothesis (both normalised).

| Set (clean) | Run | Segments | First word lost | Also lost in the whole-utterance run | WER | CER |
|---|---|---|---|---|---|---|
| VIVOS 200 | whole utterance | – | 15 (7.5 %) | – | 7.02 % | – |
| VIVOS 200 | pre-roll **300** (today) | 202 | **25 (12.5 %)** | 10 | 7.65 % | 3.74 % |
| VIVOS 200 | pre-roll **600** | 202 | **15 (7.5 %)** | 9 | **6.97 %** | 3.26 % |
| FLEURS 100 | whole utterance | – | 17 (17 %) | – | 13.65 % | – |
| FLEURS 100 | pre-roll **300** (today) | 269 | **43 (43 %)** | 13 | 17.22 % | 11.15 % |
| FLEURS 100 | pre-roll **600** | 269 | **23 (23 %)** | 11 | **16.77 %** | 10.81 % |

**What the first-word losses look like at 300 ms.** Seen in the samples, which come from the public corpora:
- the first syllable reduced to its last consonant: "âm thầm" → "m thầm", "mong anh" → "m anh", "nó liên quan" → "n liên quan";
- the first syllable missing altogether: "màu son" → "son", "mặc dù" → "dù", "ví dụ" → "dụ", "tương tự" → "ương tự".

At 600 ms most of these come back intact. What remains is mostly ASR errors that the whole-utterance run also makes ("hằng" → "hàng", "loáng" → "lóng").

**Reading.**
- The number of segments does not change (202, 269), so the pre-roll does not change *where* segments are cut, only how much audio comes before them.
- On FLEURS, 600 ms still leaves 23 % against 17 % for the whole utterance. FLEURS speakers pause inside sentences (269 segments for 100 sentences), so a sentence often starts on a soft onset. A longer pre-roll may help further (see question 3).

**Limitations of step 0.**
- Clean audio only, one seed, 450 not measured, no ghost count. All of that is what step 1 measures.
- The metric only looks at the first word of each *sentence*. Starts of later segments inside a FLEURS sentence are also clipped, but only show up in the WER.

**Rejected measurement.** I first tried a Silero-only check, timing the segment start against an energy-based speech onset. The energy onset caught breathing and room noise in the recordings (FLEURS: "clipped" by seconds), so it says nothing and is not used.

## Design

### 1. Segmenter: pre-roll separate from the vote

- `Segmenter.__init__` gets the vote window as its own constant, `START_WINDOW_MS = 300`. It is not a setting.
- `start_votes = ceil(window_frames / 2)`, so the trigger rule is unchanged: ≥ 5 of the last 10 frames.
- **History while idle.**
  - A deque of `max(window_frames, pre_roll_ms // frame_ms)` frames keeps the idle frames.
  - The vote counts only the last `window_frames` of them, through a running count kept together with a deque of the last 10 flags, so it stays O(1) per frame.
- **On trigger.** The utterance starts with the last `pre_roll_frames` idle frames, the trigger frame included.
  - At the start of a stream, or right after an utterance ended, fewer idle frames may exist. The utterance then starts with what there is: never audio from before the session, and never audio already sent in the previous utterance.
  - `speech_start` is the time of the first frame kept.
- **Unchanged:** the forced cut and its continuation (no pre-roll, no lost audio), the end after `silence_ms`, the discard of segments with no voiced frame, `reset()`, and the VAD (thresholds untouched).
- `pre_roll_ms < START_WINDOW_MS` is allowed and keeps fewer frames than the vote window. Today's behaviour is `pre_roll_ms = 300`.

### 2. Length light

`open_s` (used by the length signal of the speaker light) counts every frame of the open utterance, so a longer pre-roll would turn the light on earlier: +300 ms at a pre-roll of 600.

- Proposed: `open_s` counts from the start of the vote window, which is the trigger frame minus 9 frames. That is exactly today's value, whatever `pre_roll_ms` is. Detail in question 1.
- A new read-only `Segmenter.preroll_s` gives the extra audio, so a test checks `open_s` against the 300 ms baseline.

### 3. Forced cut (`max_segment_s`)

`max_segment_s` is not changed, as you asked. It counts frames of the utterance, pre-roll included, so at 600 ms a 15 s segment holds 0.3 s less speech before a forced cut.

I propose to keep it so. The point of the cap is the audio length given to Whisper (a 30 s window, the decode time), not speech time. Question 2.

### 4. Settings, CLI, GUI

- `Settings.pre_roll_ms` already exists (300) and stays the single knob.
- A `--pre-roll-ms` flag (CLI only, like the other segmentation flags) makes the manual A/B possible. Only the default changes, if the rule passes.

### 5. Bench: `eval/preroll_bench.py` (new, step 1)

It reuses `denoise_bench` (loaders, `noise_for`, `normalize_vi`, `error_rates`) and `asr_bench` (`load_fleurs`, `seed_decoder`, the app's `WhisperASR` + `filter_from_settings`). It never downloads anything.

- **Stream.**
  - 2 s warm-up, then for each utterance 1.5 s of gap followed by the utterance.
  - After every 10 utterances, a **10 s noise-only stretch** to count ghosts.
  - Noise (DEMAND ch01, MUSAN noise; 20 files, `--seed 0`) runs at a constant level over the whole stream, scaled to the SNR over the speech.
  - Clean = digital silence between utterances.
- **Pipeline.** The real `Segmenter` with `--pre-roll-ms` values, and one `SileroVad` per stream, frame by frame.
- **Per condition × pre-roll:**
  - WER and CER;
  - first word lost (the definition above), and the same count for the whole-utterance run on clean audio, as the floor;
  - segments: total, overlapping an utterance, and outside any utterance;
  - **ghosts** = segments wholly outside every utterance span (gaps and noise stretches) that still have text after the filter;
  - ASR ms p50 and p95 per segment;
  - mean segment audio length.
- **`--samples`:** a TSV with the reference and hypothesis of every sentence whose first word is lost, under any pre-roll. Public corpora only.
- **Grid.**

  | Set | Conditions | Pre-roll |
  |---|---|---|
  | VIVOS 200 | clean; DEMAND 10 / 5 dB; MUSAN 10 / 5 dB | 300 / 450 / 600 |
  | FLEURS 100 | clean; DEMAND 10 / 5 dB | 300 / 450 / 600 |

  About 2 h in total, run under `caffeinate -i` in the background. Output in `results/preroll_*.csv`.

### 6. Rule to change the default (yours, made measurable)

A pre-roll value P replaces 300 only if all of these hold against 300, in every cell (corpus × condition):
1. **First word lost clearly lower:** on each corpus, clean, at least 30 % fewer lost first words than at 300. In no noisy cell is the count higher than at 300. (Today's clean data: VIVOS −40 %, FLEURS −47 % at 600.)
2. **WER** not worse than at 300 by more than 0.1 point in any cell.
3. **Ghosts** not higher than at 300 in any cell.
4. **ASR p50** at most +10 % in every cell.

If several values pass, the smallest that gets within 1 point of the best first-word rate wins, because a shorter pre-roll means less audio and less ASR time. If none passes, the default stays 300 and I report back.

### 7. Tests (`tests/test_segmenter.py`)

- **Pre-roll start.** At a pre-roll of 300, 450 and 600, with enough idle frames before, the utterance starts exactly `pre_roll_ms` before the end of the trigger frame. Both the audio length and `speech_start` are checked.
- **Trigger unchanged.** The same frame pattern triggers at the same frame for every pre-roll: 5 of the last 10 voiced, scattered voiced frames, a vote just under the threshold.
- **Start of stream.** Trigger after only 6 frames with a pre-roll of 600: the utterance holds the 6 frames, no padding is added.
- **Right after an end.** Speech resumes 4 frames after an utterance ended: the new utterance holds no frame of the previous one.
- **Forced cut.** The continuation still starts on the next frame, with no pre-roll and no lost audio.
- **`open_s`.** Unchanged against a pre-roll of 300 (see design point 2).
- The existing tests pass unchanged at 300.
- **Bench smoke test.** Toy ASR plus energy VAD, offline.

### 8. Docs

- README: segmentation parameters, the step 1 result table and the decision.
- `docs/manual_test.md`: a step in (a) to compare `--pre-roll-ms 300` with the new default on sentences that start softly, for example ones beginning with "m", "n" or "v" ("Mong anh…", "Nó…", "Ví dụ…"). It checks the first word in the terminal.
- This plan: results and status.

## Steps

0. This plan (done; waiting for approval).
1. Segmenter change plus tests. The default stays 300, so the code alone changes nothing.
2. `eval/preroll_bench.py` plus its smoke test; run the grid.
3. Decide by the rule; change `Settings.pre_roll_ms` or keep 300; docs. Full test run, commit, push, no merge.

## Questions for you

1. **Length light.** Either count `open_s` from the start of the vote window, which is exactly today's value (proposed); or count it strictly from the trigger frame, which turns the light on about 0.27 s *later* than today, whatever the pre-roll.
2. **`max_segment_s`.** Either keep counting the pre-roll in the 15 s (proposed: it caps the audio Whisper gets); or count only from the vote window, which gives Whisper up to 15.3 s at a pre-roll of 600.
3. **Values to measure.** Measure only 300 / 450 / 600 as you specified; or add **900** as an extra column for information only, because FLEURS still loses more first words at 600 than when the whole utterance is transcribed (23 % against 17 %).
4. **"Clearly lower".** Is "at least 30 % fewer on each clean corpus, never more in noise" the right reading?
