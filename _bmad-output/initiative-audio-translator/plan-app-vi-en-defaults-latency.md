---
title: 'Lock in the Vietnamese → English app configuration (PhoWhisper-small default) + real-app latency log, report, manual test script, README refresh'
type: 'feature'
created: '2026-11-11'
status: 'done'
baseline_revision: '3b54550'
branch: 'feat/app-vi-en-defaults-latency'
route: 'full'
risk: 'medium'
context:
  - '{project-root}/_bmad-output/initiative-audio-translator/plan-guard-defaults.md'
  - '{project-root}/_bmad-output/initiative-audio-translator/plan-asr-vietnamese-accuracy.md'
---

## Intent

**Problem:** The benches settled the Vietnamese → English stack:
- PhoWhisper-small ASR (FLEURS WER 16.7% vs 27.3% for `small`, `results/asr_fleurs_demand.csv`);
- Silero + ASR filter;
- vibert-capu restoration (`restore=auto`);
- Argos MT.

The app still defaults to Whisper `medium`, and the README sections 1–3 and 6–9 describe the old state. There is also no per-stage latency measured in the real app: the timestamps exist on every item, but they are only printed.

**Approach:**
- Make `phowhisper-small` the default, with a safe fallback to `small` when it is not converted.
- Add an opt-in per-segment latency CSV, written from worker threads only.
- Add a report script and a manual test script for the human to run with a real mic, BlackHole and Meet.
- Bring the README in line with the code and with `results/`.

## Model check (2026-11-11, `~/.cache/audio-translator/models/`)

| Model | Dir | Size | Status |
|---|---|---|---|
| phowhisper-small | `phowhisper-small-ct2` | 464 MB (model.bin 461 MB) | converted |
| phowhisper-medium | `phowhisper-medium-ct2` | 1.4 GB | converted |
| vibert-capu | `vibert-capu-ct2` | 113 MB (model.bin 112 MB) | converted |
| nllb-600m | `nllb-600m-ct2` | 604 MB (model.bin 594 MB) | converted |
| (m2m100-418m) | `m2m100-418m-ct2` | 473 MB | converted |

Whisper `small` (the fallback) is already in the Hugging Face cache. All models are present, so no convert command is needed on this machine; the app's warning will still print one.

## Boundaries & Constraints

**Always:**
- Change only the default ASR model. Every other default stays as it is: Silero, `asr_filter=True`, `restore=auto`, `translator=offline`.
- Latency logging is off by default (`latency_log=None`). When it is off, nothing is opened or written and behaviour is unchanged.
- The CSV is written only from orchestrator worker threads: the monitor worker for played segments, and the stage workers or monitor for dropped ones. Audio callbacks keep doing only `put_nowait` and the `first_audio_out` timestamp they already set.
- A file error (cannot open, disk full) disables the log with one WARNING. It never stops the session.
- All tests are offline, using the existing fake streams.

**Never:**
- Run the app with a real mic or BlackHole. That is for the human, following `docs/manual_test.md`.
- Add dependencies.
- Change any other behaviour.
- Invent numbers in the README: every figure names its `results/` file.
- Push or open a PR without asking.

## Design

### 1. Default ASR model
- `Settings.model_size = "phowhisper-small"`, plus a new `Settings.asr_fallback_model = "small"`.
- The CLI `--model` default comes from `Settings`, and its help text is updated. The GUI dropdown already preselects `orchestrator.selected_model`, so it follows automatically.
- Fallback lives in `Orchestrator._start_session`, before `models.get`. If the model to load is in `asr.LOCAL_MODELS` and its `model.bin` is missing:
  - emit a `WARNING`: "PhoWhisper-small is not converted (<dir>); run: <convert_command>. Using 'small' for this session";
  - load `asr_fallback_model` instead.
  - `_selected` is kept, so converting later and restarting the session picks PhoWhisper up.
- An injected fixed ASR (tests) is unaffected. `--list-models` already shows "not converted" with the command.

### 2. Latency log (`latency_log.py`, new)
- `Settings.latency_log: str | None = None`; CLI `--latency-log PATH`.
- `LatencyLog(path)`:
  - opens in append mode and writes the header if the file is new or empty;
  - uses `csv.writer` and a lock, and flushes every row;
  - provides `played(clip, session_info)`, `dropped(item, reason, detail, session_info)` and `close()`.
- The orchestrator opens it at session start and closes it at session stop or shutdown.
- `session_info` is recorded once per row so each row stands alone: model actually used, VAD, filter, restorer, translator.
- **Columns:** `wall_time, session_id, segment_id, status (played|dropped), reason, asr_model, vad, asr_filter, restore, translator, forced_cut, audio_s, n_words_vi, n_words_en, speech_start, speech_end, segmented, asr_start, asr_end, mt_start, restored, mt_end, tts_start, tts_end, first_audio_out` (monotonic seconds, empty when not reached), then the stage durations in ms:
  - `segment_ms = segmented − speech_end`
  - `asr_wait_ms = asr_start − segmented`
  - `asr_ms`
  - `mt_wait_ms = mt_start − asr_end`
  - `restore_ms = restored − mt_start`
  - `mt_ms = mt_end − (restored or mt_start)`
  - `tts_wait_ms = tts_start − mt_end`
  - `tts_ms`
  - `playout_wait_ms = first_audio_out − tts_end`
  - `e2e_ms = first_audio_out − speech_end`

  The row ends with `text_vi` and `text_en`.
- **Drop reasons**, one row per dropped segment, logged where the skip already happens:
  - `asr_error`;
  - `no_speech`, with detail = the ASR filter reasons from `WhisperASR.dropped` (e.g. `blacklist`, `no_speech_prob`) when available, so ghost sentences are visible;
  - `mt_error`, `empty_translation`, `tts_error`;
  - `overload` (queue drop);
  - `playout_late` / `playout_overload` (Playout notices, via the monitor queue).
  - `text_vi` keeps the dropped text when there is one.
  - Stale items from a stopped session are not logged; they are counted in `stats()` as today.
- **Thread safety:** `WhisperASR.dropped` is read in the ASR worker right after `transcribe`. There is only one ASR worker, so no race.

### 3. `eval/latency_report.py` (new, stdlib + numpy)
- Reads one or more CSVs. `--session N` filters to one session.
- Prints, per stage and end to end: n, mean, P50, P95, max (ms), over played segments only.
- Prints counts of dropped segments by reason (and detail), and total played / dropped.
- `--markdown` prints the table ready to paste into the README.

### 4. `docs/manual_test.md` (new, Vietnamese)
- Prerequisites:
  - BlackHole 2ch;
  - **headphones** to avoid the speaker → mic loop;
  - models present (`--list-models`);
  - a quiet room for (a).
- **Three scenarios**, each with:
  - the exact command, with `--latency-log ~/gr2_latency_<scenario>.csv`;
  - 10 prepared Vietnamese sentences (names, numbers, dates, short and long, one question);
  - how to read the result (`uv run python eval/latency_report.py …`);
  - a result sheet: sentences heard correctly by the listener, and notes on ghosts or loops.

  The scenarios:
  - (a) CLI with no call: output to BlackHole, monitored with QuickTime / Audio MIDI;
  - (b) Google Meet with a phone as the listener;
  - (c) a noisy room (fan, music or café sound from a second device), same 10 sentences.
- At the end: which files to send back, i.e. the three CSVs and the filled result sheet.

### 5. README refresh
- §1: flow diagram covering mic → (NS) → Silero VAD/segmenter → PhoWhisper + ASR filter → vibert-capu restore → Argos (Google fallback) → Edge TTS → playout → BlackHole.
- §2: module table, adding `restore.py` and `latency_log.py`, and the Silero/filter in M3/M4.
- §3: technology and licences, adding PhoWhisper (BSD-3-Clause, per its model card; to be verified), Silero VAD (MIT, via faster-whisper), vibert-capu (CC-BY-SA-4.0), NLLB-200 (CC-BY-NC-4.0, optional) and M2M100 (MIT). Licences will be checked against the local `SOURCE.json`/model cards, not guessed.
- §6: split into "old manual measurements (Whisper small/medium, 2025)" and "bench results", each with its `results/` file. Real-app latency is marked "pending manual_test.md".
- §7, §8: move solved items into §7 (ghost sentences, offline MT, restoration). §8 keeps the real open problems: background speakers, segmentation loss in MT, Meet latency unmeasured, the 25 unreviewed references, TTS online.
- §9: roadmap updated.
- §5 and §11: one test count, taken from the final `pytest` run.

## Tasks & Acceptance

**Execution:**
- [x] `config.py`, `cli.py`, `orchestrator.py`: PhoWhisper-small default + fallback warning.
- [x] `latency_log.py` (new), `config.py`, `cli.py`, `orchestrator.py`: opt-in CSV, played and dropped rows.
- [x] `eval/latency_report.py` (new): mean/P50/P95/max table + drop reasons.
- [x] Tests:
  - `test_orchestrator.py`: default model, fallback with a warning and no crash, log off writes nothing, log rows for played and dropped segments with fake streams, stage ms consistent with timestamps, file error → warning and the session continues;
  - `test_latency_log.py`, `test_latency_report.py`;
  - CLI flags.
- [x] `docs/manual_test.md`.
- [x] `README.md` §1–3, §5–9, §11.

**Acceptance Criteria:**
- With default settings and PhoWhisper-small converted, a session uses `phowhisper-small`. If it is not converted, a WARNING with the convert command is shown, `small` is used and the session runs.
- With `--latency-log PATH`, each played segment writes one row whose `e2e_ms` equals `(first_audio_out − speech_end)·1000`. Each skipped segment writes one row with its reason. Nothing is written from an audio callback.
- `uv run --group eval pytest -q`: all pass.
- `eval/latency_report.py` on a CSV produced by the fake-stream test prints the table.

## Decisions (human, 2026-11-11)

1. Fallback = `small`. The warning must be visible in the GUI (red error line) and the CLI; the latency log names the model that really ran (`small`), not the selection.
2. No GUI field; `uv run audio-translator --latency-log PATH` opens the GUI with logging (documented in `docs/manual_test.md`).

## Implementation Notes

- `Orchestrator._usable_model` emits `WARNING` with `data.model_fallback/model/fallback/command`; `gui.on_event` puts it in `error_label`. `_selected` keeps `phowhisper-small`, so a later conversion is picked up at the next Start.
- `latency_log.LatencyLog`: append + header once, lock, flush per row. One row per segment (first outcome wins, `_lat_done`). `no_speech` detail = ASR filter reasons (`repetition` excluded, it keeps the text). `_lat_seg` carries audio length / forced cut from the segment worker. Rows are written by the stage workers (drops) and the monitor worker (played, playout notices).
- `gui.parse_args`: `--latency-log`, unknown args left for Qt.
- Licences checked locally: PhoWhisper bsd-3-clause (HF model card in cache), vibert-capu cc-by-sa-4.0 (SOURCE.json), Argos vi-en 1.9 CC-BY-4.0 (package README, OPUS-MT), faster-whisper/ctranslate2/onnxruntime MIT (package metadata), edge-tts LGPLv3.
- 288 tests pass (was 273).

## Open questions for the human (answered above)
1. Should `asr_fallback_model` be `small` (485 MB, cached here), or would you rather the start fail with the convert command? Plan: `small`, as you asked.
2. Should the `--latency-log` path also be a GUI field? Plan: no, CLI + Settings only. The GUI can read `Settings.latency_log` later.
