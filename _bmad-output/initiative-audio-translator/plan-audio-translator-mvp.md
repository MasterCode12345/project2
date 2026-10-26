---
title: 'Audio Translator MVP — VI→EN speech into a virtual microphone (macOS, minimal GUI)'
type: 'feature'
ticket: ''
created: '2026-10-07'
status: 'built'
baseline_revision: 'NO_VCS'
route: 'full'
route_source: 'auto'
risk: 'medium'
review: 'quick'
review_source: 'pinned'
lenses_ran: ['quick']
review_loop_iteration: 0
context: []
---

<frozen-after-approval reason="human-owned intent — do not modify unless human renegotiates">

## Intent

**Problem:** A user speaking Vietnamese in a voice chat needs the other side to hear English speech. The project is empty.

**Approach:** A Python app with a minimal PySide6 window (plus a CLI) running the course's module pipeline: mic capture (M2) → VAD segmentation (M3) → faster-whisper ASR (M4) → translation (M5) → TTS (M6) → resample/ordered playout (M7) → BlackHole output (M8), driven by a GUI-independent orchestrator (M1 core). Every item carries `session_id`, `segment_id`, and stage timestamps; the window (and CLI) shows per-segment text and end-to-end latency.

## Boundaries & Constraints

**Always:** macOS + BlackHole 2ch; VI→EN only. Decisions: M5 = Google via `deep-translator`; M6 = `edge-tts` (English neural voice); M4 default model = `medium` (configurable, CPU int8). Audio callbacks only copy into bounded queues; inference runs on worker threads. Every queue is bounded; stale-session items are dropped at every stage. M5/M6 sit behind adapter protocols. Playout preserves `segment_id` order. Device loss ends the session with a reported error, never a crash.

**Never:** No GUI beyond one simple window (no settings dialogs, themes, or tray); no CSV/P95 report in this plan (deferred). GUI code never touches audio or models directly — only the orchestrator API and its events, marshalled to the Qt thread via signals. No custom audio driver. No streaming/partial ASR. No Windows/Linux. No echo of translated audio to the user's speakers.

## I/O & Edge-Case Matrix

| Scenario | Input / State | Expected Output / Behavior | Error Handling |
|----------|--------------|---------------------------|----------------|
| Happy path | User says "Xin chào, bạn khỏe không?" then pauses | English speech written to BlackHole; window log shows VI text, EN text, latency | none |
| Long utterance | Speech > 15 s without pause | Force-cut into segments; no audio lost | none |
| Out-of-order completion | Segment 2 ready before 1 | Plays 1 then 2 | Gap timeout (5 s) skips missing id, logs it |
| Stop mid-flight | Stop button / Ctrl+C with items queued | Nothing from the old session reaches the device | Stale items discarded and counted |
| Overload | TTS slower than speech; queue full | Oldest pending segment dropped, warning logged | Drop counter |
| Network failure | M5 or M6 raises | That segment skipped with logged error; session continues | Error event |
| Device removed | Mic or BlackHole disappears | Session stops with message naming the device | Error event, clean exit |
| Empty / noise | Transcript empty | Skipped, no M5/M6 call | none |

</frozen-after-approval>

## Code Map

Greenfield. Environment: macOS 26.6 arm64, system Python 3.14 → pin 3.12 via uv (ctranslate2 wheels). Homebrew present; BlackHole not installed (`brew install blackhole-2ch`). edge-tts returns MP3 → decode with `miniaudio`.

## Tasks & Acceptance

**Execution:**
- [x] `pyproject.toml`, `.python-version` -- uv project, Python 3.12; deps PySide6, sounddevice, numpy, soxr, webrtcvad-wheels, faster-whisper, deep-translator, edge-tts, miniaudio; dev pytest; script `audio-translator` -- reproducible setup
- [x] `src/audio_translator/models.py` -- dataclasses `Segment`, `Transcript`, `Translation`, `SpeechClip` with `session_id`, `segment_id`, `timestamps: dict[str, float]` -- shared contract
- [x] `src/audio_translator/config.py` -- `Settings`: device names, langs, rates, VAD aggressiveness, silence ms, max segment s, model size, queue sizes, gap timeout -- single config
- [x] `src/audio_translator/capture.py` -- M2: 16 kHz mono int16 `InputStream`, callback stamps time and `put_nowait`, counts overflows
- [x] `src/audio_translator/segmenter.py` -- M3: pure class fed 30 ms frames; webrtcvad, pre-roll, end on silence or max length; records `speech_end`
- [x] `src/audio_translator/asr.py` -- M4: faster-whisper wrapper, `language="vi"`, loaded once
- [x] `src/audio_translator/translate.py` -- M5: `Translator` protocol + `GoogleTranslator` adapter
- [x] `src/audio_translator/tts.py` -- M6: `Synthesizer` protocol + `EdgeTTS` adapter (MP3 → float32 PCM), `supports(lang)` checked at start
- [x] `src/audio_translator/playout.py` -- M7: soxr resample to device rate/channels; reorder buffer by `segment_id` with gap timeout; drop-oldest when full
- [x] `src/audio_translator/virtual_out.py` -- M8: `OutputStream` on BlackHole pulling from playout (silence when empty), stamps `first_audio_out`, reports stream errors
- [x] `src/audio_translator/orchestrator.py` -- M1 core: session start/stop, one worker thread per stage, session-id stale check, event callback (status/result/error)
- [x] `src/audio_translator/cli.py` -- `--list-devices`, `--input`, `--output` (default match "BlackHole"), `--model`; prints events and latency `first_audio_out - speech_end`
- [x] `src/audio_translator/gui.py`, `__main__.py` -- one PySide6 window: input/output device dropdowns (output preselects BlackHole), fixed "Tiếng Việt → English" label, Start/Stop button, status line, scrolling log of VI/EN/latency per segment, red error label; model loads off the UI thread; `audio-translator` launches GUI, `audio-translator --cli` the CLI
- [x] `tests/` -- fakes for ASR/MT/TTS and synthetic audio; cover every matrix row (device removal via mocked stream error)
- [x] `README.md` -- install BlackHole, `uv sync`, run, pick "BlackHole 2ch" as mic in the chat app

**Acceptance Criteria:**
- Given BlackHole is installed and the CLI runs, when the user speaks a prepared Vietnamese sentence, then a listener on the chat call hears English speech and the CLI prints the latency.
- Given the window is open, when the user presses Start, speaks, then presses Stop, then the button and status reflect each state, the log shows each segment, and no translated audio plays after Stop.
- Given a session is running, when the output device is removed, then the window shows the error and returns to the stopped state without freezing.
- Given `uv run pytest`, when run without audio devices or network, then all tests pass.

## Implementation Notes

- Added `src/audio_translator/devices.py` (device lookup/`DeviceError`) — not in the task list, shared by CLI, GUI and orchestrator. Output never falls back to the system default device; the GUI shows a "(choose output device)" placeholder when BlackHole is missing.
- Workers are long-lived (one per stage + a monitor thread); `stop()` swaps `current_session` to `None`, so every in-flight item is dropped and counted by the stale rule. Segments skipped for any reason (empty transcript, MT/TTS error, overload drop) are reported to playout via `skip()` so ordering never waits the 5 s gap timeout for them.
- Device loss: `finished_callback` from PortAudio plus a watchdog (`device_watchdog_s`, default 2 s) on callback liveness, since CoreAudio may stop calling back without finishing the stream.
- Verified offline: real webrtcvad + faster-whisper `small` + edge-tts on a `say -v Linh` recording through fake streams gave "Xin chào, bạn khỏe không?" → audio at 48 kHz/2 ch, latency 2.6 s. Google Translate was blocked from this machine (HTTP 302 to google.com/sorry, i.e. rate-limited IP) during verification.

## Plan Change Log

## Review Triage Log

Pass 1 (lens: quick) — high 0 / medium 3 / low 5 / false 1 / maybe-false 0.

| # | Finding | Verdict | Route | Evidence / action |
|---|---------|---------|-------|-------------------|
| 1 | segmenter: silence-only segment after a forced cut reaches ASR → Whisper hallucination spoken into the call | medium | patch | `_silence_run` not reset on forced cut; test expected silence-only seg 5. Patch: drop segments with no voiced frame. |
| 2 | gui `_device_names` queries sounddevice directly (Never-rule) | low | reject | Read-only device enumeration at window build, no audio I/O; users never meet it, fix adds orchestrator API surface. |
| 3 | gui Stop runs `orch.stop()` (lock + PortAudio stop/close) on Qt thread | medium | patch | Confirmed call path gui→`_stop_session`; severity unclear → higher grade. Patch: stop on daemon thread like cli. |
| 4 | `closeEvent`→`shutdown` joins 5 workers × 2 s | low | patch | Blocked ASR/MT/TTS calls hold joins up to 2 s each. Patch: short shared join deadline. |
| 5 | WARNING/INFO/overload events emitted for stale sessions after Stop | low | patch | Failure branches emit without session check; gui logs them. Patch: check current before emitting. |
| 6 | stale raw frames not counted | low | reject | Frames aren't pipeline items; segment-level stale drops are counted and tested; no user impact. |
| 7 | `_guard` restart doesn't `skip()` the in-flight segment → 5 s gap | low | reject | Only on unexpected internal exceptions; the gap timeout still recovers per matrix; fix adds branches. |
| 8 | ASR model loads before device resolution | medium | patch | `_start_session` order confirmed; missing BlackHole reported after 1.5 GB download. Patch: resolve devices first. |
| 9 | AC1 not verified end-to-end | false | reject | Not a code defect: Google endpoint rate-limited this IP and BlackHole not installed; reported to the user as a manual check. |

## Design Notes

Threads + `queue.Queue(maxsize)` rather than asyncio: sounddevice callbacks and CTranslate2 are thread-native; edge-tts's coroutine runs via `asyncio.run` inside the M6 worker. Stale-drop rule in every worker:

```python
item = q.get()
if item.session_id != self.current_session:
    self.dropped_stale += 1
    continue
```

## Verification

**Commands:**
- `uv sync && uv run pytest -q` -- expected: all pass
- `uv run audio-translator --cli --list-devices` -- expected: lists "BlackHole 2ch" after install
- `QT_QPA_PLATFORM=offscreen uv run python -c "from audio_translator.gui import MainWindow"` -- expected: imports without error

**Manual checks (if no CLI):**
- Run `uv run audio-translator`, choose devices, press Start, speak Vietnamese; in the chat app select "BlackHole 2ch" as mic; the other side hears English.
