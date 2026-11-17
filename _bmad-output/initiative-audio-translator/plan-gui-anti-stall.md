---
title: 'GUI checkbox for anti-stall (c4b): apply the decoding settings to a model that is already loaded'
type: 'feature'
created: '2026-11-17'
status: 'done — option A; checkbox off by default (c4b); branch pushed, not merged'
baseline_revision: 'f355dcf'
branch: 'feat/gui-anti-stall'
route: 'quick'
risk: 'low'
---

## Intent

The supervisor approved the current latency and left segment merging and anti-stall to the student. So:
- **Anti-stall** gets a checkbox in Cài đặt, "Chống kẹt nhận dạng (thử nghiệm)", off by default; on = preset `c4b`.
- **Segment merging** before MT is not done; it is recorded as an extension. Only the docs change for this.

## Finding: toggling between sessions has no effect today (reproduced)

`ModelManager` keeps one `WhisperASR` per model name. That object is built by `Orchestrator._make_recognizer` from the settings **at load time**, and `WhisperASR.load()` is the only place that attaches the `DecodeLimiter`. The GUI writes its settings right before `start()`, so a model that is already loaded ignores the new anti-stall settings.

Reproduced with a fake stream and a fake `WhisperModel`. The model manager and orchestrator were real; the script is in the scratchpad, not in the repo:

```
session 1: settings=off  recognizer decode='base' limiter=no
session 2: settings=c4b  recognizer decode='base' limiter=no   <- checked, no effect
session 3: settings=off  recognizer decode='base' limiter=no
```

The other direction fails the same way: an app opened with `--asr-anti-stall c4b` keeps its limiter after the box is unchecked.

## Options

**A (proposed): re-apply the decoding options to the model at every session start.** No reload.

- `asr.py`:
  - `decoding_from_settings(s)` maps the `asr_*` decoding fields of `Settings` to `WhisperASR` options. It is the single place for that mapping: `_make_recognizer` uses it too.
  - New `WhisperASR.set_decoding(**opts)` sets those attributes. If the model is loaded, it also:
    - attaches a `DecodeLimiter` when `uses_limiter` is true and none is attached;
    - updates the rule flags of an attached one;
    - **detaches** it (puts the inner CTranslate2 model back) when it is no longer needed.

  The model weights are untouched, so nothing reloads and the call takes microseconds.
- `orchestrator.py` (**needs your OK**):
  - In `_start_session`, right after `rec = self._models.get(name, pin=True)`, call `rec.set_decoding(**decoding_from_settings(s))` when the recognizer has that method. An injected ASR (tests, fakes) is used as-is, as today.
  - `_make_recognizer` uses `decoding_from_settings` (same result as today).
  - In `_open_latency_log`, add a session field `asr_anti_stall`: `off` / `c4a` / `c4b` / `custom`, read from the recognizer that really runs. A preset name is written only if its limiter is really attached; otherwise `off`.
- **Thread safety.** `set_decoding` runs while no session runs. A segment of the previous session may still be inside the ASR worker; if it sees the limiter change mid-call, nothing breaks, and its result is dropped by session id anyway. A lock would make Start wait for that segment, up to the length of a stall (about 36 s), so there is none.
- `model_manager.py`: **unchanged**.

**B: reload the model when the decoding options changed.**
- Simple semantics, but every toggle costs a model reload: about 1–2 s for phowhisper-small, more for medium.
- It needs a `model_manager.py` change: either a cache key of name + options, or an evict API.
- Not proposed.

## GUI

- `QCheckBox` "Chống kẹt nhận dạng (thử nghiệm)" in Cài đặt. Tooltip: "Giảm thời gian kẹt nhận dạng lâu nhất (khoảng 36 s xuống khoảng 7,5 s trong bench), nhưng trong nhiễu rất nặng có thể mất vài câu. Có hiệu lực từ lần Bắt đầu tiếp theo." The numbers come from `results/antistall_step1_c3_c4.csv`: FLEURS clean, max 35.9 s → 7.5 s with C4b; DEMAND 0 dB loses whole sentences.
- **Initial state** from the settings (the preset is detected from the `asr_*` fields):

  | Detected preset | Box | Behaviour |
  |---|---|---|
  | off | unchecked | |
  | c4b | checked | |
  | c4a | checked | c4a is kept until the user clicks |
  | other non-default fields (code only) | checked | kept as they are until clicked |

- **On start** (`_apply_settings`, only if the user clicked the box): reset every anti-stall field to its default, then apply `c4b` or `off`.
- **Locked while starting/running/stopping**, like the other settings; it takes effect at the next Start.

## Latency log and report

- New column `asr_anti_stall` (a session field, next to `asr_decode`).
- `latency_report.py` shows it on the config line: `config: … asr_decode=… anti_stall=c4b …`, so each session shows whether anti-stall was on.
- Old logs without the column show `?`.

## Tests

- **GUI:**
  - initial state for off / c4b / c4a (c4a kept when untouched; c4b after uncheck and check again; off after uncheck);
  - locked in starting/running/stopping, enabled when stopped;
  - the tooltip text.
- **`WhisperASR.set_decoding`** (fake `faster_whisper`): attach / update / detach on a loaded model; before load, only attributes change, and `load()` attaches as today.
- **Orchestrator end to end** (fake streams, fake `WhisperModel`, real `ModelManager` + `_make_recognizer`):
  - session 1 off → no limiter; session 2 c4b → limiter attached and `decode` = the c4b label; session 3 off → detached;
  - the model loaded only once;
  - the latency log writes `off`, `c4b`, `off` for the three sessions, and the report prints them.

## Docs

- README, README.en.md, `plan-asr-anti-stall.md`:
  - **segment merging before MT** not done (latency first; the supervisor approved the current latency; translation quality in real use is good enough), listed as an extension;
  - **anti-stall**: a GUI checkbox, off by default; pre-roll 600 ms already lowered ASR p95 in noise from 5–7.5 s to 1.4–2.4 s.
- `docs/manual_test.md`: a step that toggles the box between two sessions and checks `anti_stall=` in `latency_report`.

## Approval (human, 2026-11-17)

**Option A**, with the 3 orchestrator changes of the plan; `model_manager.py` unchanged. Added requirements:
1. **Safety while an old segment is decoding.**
   - `set_decoding` must not change the limiter in use. It builds a new limiter (or a new configuration) and swaps one reference, so a running `transcribe` call never changes rules halfway.
   - The values read after decoding (`last_hit_limit`, `last_all_truncated`, …) must not fail if the limiter was just removed or replaced.
   - An error while decoding an old segment must not stop the ASR thread.
   - Test with a slow fake model: the ASR thread stays alive, the old segment is dropped, the new session runs the new rules.
2. **The right model:** the settings go to the model the session really uses, also the fallback (`small`). With two models loaded, whichever one the next session uses gets the new settings.
3. **`asr_anti_stall`** in the latency log comes from the model's real state in that session.

## Implementation

- **`asr.py`.**
  - `Decoding` is a frozen dataclass with the 10 decoding options; `DECODING_OPTIONS`; `decoding_from_settings`; `anti_stall_label`.
  - `WhisperASR` keeps one `Decoding`. The 10 option attributes stay readable and settable; setting one is `set_decoding`.
  - `set_decoding(**options)`, under a short lock and never while decoding:
    - builds the new `Decoding` and a **new** `DecodeLimiter` around the CTranslate2 model (or no limiter);
    - swaps both together (`WhisperModel.model` and the options);
    - returns False when nothing changed; the limiter in place is then kept.
- **A running call keeps its rules.**
  - faster-whisper reads `self.model` again for every attempt (faster_whisper 1.2.1 `transcribe.py:1446`, inside the temperature loop). It never assigns to `self` after `__init__`, and `WhisperModel` has no `__del__`, `__slots__` or copy hooks.
  - So `transcribe_segments` takes, at its start and under the lock, the options and a shallow copy of the `WhisperModel`. The copy shares the weights and holds the limiter (or none) of that moment. Every attempt of that call goes through that object, even if `set_decoding` swapped the one on the model meanwhile.
  - Everything read after decoding uses the call's own limiter and options, never `self.limiter`.
- **The ASR thread** already catches an exception from `transcribe`. `_skip_report` ignores segments of an old session, so an old segment that fails or finishes after Stop is dropped silently. Unchanged; covered by a test.
- **`orchestrator.py`, 3 changes.**
  - `_make_recognizer` uses `decoding_from_settings`.
  - `_start_session` calls `rec.set_decoding(**decoding_from_settings(s))` on the recognizer finally chosen: after the fallback and after the "failed pending selection" path. A recognizer without the method (fakes, an injected ASR) is left as it is.
  - `_open_latency_log` writes `asr_anti_stall` = `WhisperASR.anti_stall`, i.e. the attached limiter and the options at session start. A preset name only with its limiter attached.
- **`latency_log.py` / `latency_report.py`.**
  - Session field `asr_anti_stall` after `asr_decode`.
  - The report shows `anti_stall=` on each config line, and a line `anti-stall per session: 1=off, 2=c4b, …`.
- **`config.py`:** `ANTI_STALL_FIELDS`, `anti_stall_preset(settings)`, `apply_anti_stall_preset(settings, name)` (every decoding field back to its default first).
- **GUI.**
  - `QCheckBox` "Chống kẹt nhận dạng (thử nghiệm)" with its tooltip.
  - The initial state comes from `anti_stall_preset`.
  - `clicked` (user only) marks it touched. Only then does Start apply `c4b` or `off`; untouched, `c4a` stays.
  - Locked with the other settings.
- **Not in scope:** the ASR filter and the initial prompt of a cached model are still those of its load time. The GUI changes neither.

## Results

**Fake-stream run, three sessions (off → c4b → off).**
- Real `ModelManager`, orchestrator and `_make_recognizer`; fake streams and a fake `WhisperModel`. One sentence spoken per session.
- The script is in the scratchpad, not in the repo; the same flow is in `tests/test_anti_stall_switch.py`.

```
session 1: settings=off  recognizer decode='base' limiter=no decoded_through=CT2 last_call=off
session 2: settings=c4b  recognizer decode='cap20+32,budget4s,reject:trunc+loop' limiter=yes decoded_through=DecodeLimiter last_call=c4b
session 3: settings=off  recognizer decode='base' limiter=no decoded_through=CT2 last_call=off
model loads: 1
latency log: session 1 asr_anti_stall=off asr_decode=base
latency log: session 2 asr_anti_stall=c4b asr_decode=cap20+32,budget4s,reject:trunc+loop
latency log: session 3 asr_anti_stall=off asr_decode=base
anti-stall per session: 1=off, 2=c4b, 3=off
```

Before the change, session 2 had `limiter=no` and `decode='base'` (Finding above).

**Tests** (`tests/test_anti_stall_switch.py`, `tests/test_gui_redesign.py`):
- **Settings helpers:** the presets round-trip, `custom` is detected, and `off` resets every field.
- **`set_decoding`:**
  - before load, only the options change;
  - attach, then replace with a new object (the old limiter is unchanged);
  - "unchanged" keeps the same limiter;
  - detach;
  - an attribute assignment goes through it;
  - an unknown option is refused;
  - labels: a preset name only with its limiter attached.
- **A decode in progress** (off → c4b and c4b → off):
  - both attempts of the running call go through its own object, and its after-call values come from its own limiter;
  - `set_decoding` returns at once (under 0.5 s);
  - the next call uses the new rules.
- **Three sessions on one loaded model:** limiter no / yes / no; loaded once; log and report as above.
- **The fallback model** (phowhisper-small not converted → small) gets c4b; the log shows `small` / `c4b`.
- **Two models loaded:**
  - the one used next gets the settings; the other is left alone until it is used;
  - the first is switched back and gets c4b, then off;
  - no reload.
- **Settings changed while an old segment decodes slowly** (also when that old call raises):
  - Start does not wait;
  - the old call keeps the old rules to its last attempt and is dropped, never reported;
  - the ASR thread stays alive;
  - session 2 decodes through a limiter;
  - the log has only session 2, with `c4b`.
- **GUI:**
  - initial state off / c4b / c4a; c4a is kept when untouched; click → c4b, click again → off;
  - locked in starting / running / stopping, and a disabled box ignores clicks;
  - the tooltip names 36 s / 7.5 s;
  - the "no English on the UI" check now covers checkboxes too.
- **Mutation checks** (temporary, reverted):
  - without the orchestrator call, 5 of the new tests fail;
  - without the per-call view, 4 fail;
  - if `set_decoding` never swaps the limiter, 8 fail.

## Decisions recorded (docs)

- **Segment merging before MT: not done.** It would wait for the next segment, so it adds latency. Latency comes first (the supervisor approved the current latency), and translation quality in real use is good enough. Listed as an extension in README §9.
- **Anti-stall:** a GUI checkbox, off by default (c4b). The 600 ms pre-roll already lowered ASR p95 in noise from 5–7.5 s to 1.4–2.4 s (`results/preroll_*.csv`).
