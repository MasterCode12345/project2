---
title: 'Model manager — cached, background-loaded Whisper models with a selection API'
type: 'feature'
ticket: ''
created: '2026-10-07'
status: 'built'
baseline_revision: 'NO_VCS'
route: 'full'
route_source: 'auto'
risk: 'low'
review: 'quick'
review_source: 'pinned'
lenses_ran: ['quick']
review_loop_iteration: 0
context: []
---

<frozen-after-approval reason="human-owned intent — do not modify unless human renegotiates">

## Intent

**Problem:** The ASR model is fixed at start (`Settings.model_size`) and loaded once; trying another model means restarting and waiting for a full load (and a multi-GB download for an uncached model).

**Approach:** A model manager that keeps recently used Whisper models loaded in memory (bounded LRU), loads/downloads in the background, and reports which models are downloaded/loaded. The orchestrator exposes `list_models()` / `select_model(name)` and `MODEL` events so a front end can switch fast; the CLI gets `--list-models`. The GUI dropdown is a separate follow-up.

## Boundaries & Constraints

**Always:** Whisper ASR only. Catalog = multilingual models `tiny`, `base`, `small`, `medium`, `large-v3-turbo`, `large-v3` (`.en`/`distil-*` are English-only, excluded). Model can be changed only while no session is running (`select_model` while STARTING/RUNNING is refused with a clear message). Loading/downloading never runs in the caller's thread of `select_model` or in an audio callback. At most `max_loaded_models` (default 2; 8 GB RAM) stay resident; LRU eviction never evicts the model a session is using. A failed load/download emits an error event and leaves the previous model selected and usable.

**Never:** No GUI changes in this plan. No translation/TTS selection. No custom model paths, GPU/Metal, or deleting models from disk.

## I/O & Edge-Case Matrix

| Scenario | Input / State | Expected Output / Behavior | Error Handling |
|----------|--------------|---------------------------|----------------|
| Cached switch | `small`, `medium` loaded; `select_model("small")` | Ready immediately; no load call | none |
| Background preload | `select_model("large-v3-turbo")` while stopped | Returns at once; `MODEL` events loading → loaded | none |
| Not downloaded | Model not on disk | Status `not-downloaded` with approx size; selecting downloads then loads in background | Download failure → error event, previous model kept |
| Start during preload | `start()` while selected model still loading | Session waits for that same load (STARTING), no second load | Stop during the wait aborts as today |
| Eviction | Cap 2, third model loaded | LRU released; in-use model never evicted | none |
| Rapid re-selection | Several `select_model` calls quickly | Only the last becomes selected; earlier loads land in cache or are dropped, never applied | none |
| Select while running | `select_model` during a session | Refused; current session unaffected | Warning event / False return |
| Load failure | Unknown/corrupt model | Error event; previous model still selected | Error event |

</frozen-after-approval>

## Code Map

- `src/audio_translator/asr.py` -- `WhisperASR(model_size, …)` with `load()`/`loaded`/`transcribe`; reuse as the per-model unit, transcribe unchanged.
- `src/audio_translator/orchestrator.py` -- `_components()` builds one `WhisperASR` from `settings.model_size`; `_start_session` emits "Loading ASR model…" then `self._asr.load()`; `_asr_worker` calls `self._asr.transcribe`. Keep the `asr=` injection seam (tests rely on it).
- `src/audio_translator/cli.py` -- `--model` (free text) → `Settings.model_size`.
- `src/audio_translator/config.py` -- `Settings.model_size`, `asr_device`, `compute_type`, `beam_size`.
- `faster_whisper.utils.download_model(name, local_files_only=True)` raises when not cached → "downloaded" check. HF cache `~/.cache/huggingface/hub` holds `small`, `medium` now.
- `tests/conftest.py` -- `Harness`, `FakeASR`, `EventLog` for orchestrator tests.

## Tasks & Acceptance

**Execution:**
- [x] `src/audio_translator/model_manager.py` -- `ModelManager`: catalog (name, approx size), `status(name)` (not-downloaded / downloaded / loading / loaded / failed), `request(name)` background load into bounded LRU, `get(name)` blocking (worker threads only), `pin/unpin`; injectable recognizer factory and downloaded-check; change callback -- caching core
- [x] `src/audio_translator/config.py` -- `max_loaded_models: int = 2` -- config
- [x] `src/audio_translator/models.py` -- `MODEL` event kind -- contract
- [x] `src/audio_translator/orchestrator.py` -- own a `ModelManager` (unless `asr=` injected); `list_models()`, `select_model(name) -> bool`; `_start_session` gets + pins the selected recognizer, unpins on stop; forward manager changes as `MODEL` events -- wiring
- [x] `src/audio_translator/cli.py` -- `--list-models`; `--model` choices = catalog -- CLI
- [x] `tests/test_model_manager.py`, `tests/test_orchestrator.py` -- fake factory with gates; every matrix row -- regression
- [x] `README.md` -- models, sizes, memory note, `--list-models` -- docs

**Acceptance Criteria:**
- Given two models were loaded once in this process, when `select_model` switches between them, then each is ready in under 1 s without reloading.
- Given `uv run audio-translator --cli --list-models`, then the catalog prints with `small` and `medium` marked downloaded.
- Given `uv run pytest`, when run offline with no audio devices, then all tests pass.

## Implementation Notes

- `ModelManager.get(name, pin=True)` pins atomically so a racing background load cannot evict the session's model between get and pin. Eviction drops the last reference (no explicit unload API in faster-whisper).
- Orchestrator gained a `model_manager=` injection seam (tests use `FakeModelFactory` in `conftest.py`); with `asr=` injected, `list_models()` is `[]` and `select_model` returns False with a warning.
- `select_model` returns True when accepted (applied at once if loaded, otherwise applied when its background load finishes, latest call wins). Failures/unknown names emit `ERROR` and keep the previous model.
- Verification finding: on this machine `medium` is NOT fully downloaded — its HF snapshot lacks `model.bin` (a stalled `*.incomplete` blob, ~268 MB of ~1.5 GB). `--list-models` therefore correctly reports `medium` as `not-downloaded`; only `small` is complete.

## Plan Change Log

## Review Triage Log

Pass 1 (lens: quick) — high 0 / medium 2 / low 2 / false 1 / maybe-false 0.

| # | Finding | Verdict | Route | Evidence / action |
|---|---------|---------|-------|-------------------|
| 1 | AC2: `--list-models` shows `medium` not-downloaded | false | reject | Code is correct: local HF cache holds only a 268 MB `.incomplete` blob for medium (interrupted download); reported to user. |
| 2 | Start during preload + load failure fails the session instead of falling back | medium | patch | `_start_session` picks `_pending`, `get` raises → `_fail_start`; violates "previous model usable". Patch: fall back to `_selected`. |
| 3 | LRU cap enforced only after the new model loads (cap+1 resident) | medium | patch | `_evict_locked` only in success branch of `_load`; 8 GB target can exceed with large-v3. Patch: evict to cap-1 before loading. |
| 4 | `Settings.model_size` outside catalog only fails at start | low | reject | CLI restricts choices; default is in catalog; only programmatic misuse; fix adds a guard. |
| 5 | LOADED event for a just-selected model says `selected: False` | low | patch | `_notify(LOADED)` precedes `job.callbacks`. Patch: run callbacks first. Unlocked `_selected` read race is negligible (reject part). |

## Design Notes

"Latest wins" selection, so a slow earlier load never overrides a later choice:

```python
token = self._select_token = object()
def on_loaded(name):
    if self._select_token is token:
        self._selected = name
```

## Verification

**Commands:**
- `uv run pytest -q` -- expected: all pass
- `uv run audio-translator --cli --list-models` -- expected: catalog, `small`/`medium` downloaded
