---
title: 'GUI model dropdown — choose the Whisper model in the main window'
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

**Problem:** The orchestrator can now list, preload and cache Whisper models (`list_models()`, `select_model()`, `MODEL` events), but the window has no way to use it; the user must use the CLI.

**Approach:** Add a "Model:" dropdown to the main window that shows every catalog model with its status, preloads the chosen one in the background via `select_model`, follows `MODEL`/`ERROR` events, and is locked while a session runs.

## Boundaries & Constraints

**Always:** GUI talks only to the orchestrator API (`list_models`, `select_model`, `selected_model`) and its events, on the Qt thread. Status text in Vietnamese: `đã tải` (downloaded), `chưa tải (~1.5 GB)` (not-downloaded), `đang tải…` (loading), `trong RAM` (loaded), `lỗi` (failed); the selected model is marked `✓`. Dropdown is enabled only in STOPPED/IDLE. A refused or failed selection returns the dropdown to `orch.selected_model`. Only user interaction triggers `select_model` (never programmatic index changes).

**Never:** No change to orchestrator, model manager or CLI. No settings persistence. No translation/TTS selection.

## I/O & Edge-Case Matrix

| Scenario | Input / State | Expected Output / Behavior | Error Handling |
|----------|--------------|---------------------------|----------------|
| Initial view | Window opens, `medium` selected, `small` downloaded | Dropdown lists 6 models with status; `medium ✓` current | none |
| Pick cached/downloaded model | User picks `small` | `select_model("small")` called once; label updates as `MODEL` events arrive; status line shows loading then selected | none |
| Pick not-downloaded model | User picks `large-v3` | Status line says downloading ~3.1 GB; window responsive | — |
| Load failure | `ERROR` "Could not load model…" | Error label shows it; dropdown returns to previous selected model | Error shown |
| Refused selection | `select_model` returns False | Dropdown returns to `selected_model` | Warning already logged |
| Session running | State STARTING/RUNNING/STOPPING | Dropdown disabled; re-enabled on STOPPED | none |
| No model API | Orchestrator without `list_models` (injected ASR) → empty list | Model row shows disabled "(cố định)" placeholder | none |

</frozen-after-approval>

## Code Map

- `src/audio_translator/gui.py` -- `MainWindow.__init__` builds `form` rows (Microphone / Output / Language); `on_event` dispatches STATUS/RESULT/ERROR/WARNING/INFO; `_apply_state` enables/disables combos by state. Add the model row, a `MODEL` branch and model-combo enablement here.
- `src/audio_translator/orchestrator.py` -- `list_models() -> list[ModelInfo]` (name, size_mb, status, selected; `size_label` property), `select_model(name) -> bool`, `selected_model`; `MODEL` events carry `data={"model","status","detail","selected"}`; selected confirmation has `detail == "selected"`. Load failure emits `ERROR` with `data={"model": name}`. Read-only for this plan.
- `src/audio_translator/model_manager.py` -- status constants `NOT_DOWNLOADED`, `DOWNLOADED`, `LOADING`, `LOADED`, `FAILED`; `CATALOG`.
- `tests/test_frontends.py` -- `FakeOrch` (no model API yet), `_window(qapp)`; extend FakeOrch with `list_models`/`select_model`/`selected_model`.

## Tasks & Acceptance

**Execution:**
- [x] `src/audio_translator/gui.py` -- add `model_combo` row "Model:" after Language; populate from `orch.list_models()` (guard missing API); item data = model name; connect `activated` → `orch.select_model`; on `MODEL` events refresh item texts and status line, set current to the selected model on `detail == "selected"`; on `ERROR` with `data["model"]` or False return, reset to `orch.selected_model`; enable only when stopped in `_apply_state` -- the feature
- [x] `tests/test_frontends.py` -- FakeOrch model API + tests covering every matrix row -- regression
- [x] `README.md` -- one line: model can be chosen in the window when stopped -- docs

**Acceptance Criteria:**
- Given the window is stopped, when the user picks another model, then the window stays responsive and the dropdown shows it as selected once loaded.
- Given `uv run pytest`, when run offline with no audio devices, then all tests pass.

## Implementation Notes

## Plan Change Log

## Review Triage Log

Pass 1 (lens: quick) — high 0 / medium 1 / low 4 / false 0 / maybe-false 0.

| # | Finding | Verdict | Route | Evidence / action |
|---|---------|---------|-------|-------------------|
| 1 | Re-picking the current model during a pending load skips `select_model`; pending load later wins / is used by Start | medium | patch | Early return in `on_model_activated`; orchestrator `_start_session` uses `_pending or _selected`. Patch: always call `select_model`. |
| 2 | Status line ends on "ready"/"released" instead of "selected" with real event order | low | patch | `_load` runs callbacks (selected) before LOADED "ready" notify; GUI writes every MODEL message. Patch: status line only for loading/selected. |
| 3 | `list_models()` disk checks on Qt thread per MODEL event | low | reject | Local HF cache lookups only (no network), a few per load; no user-visible stall; fix needs caching/threading. |
| 4 | Empty `list_models()` path of "No model API" row untested | low | patch | Test uses FakeOrch without the attribute. Patch: add empty-list test. |
| 5 | Extra blank line in tests | low | patch | PEP 8 E303. Patch: remove. |

## Verification

**Commands:**
- `uv run pytest -q` -- expected: all pass
- `QT_QPA_PLATFORM=offscreen uv run python -c "from audio_translator.gui import MainWindow"` -- expected: no error

**Manual checks (if no CLI):**
- `uv run audio-translator`: dropdown shows 6 models with status; pick `small` while stopped → status shows loading then selected; press Start → dropdown greys out.
