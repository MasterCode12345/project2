---
title: 'GUI redesign for real Zoom/Meet calls: full mode + always-on-top compact mode, Vietnamese strings'
type: 'feature'
created: '2026-11-14'
status: 'approved — in progress'
baseline_revision: '59f0cdd'
branch: 'feat/gui-redesign'
route: 'full'
risk: 'medium'
context:
  - '{project-root}/_bmad-output/initiative-audio-translator/plan-speaker-indicator.md'
---

## Intent

The supervisor's feedback is that the app is used *during* a real Zoom/Meet call. The window must therefore:
- be easy to read at a glance;
- use few technical words;
- be able to shrink to a small always-on-top strip that does not cover the call.

Only the GUI changes. The pipeline, the orchestrator, the `Event` format, `Settings` defaults, the CLI and the latency log stay as they are.

## Sketches

### Full mode (main window, about 640 × 720)

```
┌────────────────────────────────────────────────────────────────┐
│ ⛔ Không tìm thấy BlackHole. Cài bằng: brew install blackhole-2ch │ ← red banner, only when there is an error
├────────────────────────────────────────────────────────────────┤
│ Tiếng Việt → Tiếng Anh  [  ▶ Bắt đầu  ]  Đã dừng     [Thu gọn] │ ← language label (future: pair dropdown),
│                                                                │   big Start/Stop button, state
│  ╭─────╮                                                       │
│  │  ✋  │  ĐỎ: Dừng lại một chút (câu 12,4 s)                   │ ← big light (≈ 72 px) + main reason
│  ╰─────╯  Cũng: Đang xử lý, nói chậm lại                       │   + secondary reasons (tooltip: all)
├────────────────────────────────────────────────────────────────┤
│ Bản dịch                                    [Sao chép tất cả] [Xóa] │
│ ┌────────────────────────────────────────────────────────────┐ │
│ │ xin chào bạn                                     (grey, small) │ │
│ │ Hello, friend.                                      2,95 s │ │ ← EN larger; latency on the right
│ │ ──────────────────────────────────────────────────────── │ │
│ │ tôi là sinh viên đại học bách khoa hà nội                  │ │
│ │ I was a student at the University of Hanoi.         3,01 s │ │
│ │                                     (auto-scroll to newest │ │
│ │                                      unless scrolled up;   │ │
│ │                                      keeps the last 200)   │ │
│ └────────────────────────────────────────────────────────────┘ │
│ Câu gần nhất: 3,0 s · Trung vị phiên: 3,0 s · Số câu: 2        │
├────────────────────────────────────────────────────────────────┤
│ ▸ Cài đặt                                       (collapsed)    │
│   Micro:              [(Mặc định của hệ thống)        ▾]       │
│   Đầu ra (micro ảo):  [BlackHole 2ch                   ▾]       │
│                       Trong Zoom/Meet, chọn Microphone = BlackHole 2ch │
│   Model nhận dạng:    [phowhisper-small — trong RAM ✓  ▾]       │
│   Bộ dịch:            [Offline (Argos)                 ▾]       │
│   Lọc nhiễu:          [Tắt                             ▾]       │
│   (locked while running; takes effect at the next Start)       │
├────────────────────────────────────────────────────────────────┤
│ ▸ Nhật ký kỹ thuật                              (hidden)        │
│   model: Model 'phowhisper-small': ready                        │
│   info: Latency log: …                                          │
└────────────────────────────────────────────────────────────────┘
```

### Compact mode (separate small window, about 360 × 120, always on top)

```
┌──────────────────────────────────────────────┐
│ ╭───╮ VÀNG: Câu dài, hãy ngắt nghỉ            │
│ │ ⚠ │ "I was a student at the University o…"  │ ← newest EN sentence, one line, elided;
│ ╰───╯                     [■ Dừng] [Mở rộng] │   full text in the tooltip
└──────────────────────────────────────────────┘
```

- **"Thu gọn"** hides the main window and shows the compact window. The compact window is a separate top-level `QWidget` with `Qt.WindowStaysOnTopHint | Qt.Tool`.
- **"Mở rộng"** hides the compact window and shows the main window again; the main window never carries the on-top flag. Closing the compact window counts as "Mở rộng", so the app is never lost.
- Both windows are fed by **one** update path, so the light, the reason and the newest sentence are always the same in both.
- The compact Stop button is the same action as the main button. It shows "▶ Bắt đầu" when stopped, so a session can also be started from the compact window.
- macOS: an always-on-top window may not appear over an app in native full-screen (Meet in full-screen Chrome). This is documented, not worked around (no pyobjc). `docs/manual_test.md` records the result in a normal window and in full-screen.

## Behaviour details

### Error banner
- Red, at the top, hidden when there is nothing to show.
- These messages are kept and already Vietnamese:
  - missing BlackHole, with the brew command;
  - model not converted, temporary fallback, with the convert command.
- **Other errors come from the orchestrator as English text**, e.g. "Could not start: Output device … not found", and the orchestrator must not change. The banner shows a **Vietnamese headline** picked by a few known patterns:

  | Pattern | Headline |
  |---|---|
  | `Could not start` | "Không khởi động được" |
  | `Could not load model` | "Không tải được model" |
  | `stopped responding` | "Thiết bị âm thanh ngừng hoạt động (bị rút ra?)" |
  | `Cannot open … device` | "Không mở được thiết bị" |
  | anything else | "Đã xảy ra lỗi" |

  The original English message goes into the banner's tooltip and the technical log. → **Question 1.**
- The banner clears at the next Start, and when the user closes it with its "×".

### State text
- Vietnamese words from the `STATUS` state, not the orchestrator's English message:
  - Đã dừng / Đang khởi động… / Đang chạy / Đang dừng…
- Progress such as "Loading ASR model 'x'…" or "Downloading translation model (~66 MB)…" is mapped to "Đang tải model nhận dạng…" / "Đang tải model dịch (~66 MB)…". The raw text goes to the technical log.
- The model-selection messages that showed while stopped get Vietnamese words as well:
  - "Đang tải model 'small'…";
  - "Đã chọn model 'small'";
  - "Đang tải về model 'x' (~3,1 GB)…".

### Light
- Same `HEALTH` handling as now: levels, the main reason plus a "Cũng: …" line and a tooltip, ignoring events from an old session, grey when not running.
- Only the size and position change. The levels, thresholds and smoothing live in the orchestrator / `health.py` and are not touched.

### Translation pane
- A `QListWidget` with one custom item widget per sentence:
  - the VI line small and grey;
  - the EN line larger;
  - the latency right-aligned ("3,0 s"; "–" when unknown).
- Auto-scrolls to the newest sentence only when the view was already at the bottom.
- Keeps at most `MAX_SENTENCES = 200`; the oldest are removed.
- "Sao chép tất cả" copies "VI\nEN\n\n" per sentence to the clipboard. "Xóa" clears the pane and the statistics.
- Statistics line: last latency · median of the latencies **of the current session** (reset at each Start) · number of sentences in the session.

### Settings section (collapsed by default)

| Setting | Values |
|---|---|
| Micro | the system input devices |
| Đầu ra (micro ảo) | the output devices, plus the help line "Trong Zoom/Meet, chọn Microphone = BlackHole 2ch". The "never default to real speakers" rule is unchanged. |
| Model nhận dạng | the existing dropdown and select API, enabled only when stopped, as now |
| Bộ dịch | "Offline (Argos)" · "NLLB-600M (offline, cần convert một lần)" · "M2M100-418M (offline, cần convert một lần)" · "Google (cần mạng)" |
| Lọc nhiễu | "Tắt" · "logMMSE (thử nghiệm)" |

- **Bộ dịch** and **Lọc nhiễu** need **no orchestrator change**. The orchestrator re-reads `self.settings.translator` (`_prepare_translator`) and `self.settings.ns_mode` (`noise_suppressor_from_settings`) at every session start.
  - The GUI writes these two fields into `orch.settings` just before calling `start()`, only while stopped.
  - The `Settings` *defaults* are not changed.
  - An NLLB/M2M100 model that is not converted already falls back to Google with a WARNING (existing behaviour). That warning shows in the banner as "Model dịch … chưa có, đang dùng Google".
- Every setting control is disabled from Start until STOPPED (or IDLE), and applies from the next Start.
- VAD, restore, ASR filter and token cap stay CLI-only.

### Technical log (hidden by default)
- A collapsible section holding the `QPlainTextEdit` that today receives MODEL / INFO / WARNING / ERROR lines, together with the raw orchestrator texts mentioned above.
- RESULT lines no longer go there; they go to the translation pane.

### Strings
- A new module, `gui_strings.py`, holds one `VI` dict with every visible string; `MainWindow` reads only from it, so a UI language can be added later.
- Device and model *names* stay as they are.

## Files

| File | Change |
|---|---|
| `src/audio_translator/gui.py` | `MainWindow` reorganised (banner, control row, light, translation pane, settings, technical log). `CompactWindow` wiring. `parse_args`/`main` unchanged. `EventBridge` with `QueuedConnection` and `closeEvent` → `orch.shutdown()` are kept, and closing also closes the compact window. |
| `src/audio_translator/gui_widgets.py` (new) | `LightWidget` (disc + main + secondary reasons, two sizes), `TranscriptView` (sentence blocks, cap, auto-scroll, copy/clear, statistics), `CollapsibleSection`, `CompactWindow`. |
| `src/audio_translator/gui_strings.py` (new) | `VI` dict, error-headline patterns, state words. |
| `README.md`, `README.en.md` | new GUI description, compact mode, the full-screen limitation |
| `docs/manual_test.md` | new scenario (f): compact mode in a Meet call (normal window vs full-screen), what to record |

No change to: `orchestrator.py`, `models.py`, `config.py`, `cli.py`, `latency_log.py`, `health.py`.

## Tests

**Existing tests, updated for the new texts while keeping their meaning, none deleted:**
- `tests/test_frontends.py`:
  - `button.text()` "Start"/"Stop" → "▶ Bắt đầu"/"■ Dừng";
  - `status_label` "Listening" → the state word "Đang chạy";
  - the result text moves from `w.log` to the translation pane;
  - "loading", "selected", "downloading (~3.1 GB)" → their Vietnamese forms in `status_label`, with the raw English still in the technical log;
  - "Could not load model" in `error_label` → the Vietnamese headline in the banner, with the raw text in its tooltip;
  - `lang_label` "Tiếng Việt → English" → "Tiếng Việt → Tiếng Anh".
- `tests/test_health.py`: the GUI light tests keep the same assertions on colour, icon and text, through `LightWidget`.

**New, `tests/test_gui_redesign.py`, offscreen:**
- compact ↔ full: visibility of both windows; the on-top flag on the compact window only; closing the compact window expands.
- Light and newest sentence in sync: one `HEALTH` + one `RESULT` update both windows; the EN line is elided with the full text in the tooltip; an old-session `HEALTH` changes neither.
- Compact Stop button: calls `orch.stop()`; shows "Bắt đầu" after STOPPED and starts a session.
- Translation pane:
  - block content (VI, EN, latency "0,9 s");
  - cap 200 (201 results → 200 kept, the oldest gone);
  - no auto-scroll when scrolled up, auto-scroll when at the bottom;
  - copy all, clear;
  - statistics line (last / median / count; reset at a new session).
- Technical log hidden by default and receiving MODEL / INFO / WARNING / ERROR lines.
- Settings locked from Start to STOPPED.
- Translator and noise choices written into `orch.settings` at Start (and not while running); defaults unchanged.
- No English left on the UI: walk every `QLabel`, `QPushButton`, `QToolButton` and combo item of both windows in each state (stopped, running, error), and check them against a list of English words ("Start", "Stop", "Stopped", "Running", "Microphone", "Output", "Language", "Model:", "Listening", "Loading", "error"…). Device and model names are excluded.

Every existing test must pass: `uv run --group eval pytest -q`.

## Commits (on `feat/gui-redesign`, pushed, not merged)

1. `gui_strings.py` + `gui_widgets.py` + their tests.
2. `gui.py` full mode + updated existing tests.
3. Compact mode + tests.
4. Docs (README, README.en, manual_test (f)).

## Decisions (human, 2026-11-14)

1. **Errors.**
   - The banner shows a Vietnamese headline **plus a short hint**, e.g. "Không khởi động được – kiểm tra BlackHole đã cài và chọn lại Đầu ra trong Cài đặt".
   - The English original goes only into the tooltip and the technical log.
   - A **"Chi tiết"** button on the banner opens the technical log.
   - An unmatched error shows "Đã xảy ra lỗi – bấm Chi tiết để xem". English is never shown on the banner.
2. **Translator / noise.** Writing `orch.settings` before Start is approved.
   - The dropdowns start **from `orch.settings`**, so `--translator nllb-600m` shows NLLB; the GUI accepts `--translator` and `--ns`.
   - A WARNING that MT fell back to Google shows a **yellow banner**: "Đang dịch bằng Google (cần mạng) vì <bộ dịch> chưa sẵn sàng". Tests cover both points.
3. **Compact button.**
   - "▶ Bắt đầu" when stopped; disabled while stopping.
   - Starting from the compact window uses the current settings.
4. **Also:**
   - **Quitting:** Cmd+Q while compact (main hidden) still calls `orch.shutdown()` through `QApplication.aboutToQuit`. Shutdown is idempotent and tested.
   - **First position:** the compact window first opens at the **top-right** of the screen, away from the Meet/Zoom controls at the bottom.
5. **Implementation note (macOS):** the compact window must **not** use `Qt.Tool`. On macOS a tool window hides when the app is not active, i.e. as soon as the user clicks into Meet. It uses `Qt.Window | Qt.WindowStaysOnTopHint` only.

## Questions for you (answered above)

1. **English errors from the orchestrator:** a Vietnamese headline in the banner, with the original English only in the tooltip and the technical log. Is that acceptable, or should the full English detail also show as a small second line in the banner? The orchestrator must not change, so I cannot translate every message at the source.
2. **Translator and noise settings** are applied by writing `orch.settings.translator` / `orch.settings.ns_mode` before `start()`. The orchestrator code is not modified; it already re-reads both at each start. Is that within "do not change the orchestrator"?
3. **Compact Stop button:** when stopped, it becomes "Bắt đầu" (proposed), rather than staying a disabled "Dừng". Do you agree?
