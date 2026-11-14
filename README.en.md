# Audio Translator (VI → EN virtual microphone, macOS)

> Vietnamese README with architecture, measurements, known issues and roadmap: [README.md](README.md).

Speak Vietnamese into your mic; the other side of a voice chat hears English.
Pipeline: mic capture → VAD segmentation → faster-whisper ASR → offline
translation (Argos vi→en model on CTranslate2; Google Translate as fallback) → Edge TTS → resample/ordered playout → **BlackHole 2ch** (virtual mic).

## Requirements

- macOS (Apple Silicon or Intel)
- [uv](https://docs.astral.sh/uv/) (`brew install uv`) — the project pins Python 3.12
- Network access for Edge TTS (and for the one-time translation-model download)

## Install

```sh
brew install blackhole-2ch    # virtual audio device; a reboot/log-out may be needed
uv sync
uv run audio-translator --cli --list-devices   # "BlackHole 2ch" should be listed
```

The default ASR model is `phowhisper-small` (Vietnamese fine-tune of Whisper,
FLEURS WER 16.7% vs 27.3% for `small`, `results/asr_fleurs_demand.csv`). Convert it once
(torch/transformers are only installed for this command):

```sh
uv run --with transformers --with torch python scripts/convert_phowhisper.py small
```

If it is not converted, the app warns (red line in the GUI, WARNING in the CLI) with this
command and runs Whisper `small` (downloaded on first use, ≈ 485 MB) for the session.

Per-segment latency of a real session: `--latency-log PATH` (CLI and GUI), summarised by
`uv run python eval/latency_report.py PATH`; manual test script: `docs/manual_test.md`.

Speaker indicator (green / yellow / red, observation only): it shows the highest of three signals: utterance length (yellow ≥ 8 s, red ≥ 12 s), ASR backlog (yellow with ≥ 1 segment queued or ASR > 3 s; red with ≥ 2 queued or ASR > 6 s), and an unclear last segment (filtered, ASR time abnormal for its length, or a temperature fallback). The GUI shows a coloured disc with an icon and text; the CLI prints a line only when the level changes; `--latency-log` adds the columns `health_max` and `health_reason`. The thresholds live in `Settings.indicator_*`.

## Models

Only multilingual Whisper models are offered (`.en` / `distil-*` are English-only):

| Model | Approx. size |
|-------|-------------|
| `tiny` | 75 MB |
| `base` | 145 MB |
| `small` | 485 MB |
| `medium` | 1.5 GB |
| `large-v3-turbo` | 1.6 GB |
| `large-v3` | 3.1 GB |
| `phowhisper-small` (default, converted locally) | 485 MB |
| `phowhisper-medium` (converted locally) | 1.5 GB |

```sh
uv run audio-translator --cli --list-models   # size + not-downloaded / downloaded per model
```

A model that is not downloaded is fetched from Hugging Face on first use
(cached in `~/.cache/huggingface/hub`). Recently used models stay loaded in
memory so switching back is instant; at most `max_loaded_models` (default 2,
sized for 8 GB RAM) are kept, least recently used first out — the model a
running session uses is never released. The model can be changed only while no
session is running (`Orchestrator.list_models()` / `select_model(name)` for
front ends; loading happens in the background and is reported as `model` events).
In the window, pick the model from **Cài đặt → Model nhận dạng** (shows each model's
status; enabled only while stopped).

## Run

GUI (Vietnamese UI; every string lives in `gui_strings.py`):

```sh
uv run audio-translator
uv run audio-translator --translator nllb-600m
```

The second command opens the GUI with a different initial translator.

Press **▶ Bắt đầu** (Start), speak Vietnamese, then press **■ Dừng** (Stop).

**Full mode**, from top to bottom:
- a red error banner: a Vietnamese headline plus a hint. The English original is in the tooltip; **Chi tiết** (Details) opens the technical log.
- a yellow banner when translation fell back to Google;
- the control row;
- a large speaker light;
- the translation pane: one block per sentence (VI small grey, EN large, latency on the right). It auto-scrolls unless you scrolled up, keeps the last 200 sentences, and has copy-all / clear buttons and a line with the last / session-median latency and the sentence count;
- **Cài đặt** (Settings, collapsed): microphone, output (virtual mic), ASR model, translator, noise suppression. They are locked while running and apply from the next Start. VAD, restore, ASR filter and the token cap are CLI-only.
- **Nhật ký kỹ thuật** (Technical log, hidden).

**Thu gọn** (Compact) switches to a small always-on-top window (about 360×120):
- it shows the light, the main reason, the newest English sentence (elided; full text in the tooltip), Start/Stop and **Mở rộng** (Expand);
- it first opens at the top-right of the screen, away from the call controls;
- closing it returns to the full window; Cmd+Q still shuts down cleanly;
- on macOS it may not float above an app in native full-screen (e.g. Meet in full-screen Chrome). Keep the call in a normal window; see scenario (f) in `docs/manual_test.md`.

CLI:

```sh
uv run audio-translator --cli [--input "MacBook Pro Microphone"] [--output BlackHole] [--model medium] [--list-models]
```

Ctrl+C stops the session (press twice to abort while the model is loading).

## Translation backend

Translation runs **offline** by default with the Argos Translate Vietnamese→English
model, executed directly by CTranslate2 + SentencePiece (the `argostranslate`
library is not used). The model package (~66 MB) is downloaded once, at the first
session start ("Downloading translation model (~66 MB)…"), and cached in
`~/.cache/audio-translator/argos` (`Settings.translation_cache_dir`). After that
no translation request leaves the machine.

Google Translate is the fallback: if the model cannot be downloaded or loaded
the session starts with Google and a warning (offline is retried on the next
start), and if offline translation fails for one segment that segment alone is
sent to Google. To use Google only:

```sh
uv run audio-translator --cli --translator google
```

## Use it in a chat app

In Zoom / Meet / Discord / Teams, open audio settings and select
**"BlackHole 2ch" as the microphone**. Translated speech is written only to
BlackHole; it is never played on your own speakers.

macOS will ask for microphone permission for your terminal the first time.

## Tests

```sh
uv run pytest -q
```

Tests use fake audio streams, ASR, translator and TTS — no devices or network needed.
