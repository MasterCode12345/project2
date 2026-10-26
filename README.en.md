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

The first start downloads the Whisper model (`medium` ≈ 1.5 GB by default; use
`--model small` for a faster, less accurate start).

## Models

Only multilingual Whisper models are offered (`.en` / `distil-*` are English-only):

| Model | Approx. size |
|-------|-------------|
| `tiny` | 75 MB |
| `base` | 145 MB |
| `small` | 485 MB |
| `medium` (default) | 1.5 GB |
| `large-v3-turbo` | 1.6 GB |
| `large-v3` | 3.1 GB |

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
In the window, pick the model from the **Model:** dropdown (shows each model's
status; enabled only while stopped).

## Run

GUI (one window):

```sh
uv run audio-translator
```

Pick your microphone, keep **BlackHole 2ch** as output, press **Start**, speak
Vietnamese, press **Stop**. The log shows the Vietnamese text, English text and
end-to-end latency (`first_audio_out − speech_end`) per segment.

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
