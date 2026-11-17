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

**Glossary of proper names** (on by default; `--no-glossary` to turn it off):
- `src/audio_translator/glossary_vi_en.tsv` holds the names; `--glossary PATH` adds your own file, which also overrides identical terms.
- Before MT each name is replaced by a placeholder (`ZQX1`…), and restored to the official English name after MT.
- A sentence whose placeholders do not survive is translated again without the glossary; sentences without a name are unchanged.
- Results (`results/mt_glossary_vi_en.csv`): Argos term accuracy 0 → 100 %; NLLB and M2M100 also reach 100 %; no placeholder leaked; FLEURS unchanged.
- Limitations: sentences with a placeholder sometimes lose an article or a preposition around the name; with two terms, their relation can change ("thuộc", "belongs to", → "is"). The table is for proper names only. Possible extension (not done): a placeholder with its class noun, e.g. "trường ZQX1".

**First word of an utterance** (`--pre-roll-ms`, default 600): an utterance still starts when ≥ 5 of the last 10 frames (300 ms) are voiced, but it now keeps the 600 ms of audio before that, so a soft first syllable is no longer cut ("đồ án" → "án"). The length light and the 15 s cap are unchanged. Measured on a continuous stream of VIVOS/FLEURS through the app's Segmenter + Silero + PhoWhisper (`eval/preroll_bench.py`, `results/preroll_*.csv`): first word lost 9.5 → 7.0 % (VIVOS clean; best possible 8.0 %) and 27 → 13 % (FLEURS clean); WER better in every cell at seed 0 (FLEURS DEMAND 10 dB, seed 2: +0.73); ASR p95 in noise 5–7.5 s → 1.4–2.4 s.

**Anti-stall** (experimental, **off** by default): the checkbox "Chống kẹt nhận dạng (thử nghiệm)" in Cài đặt is the preset `c4b` (`--asr-anti-stall c4b`). Its tooltip states the trade-off: the longest ASR stall drops from about 36 s to about 7.5 s in the bench, but in very heavy noise a few sentences can be lost. Opening the app with `--asr-anti-stall c4a` shows the box checked and keeps `c4a` until it is clicked. A model that is already loaded (also the `small` fallback) gets the new setting at the next Start, without a reload; a segment of the previous session still decoding keeps its old rules and is dropped. The latency log column `asr_anti_stall` (`off` / `c4a` / `c4b` / `custom`, from the model's real state) and `latency_report.py` (`anti-stall per session: …`) show it per session. The 600 ms pre-roll (default since 2026-11-16) already lowered ASR p95 in noise from 5–7.5 s to 1.4–2.4 s.

**Segment merging before translation: not done** (decision 2026-11-17). It would wait for the next segment and so add latency; latency comes first (the supervisor approved the current latency) and translation quality in real use is good enough. Listed as an extension.

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
