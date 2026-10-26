---
title: 'Offline Vietnamese→English translation (Argos model via CTranslate2) with Google fallback'
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

**Problem:** Google Translate (via `deep-translator`) rate-limits this machine's IP — every segment fails with "You made too many requests to the server", so nothing is spoken into the call.

**Approach:** Make an offline Vietnamese→English translator the default: the Argos Translate `vi→en` model package run directly with CTranslate2 + SentencePiece (no `argostranslate` library). The package (~66 MB) is downloaded once on first use, off the UI thread. Google stays available as an optional backend and as automatic fallback when offline translation is unavailable.

## Boundaries & Constraints

**Always:** Default `Settings.translator = "offline"`; alternative `"google"`. In offline mode, if the model package cannot be obtained or loaded, fall back to Google with a warning (session still starts); a per-segment offline error also falls back to Google for that segment. Package download/extract happens in the session-start thread (never Qt thread or audio callback), is atomic (temp dir → rename) and reused from a local cache dir; metadata must say `from_code=vi`, `to_code=en`. Strip SentencePiece `▁` markers so output is clean text. Translation adapter stays behind the `Translator` protocol.

**Never:** Do not add the `argostranslate` package (pulls torch/stanza/spacy, several GB). No GUI changes. No other language pairs. No Claude API.

## I/O & Edge-Case Matrix

| Scenario | Input / State | Expected Output / Behavior | Error Handling |
|----------|--------------|---------------------------|----------------|
| Offline happy path | Package cached; "Xin chào, bạn khỏe không?" | English text, no network call | none |
| First run | Package not cached | STATUS "Downloading translation model (~66 MB)…", then session runs offline | none |
| Download fails | No network / bad URL / corrupt zip / wrong metadata | Session starts with Google; warning names the cause; no partial package left in cache | Warning event |
| Offline segment error | Offline translate raises for one segment | That segment translated by Google; next segment tries offline again | Warning event; if Google also fails, segment skipped as today |
| Google mode | `translator="google"` | Behaves exactly as before; no package download | unchanged |
| Multi-sentence text | "Câu một. Câu hai?" | Each sentence translated, joined with spaces, no `▁` | none |
| Empty result | Translator returns "" | Segment skipped as today (no TTS) | none |

</frozen-after-approval>

## Code Map

- `src/audio_translator/translate.py` -- `Translator` protocol, `GoogleTranslator(source, target)`; add offline + fallback adapters here.
- `src/audio_translator/orchestrator.py` -- `_components()` builds `GoogleTranslator(s.source_lang, s.target_lang)` when `self._translator is None`; `_start_session` runs off-thread and already emits STATUS "Loading ASR model…" — prepare the offline package here; `_mt_worker` calls `self._translator.translate(tr.text)` and handles exceptions/empty. Injected `translator=` used by tests — keep.
- `src/audio_translator/config.py` -- add `translator: str = "offline"`, `translation_cache_dir` (default `~/.cache/audio-translator/argos`).
- `src/audio_translator/cli.py` -- add `--translator {offline,google}`.
- Verified during planning: index `https://raw.githubusercontent.com/argosopentech/argospm-index/main/index.json` lists `vi→en` 1.9 at `https://argos-net.com/v1/translate-vi_en-1_9.argosmodel` (65,895,840 bytes). Zip layout: `translate-vi_en-1_9/{metadata.json, sentencepiece.model, model/{model.bin,config.json,shared_vocabulary.json}, stanza/…}`. `ctranslate2.Translator(dir+"/model", device="cpu", compute_type="int8")` loads in ~0.1 s; `sp.encode(text, out_type=str)` → `translate_batch([toks], beam_size=4)` → join hypothesis tokens, replace `▁` with space, strip — ~50–100 ms per sentence. `ctranslate2` 4.8.2 already installed (via faster-whisper); `sentencepiece` 0.2.2 must be added.
- `tests/conftest.py` -- `FakeTranslator`, `Harness(translator=...)`.

## Tasks & Acceptance

**Execution:**
- [x] `pyproject.toml` -- add `ctranslate2>=4.0` (explicit) and `sentencepiece>=0.2` -- dependencies
- [x] `src/audio_translator/config.py` -- `translator`, `translation_cache_dir`, `argos_vi_en_url` -- config
- [x] `src/audio_translator/translate.py` -- `ensure_argos_package(cache_dir, url, opener=...) -> Path` (download, verify zip + metadata, atomic install, reuse); `OfflineTranslator(package_dir)` (lazy load, regex sentence split, `▁` cleanup); `FallbackTranslator(primary, fallback, on_fallback)` -- adapters
- [x] `src/audio_translator/orchestrator.py` -- `_components`/`_start_session` choose backend from settings; in offline mode emit download STATUS, prepare package, wrap with Google fallback; on preparation failure warn and use Google -- wiring
- [x] `src/audio_translator/cli.py` -- `--translator` -- CLI
- [x] `tests/test_translate.py`, `tests/test_orchestrator.py` -- fake opener serving a tiny zip, fake CT2/SPM seams; every matrix row; no network -- regression
- [x] `README.md` -- offline translation default, first-run download, `--translator google` -- docs

**Acceptance Criteria:**
- Given network blocks Google, when the user speaks after the offline package is cached, then English speech is produced with no Google request.
- Given `uv run pytest`, when run offline with no audio devices, then all tests pass.

## Implementation Notes

- `ensure_argos_package(cache_dir, url, opener=None, *, source, target, on_download=None)` raises `PackageError`; downloads+extracts in a `.download-*` temp dir inside the cache, verifies zip CRC, path safety, required files and `from_code=vi`/`to_code=en`, then renames into `<cache>/translate-vi_en-1_9`. A broken cached install is removed and re-fetched.
- `OfflineTranslator.load()` is called in the session-start thread so load failures fall back to Google at start; translate() also lazy-loads.
- Orchestrator: `_prepare_translator()` runs after device resolution, before ASR load. Seams for tests: `package_preparer`, `offline_translator_factory`, `google_translator_factory`. An injected `translator=` bypasses selection. A loaded offline translator is reused across sessions; after a preparation failure the next session retries offline. Google client is created lazily (only on first fallback).

## Plan Change Log

- 2026-10-07: default downloader sends `User-Agent: audio-translator/0.1` — argos-net.com answers HTTP 403 to urllib's default `Python-urllib` UA (found during verification).
- 2026-10-07: `ensure_argos_package` gained `on_download` callback so the orchestrator emits the download STATUS only when actually downloading.

## Review Triage Log

Pass 1 (lens: quick) — high 0 / medium 0 / low 6 / false 0 / maybe-false 0.

| # | Finding | Verdict | Route | Evidence / action |
|---|---------|---------|-------|-------------------|
| 1 | Cached package missing `config.json`/`shared_vocabulary.json` passes check; offline never recovers | low | patch | `_REQUIRED` lists only 3 files; load failure path keeps cache. Patch: add both files. |
| 2 | Killed download leaves `.download-*` temp dirs forever | low | patch | Start thread is daemon; `finally` skipped on hard exit; no sweep. Patch: sweep leftovers on entry. |
| 3 | Stop during translation-model download honoured only after ASR load | low | patch | `_abort_start` checked only after both loads. Patch: check after `_prepare_translator`. |
| 4 | Per-segment fallback warning lacks session/segment ids | low | reject | Rarely seen; fix adds parameters to the translator adapter surface. |
| 5 | Offline ignores configured `source_lang`/`target_lang` | low | patch | `ensure_argos_package` called with defaults. Patch: pass settings languages. |
| 6 | Test gaps: offline `load()` at start untested (HTTP-error variant already covered: HTTPError is OSError) | low | patch | Fake has no `load`. Patch: add load-failure test. |

## Verification

Results (2026-10-07): `uv run pytest -q` → 93 passed. Real verification command → `'Hello, how are you?'` (download ~9 s, package at `~/.cache/audio-translator/argos/translate-vi_en-1_9`, no temp leftovers). Orchestrator e2e with the real cached package and a Google factory that raises → RESULT "Hello, how are you?", no warnings.

**Commands:**
- `uv sync && uv run pytest -q` -- expected: all pass
- `uv run python -c "from audio_translator.translate import ensure_argos_package, OfflineTranslator; from audio_translator.config import Settings; s=Settings(); t=OfflineTranslator(ensure_argos_package(s.translation_cache_dir, s.argos_vi_en_url)); print(t.translate('Xin chào, bạn khỏe không?'))"` -- expected: an English greeting without `▁`
