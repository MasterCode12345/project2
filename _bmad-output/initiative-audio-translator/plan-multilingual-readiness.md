---
title: 'Multilingual readiness: what is tied to Vietnamese → English, the candidate pairs, and the path to "a new pair = configuration"'
type: 'plan'
created: '2026-11-19'
status: 'plan only — no code change; waiting for the supervisor to choose the pairs'
baseline_revision: '1d95475'
branch: 'plan/multilingual-readiness'
route: 'full'
risk: 'medium (depends on the pairs chosen)'
---

## Intent

The supervisor has not chosen the language pairs yet. This plan prepares the work, so that once the pairs are fixed the implementation starts at once. It contains:
1. every place tied to Vietnamese → English, with its file, how hard it is tied, and how to generalise it;
2. the six candidate pairs, with the facts needed to choose: ASR, MT, TTS, data, what must be turned off or replaced, disk/RAM, effort and risk;
3. an architecture where a new pair is a configuration entry, and the order of work;
4. the questions for the supervisor.

**How the facts were gathered (2026-11-19)**, without downloading any model:
- **Code:** read `languages.py`, `plan-mt-multilingual-models.md` and the whole pipeline (`config`, `orchestrator`, `asr`, `segmenter`, `restore`, `glossary`, `translate`, `tts`, `health`, `gui*`, `cli`, `latency_log`, `eval/*`, `scripts/*`).
- **Argos index** (`argospm-index/index.json`): package list; sizes from HTTP `Content-Length`.
- **Edge TTS:** `edge_tts.list_voices()`.
- **FLEURS** (`google/fleurs` on Hugging Face): configs and test audio sizes.
- **Google codes** (`deep_translator`): the codes it uses.
- **RAM:** peak RSS measured on this MacBook (8 GB, int8, CPU). Each component was loaded in its own process and used once; GUI/Qt not included.
- **Two quick checks of the glossary code with Chinese text** (Finding G below).

## 1. What is tied to Vietnamese → English today

**How hard it is tied:**
- **none:** already driven by `source_lang` / `target_lang` or `languages.py`.
- **low:** a value or label to change.
- **medium:** works but wrong or ineffective for another language.
- **high:** another language breaks or silently degrades.

| # | What | File(s) | Tied | What happens with another pair | How to generalise |
|---|---|---|---|---|---|
| 1 | Language choice | `config.py` (`source_lang="vi"`, `target_lang="en"`) | low | Fields exist, but nothing sets them: no CLI flag, no GUI control. | `--source/--target` + `--list-pairs` (CLI); a pair dropdown, locked while running; changing it = a new session. |
| 2 | Language table | `languages.py` (rows `vi`, `en` only; normalisers `vi`, `en`) | low | Built for this: one row per language. **Nothing in the app reads it yet**, only `mt_bench` / `translate` (codes). | Add rows `zh` / `ja` / `ko` / `fr` (codes below); the app reads every per-language detail from it. |
| 3 | ASR model | `config.py` (`model_size="phowhisper-small"`, fallback `small`); `asr.LOCAL_MODELS`; `model_manager.CATALOG`; `Language.asr_model` unused | **high** | PhoWhisper is Vietnamese only. With another source language the app still starts and recognises garbage, with no warning. `.en` / distil English models are not in the catalog. | Each catalog model gets the languages it supports (`phowhisper-*`: `{vi}`). The session picks `Language(source).asr_model` unless the user chose a compatible one, and refuses or warns on an incompatible one. Add `small.en` / `distil-small.en` for an English source. |
| 4 | Casing / punctuation restore | `restore.py` (`RESTORE_MODELS["vibert-capu"]` = `("vi",)`, `auto_restorer_name(lang)`) | none / low | `auto` already resolves to **off** for another source. Multilingual Whisper writes punctuation and casing itself (only PhoWhisper writes lowercase without punctuation), so off is right. `sent` is Latin-script only. | Keep `auto`. Re-measure only if a new source model writes no punctuation. |
| 5 | Ghost-sentence blacklist | `asr.py` `BLACKLIST_PATH = hallucinations_vi.txt` (always loaded); `scripts/collect_hallucinations.py` (`language="vi"`) | medium | Vietnamese phrases never match another language: no crash, but no protection. Each language has its own well-known Whisper ghosts. | `hallucinations_<lang>.txt` chosen by the source language (empty file = only the confidence thresholds). `collect_hallucinations.py --language`, then build the list from MUSAN as for Vietnamese. |
| 6 | Junk tokens | `asr.py` `JUNK_TOKENS_PATH = junk_tokens_vi.txt` ("unk", "ofc"; found on PhoWhisper) | medium | For an English source, "ofc" is a real word ("of course") and would be deleted. | Per model/language file; off unless measured for that model. |
| 7 | Loop / repeat handling | `asr.py` `collapse_repeats`, `has_loop_tail`, `trim_truncated_tail` (word groups split on spaces); `strip_cap_tail` (Vietnamese "a." tail) | medium (high for zh/ja source) | Chinese and Japanese have no spaces: a whole loop is one "word", so these never fire. The anti-stall loop rule (C4b, 4-word groups) is blind there. | One tokenisation helper per language: words for space-separated scripts, character n-grams for CJK. Used by all four functions. |
| 8 | Anti-stall caps | `config.ANTI_STALL_PRESETS` (20 real tokens/s + 32, measured on VIVOS/FLEURS **vi**) | medium | CJK text costs more Whisper tokens per second of speech, so the cap could cut normal speech. | Measure the token rate per language on FLEURS (`asr_bench`, `max_tok_per_s` column); a cap per source language; offer the checkbox only where measured. |
| 9 | Glossary | `glossary.py` (`glossary_<src>_<tgt>.tsv` per pair; `MIN_SYLLABLES=3` via `split()`; matching `(?<!\w)…(?!\w)` with `[ \t]+` between syllables; placeholder restore `(?<!\w)ZQX1(?!\w)`); `glossary_vi_en.tsv` | **high for CJK** (low for Latin) | Finding G below: CJK as target, CJK as source, and the syllable rule. The `zqx` placeholder form was chosen on vi→en only. | ASCII lookarounds for placeholders (`(?<![A-Za-z0-9])`); CJK matching without word boundaries (longest-first already prevents partial overlaps); minimum length in characters for CJK. A glossary file per pair, and `glossary_bench` rerun per pair before it is on by default. |
| 10 | Argos package | `config.py` `argos_vi_en_url` (one URL); `translate.ensure_argos_package`; `OfflineTranslator` (one package); `PivotTranslator` + `plan_route` **exist but are not wired** in `orchestrator._prepare_translator` | **high** | Any other pair: the vi→en package fails the `from_code`/`to_code` check, so the session warns and runs on **Google** (online). | A table of Argos packages per direction (URL + pinned version). The route is `direct`, else `pivot:en` with two packages (`plan_route` + `PivotTranslator`), else NLLB/M2M or Google. |
| 11 | Multilingual MT | `translate.CT2MultilingualTranslator` (NLLB-600M, M2M100-418M; codes from `languages.py`) | none | Direct between any two listed languages once their rows exist. | Only the `languages.py` rows. |
| 12 | Google fallback | `translate.GoogleTranslator` (codes from `languages.py["google"]`) | none | Works once the rows exist (Chinese is `zh-CN` / `zh-TW`). | Rows only. |
| 13 | TTS voice | `config.py` `tts_voice="en-US-AriaNeural"`; `tts.EdgeTTS` default; `Language.edge_voice` unused | low | The orchestrator already refuses to start if the voice does not match the target (`supports`), so a pair change without a voice change fails cleanly. | Voice = `Language(target).edge_voice` unless set explicitly; an optional voice choice in Cài đặt. |
| 14 | Sentence split before MT | `translate._SENTENCE_END = (?<=[.!?…])\s+` | low | CJK `。！？` are followed by no space, so there is no split: the whole segment becomes one MT item. No crash. | Add the CJK sentence ends without requiring a space. |
| 15 | UI labels | `gui_strings.VI["lang_pair"] = "Tiếng Việt → Tiếng Anh"` (fixed label); CLI prints `VI:` / `EN:` (`cli.py:144`); the GUI itself is Vietnamese | low | Wrong label only. The UI language (Vietnamese) is a separate matter and stays. | The label comes from the chosen pair (a dropdown); CLI prints the language codes. |
| 16 | Latency log | `latency_log.py` columns `text_vi`, `text_en`, `n_words_vi`, `n_words_en` (word counts via `split()`) | low | Names are wrong for another pair; word counts are meaningless for CJK. | Add `src_lang` / `tgt_lang` session fields; keep the old column names for compatibility (or rename with report support); count characters for CJK. |
| 17 | Segmentation | `config.py`: Silero 0.5, `pre_roll_ms=600`, `silence_ms=600`, `max_segment_s=15` (measured on VIVOS / FLEURS vi) | low | The mechanism does not depend on the language; the values were tuned on Vietnamese. | Rerun `preroll_bench` on FLEURS of the new source language (it needs a character-based "first word lost" for CJK). |
| 18 | Speaker indicator | `config.py` `indicator_*` (ASR 3 s / 6 s, unclear ratio 1.5, from PhoWhisper-small latencies) | low–medium | A bigger model (e.g. turbo for a Chinese source) is slower, so the light turns yellow too often. | Calibrate per ASR model (the latency log already has what is needed). |
| 19 | ASR filter thresholds | `config.py` (`no_speech_prob 0.6`, `avg_logprob −1.0`, `compression_ratio 2.4`: Whisper's own defaults) | none / low | Language-independent in Whisper. The compression ratio of CJK text differs, but Whisper uses the same 2.4 for every language. | Keep; check the ghost counts per language with the guard bench. |
| 20 | Evaluation | `eval/denoise_bench.normalize_vi` (used by `asr_bench`, `preroll_bench`); VIVOS loader; `eval/data/mt_custom_vi_en.tsv`, `mt_glossary_vi_en.tsv`; `mt_bench` FLEURS pairing (generic, via `languages.py`) | medium | WER on space-split words is wrong for zh/ja (needs CER); the custom and glossary sets exist for vi→en only. | Normaliser and metric from `languages.py` (CER primary for zh/ja); FLEURS for each language; a custom set and a glossary set per pair, **reviewed by a speaker of the target language**. |
| 21 | BLEU tokenizer | `languages.py` `bleu_tokenize` (`13a`) | low | sacrebleu has `zh` built in; `ja-mecab` / `ko-mecab` need MeCab packages, i.e. new eval dependencies. | Use chrF (tokenisation-free) as primary for ja/ko, plus sacrebleu `char` or `flores200`; avoid new libraries. |

**Finding G (checked with the real `glossary.py`, Chinese text):**
- **CJK target.** `restore("我是ZQX1的学生。")` returns **None**: the placeholder is glued to CJK characters, which count as `\w`, so every such sentence falls back to plain translation. Safe, but the glossary does nothing. With spaces around the placeholder it works.
- **CJK source.** `find("我是河内百科大学的学生。")` finds **nothing**; it finds the term only when spaces surround it.
- **Syllable rule.** The "≥ 3 syllables" rule counts `split()` parts, so any Chinese term counts as 1 and is refused.

## 2. Candidate pairs

Facts common to all pairs (2026-11-19):
- **Argos** only has packages to and from **English**: no direct vi↔zh/ja/ko/fr. So every non-English pair is **two hops through English**: two packages, errors compound, and MT time is roughly doubled (estimate).
  - Packages that exist (size): `vi→en` 1.9 (66 MB), `en→vi` 1.9 (68 MB), `en→zh` 1.9 (71 MB, also `en→zt` traditional), `zh→en` 1.9 (74 MB), `en→fr` 1.9 (65 MB), `en→ja` **1.1** (120 MB), `en→ko` **1.1** (121 MB).
  - ja/ko are an older generation of packages, so their quality is unknown and must be measured.
- **NLLB-600M / M2M100-418M** translate directly between all six languages. They are already converted here; only `languages.py` rows are needed. NLLB is CC-BY-NC (fine for the thesis) and measured at 552/1160 ms mean/P95 per segment on vi→en, above the 500 ms target. Argos measured 97/214 ms.
- **Google fallback** exists for every candidate (`vi`, `en`, `zh-CN` / `zh-TW`, `ja`, `ko`, `fr`).
- **Edge TTS voices**:

  | Locale | Voices | Examples |
  |---|---|---|
  | vi-VN | 2 | HoaiMy (F), NamMinh (M) |
  | zh-CN | 6 | Xiaoxiao (F), Yunxi (M), … |
  | ja-JP | 2 | Nanami (F), Keita (M) |
  | ko-KR | 3 | SunHi (F), InJoon (M), … |
  | fr-FR | 5 | Denise (F), Henri (M), … |
  | en-US | 17 | |

- **FLEURS** has all six languages (`vi_vn`, `en_us`, `cmn_hans_cn`, `ja_jp`, `ko_kr`, `fr_fr`).
  - The sentences are aligned across languages (FLoRes ids), so the "source audio + target text" pairing used for vi→en works for every pair.
  - Test audio sizes: en 290 MB, zh 525 MB, ja 449 MB, ko 214 MB, fr 349 MB (vi already here).
- **ASR quality, a priori.**
  - Measured here: Whisper `small` gets 27.3 % WER on FLEURS **vi**, and PhoWhisper-small 16.7 %.
  - Vietnamese is low-resource for Whisper. The Whisper paper (Radford et al., 2022) lists its training hours per language, rounded:

    | Language | Training hours |
    |---|---|
    | English | ≈ 438 000 |
    | Chinese | ≈ 23 000 |
    | French | ≈ 9 800 |
    | Korean | ≈ 8 000 |
    | Japanese | ≈ 7 000 |
    | Vietnamese | ≈ 700 |

  - So plain Whisper is expected to do clearly better on those sources than on Vietnamese. **Not measured yet**: the first step for a source other than Vietnamese is `asr_bench` on its FLEURS (CER for zh/ja).
- **RAM**, peak RSS per component, measured:

  | Component | Peak RSS |
  |---|---|
  | PhoWhisper-small / Whisper `small` | 0.9 GB |
  | `medium` / PhoWhisper-medium / `large-v3-turbo` | 1.6–1.75 GB |
  | vibert-capu | 0.14 GB |
  | Argos (each package) | 0.15 GB |
  | NLLB-600M | 1.0 GB |
  | M2M100 | 0.8 GB |

  Today's vi→en pipeline is about **1.2 GB** (+ GUI). Every candidate below stays under about **2.1 GB**, well inside 8 GB, even with two ASR models resident (`max_loaded_models=2`).
- **Shared groundwork, done once for every pair:** pair profile, pivot routing, CLI/GUI pair choice, per-language resources. Rough estimate: **3–5 days**. The per-pair estimates below come on top of it, are rough, and include measurement.

### Summary

| Pair | ASR | MT offline | TTS | Turn off / replace | New download · peak RAM | Effort · risk |
|---|---|---|---|---|---|---|
| **en → vi** | Whisper `small` (or `small.en`); English is Whisper's best language | Argos `en→vi` **direct** (68 MB); NLLB/M2M direct; Google | vi-VN, 2 voices | restore **off**; blacklist → English list; junk tokens **off**; new `glossary_en_vi.tsv` (reverse names) | small 485 MB + 68 MB · ≈ 1.1 GB | 2–3 days · **low** |
| **vi → fr** | PhoWhisper-small (unchanged) | Argos `vi→en→fr` (pivot, +65 MB); NLLB/M2M direct; Google | fr-FR, 5 voices | ASR side unchanged; `glossary_vi_fr.tsv` (French official names); placeholder re-measured (French elision "de"/"d'") | +65 MB · ≈ 1.35 GB (NLLB ≈ 2.0 GB) | ≈ 2 days · **low–medium** |
| **vi → zh** | PhoWhisper-small (unchanged) | Argos `vi→en→zh` (pivot, +71 MB); NLLB/M2M direct; Google `zh-CN` | zh-CN, 6 voices | ASR side unchanged; glossary: CJK target fix (Finding G) + `glossary_vi_zh.tsv`; BLEU `zh` | +71 MB · ≈ 1.35 GB (NLLB ≈ 2.0 GB) | 3–4 days · **medium** |
| **vi → ja** | PhoWhisper-small (unchanged) | Argos `vi→en→ja` (pivot, **package 1.1**, +120 MB); NLLB/M2M direct; Google | ja-JP, 2 voices | as vi → zh; BLEU needs MeCab, so chrF instead | +120 MB · ≈ 1.4 GB (NLLB ≈ 2.0 GB) | 3–4 days · **medium–high** (Argos 1.1 quality unknown; NLLB would break the 500 ms target) |
| **vi → ko** | PhoWhisper-small (unchanged) | Argos `vi→en→ko` (pivot, **package 1.1**, +121 MB); NLLB/M2M direct; Google | ko-KR, 3 voices | as vi → ja | +121 MB · ≈ 1.4 GB (NLLB ≈ 2.0 GB) | 3–4 days · **medium–high** (same as ja) |
| **zh → vi** | Whisper `small` / `medium` / `large-v3-turbo`, to choose by CER on FLEURS zh; simplified/traditional output to control | Argos `zh→en→vi` (pivot, 74 + 68 MB); NLLB/M2M direct; Google | vi-VN, 2 voices | restore off; Chinese blacklist; junk off; loop/anti-stall on characters; glossary CJK source fix; CER metrics; pre-roll and indicator re-checked | small 485 MB (turbo 1.6 GB, **> 1 GB: ask**) + 142 MB · ≈ 1.2 GB (turbo ≈ 2.1 GB) | 1–2 weeks · **high** |

### Per pair

**en → vi (English speaker → Vietnamese listeners).**
- **ASR:**
  - Whisper `small` is in the catalog. `small.en` / `distil-small.en` are English-only and usually better for English, but are not in the catalog yet (faster-whisper offers them).
  - PhoWhisper must not be used. Expected quality: the best of all candidates; measure on FLEURS `en_us`.
- **MT:** Argos `en→vi` 1.9, direct, so no pivot; the latency should be like today's. NLLB/M2M direct; Google fallback.
- **TTS:** vi-VN, 2 voices (HoaiMy F, NamMinh M).
- **Data:** FLEURS `en_us` audio + `vi_vn` text (both FLoRes-aligned; the vi text is already here); a new custom set en→vi.
- **Turn off / replace:**
  - restore off (Whisper punctuates English);
  - Vietnamese blacklist → an English list (build with `collect_hallucinations.py`);
  - junk tokens off ("ofc" is English);
  - `glossary_en_vi.tsv`: the same names reversed, e.g. "Hanoi University of Science and Technology" → "Đại học Bách khoa Hà Nội";
  - normaliser `en` already exists.
- **Disk · RAM:** Whisper small 485 MB + Argos 68 MB; ≈ 1.1 GB peak.
- **Effort · risk:** 2–3 days, low. Main risk: English names and code-switching in the Vietnamese voice output (only 2 vi voices).

**vi → fr.**
- **ASR:** unchanged. PhoWhisper, vibert-capu, the Vietnamese blacklist, junk tokens and pre-roll all stay valid: the source is still Vietnamese.
- **MT:** Argos pivot vi→en (have) → en→fr (65 MB). French is high-resource and Latin-script; double MT time (estimate ≈ 0.2 s mean per segment, to measure). NLLB/M2M direct; Google fallback.
- **TTS:** fr-FR, 5 voices.
- **Data:** FLEURS vi audio (have) + `fr_fr` text; a new custom set vi→fr (needs a French reader to review); normaliser `fr` (accents, apostrophes; like `en`).
- **Turn off / replace:**
  - `glossary_vi_fr.tsv` with the official French names (to confirm with the school);
  - rerun `glossary_bench`: French elision ("de" + vowel → "d'") may interact with the placeholder.
- **Disk · RAM:** +65 MB; ≈ 1.35 GB (with NLLB ≈ 2.0 GB).
- **Effort · risk:** ≈ 2 days once the pivot exists; low–medium (pivot quality).

**vi → zh.**
- **ASR:** unchanged (as vi → fr).
- **MT:** Argos pivot vi→en → en→zh (simplified; `zt` = traditional also exists). NLLB/M2M direct; Google `zh-CN`.
- **TTS:** zh-CN, 6 voices.
- **Data:** FLEURS vi audio + `cmn_hans_cn` text; sacrebleu `zh` tokenizer (built in) + chrF; a custom set vi→zh (needs a Chinese reader).
- **Turn off / replace:**
  - glossary: ASCII placeholder lookarounds (Finding G), `glossary_vi_zh.tsv` with official Chinese names, `glossary_bench` rerun;
  - sentence split and word counts in the log (rows 14, 16).
- **Disk · RAM:** +71 MB; ≈ 1.35 GB (NLLB ≈ 2.0 GB).
- **Effort · risk:** 3–4 days, medium: two MT hops, a CJK target, and a reviewer needed.

**vi → ja / vi → ko.**
- As vi → zh, with these differences:
  - Argos `en→ja` / `en→ko` are **version 1.1** (120 / 121 MB, an older generation): their quality is the main unknown.
  - If they are too weak, NLLB-600M is the offline alternative. It was measured on vi→en at 552 / 1160 ms mean / P95 per segment, which breaks the 500 ms target.
  - BLEU for ja/ko properly needs MeCab tokenisers, i.e. new eval dependencies; chrF / sacrebleu `char` instead.
- **TTS:** ja-JP 2 voices; ko-KR 3 voices.
- **Data:** FLEURS `ja_jp` / `ko_kr` text.
- **Disk · RAM:** +120 / +121 MB; ≈ 1.4 GB (NLLB ≈ 2.0 GB).
- **Effort · risk:** 3–4 days each, medium–high.

**zh → vi (Chinese speaker → Vietnamese listeners).**
- **ASR:**
  - Plain Whisper; PhoWhisper is Vietnamese only.
  - Choose between `small` (0.9 GB), `medium` (1.7 GB) and `large-v3-turbo` (1.75 GB; download 1.6 GB, **over 1 GB, so ask first**) by CER and latency on FLEURS `cmn_hans_cn` (525 MB).
  - Known risk: Whisper mixes simplified and traditional characters in its output. Control it with an initial prompt or a conversion step; no new library without asking.
- **MT:** Argos pivot zh→en (74 MB) → en→vi (68 MB). NLLB/M2M direct; Google fallback.
- **TTS:** vi-VN, 2 voices.
- **Data:** FLEURS `cmn_hans_cn` audio + `vi_vn` text; CER (character normaliser).
- **Turn off / replace** (CJK on the ASR side touches the most code):
  - restore off; Chinese blacklist; junk tokens off;
  - loop/repeat handling and the anti-stall loop rule on characters (rows 7, 8);
  - glossary CJK source matching and the length rule (row 9);
  - `preroll_bench` "first word lost" on characters; indicator thresholds for a bigger model (row 18).
- **Disk · RAM:** small 485 MB (or turbo 1.6 GB) + 142 MB; ≈ 1.2 GB (turbo ≈ 2.1 GB).
- **Effort · risk:** 1–2 weeks, high.

## 3. Architecture: a new pair = configuration

**Goal.** One place decides everything that depends on the pair. Adding a measured pair is a data change (a `languages.py` row + a `PAIRS` entry + resource files), not a code change.

1. **Language rows** (`languages.py`, the existing `Language` dataclass, extended). Per language:
   - codes (whisper, argos, google, nllb, m2m100), FLEURS config, normaliser, BLEU tokenizer, Edge voice (already there);
   - new: `script` (`spaced` | `cjk`, drives tokenisation and matching);
   - sentence-end characters;
   - ASR models in order of preference, and which models support the language;
   - resource files: blacklist, junk tokens (optional);
   - the anti-stall token cap, if measured.
2. **Pair table** `PAIRS[(src, tgt)]`:
   - status `supported` / `experimental`;
   - Argos route (`direct` package or `pivot:en` with both packages: URL + pinned version);
   - default MT backend;
   - glossary file (optional);
   - voice override (optional);
   - measured results (file names) for the README.

   Plain Python data in `languages.py`, so no new library (`tomllib` would also be stdlib, but Python data keeps the type checks).
3. **`resolve_pair(settings) -> PairProfile`**, called once at session start. The orchestrator reads ASR model, restore, filters, glossary, MT route and voice from the profile instead of from scattered `Settings` fields and file constants. vi→en must resolve to exactly today's behaviour: the profile is a refactor proven by the existing 467 tests.
4. **Text helpers by script:** one `tokens(text, lang)` used by `collapse_repeats` / `has_loop_tail` / `trim_truncated_tail`, the glossary matching and length rule, sentence split, log counts and `first_word_lost`. Placeholders use ASCII lookarounds.
5. **MT routing:** wire `plan_route` + `PivotTranslator` into `_prepare_translator`. Argos `direct` → `pivot:en` → (if chosen) NLLB/M2M → Google, with the warnings the app already gives. The same cache rules apply per package.
6. **Front ends:**
   - CLI: `--source`, `--target`, `--list-pairs`.
   - GUI: a pair dropdown in Cài đặt, locked while running; changing it starts a new session. The voice follows the target.
   - The labels come from the pair.
7. **Evaluation per pair:** `languages.py` drives the normaliser and metric (WER or CER, BLEU tokenizer, chrF). `asr_bench`, `preroll_bench`, `mt_bench` and `glossary_bench` take `--source/--target`. Custom and glossary sets live as `eval/data/mt_custom_<src>_<tgt>.tsv` with `reviewed` flags, as today.

**Order of work.**
1. **Steps 1–5 with vi→en only**, no visible change: refactor, behaviour identical, all tests pass. This is the groundwork every pair needs, so it can start before the supervisor chooses.
2. **The first pair chosen**, or en→vi if the supervisor wants something soon, because it is the cheapest: no pivot, Latin script, data already half here.
   1. `languages.py` row + `PAIRS` entry.
   2. Measure: ASR on FLEURS, MT with `mt_bench`, glossary with `glossary_bench`, latency in a fake-stream session.
   3. Manual test with a real microphone.
3. **Further Vietnamese-source pairs:** vi→fr, then vi→zh (CJK target), then vi→ja / vi→ko (Argos 1.1 check first).
4. **CJK-source pairs (zh→vi) last:** they need step 4 everywhere on the ASR side and an ASR model choice.

Each pair keeps the rules used so far: measure before turning anything on by default, fixed seeds, no conclusion from small samples, and ask before downloads over 1 GB.

## 4. Questions for the supervisor

1. **Which pairs, and in which order?** One-way (e.g. vi → X) or both directions? How many are needed for the thesis or the demo?
2. **Who speaks which language?** The app translates **the local user's own speech** into a virtual microphone for the other side.
   - **Trung → Việt** means the person at this Mac speaks Chinese.
   - If the intent is instead to translate **what the other side says** (incoming call audio, heard in Vietnamese on headphones), that is a different audio route: capture the call output, play to headphones. It is a separate piece of work to plan.
3. **Quality and latency bar per pair:** the same as vi→en (MT ≤ 500 ms per segment, WER/BLEU not worse than measured), or simply "works in a demo"? For ja/ko this decides between Argos 1.1 (fast, unknown quality) and NLLB (better, but measured at 552 / 1160 ms mean / P95 per segment on vi→en).
4. **Chinese:** Mandarin with simplified characters (mainland), or traditional (Taiwan)? Argos and Edge have both.
5. **Licences and online services:** is NLLB (CC-BY-NC) acceptable as an offline MT for some pairs? Are Google (fallback) and Edge TTS (online) still acceptable for the new pairs?
6. **Reviewers:** who can review the reference translations and glossary names in Chinese / Japanese / Korean / French? Without one, the custom sets stay "unreviewed" and only FLEURS can be used to conclude.
7. **Official names:** the official names of HUST and SoICT in the target languages (the school's international pages), for the glossaries.
8. **Hardware:** still the 8 GB MacBook on CPU? For a Chinese source, is a bigger ASR model (turbo, about 1.75 GB RAM, slower) acceptable?
9. **Switching:** must the pair be switchable in the GUI during a demo, or is one pair per run (a command-line flag) enough?
