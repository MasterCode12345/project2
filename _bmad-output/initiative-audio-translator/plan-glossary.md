---
title: 'Glossary for proper names (vi → en): protect terms with placeholders, fall back safely'
type: 'feature'
created: '2026-11-15'
status: 'done — on by default (rule passed); branch pushed, not merged'
baseline_revision: '4c6c2fe'
branch: 'feat/glossary'
route: 'full'
risk: 'medium'
---

## Intent

Proper names are translated word by word. In `eval/data/mt_custom_vi_en.tsv` c01, "Đại học Bách khoa Hà Nội" becomes:
- Argos (default): "University of Hanoi Encyclopedia";
- NLLB: "Hanoi Polytechnic University".

The official name is "Hanoi University of Science and Technology". A user-editable glossary per language pair should force the official translation, without changing anything for sentences that contain no term.

## Design

### 1. Glossary files

**Default file.**
- `src/audio_translator/glossary_vi_en.tsv`, one entry per line: `Vietnamese term<TAB>English`; `#` starts a comment; after a second TAB, an optional note (source).
- The name carries the language pair, so other pairs are added as `glossary_<src>_<tgt>.tsv`.

**User file.**
- `--glossary PATH` (CLI and GUI), and `Settings.glossary_path`.
- Its entries are **added to** the default ones and **override** an identical source term.
- A broken line raises a clear error at session start, giving the file and the line, and the session does not start. → **Question 1.**

**Loading checks.**
- A source term must have **≥ 2 words**: single generic words such as "bách khoa" or "trường" are refused.
- Duplicates are reported.

**Initial entries.** Requested by you:

| Vietnamese | English |
|---|---|
| Đại học Bách khoa Hà Nội | Hanoi University of Science and Technology |
| Trường Công nghệ Thông tin và Truyền thông | School of Information and Communication Technology |
| Trần Hải Anh | Tran Hai Anh |
| Trần Trung Nghĩa | Tran Trung Nghia |

**Proposed by me (marked "đề xuất"; keep or drop):**

| Vietnamese | English | Why |
|---|---|---|
| Trường Đại học Bách khoa Hà Nội | Hanoi University of Science and Technology | the full form with "Trường"; longest match wins, so "Trường" is not translated separately as "school" |
| Bách khoa Hà Nội | Hanoi University of Science and Technology | the usual short form; with "Hà Nội" it can only mean the university |
| Viện Công nghệ Thông tin và Truyền thông | School of Information and Communication Technology | SoICT's former name (Viện), still common in speech |
| Đại học Quốc gia Hà Nội | Vietnam National University, Hanoi | the other big university name likely in a student's talk |
| đồ án tốt nghiệp | graduation project | the project's own topic; Argos and NLLB vary ("graduation plan", "thesis") |
| giảng viên hướng dẫn | academic supervisor | same reason |

"bách khoa" alone, "Hà Nội" alone and "trường" alone are **not** entries: they are generic, and the ≥ 2-word rule enforces it.

### 2. Matching

- **Where:** on the text **after restoration** (`restore=auto`, casing and punctuation), right before MT, in the MT worker.
- **How:**
  - Unicode **NFC** on both sides;
  - **case-insensitive** (`casefold`);
  - **Vietnamese tone marks kept** ("Hải" ≠ "Hai");
  - whole words only (word boundaries on both ends);
  - **longest match first**, no overlaps: candidates are sorted by length in words, then by length in characters.
- **Not matched across punctuation:** "Bách khoa, Hà Nội" is not a term.

### 3. Placeholder protection

`GlossaryTranslator` wraps the session's translator (Argos + Google fallback, NLLB/M2M100 + fallback, or Google).

**Per sentence:**
1. Find the terms; if there are none, call the inner translator **unchanged**, so the result is identical to today. Tested.
2. Replace each term occurrence k by a placeholder P(k).
3. Translate.
4. Check that **each placeholder appears exactly once** in the output, in the form produced. If so, replace it by the English term.
5. If any placeholder is missing, duplicated or deformed, **retranslate the original sentence without the glossary**, and return that. It is counted as a fallback.
6. A final guard ensures that the output given to TTS never contains a placeholder pattern, even a partial one. If it does, step 5 applies.

**Placeholder forms to compare** (step 1 of the measurement; each backend keeps the form that survives best):

| Id | Form | Idea |
|---|---|---|
| P1 | `X1`, `X2`, … | short Latin token |
| P2 | `ZQX1` | rare letter cluster, not a word in either language |
| P3 | `[1]` | bracketed number |
| P4 | `__T1__` | code-like token |
| P5 | the **English term itself** inline: "tôi học ở Hanoi University of Science and Technology" | no placeholder; relies on the model copying a Latin span. Check: the exact English term appears in the output |

Note on P5: if it is chosen, step 4 checks for the English term instead of a placeholder; the fallback stays the same.

**Grammar risk.** A placeholder can change the words around it, e.g. a missing article ("at X1" → "at the Hanoi University…"). The BLEU/chrF check on the term sentences and on mt_custom covers it.

### 4. Measurement (`eval/mt_bench.py` + new set)

**New set: `eval/data/mt_glossary_vi_en.tsv`.**
- About 20 sentences with the glossary terms, all marked **`reviewed = no`** (your review).
- Mixed positions (start / middle / end), casing as ASR + restore produce it (lowercase / capitalised), and 2 terms in one sentence.
- Plus 3 sentences containing the generic words alone ("bách khoa toàn thư", "trường học ở Hà Nội"), which must **not** be matched.

**Metrics.**
- **Placeholder survival rate** per form × backend (Argos, NLLB-600M, M2M100-418M, Google) on the glossary set. → choose a form per backend.
- **Term accuracy**, before and after, per backend: the share of expected English terms that appear verbatim (case-insensitive) in the hypothesis.
- **Fallback count**: sentences that lost a placeholder.
- **BLEU / chrF** on the glossary set (unreviewed, so only indicative), on mt_custom (unreviewed as well) and on FLEURS (100 sentences, reviewed references):
  - FLEURS should contain almost no glossary terms → expected identical. This verifies "no term → no change" at scale.
- **mt_ms:** mean / P95 of translation time, before and after.
  - Matching costs microseconds.
  - A fallback costs a second translation, so P95/max can rise on the term sentences.

**Runs.**
- mt_bench text input (references as source; no ASR), seed fixed.
- Google is called on the glossary set + mt_custom only (about 70 sentences × forms). FLEURS × Google already exists in `results/mt_vi_en*.csv`; I re-run it with the glossary only if FLEURS contains a term. → **Question 2.**

**Decision rule (yours):** on by default only if:
- term accuracy improves clearly with the default backend (Argos), e.g. from about 0–30 % to ≥ 90 %;
- BLEU and chrF on FLEURS and mt_custom are **not worse**;
- the mt_ms increase is small (reported).

Otherwise it stays optional (`--glossary`).

## Other changes

- **Latency log:** new column `glossary`. It lists the terms used in that segment (`;`-separated source terms), `fallback` when the placeholders were lost, or empty.
  - The orchestrator keeps it per segment, like `audio_s`; the MT worker reads it from the wrapper right after `translate`. There is one MT worker, so there is no race.
  - `latency_report.py` counts the segments that used the glossary.
- **Settings:**
  - `glossary: bool` (default **False** until the rule passes);
  - `glossary_path: str | None` (the user file).
- **CLI and GUI:**
  - `--glossary PATH` (implies on);
  - `--no-glossary`.
  - No GUI editor, as asked.
- **Docs:**
  - README: how the glossary works; how to add a term (edit the TSV or pass `--glossary`); the ≥ 2-word rule.
  - `docs/manual_test.md`: a short scenario that says the four names in a sentence and checks the EN line / log column.

## Files

| File | Change |
|---|---|
| `src/audio_translator/glossary.py` (new) | load / merge / validate, `find_terms`, `protect`/`restore`, `GlossaryTranslator` |
| `src/audio_translator/glossary_vi_en.tsv` (new) | the entries above |
| `src/audio_translator/config.py` | `glossary`, `glossary_path` |
| `src/audio_translator/orchestrator.py` | wrap the translator in `_prepare_translator` when on; per-segment `glossary` value for the log |
| `src/audio_translator/latency_log.py`, `eval/latency_report.py` | `glossary` column, count |
| `src/audio_translator/cli.py`, `gui.py` | `--glossary`, `--no-glossary` |
| `eval/mt_bench.py` | `--glossary [PATH]`, `--placeholder P1..P5`, term accuracy / fallback / survival columns |
| `eval/data/mt_glossary_vi_en.tsv` (new) | about 20 + 3 sentences, unreviewed |
| `results/mt_glossary_*.csv` | the measurements |
| README, `docs/manual_test.md`, plan | |

## Tests

- **Loading:** default + user file merge and override; refusal of 1-word terms; a bad line gives an error with file:line; comments.
- **Matching:**
  - NFC (a decomposed "ạ"); case-insensitive; tone marks matter ("Trần Hải Anh" vs "Tran Hai Anh"); whole words only;
  - longest match ("Trường Đại học Bách khoa Hà Nội" over "Đại học Bách khoa Hà Nội" over "Bách khoa Hà Nội"); two terms in one sentence;
  - no match for "bách khoa toàn thư".
- **No term → the inner translator gets the identical string and the result is identical.**
- **Placeholder restored** → English term in the output. **Placeholder lost / duplicated / deformed** → fallback to the plain translation, with no placeholder in the output and the fallback counted.
- Orchestrator with fake streams: the glossary column in the latency log; off by default → no wrapper.
- The CLI and GUI flags.
- The bench: term accuracy and survival columns (smoke).

## Decisions (human, 2026-11-15)

0. **Length rule.** Vietnamese is written syllable by syllable, so "bách khoa" and "Hà Nội" already have 2 "words".
   - New rule: a term needs **≥ 3 syllables**.
   - A shorter one is accepted only with an explicit `short_ok` in column 3.
   - Tests: "bách khoa", "Hà Nội" and "bách khoa toàn thư" must not match.
1. **Proper names only** (people, schools, organisations).
   - Kept: Trường Đại học Bách khoa Hà Nội, Bách khoa Hà Nội, Viện Công nghệ Thông tin và Truyền thông, Đại học Quốc gia Hà Nội → Vietnam National University, Hanoi.
   - **Not** in the default file: "đồ án tốt nghiệp", "giảng viên hướng dẫn". Measure whether the translators get them right; add them only if clearly wrong, then with "supervisor".
   - No "Đại học Bách khoa" alone: several universities share the name.
2. **Errors in the glossary files.**
   - Default file: a format test; an error fails the test.
   - User file (`--glossary`): a bad line is skipped with a yellow GUI warning / CLI line giving file:line; valid lines are used.
   - A **missing path** refuses the session start with a clear error.
3. **Google** may be called for the measurement, spread out. If blocked or failing: "not measurable", without waiting. The default decision rests on **Argos and NLLB**.
4. **Rule for turning it on by default.** All of:
   - Argos term accuracy ≥ 90 %;
   - no placeholder ever leaks;
   - sentences without terms translate identically;
   - BLEU/chrF on mt_custom and FLEURS not worse;
   - mt_ms increase ≤ 10 %.
5. **For each term sentence**, print the before/after translations side by side for human reading.

## Results (2026-11-15) — `results/mt_glossary_vi_en.csv`, `results/mt_glossary_vi_en_samples.tsv`

**Sets.** Text input, no ASR:
- the glossary set: 27 sentences; 20 contain terms (25 expected terms), 3 are generic, 4 are phrase checks; unreviewed;
- mt_custom: 25 sentences, 1 with a term (c01), unreviewed;
- FLEURS: 100 sentences, seed 0, **0 with a term**.

**Glossary set, by backend and placeholder form.**

| Backend | Term accuracy before | x | zqx | bracket | under | inline |
|---|---|---|---|---|---|---|
| Argos (default) | 0 % | 96 % (1 fallback) | **100 % (0 fallbacks)** | 100 % | 0 % (20 fallbacks) | 32 % (13) |
| NLLB-600M | 12 % | 92 % (1) | **100 % (0)** | 84 % (2) | 64 % (8) | 60 % (8) |
| M2M100-418M | 8 % | 100 % (0) | **100 % (0)** | 68 % (5) | 64 % (6) | 64 % (7) |
| Google | not measurable: blocked ("too many requests"), not retried, as agreed | | | | | |

**Chosen form: `zqx` (`ZQX1`…), the only one at 100 % with no fallback on all three local backends.**

**Other checks, for `zqx`.**
- **Leaked placeholders: 0** in every backend × form.
- **Sentences without terms identical:** 7/7 (glossary set), 24/24 (mt_custom), 100/100 (FLEURS), for every backend.
- **BLEU / chrF:**

  | Set | Argos | NLLB | M2M100 |
  |---|---|---|---|
  | Glossary set | 18.6 → 57.9 / 45.0 → 76.2 | 27.9 → 58.6 | 27.4 → 57.7 |
  | mt_custom | 42.5 → 44.4 / 59.3 → 61.3 | 52.0 → 53.6 | 44.9 → 47.2 |
  | FLEURS | unchanged (27.70 / 54.37) | unchanged | unchanged |

- **mt_ms.**
  - `Glossary.find` costs **0.004 ms per sentence**: about 0.002 % of an Argos FLEURS sentence (about 197 ms).
  - FLEURS through the wrapper (form x, all 100 sentences): 197.5 → 203.0 ms (+2.8 %, within run-to-run noise).
  - Term sentences got *faster* (Argos 86.7 → 82.6 ms), because the placeholder is shorter than the name.
- **The two common phrases** (not in the glossary):
  - "đồ án tốt nghiệp": Argos right ("graduation project", 2/2); NLLB "graduate project" / "graduation papers"; M2M100 "graduate text" / "graduation scheme".
  - "giảng viên hướng dẫn" → "instructor" (Argos, NLLB) or "tutor" (M2M100), never "supervisor".
  - With the default Argos these are not clearly wrong, so they were **not added**, per the proper-names-only rule. Your call.

**Decision rule (yours):** Argos ≥ 90 % (100 %) ✓, no leak ✓, identical without terms ✓, BLEU/chrF not worse ✓, mt_ms ≤ +10 % ✓. **Turned on by default:** `Settings.glossary = True`, `glossary_placeholder = "zqx"`; `--no-glossary` turns it off.

**Naturalness, read from the samples (Argos, zqx).** The names are right, but some sentences lose an article or a preposition around the placeholder:
- "I'm School of Information and Communication Technology, 6th class";
- "School of … has lots of labs" (no "The");
- "I am Tran Trung Nghia student School of …".

One sentence changes meaning: "Trường CNTT&TT thuộc Đại học Bách khoa Hà Nội" → "… **is** Hanoi University of Science and Technology" (should be "belongs to"). Without the glossary, the same sentence named the wrong school ("University of the Encyclopedia of Hanoi"). The full before/after list is in the samples file.

**Limitations.**
- 20 term sentences; references unreviewed.
- Google not measured. In the app, Google is only the per-segment fallback of Argos; the wrapper verifies the placeholders whatever the backend, so a Google fallback that damages one only causes a plain retranslation.
- Sentences with a placeholder sometimes lose an article or a preposition around the name ("I'm School of …", "student School of …").
- A sentence with two terms can have its relation changed: "Trường CNTT&TT thuộc Đại học Bách khoa Hà Nội" → "… **is** Hanoi University of Science and Technology" ("thuộc" = "belongs to").
- "đồ án tốt nghiệp" and "giảng viên hướng dẫn" stay out of the shipped table (your decision: the table is for proper names only).

**Possible extension (not done now):** a placeholder with its class noun kept in the source, e.g. "trường ZQX1" / "thầy ZQX2", so the MT sees a noun phrase and keeps articles, prepositions and relations. To be measured with the same bench before any change of default.

## Questions for you (answered above)

1. **A broken user glossary file:** refuse to start the session with the error (proposed: you notice at once), or warn and run without the user file?
2. **Google:** may I call it for the glossary set + mt_custom, × the placeholder forms (about 70 sentences × 5 forms ≈ 350 requests, spread out)? FLEURS × Google only if FLEURS contains a term.
3. **The six proposed entries:** keep all of them? In particular "academic supervisor" for "giảng viên hướng dẫn" (alternatives: "supervisor", "advisor").
