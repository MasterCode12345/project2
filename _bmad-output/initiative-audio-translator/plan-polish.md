---
title: 'Polish: compact window position; junk-token list (unk, ofc) in the ASR filter'
type: 'fix'
created: '2026-11-14'
status: 'done (branch pushed, not merged)'
baseline_revision: '1ed5aee'
branch: 'fix/polish'
---

## 1. Compact window position
In the real Meet test, the compact window opened at the top-right and covered Chrome's toolbar and its "⋮" menu. It now first opens **80 px below the top edge**, still flush right with a 16 px margin (`CompactWindow.TOP_OFFSET`, `RIGHT_MARGIN`). A position the user drags it to is kept for the rest of the run, as before.

## 2. Junk tokens: "ofc" joins "unk"
**One list.** The tokens live in `src/audio_translator/junk_tokens_vi.txt`, one per line, with their evidence after a TAB. Adding a token needs no code change. `asr.strip_junk` removes a listed token only when it stands alone as a word, in any case and any position; "office", "junk" and "unknown" stay. The filter reason is now "junk", and `Settings.asr_drop_unk` is renamed `asr_drop_junk`.

**Counts for "ofc"** (stand-alone word, ignoring case):
- References: VIVOS test 0, VIVOS train 0, FLEURS test 0.
- `results/mt_asr_cache_vi.tsv`: 1 of 1 389 lines ("unk bao giờ phẩy. ofc đâu phẩy.").
- Stall-bench per-segment output (scratch TSVs behind `results/stall_*.csv`):
  - FLEURS clean: 1 of 244 segments in base and in temps ("ba mươi mốt dặm. ofc ba mươi mốt dặm."); 0 in cap and in both;
  - VIVOS × DEMAND and VIVOS × MUSAN: 0 in every variant.
- Real logs: 1 segment, `latency_compact_20261114_1518.csv` ("cảm ơn bạn. ofc thật sự…" → "Thank you. OFC is really…").

**WER before → after.** Recomputed from the cached transcripts. Before = the current default, with "unk" already removed; after = "unk" and "ofc" removed.

| Variant | Before | After |
|---|---|---|
| FLEURS clean, base | 15.65 | **15.62** |
| FLEURS clean, temps | 16.37 | **16.34** |
| Every other cell (4 variants × DEMAND/MUSAN × 5 SNRs, plus FLEURS cap/both) | – | unchanged |

**It is never worse, so the token is on by default.**

Limitation: these are single occurrences, so the evidence only shows that removing "ofc" is harmless, not that it is common. Related forms such as "ofo" and "of." (seen once each in a probe on music) are not in the list; they would need the same check first.

381 tests pass.
