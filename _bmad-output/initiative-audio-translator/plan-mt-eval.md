---
title: 'M5 — đo chất lượng dịch (mt_bench), bảng ngôn ngữ trung tâm, nghiên cứu model dịch thay Argos (giai đoạn 4A)'
type: 'feature'
ticket: ''
created: '2026-10-31'
status: 'done'
baseline_revision: '8ce7e3aae35b1bee207f009ca2d5525e65435839'
route: 'full'
route_source: 'auto'
risk: 'low'
review: ''
review_source: ''
lenses_ran: []
review_loop_iteration: 0
context: []
---

<frozen-after-approval reason="human-owned intent — do not modify unless human renegotiates">

## Intent

**Problem:** ASR đã tốt lên nhiều (WER VIVOS sạch: small 22,2% → phowhisper-small 6,9%, phowhisper-medium 4,9%; số từ bench bước 1, bench vẫn đang chạy). Giờ khâu dịch M5 (Argos vi→en) là điểm yếu chính, nhưng chưa có số đo nào. Ví dụ đã thấy: "Đại học Bách Khoa Hà Nội" bị dịch thành "University of Hanoi".

**Approach:** Viết bench `eval/mt_bench.py` trả lời hai câu hỏi:
- (A) Dịch tốt đến đâu khi đầu vào là câu nguồn chuẩn?
- (B) Dịch tệ đi bao nhiêu khi đầu vào là chữ do ASR (`small`, `phowhisper-small`) nhận từ audio FLEURS?

Chỉ số dùng BLEU và chrF (sacrebleu), cùng độ trễ dịch. Thêm bảng ngôn ngữ trung tâm `languages.py` và hàm `supports(src, tgt)` cho các backend để code mới không ghi cứng vi/en. Nghiên cứu và đề xuất 2–3 model dịch offline đa ngôn ngữ. Giai đoạn này chưa tích hợp model mới, không đổi hành vi app. **Dừng sau 4A** để người dùng chọn model.

**Decisions (human, 2026-10-31):**
1. Tải FLEURS `en_us/test.tsv` (368 KB, chỉ chữ) vào `~/data/fleurs_en/`.
2. Thêm điều kiện `text_norm`: câu nguồn chuẩn nhưng chuẩn hóa như đầu ra ASR (chữ thường, bỏ dấu câu). So text → text_norm → asr để tách lỗi mất định dạng khỏi lỗi nhận dạng.
3. CSV có thêm cột `corpus` ở đầu.
4. Plan được duyệt; những gì thêm vào thấy hợp lý thì được làm.

## Boundaries & Constraints

**Always:**
- **Ngôn ngữ đi qua tham số:** bench và backend nhận `--src`/`--tgt`. Mọi mã riêng của từng backend lấy từ `languages.py`.
- **Dữ liệu ngoài repo:** script nhận đường dẫn, không tự tải. Thiếu dữ liệu thì báo lỗi rõ và exit 2, giống `asr_bench`.
- **Tập câu cố định:** sắp theo id. `--limit N` lấy N câu bằng RNG có `--seed` (mặc định 0), nên chạy lại ra cùng tập. Mỗi id FLEURS dùng một file audio, chọn tất định là file có tên nhỏ nhất.
- **Điều kiện ASR:** Whisper nhận nguyên câu audio sạch, không qua VAD/segmenter, không nhiễu. Cấu hình giống bench bước 1: CPU int8, beam 5, `language` lấy từ bảng ngôn ngữ, không prompt. Như vậy B đo đúng lỗi nhận dạng truyền sang dịch, không lẫn lỗi cắt câu.
- **Một model ASR mỗi lúc** (máy 8 GB). Chữ ASR được ghi ra cache TSV để các backend dùng chung và chạy lại không phải nhận dạng lại.
- **Google lỗi theo từng câu:** câu lỗi được đếm vào `n_failed`, câu đó bị bỏ khỏi điểm của backend ấy, và lượt chạy vẫn tiếp tục. `n_sents` là số câu thực sự được chấm.
- **sacrebleu** (Apache-2.0) chỉ thêm vào group `eval`. BLEU tính theo corpus với cấu hình mặc định (cased, `tok` lấy từ bảng ngôn ngữ, `13a` cho tiếng Anh). chrF dùng chrF2 mặc định (`nc=6, nw=0`). Chữ ký (signature) sacrebleu được in ra log và README.
- **Hạn chế ghi kèm kết quả:** FLORES vốn dịch từ tiếng Anh sang tiếng Việt, nên chiều vi→en dùng câu nguồn kiểu "translationese". Chỉ có một tham chiếu. Tập nhỏ không được đem ra kết luận.

**Never:**
- Không thêm torch, transformers hay COMET vào app hoặc group nào. COMET chỉ ghi là hướng mở rộng.
- Không tích hợp model dịch mới, không đổi mặc định hay hành vi của `Orchestrator`, `cli`, `gui`.
- Không tải gì trên 1 GB.
- Không gộp tập câu tự viết vào kết luận khi bản dịch tham chiếu chưa được người dùng duyệt.
- Không push.

## I/O & Edge-Case Matrix

| Scenario | Input / State | Expected Output / Behavior | Error Handling |
|----------|--------------|---------------------------|----------------|
| Ghép cặp FLEURS | vi `test.tsv` (857 dòng, 347 id) + en `test.tsv` | một cặp cho mỗi id có ở cả hai bên; log in ra số id chỉ có ở một bên | một id có hai câu raw khác nhau ở cùng một bên → dừng và báo id đó |
| Điều kiện text | câu nguồn chuẩn (cột raw, giữ hoa/dấu câu) | dòng CSV `condition=text`, `asr_model=-` | — |
| Điều kiện text_norm | `normalize(src, src_lang)` của câu nguồn | dòng CSV `condition=text_norm`, `asr_model=-` | — |
| Điều kiện ASR | wav FLEURS (float32 16 kHz, đọc bằng `read_wav`) | dòng CSV `condition=asr` cho từng `asr_model` | model chưa có → exit 2 kèm lệnh convert |
| Google bị chặn | mọi lệnh gọi đều ném lỗi | `n_failed = n`, cột điểm để trống, lượt chạy vẫn xong | không traceback |
| Backend không hỗ trợ cặp | `supports(src,tgt)` = False | dòng đó bị bỏ qua và log ghi lý do; có pivot qua `en` thì `backend=argos+pivot:en` | ngôn ngữ không có trong bảng → exit 2 |
| Tập tự viết | `eval/data/mt_custom_vi_en.tsv`, cột `reviewed` | các dòng ghi `corpus=custom_unreviewed` cho tới khi mọi câu là `yes` | — |
| `--smoke` | không mạng, không model | CSV và file mẫu đầy đủ từ dữ liệu + translator + ASR giả | — |

</frozen-after-approval>

## Code Map

- `src/audio_translator/translate.py` -- `Translator` protocol; `OfflineTranslator` (Argos, CT2 + SentencePiece), `GoogleTranslator`, `FallbackTranslator`
- `src/audio_translator/config.py` -- `source_lang="vi"`, `target_lang="en"`, `tts_voice`, `argos_vi_en_url` (giữ nguyên)
- `src/audio_translator/orchestrator.py` -- `_prepare_translator` (không sửa ở 4A)
- `eval/denoise_bench.py` -- `normalize_vi`, `read_wav` (đọc được float WAV); `eval/asr_bench.py` -- `load_fleurs`, `TimedASR`, `latency_stats`, cách ghi dòng dần
- `scripts/convert_phowhisper.py` -- mẫu cho script convert model dịch một lần ở 4B

## Tasks & Acceptance

**Execution:**
- [ ] `src/audio_translator/languages.py` -- `@dataclass(frozen=True) Language` gồm: `code`, `name`, `whisper`, `asr_model` (vi → `phowhisper-small`, ngôn ngữ khác → `small`), `argos`, `google`, `nllb` (vd `vie_Latn`, để dành cho 4B), `edge_voice`, `fleurs` (`vi_vn`/`en_us`), `normalizer` (tên hàm), `bleu_tokenize` (`13a`; sau này tiếng Trung là `zh`, tiếng Nhật là `ja-mecab`…). Bảng `LANGUAGES` gồm hai dòng vi và en. Thêm `get_language(code)` (ValueError nếu không có) và `normalize(text, lang)`: `vi` giữ đúng logic `normalize_vi`; `en` chuyển chữ thường, bỏ dấu câu, giữ dấu nháy trong từ (`don't`). Bảng dispatch theo tên để sau này thêm normalizer tách chữ cho ngôn ngữ châu Á -- thêm ngôn ngữ = thêm một dòng
- [ ] `src/audio_translator/translate.py` -- thêm (không đổi hành vi cũ): `supports(src, tgt)` cho `OfflineTranslator` (theo `metadata.json` của package) và `GoogleTranslator` (theo bảng ngôn ngữ); `PivotTranslator(first, second)`; `plan_route(supports, src, tgt, pivot="en")` → `direct` | `pivot:en` | `None`. Quy tắc: chỉ dùng pivot khi không có model dịch thẳng, và luôn ghi rõ tên route
- [ ] `eval/mt_bench.py` -- loader FLEURS ghép theo id; loader tập tự viết; `--src/--tgt/--fleurs-src-dir/--fleurs-tgt-tsv/--custom-tsv/--backends argos,google/--asr-models small,phowhisper-small/--asr-cache/--limit/--seed/--out/--samples-out/--smoke`. Độ trễ đo từng lệnh dịch (perf_counter) sau một câu khởi động không tính giờ. CSV ghi từng dòng ngay khi xong
- [ ] `eval/data/mt_custom_vi_en.tsv` -- 25 câu có tên riêng, số, ngày giờ, đơn vị, từ mượn; cột `id, src, ref, category, reviewed=no`; ghi rõ "CẦN DUYỆT" trong header
- [ ] `pyproject.toml` / `uv.lock` -- thêm `sacrebleu` vào group `eval`
- [ ] `tests/test_languages.py`, `tests/test_translate.py` (bổ sung), `tests/test_mt_bench.py` -- bảng ngôn ngữ + normalizer, supports/pivot, ghép cặp (thiếu id, id mâu thuẫn), tập con tất định, n_failed của Google, smoke CSV + file mẫu, thiếu dữ liệu → exit 2
- [ ] `README.md` -- mục "Đo chất lượng dịch": lệnh chạy, signature, bảng kết quả, hạn chế

**Acceptance Criteria:**
- Given toàn bộ test, when `uv run --group eval pytest -q`, then tất cả pass (hiện có 195 test ở worktree bước 2; ở nhánh này là số test của main cộng test mới).
- Given `--limit 50`, when chạy hai lần, then CSV giống hệt nhau (trừ cột độ trễ).
- Given chạy đủ, when xem CSV, then có các cột `corpus, src_lang, tgt_lang, backend, condition, asr_model, n_sents, bleu, chrf, mean_latency_ms, p95_latency_ms, n_failed` và một file mẫu khoảng 20 câu (nguồn, tham chiếu, chữ ASR của từng model, bản dịch).

## Design Notes

**Ghép cặp FLEURS:** cột 0 của `test.tsv` là id câu FLoRes, giống nhau giữa các ngôn ngữ. Cột 2 là câu raw (giữ hoa, dấu câu), cột 3 là câu đã chuẩn hóa. Câu nguồn và câu tham chiếu đều lấy cột raw. Phía vi đã kiểm tra: 857 dòng, 347 id, mỗi id chỉ có một câu raw. Phía en (`en_us/test.tsv`) **chưa có**: file 368 KB, chỉ là chữ, không cần audio. License CC-BY-4.0. Lệnh tải được ghi trong README; người dùng duyệt thì tôi mới tải.

**Đề xuất model (sơ bộ, sẽ chốt kèm số đo cuối 4A):**

| Model | Cặp | License | Checkpoint gốc → CT2 int8 (ước tính) | Ghi chú |
|---|---|---|---|---|
| NLLB-200-distilled-600M | 200 ngôn ngữ, mọi chiều | CC-BY-NC-4.0 (được dùng cho đồ án, cấm thương mại) | 2,46 GB fp32 → ~0,6 GB, RAM ~1 GB | CT2 hỗ trợ chính thức; ứng viên chính |
| M2M100-418M | 100 ngôn ngữ | MIT | 1,94 GB → ~0,5 GB | nhẹ, license thoáng; thường kém NLLB |
| MADLAD-400-3B-MT | 400+ ngôn ngữ | Apache-2.0 | ~3 GB int8, RAM ~3,5 GB | có bản CT2 sẵn; có thể quá chậm so với mục tiêu ≤ 500 ms và sát RAM 8 GB, chỉ để làm mốc chất lượng |
| (tham khảo) VinAI Translate vi2en-v2 | chỉ vi↔en | AGPL-3.0 | mBART ~420M | chuyên tiếng Việt, không đa ngôn ngữ |

Tất cả đều convert được bằng `ct2-transformers-converter` qua một script chạy một lần (torch chỉ có trong `uv run --with`), lúc chạy app chỉ cần ctranslate2 và sentencepiece như Argos. Tốc độ và RAM trong bảng là ước tính, chưa đo.

Nguồn:
- [NLLB card](https://huggingface.co/facebook/nllb-200-distilled-600M)
- [CT2 Transformers guide](https://opennmt.net/CTranslate2/guides/transformers)
- [MADLAD-3B card](https://huggingface.co/google/madlad400-3b-mt)
- [M2M100-418M card](https://huggingface.co/facebook/m2m100_418M)
- [VinAI Translate](https://github.com/VinAIResearch/VinAI_Translate)
- [sacrebleu](https://github.com/mjpost/sacrebleu)

**Hướng mở rộng:** COMET (cần torch, chạy riêng ngoài app); chrF++ (`nw=2`).

## Verification

**Commands:**
- `uv run --group eval pytest -q` -- expected: tất cả pass
- `uv run --group eval python eval/mt_bench.py --smoke --out /tmp/x.csv` -- expected: exit 0, CSV đủ cột
- `uv run --group eval python eval/mt_bench.py --src vi --tgt en --fleurs-src-dir ~/data/fleurs_vi --fleurs-tgt-tsv ~/data/fleurs_en/test.tsv --custom-tsv eval/data/mt_custom_vi_en.tsv --limit 50 --out results/mt_fleurs_vi_en_limit50.csv` -- expected: chạy xong, rồi báo người dùng trước khi chạy đủ 347 câu

## Implementation Notes

## Plan Change Log

## Review Triage Log

- 2026-10-31: FLEURS `en_us/test.tsv` downloaded (367 864 B, 647 rows, 350 ids). All 347 vi ids are paired, and every id has a single raw text on both sides. A spot-check of numbers and names (ids 1684, 1693, 1953) looked correct.
- `languages.py`: a frozen `Language` row whose `mt` mapping goes backend → code, so a new backend adds a key, not a field. Normalisers sit in a dispatch table: `vi` is identical to `denoise_bench.normalize_vi` (tested), `en` keeps word-internal apostrophes.
- `translate.py` additions only: `GoogleTranslator.supports` (static, from the table), `OfflineTranslator.supports` (from the package's metadata.json), `PivotTranslator`, `plan_route`. The app does not call them yet.
- `mt_bench.py`:
  - Added `src_wer` (WER of the translator input against the normalised source) to read B against ASR quality.
  - Argos uses only an installed package (`--argos-package`, default: the app cache), so the bench never downloads.
  - FLEURS wav (float, int16 scale) is converted to int16 before `WhisperASR.transcribe`, because transcribe only rescales int16 input.
  - sacrebleu signatures are available only after one evaluation, so they are printed at the end of a run.
- The 50-sentence trial ran at the same time as the step-1 ASR bench (task bch4emju5, phowhisper-medium rows). Its latency columns are pessimistic, and the step-1 latency of the rows measured during the trial is suspect.
- Added `bleu_lc`/`chrf_lc` (sacrebleu `lowercase=True`), approved by the human: the app's output is speech.
- Custom set: 9 draft references revised by Claude (c04, c08, c10, c11, c14, c16, c19, c20, c24). For c19 the source was made unambiguous ("với bạn thân"). Every row is still `reviewed=no` and waits for the human.
- Full run, FLEURS 347 (Argos; Google got HTTP 429 on every call, including manual spaced calls):
  - text: 27.83 BLEU / 55.37 chrF
  - text_norm: 18.14 / 49.70
  - asr small (WER 21.0%): 12.69 / 41.67
  - asr phowhisper-small (WER 13.0%): 15.75 / 45.44
  - A rerun from the ASR cache gave identical FLEURS scores. The latency columns are measured under CPU contention from the step-1 bench.
- Anomaly: unpunctuated input merges sentences, and Argos can drop a whole clause (FLEURS id 1662). Names are mistranslated as words ("Hải Phòng" → "the Customs"), worse in lowercase.
- 4A complete: waiting for the human to choose the model for 4B.
