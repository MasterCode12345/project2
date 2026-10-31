---
title: 'M5 — multilingual offline MT backends (NLLB-200-600M, M2M100-418M) via CTranslate2 (giai đoạn 4B)'
type: 'feature'
ticket: ''
created: '2026-11-01'
status: 'in-progress'
baseline_revision: '6597742'
route: 'full'
route_source: 'auto'
risk: 'medium'
review: ''
review_source: ''
lenses_ran: []
review_loop_iteration: 0
context: ['{project-root}/_bmad-output/initiative-audio-translator/plan-mt-eval.md']
---

<frozen-after-approval reason="human-owned intent — do not modify unless human renegotiates">

## Intent

**Problem:** 4A (`results/mt_vi_en.csv`, FLEURS 347 câu) cho thấy Argos chỉ đạt 27,8 BLEU trên câu chuẩn. Trên đầu vào thực tế của app (chữ thường, không dấu câu) chỉ còn 18,1, và với chữ PhoWhisper là 15,8. Argos cũng làm hỏng tên riêng. Mỗi cặp ngôn ngữ của Argos lại cần một package riêng.

**Approach:** Thêm backend `CT2MultilingualTranslator`, một class chạy được cả NLLB-200-distilled-600M và M2M100-418M. Model được convert sang CTranslate2 một lần bằng `scripts/convert_mt_model.py` (torch chỉ có trong `uv run --with`). Lúc chạy chỉ cần ctranslate2 và sentencepiece, giống Argos. Backend được chọn qua `--translator` / `Settings.translator`, có Google dự phòng như Argos. Đo lại bằng `mt_bench` trên đúng dữ liệu 4A. **Không đổi mặc định** cho tới khi người dùng đồng ý.

**Decisions (human, 2026-11-01):**
1. Model NLLB-600M (chính) + M2M100-418M (so sánh).
2. Làm các đề xuất hợp lý, có logic.
3. Push được phép.
4. Tải checkpoint trên 1 GB (NLLB 2,48 GB, M2M100 1,94 GB) vẫn phải hỏi riêng.

## Boundaries & Constraints

**Always:**
- **Ngôn ngữ:** mã ngôn ngữ của từng model lấy từ `languages.py` (`mt["nllb"]`, `mt["m2m100"]`). Không ghi cứng vi/en.
- **`supports(src, tgt)`:** đúng khi cả hai ngôn ngữ có mã cho họ model đó.
- **Model ghim revision:** NLLB `f8d333a0…`, M2M100 `55c2e61b…`. Chỉ tải các file cần (`allow_patterns`), không tải `rust_model.ot`. Output int8 ghi vào `~/.cache/audio-translator/models/<name>-ct2/` kèm `SOURCE.json`; ghi vào thư mục tạm rồi mới rename.
- **Model chưa convert:** app cảnh báo kèm lệnh convert rồi dùng Google, giống khi Argos lỗi.
- **Test offline:** mọi test dùng ct2/spm giả.

**Never:**
- Không đổi `Settings.translator` mặc định (`offline` = Argos).
- Không thêm torch hay transformers vào dependency.
- Không convert khi bench ASR đang chạy: convert cần khoảng 5 GB RAM, máy chỉ có 8 GB.
- Không sửa GUI.

## I/O & Edge-Case Matrix

| Scenario | Input / State | Expected Output / Behavior | Error Handling |
|----------|--------------|---------------------------|----------------|
| NLLB dịch | "Xin chào. Bạn khỏe không?" | mỗi câu một batch item: `[vie_Latn] + pieces + [</s>]`, target_prefix `[eng_Latn]`; bỏ token ngôn ngữ ở đầu kết quả | — |
| M2M100 dịch | như trên | `[__vi__] + pieces + [</s>]`, prefix `[__en__]` | — |
| Chưa convert | thư mục model không có `model.bin` | `load()` báo RuntimeError kèm lệnh convert; app chuyển sang Google | cảnh báo WARNING |
| Cặp không hỗ trợ | ngôn ngữ thiếu mã trong bảng | `supports` = False; constructor báo ValueError | — |
| Câu rỗng | "" hoặc chỉ có khoảng trắng | trả về "" | — |

</frozen-after-approval>

## Code Map

- `src/audio_translator/translate.py` -- thêm `MT_MODELS`, `mt_model_dir`, `mt_convert_command`, `CT2MultilingualTranslator`
- `src/audio_translator/orchestrator.py` -- `_prepare_translator`: nhánh cho model đa ngôn ngữ, có Google dự phòng; model đã load được giữ lại giữa các session
- `src/audio_translator/cli.py` -- thêm lựa chọn cho `--translator`
- `scripts/convert_mt_model.py` -- convert một lần (mẫu: `convert_phowhisper.py`)
- `eval/mt_bench.py` -- thêm backend `nllb-600m`, `m2m100-418m`

## Tasks & Acceptance

**Execution:**
- [ ] `translate.py` + `tests/test_translate.py` -- backend, định dạng token cho từng họ model, chưa convert, câu rỗng, supports
- [ ] `scripts/convert_mt_model.py` + `tests/test_convert_mt_model.py` -- revision ghim, allow_patterns, thư mục tạm, SOURCE.json, `--force`
- [ ] `orchestrator.py`, `cli.py` + tests -- chọn backend, dự phòng khi chưa convert, giữ mặc định
- [ ] `eval/mt_bench.py` + tests -- backend mới; model chưa convert thì bỏ qua kèm log
- [ ] Sau khi người dùng cho tải: convert, chạy `mt_bench` đủ 347 câu, cập nhật README

**Acceptance Criteria:**
- Given chưa đổi gì trong cấu hình, when chạy app, then dịch vẫn dùng Argos như trước.
- Given `--translator nllb-600m` và model đã convert, when nói tiếng Việt, then dịch bằng NLLB, lỗi từng câu thì dùng Google.
- Given `uv run --group eval pytest -q`, then tất cả pass.

## Implementation Notes

- 2026-11-01: The HF API was checked for both repos.
  - NLLB `pytorch_model.bin` is 2 460 457 927 B and M2M100 is 1 935 796 948 B. M2M100 also ships `rust_model.ot` (1.9 GB), which `ALLOW_PATTERNS` excludes.
  - Licenses read from the cards: NLLB cc-by-nc-4.0, M2M100 mit.
- One class serves both families. The family selects the `languages.py` column and the language-token format. Output starts with the forced target token, which is stripped.
- Orchestrator: the multilingual model is cached under key (name, src, tgt) and reused across sessions. A load failure (not converted) emits a WARNING naming the convert command, and the session uses Google. The next session retries.
- 206 tests pass. Conversion and measurement are waiting for the human's OK on the >1 GB download.

## Plan Change Log

## Review Triage Log

## Verification

**Commands:**
- `uv run --group eval pytest -q` -- expected: tất cả pass
- `uv run --with transformers --with torch --with sentencepiece python scripts/convert_mt_model.py nllb-600m m2m100-418m` -- expected: hai thư mục `-ct2` (chỉ chạy sau khi người dùng cho tải)
- `uv run --group eval python eval/mt_bench.py ... --backends argos,nllb-600m,m2m100-418m --out results/mt_vi_en_models.csv` -- expected: có dòng của cả ba backend
