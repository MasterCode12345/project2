---
title: 'Vietnamese ASR accuracy — PhoWhisper (CTranslate2), large-v3-turbo, initial_prompt, ASR bench'
type: 'feature'
ticket: ''
created: '2026-10-31'
status: 'built'
baseline_revision: 'a03a1d218190bfc1166f10e817d66efcb18915c4'
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

**Problem:** Whisper `small` nghe sai tiếng Việt nhiều (WER ≈ 22% trên VIVOS sạch, bench `a03a1d2`). Đây là lỗi ưu tiên số 1 của pipeline.

**Approach:** Thêm PhoWhisper small/medium (VinAI, fine-tune cho tiếng Việt; tự convert sang CTranslate2 để chạy bằng faster-whisper) vào catalog model. Thêm tùy chọn câu mồi tiếng Việt (`initial_prompt`). Viết bench ASR trên đúng giao thức VIVOS hiện có, so Whisper small, PhoWhisper small/medium và large-v3-turbo (đã có trong catalog), có và không có câu mồi. Đo WER/CER và độ trễ ASR trên âm thanh sạch và có nhiễu. Mục tiêu: WER thấp hơn rõ rệt so với 22%.

**Decisions (human, 2026-10-31):** (1) Câu mồi: đo một câu trung tính `"Xin chào, sau đây là đoạn hội thoại bằng tiếng Việt."`. (2) Thêm FLEURS tiếng Việt (`google/fleurs` `vi_vn` test, CC-BY-4.0) làm tập out-of-domain, chạy cùng bench với nhiễu DEMAND. (3) Giữ plan một khối dù > 1600 token.

## Boundaries & Constraints

**Always:** Model chạy qua `WhisperASR`/faster-whisper như hiện tại (CPU int8, beam 5, `language="vi"`, `vad_filter=False`). PhoWhisper được convert từ repo gốc `vinai/PhoWhisper-{small,medium}` ở revision ghim (small `a86b604c…`, medium `55a7e3eb…`, BSD-3). torch/transformers chỉ dùng trong môi trường convert tạm (`uv run --with …`), không thêm vào dependency của project. Không có câu mồi thì hành vi giữ nguyên như cũ. Bench dùng lại giao thức của `eval/denoise_bench.py` (clip, chấm theo khoảng có câu nói, 100 câu luân phiên theo người nói), không có stage lọc nhiễu. Mọi test chạy offline, không cần model thật. Báo cáo ghi rõ PhoWhisper đã được train trên VIVOS train (paper Bảng 1), nên VIVOS test là in-domain với nó.

**Never:** Không đổi model mặc định hay bật câu mồi mặc định trước khi có số liệu bench. Không dùng bản CT2 do bên thứ ba convert. Không đưa weights vào repo. Không thêm GUI control. Không tải dataset trong test.

## I/O & Edge-Case Matrix

| Scenario | Input / State | Expected Output / Behavior | Error Handling |
|----------|--------------|---------------------------|----------------|
| Chọn PhoWhisper đã convert | `--model phowhisper-small`, thư mục CT2 có đủ file | Load từ thư mục local, `--list-models` hiện `downloaded` | none |
| Chọn PhoWhisper chưa convert | Thư mục không có | `--list-models` hiện `not-downloaded`; khi load: lỗi nêu đường dẫn và lệnh convert | RuntimeError → error event sẵn có |
| Convert | `scripts/convert_phowhisper.py small` | Ghi `model.bin` + tokenizer/preprocessor vào thư mục cache, ghi lại revision nguồn; chạy lại thì bỏ qua (`--force` để làm lại) | Lỗi mạng/convert → exit ≠ 0, không để lại thư mục dở dang |
| Câu mồi | `asr_initial_prompt="…"` | Truyền vào `transcribe(initial_prompt=…)` | none |
| Không câu mồi | `None` (mặc định) | Gọi y như cũ | none |
| Bench smoke | `eval/asr_bench.py --smoke` | CSV đủ cột cho mọi model × prompt × SNR, dùng ASR giả | none |
| FLEURS | `--fleurs-dir` có `test.tsv` + `test/*.wav` | 100 câu khác nhau (luân phiên theo mã câu), ref = cột transcription đã chuẩn hóa | Thiếu file → báo đường dẫn, exit ≠ 0 |
| Bench thiếu model | Model PhoWhisper chưa convert | Báo rõ và bỏ qua model đó (hoặc exit ≠ 0 nếu không còn model nào) | Error message |

</frozen-after-approval>

## Code Map

- `src/audio_translator/asr.py` -- `WhisperASR(model_size, …)` → `WhisperModel(self.model_size, …)` (nhận tên hoặc đường dẫn); `transcribe` gọi `self._model.transcribe(…, condition_on_previous_text=False)`. Thêm tham số `initial_prompt`; ánh xạ tên catalog → đường dẫn local ở đây hoặc trong model_manager.
- `src/audio_translator/model_manager.py:19` -- `CATALOG: dict[str, int]` (tên → MB), `hf_is_downloaded(name)` dùng `download_model(name, local_files_only=True)`. Thêm `phowhisper-small` (~490 MB fp16), `phowhisper-medium` (~1530 MB); trạng thái tải = thư mục local có `model.bin`.
- `src/audio_translator/orchestrator.py:191` -- `_make_recognizer(name)` tạo `WhisperASR(name, language, device, compute_type, beam_size)` → thêm `initial_prompt=s.asr_initial_prompt`.
- `src/audio_translator/config.py:44` -- `model_size`, `beam_size`, `compute_type`; thêm `asr_initial_prompt: str | None = None`, `phowhisper_dir = "~/.cache/audio-translator/models"`.
- `src/audio_translator/cli.py:24` -- `--model choices=list(CATALOG)` tự có model mới; thêm `--asr-prompt TEXT`; `--list-models` (line ~86) in thêm gợi ý lệnh convert cho model local chưa có.
- `eval/denoise_bench.py` -- tái dùng `load_vivos` (luân phiên theo người nói), `load_noises`, `build_clip`, `run_pipeline(stage=None)`, `ArmRun.text_in`, `normalize_vi`, `error_rates`, `ToyASR`, `smoke_data`. Không sửa giao thức.
- FLEURS `vi_vn` test: `data/vi_vn/test.tsv` (857 dòng: sentence id, file, raw, transcription chuẩn hóa, …, num_samples, gender; không có speaker id) + `audio/test.tar.gz` (544 MB). Câu dài hơn VIVOS (thường > 10 s) → có thể bị force-cut ở 15 s; `text_in` ghép mọi đoạn chồng lên câu.
- Dữ liệu: `~/data/vivos`, `~/data/demand_ch01` (17 file), `~/data/musan/noise`. Máy: M1, 8 GB RAM → bench nạp lần lượt từng model, giải phóng trước khi nạp model kế tiếp.
- `ctranslate2` 4.8.2 có sẵn `ct2-transformers-converter` (`--copy_files tokenizer.json preprocessor_config.json --quantization float16`).

## Tasks & Acceptance

**Execution:**
- [ ] `scripts/convert_phowhisper.py` -- convert `vinai/PhoWhisper-{size}` @ revision ghim → `{phowhisper_dir}/phowhisper-{size}-ct2/` (ghi atomic, file `SOURCE` lưu repo + revision); hướng dẫn chạy bằng `uv run --with transformers --with torch` -- tự convert
- [ ] `src/audio_translator/asr.py`, `model_manager.py`, `config.py`, `cli.py`, `orchestrator.py` -- model local trong catalog, trạng thái tải, lỗi rõ khi thiếu, `initial_prompt` -- tích hợp
- [ ] `eval/asr_bench.py` -- corpus {VIVOS, FLEURS} × model × {không mồi, có mồi} × SNR {clean, 20, 10, 5, 0} × noise dir; loader FLEURS (`--fleurs-dir`); cột: `model, prompt, snr_db, n_utts, wer, cer, asr_ms_mean, asr_ms_p95, asr_rtf, load_s`; `--smoke` offline -- đo đạc
- [ ] `tests/test_asr.py`, `tests/test_model_manager.py`, `tests/test_cli.py`, `tests/test_asr_bench.py` -- mọi dòng matrix -- regression
- [ ] `README.md` -- PhoWhisper (nguồn, license, lệnh convert, lưu ý in-domain), `--asr-prompt`, bench ASR, bảng kết quả -- docs
- [ ] Tải FLEURS vi_vn test về `~/data/fleurs_vi` (ngoài repo); chạy convert small/medium + bench thật: VIVOS × {DEMAND, MUSAN}, FLEURS × DEMAND (clean chạy một lần mỗi corpus), ghi `results/asr_*.csv` -- kết quả

**Acceptance Criteria:**
- Given PhoWhisper small đã convert, when chạy app `--model phowhisper-small`, then một session dịch được câu tiếng Việt như với Whisper.
- Given `uv run pytest` offline, when chạy, then tất cả pass.
- Given bench thật xong, when đọc CSV, then có WER/CER và độ trễ ASR cho cả 4 model × có/không mồi × 5 mức SNR cho VIVOS (DEMAND, MUSAN) và FLEURS (DEMAND).

## Implementation Notes

- Implemented directly (no subagent). Local-model mapping lives in `asr.py` (`LOCAL_MODELS`, `LOCAL_MODELS_DIR`, `local_model_dir`, `resolve_model`, `convert_command`) because `model_manager` imports `asr` (avoids a cycle). Deviation: no `Settings.phowhisper_dir` field — the models root is the module constant `asr.LOCAL_MODELS_DIR` (`~/.cache/audio-translator/models`), the convert script writes only there (review removed its `--out-dir`, which the app never read). `hf_is_downloaded` checks `<dir>/model.bin` for local models.
- `WhisperASR(initial_prompt=…)`; empty string normalised to None, so "no prompt" calls faster-whisper exactly as before (`initial_prompt=None` is its default).
- `scripts/convert_phowhisper.py`: `huggingface_hub.snapshot_download(repo, revision)` → `TransformersConverter(src, copy_files=[tokenizer.json, preprocessor_config.json]).convert(tmp, quantization="float16")` → `SOURCE.json` → rename; failure or missing files remove the temp dir. Converted on this Mac: small 464 MB, medium 1.4 GB.
- `eval/asr_bench.py` imports `denoise_bench` helpers (clip protocol, `run_pipeline(stage=None)`, `ArmRun.text_in`, `overlaps`, metrics). `TimedASR` times each `transcribe`; calls are zipped with `ArmRun.segments` (same order) so latency covers scored segments only. Rows appended + flushed one by one. One model resident at a time (`del` + `gc.collect()`). FLEURS ref = the normalised `transcription` column (col 3, per the frozen matrix; first draft used the raw column, fixed in review), then `normalize_vi`; one recording per sentence id, recording index = sentence index mod #recordings. MUSAN run uses `--snrs 20,10,5,0` (clean rows are noise-independent and come from the DEMAND run).
- Existing catalog tests (`test_model_manager`, `test_frontends` CLI list + GUI dropdown) were hard-coded to the 6 hub models; updated to the 8-entry catalog. The GUI dropdown now lists the PhoWhisper entries (no new control); picking an unconverted one fails with the convert-command error through the existing model-failure path.
- Speed spot-check (M1 CPU int8, VIVOS utterances ~1.4–3.4 s): small ≈1.0 s, phowhisper-small ≈1.0 s, phowhisper-medium ≈2.6 s, large-v3-turbo ≈3.6 s per call; turbo first load 200 s (download).
- Acceptance "app session with `--model phowhisper-small`" needs a mic + BlackHole: not run here (headless); covered by the orchestrator/CLI wiring tests and a direct `WhisperASR("phowhisper-small")` transcription check.

## Plan Change Log

## Review Triage Log

Pass 1 (lens: quick) — high 0 / medium 1 / low 3 / false 2 / maybe-false 1 (+1 rejected: plan edit). Plus one defect found during re-verification.

| # | Finding | Verdict | Route | Evidence / action |
|---|---------|---------|-------|-------------|
| 1 | AC3 unmet: bench "stopped early", only 2 rows | false | — | Bench process still running (pgrep shows the VIVOS run; rows keep being appended). Reviewed mid-run. |
| 2 | README results table missing; roadmap `[x]` before measuring | false | — | Depends on #1; table is added when the CSVs are complete. |
| 3 | AC1 (live app session with `--model phowhisper-small`) not verified | maybe-false | defer to human | Needs mic + BlackHole; settled by one manual CLI session. Wiring tests + direct `WhisperASR("phowhisper-small")` transcription pass. |
| 4 | FLEURS ref used raw column, frozen matrix says normalised `transcription` column | medium | patch | Confirmed `cols[2]` in `load_fleurs`. Fixed to `cols[3]` (needs ≥ 4 columns); test now expects the normalised text. FLEURS run had not started yet. |
| 5 | `convert_phowhisper.py --out-dir` writes where the app never looks | low | patch | Confirmed: `resolve_model`/`hf_is_downloaded` read only `LOCAL_MODELS_DIR`. Removed `--out-dir`; test asserts main writes where `resolve_model` finds it and rejects `--out-dir`. |
| 6 | Empty `--prompt` labels the "on" pass `off` | low | patch | Confirmed (`"on" if prompt else "off"`). `main` now rejects an empty prompt when mode `on` is requested; test added. |
| 7 | GUI shows unconverted PhoWhisper as `chưa tải (~485 MB)` | low | patch | Confirmed in `model_item_text`. Local models now read `chưa convert (xem --list-models)`; GUI test asserts it. |
| 8 | Plan task list names `tests/test_cli.py`, checkboxes unticked, change log empty | — | rejected | Fix is an edit of this build's plan. |
| 9 | (self, re-verification) FLEURS WAVs are IEEE float (format 3): stdlib `wave` raises, the FLEURS run would crash | high | patch | Reproduced on `~/data/fleurs_vi/test/*.wav`. `read_wav` falls back to a small RIFF parser for float WAVs; test with mono/stereo float files. Real FLEURS load: 100 utts, mean 12.8 s; VAD coverage of quiet recordings (peak ≈ 400–500) same as loud ones, no utterance missed. |

## Design Notes

Độ trễ ASR được đo là thời gian `transcribe` trên mỗi đoạn VAD đã chồng lên câu nói (mean, P95, và RTF = tổng thời gian / tổng độ dài audio). Nó là thành phần chính của độ trễ end-to-end (README: ≈ 2,57 s với `small`). `load_s` là thời gian nạp model một lần, ghi để tham khảo.

## Verification

**Commands:**
- `uv run pytest -q` -- expected: all pass
- `uv run --group eval python eval/asr_bench.py --smoke --out /tmp/asr_smoke.csv` -- expected: CSV đủ dòng
- `uv run --with transformers --with torch python scripts/convert_phowhisper.py small` -- expected: thư mục CT2 có `model.bin`, `tokenizer.json`, `preprocessor_config.json`, `SOURCE`
- `uv run audio-translator --list-models` -- expected: `phowhisper-small  downloaded`
