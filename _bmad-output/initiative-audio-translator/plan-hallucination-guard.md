---
title: 'Hallucination guard — Silero VAD option, Whisper confidence filter + Vietnamese hallucination blacklist, re-measure'
type: 'feature'
ticket: ''
created: '2026-10-31'
status: 'done'
baseline_revision: '0d42a1b680d4f3d9a302ae7659c2bdf61ef52bbc'
route: 'full'
route_source: 'auto'
risk: 'medium'
review: ''
review_source: ''
lenses_ran: []
review_loop_iteration: 0
context:
  - '{project-root}/_bmad-output/initiative-audio-translator/plan-asr-vietnamese-accuracy.md'
---

<frozen-after-approval reason="human-owned intent — do not modify unless human renegotiates">

## Intent

**Problem:** Có nhiễu thì webrtcvad bật nhầm gần như ở mọi clip chỉ có nhiễu (`vad_false_triggers` 96–141/100, bench `a03a1d2`), và Whisper "bịa" câu cho các đoạn đó, ví dụ "Hãy subscribe cho kênh Ghiền Mì Gõ…". Câu ma bị dịch rồi phát ra micro ảo.

**Approach:** Thêm Silero VAD (dạng streaming) làm lựa chọn thay cho webrtcvad. Lọc segment của Whisper theo độ tin cậy (`no_speech_prob`, `avg_logprob`, kèm `compression_ratio`) và theo một danh sách đen các câu ảo giác tiếng Việt hay gặp. Đo lại `vad_false_triggers` và `vad_extra_segs`, thêm một cột đếm câu ma còn lọt sau bộ lọc, và kiểm tra WER để chắc bộ lọc không làm mất câu thật. Mục tiêu: giảm mạnh so với mức 96–141.

**Decisions (human, 2026-10-31):** (1) Thêm cột `ghost_texts`: chạy ASR cả trên clip chỉ có nhiễu, đếm đoạn còn ra chữ sau lọc. (2) Plan này chỉ đo; đổi mặc định quyết sau khi xem số. (3) Giữ plan một khối dù > 1600 token.

## Boundaries & Constraints

**Always:** Chưa đổi mặc định khi chưa có số đo: `vad_backend="webrtc"`, bộ lọc tắt. Khi tắt cả hai thì hành vi y như cũ. Silero dùng file `silero_vad_v6.onnx` có sẵn trong faster-whisper, chạy qua onnxruntime đã có (không thêm dependency). Nó xử lý frame 480 mẫu, giữ trạng thái `h`/`c`/ngữ cảnh 64 mẫu, mỗi session một instance. VAD chạy trong segment worker như hiện tại. Danh sách đen được xây từ nhiễu **không trùng** với nhiễu dùng để đo (ghi rõ nguồn của từng câu), so khớp trên văn bản đã chuẩn hóa. Test chạy offline, không cần model thật.

**Never:** Không thêm torch hay package `silero-vad`. Không lọc trên toàn bộ câu ghép, chỉ lọc từng segment Whisper. Không xây danh sách đen từ chính các clip dùng để đo. Không đổi giao thức chấm của bench.

## I/O & Edge-Case Matrix

| Scenario | Input / State | Expected Output / Behavior | Error Handling |
|----------|--------------|---------------------------|----------------|
| Silero trên tiếng nói | Câu VIVOS sạch | Segmenter cắt ra đúng một đoạn phủ câu | none |
| Silero trên im lặng/nhiễu dừng | Im lặng tuyệt đối, white noise nhỏ | Không bật | none |
| Căn cửa sổ 512/480 | Frame 480 liên tiếp | Mỗi frame trả một quyết định; xác suất được tính đúng một lần cho mỗi cửa sổ 512; giữ ngữ cảnh/trạng thái giữa các frame | none |
| Session mới | Stop rồi Start | VAD Silero mới, trạng thái sạch | none |
| Lọc độ tin cậy | Segment có `no_speech_prob` > ngưỡng, hoặc `avg_logprob` < ngưỡng, hoặc `compression_ratio` > ngưỡng | Bỏ segment đó, giữ các segment khác của cùng đoạn | none |
| Danh sách đen | Segment chuẩn hóa chứa một câu trong danh sách | Bỏ segment đó | none |
| Câu thật gần giống | Câu thật chỉ trùng một hai từ với danh sách | Không bị bỏ (so khớp theo cụm đủ dài) | none |
| Lọc tắt | `asr_filter=False` | Văn bản y như cũ | none |
| Mọi segment bị lọc | Đoạn chỉ có câu ma | `transcribe` trả `""`, orchestrator bỏ qua đoạn rỗng như hiện tại | none |
| Bench | `--vad silero --filter on` | Thêm cột `vad_false_triggers`, `vad_extra_segs`, `ghost_texts`; `--smoke` chạy offline | none |

</frozen-after-approval>

## Code Map

- `src/audio_translator/segmenter.py:16` -- `webrtc_vad(aggressiveness) -> VadFn` (`VadFn = (bytes, rate) -> bool`). Thêm `silero_vad(threshold=0.5, neg_threshold=None) -> VadFn` (class streaming; import `faster_whisper.vad.get_vad_model` lazy; session input `input [n,576]` = 64 ngữ cảnh + 512 mẫu float32 [-1,1], `h`/`c [1,1,128]`; output `speech_probs`, `hn`, `cn`; hysteresis: bật khi p ≥ thr, tắt khi p < thr − 0.15 như faster-whisper).
- `src/audio_translator/orchestrator.py:59,80,328` -- `vad_factory: Callable[[int], VadFn]` mặc định `webrtc_vad`, gọi `self._vad_factory(s.vad_aggressiveness)` khi tạo Segmenter mỗi session. Chọn backend theo `s.vad_backend` mà vẫn giữ seam `vad_factory` cho test.
- `src/audio_translator/asr.py` -- `WhisperASR.transcribe` ghép `s.text` của mọi segment. Thêm lọc theo segment (`no_speech_prob`, `avg_logprob`, `compression_ratio`, danh sách đen); `Recognizer` protocol giữ nguyên.
- `src/audio_translator/config.py` -- thêm `vad_backend="webrtc"`, `silero_threshold=0.5`, `asr_filter=False`, `asr_max_no_speech_prob=0.6`, `asr_min_avg_logprob=-1.0`, `asr_max_compression_ratio=2.4` (ngưỡng mặc định của Whisper gốc; faster-whisper chỉ bỏ segment khi *cả* no_speech > 0.6 *và* logprob < −1, ở đây là *một trong các* điều kiện).
- `src/audio_translator/cli.py` -- `--vad {webrtc,silero}`, `--asr-filter`.
- `src/audio_translator/hallucinations_vi.txt` (mới) -- mỗi dòng một cụm đã chuẩn hóa + nguồn; nạp một lần.
- `eval/denoise_bench.py` -- `normalize_vi` (dùng lại cho so khớp), `run_pipeline`, `ArmRun.count_in`, `Clip.noise_window_s`. `eval/asr_bench.py` -- `run`/`bench_model`/`TimedASR`; thêm `--vad`, `--filter`, cột `vad_false_triggers` (đoạn trong vùng nhiễu được chấm của clip chỉ có nhiễu), `vad_extra_segs`, `ghost_texts` (đoạn của clip chỉ có nhiễu còn ra chữ sau lọc).
- Prototype (scratchpad, 100 clip): false triggers webrtc → Silero(0.5): DEMAND 96/105/101 → 24/24/27, MUSAN 141/117/115 → 0/3/10 (20/10/0 dB), không mất câu nào. DEMAND còn sót có thể do các cảnh có tiếng người (PCAFETER, OMEETING…) — kiểm tra theo từng file nhiễu.
- Dữ liệu nhiễu để xây danh sách đen (không trùng với đo): file MUSAN `noise` không thuộc 20 file được chọn với seed 0, và `~/data/musan/music`.

## Tasks & Acceptance

**Execution:**
- [x] `src/audio_translator/segmenter.py` -- `SileroVad` streaming + `silero_vad()` -- VAD mới
- [x] `src/audio_translator/asr.py`, `hallucinations_vi.txt` -- `SegmentFilter` (ngưỡng + danh sách đen), dùng trong `transcribe` khi bật -- lọc câu ma
- [x] `src/audio_translator/config.py`, `cli.py`, `orchestrator.py` -- settings, flags, chọn VAD theo session -- tích hợp
- [x] `scripts/collect_hallucinations.py` -- chạy Whisper (small, phowhisper-small) trên đoạn VAD cắt ra từ nhiễu không trùng, in tần suất câu → chọn tay vào `hallucinations_vi.txt` kèm nguồn -- danh sách đen có căn cứ
- [x] `eval/asr_bench.py` -- `--vad`, `--filter`, 3 cột mới -- đo đạc
- [x] `tests/test_segmenter.py`, `tests/test_asr.py`, `tests/test_asr_bench.py`, `tests/test_frontends.py` -- mọi dòng matrix -- regression
- [x] `README.md` -- `--vad`, `--asr-filter`, cách xây danh sách đen, bảng kết quả -- docs
- [x] Sau khi bench ASR (Bước 1) chạy xong: đo `small` và `phowhisper-small` × VAD {webrtc, silero} × lọc {tắt, bật} × SNR {clean, 20, 10, 5, 0} trên VIVOS × {DEMAND, MUSAN}, ghi `results/guard_*.csv` -- kết quả

**Acceptance Criteria:**
- Given `--vad silero --asr-filter`, when chạy app, then câu nói vẫn được dịch và không có lỗi.
- Given `uv run pytest` offline, when chạy, then tất cả pass.
- Given bench xong, when so với webrtc + không lọc, then `vad_false_triggers` và `ghost_texts` giảm mạnh, và WER trên câu thật không tăng quá 1 điểm.

## Implementation Notes

- Implemented directly in the worktree `/Users/trantrungnghia/GR2-hallucination` (branch `feat/hallucination-guard`), so the step-1 ASR bench running from `/Users/trantrungnghia/GR2` keeps loading unchanged code.
- `segmenter.SileroVad`: buffers frames, scores each 512-sample window once with the previous 64 samples as context and the carried `h`/`c`; the frame returns the hysteresis state after the last complete window (≤ 32 ms lag). `session=` seam for tests. `vad_from_settings` + `VAD_BACKENDS`; the orchestrator builds the VAD inside `_start_session`'s try block (unknown backend → error event), keeping the injected `vad_factory` seam for tests.
- `asr.SegmentFilter.reason(seg)` (strict `>`/`<` comparisons) + `load_blacklist` (≥ 4 words, whole-word match on `normalize_text`); `WhisperASR.transcribe_segments` returns raw segments (used by the collect script); `transcribe` keeps segments the filter accepts and records `dropped`. Filter off = previous output.
- `eval/asr_bench.py`: `--vads`, `--filters` loops inside one model load; noise-only clips now go through the ASR too (`vad_false_triggers`, `ghost_texts`); `extra_segs` renamed `vad_extra_segs` (step-1 CSVs, produced by the old code, keep `extra_segs`).
- `scripts/collect_hallucinations.py`: sources = 930 files (910 MUSAN noise outside the bench's seed-0 sample of 20 + 20 MUSAN music); verified disjoint from the bench sample.
- Matrix row "Silero trên tiếng nói" is tested with a fake session (no speech audio in tests); the real-model behaviour on VIVOS was checked by the prototype (0 missed utterances in 100 × 8 conditions) and is re-measured by the bench.
- Prototype numbers (scratchpad, webrtc vs Silero 0.5, false triggers on 100 noise-only clips): DEMAND 96/105/101 → 24/24/27, MUSAN 141/117/115 → 0/3/10 at 20/10/0 dB; webrtc numbers reproduce the `a03a1d2` bench exactly.
- Collect + curate done: 913 distinct outputs from 200 disjoint MUSAN noise+music files (`results/hallucination_candidates.tsv`) → 9 phrases (≥ 4 words, each with its source), 0 matches in 13,277 real VIVOS/FLEURS sentences.
- `eval/asr_bench.py` memoizes Whisper per prompt × VAD (`memoize_segments`) so filter off/on share one call and its time.

## Results (2026-11-09, `results/guard_vivos_{demand,musan}.csv`, 100 VIVOS utts, seed 0)

- `vad_false_triggers` webrtc → Silero: DEMAND 96–109 → 24–27 (−75%), MUSAN 115–141 → 0–10 (−93–100%). Per-file check: Silero fires only on the 3/17 DEMAND scenes with real background voices (OMEETING, PCAFETER, PRESTO); 0 frames on the other 14.
- `ghost_texts` (20/10/5/0 dB), webrtc+off → Silero+filter: small DEMAND 96/104/108/97 → 4/4/5/4, MUSAN 124/102/108/112 → 0/0/0/1; phowhisper DEMAND 96/105/109/101 → 24/22/26/26, MUSAN 141/117/115/115 → 0/3/4/9. The filter is ~useless for PhoWhisper (no_speech_prob always 0, varied news-like ghosts).
- WER: small never worse than +0.2 and up to −4.5 points at 0 dB. PhoWhisper +2.2 (DEMAND 5 dB) and +4.1 (MUSAN 0 dB), i.e. the "≤ 1 point" criterion is **not strictly met** there. Per-utterance diagnosis: each regression is dominated by one PhoWhisper repetition loop (+28 / +17 words); without it Silero has fewer errors than webrtc. webrtc WER at those cells is not reproducible between runs (13.2 vs 16.0, 7.3 vs 9.0; Silero reproduces exactly), most likely faster-whisper's sampled temperature fallback on long noisy webrtc segments.
- Defaults unchanged (webrtc, filter off), pending the human decision. Follow-ups: repetition-loop guard for PhoWhisper; background-speaker triggers are a separate problem.

## Plan Change Log

## Review Triage Log

## Design Notes

Bộ lọc nằm trong `WhisperASR` nên app và bench dùng chung một đường. Silero quyết định theo cửa sổ 512 mẫu, còn Segmenter đếm theo frame 30 ms. Mỗi frame trả trạng thái hysteresis sau cửa sổ cuối cùng đã đủ mẫu, nên quyết định trễ tối đa một cửa sổ (32 ms).

## Verification

**Commands:**
- `uv run pytest -q` -- expected: all pass
- `uv run --group eval python eval/asr_bench.py --smoke --vad silero --filter on --out /tmp/guard_smoke.csv` -- expected: CSV có `vad_false_triggers`, `vad_extra_segs`, `ghost_texts`
- `uv run python scripts/collect_hallucinations.py --help` -- expected: chạy được, nêu rõ nhiễu nguồn không trùng với nhiễu đo
