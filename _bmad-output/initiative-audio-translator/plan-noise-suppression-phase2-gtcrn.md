---
title: 'Noise suppression phase 2 — GTCRN streaming ONNX enhancer behind the existing Enhancer interface'
type: 'feature'
ticket: ''
created: '2026-10-28'
status: 'draft'
baseline_revision: ''
route: 'full'
route_source: 'auto'
risk: 'medium'
review: ''
review_source: ''
lenses_ran: []
review_loop_iteration: 0
context:
  - '{project-root}/_bmad-output/initiative-audio-translator/plan-noise-suppression-phase1.md'
---

<frozen-after-approval reason="human-owned intent — do not modify unless human renegotiates">

## Intent

**Problem:** Phase 1 chỉ có log-MMSE; arm chính mà research khuyến nghị (GTCRN streaming ONNX + OA) chưa có, nên bench chưa so được learned SE với classical.

**Approach:** Thêm `GtcrnEnhancer` thỏa đúng `Enhancer` protocol và đăng ký `"gtcrn"` trong `ENHANCERS`; orchestrator, `NoiseSuppressor`, gate, mixer, segmenter giữ nguyên. Thêm arm gtcrn vào bench, cập nhật README.

## Boundaries & Constraints

**Always:** Mặc định `ns_mode` vẫn `"off"`. Không đổi `Enhancer` protocol, `NoiseSuppressor`, `SnrGate`, `ObservationMixer`, `orchestrator.py`, `segmenter.py`. Enhancer causal, frame 480 int16 vào → 480 float32 (thang int16) ra, `latency_samples` khai báo đúng độ trễ thật. STFT khớp repo gốc: n_fft 512, hop 256, sqrt periodic Hann, input chuẩn hóa /32768. Cache ONNX (`conv_cache`, `tra_cache`, `inter_cache`) và buffer reset trong `reset()`. Model chỉ được nạp khi chọn `gtcrn` (import onnxruntime lazy). Test chạy offline, không cần file model (dùng session giả); test với model thật tự skip khi không có file. File model được kiểm tra sha256 `b4718df6228e7bdf1a8a435cf98f838636eb2fd331acabf86ba87c5192ebcb87` (`stream/onnx_models/gtcrn_simple.onnx`, commit `502ebfab64da7c4a9af78dcb9c6ceef1ebb01c73`).

**Never:** Không RNNoise, DNSMOS, GUI control. Không torch/onnx/librosa. Không đổi default thành bật. Không tải model trong test.

## I/O & Edge-Case Matrix

| Scenario | Input / State | Expected Output / Behavior | Error Handling |
|----------|--------------|---------------------------|----------------|
| Căn frame/WOLA | Session giả trả `enh = mix` | Output = input trễ đúng `latency_samples` (480), sai khác ≤ 1 LSB sau warm-up | none |
| Cache được nối | Session giả ghi lại cache nhận vào | Lần gọi k+1 nhận cache_out của lần k; sau `reset()` nhận lại zeros | none |
| Độ dài frame sai | 320 sample | `ValueError` | ValueError |
| Model thật (nếu có file) | White noise 5 dB SNR trên tone | Output hữu hạn, đúng shape; năng lượng ở frame chỉ có nhiễu giảm | skip khi thiếu file |
| Thiếu / sai checksum | `ns_mode="gtcrn"`, file không có hoặc sha256 sai | Session start fail với thông báo nêu đường dẫn và cách lấy file | RuntimeError → error event (cơ chế sẵn có) |
| CLI | `--ns gtcrn` | Được chấp nhận (choices lấy từ `NS_MODES`) | none |
| Bench không có model | `--smoke` khi file không có | CSV vẫn đủ các arm phase 1; arm gtcrn bị bỏ qua kèm 1 dòng thông báo | none |

</frozen-after-approval>

## Code Map

- `src/audio_translator/denoise.py` -- `Enhancer` protocol, `LogMmseEnhancer` (mẫu WOLA streaming để theo), `ENHANCERS` dict (factory không tham số), `NS_MODES` (CLI choices dùng trực tiếp). Thêm `GtcrnEnhancer` + `"gtcrn"` ở đây; không sửa phần còn lại.
- `src/audio_translator/orchestrator.py:276` -- `noise_suppressor_from_settings(s)` gọi trong `_start_session` (chạy trên thread riêng, trong `try`) → lỗi nạp model tự thành error event. KHÔNG sửa.
- `src/audio_translator/config.py:38` -- chỉ cập nhật comment `ns_mode` ("off" | "logmmse" | "gtcrn"); thêm `ns_model_dir` chỉ nếu Open Question 1 cần.
- `src/audio_translator/cli.py:29` -- `--ns` choices = `NS_MODES`, tự có `gtcrn`; chỉ cập nhật help nếu cần.
- `eval/denoise_bench.py:37` `ARMS` (name, mix) + `make_stage` (line 139) cứng `LogMmseEnhancer` → đổi ARMS thành (name, enhancer key, mix); thêm `gtcrn`, `gtcrn+oa0.2/0.4/0.6`; lọc bỏ arm gtcrn khi model không nạp được.
- `tests/test_denoise.py`, `tests/test_bench.py` -- thêm test; `test_smoke_writes_every_arm_x_snr_row` phải vẫn pass không có model.
- `pyproject.toml` -- `onnxruntime` 1.30.0 đã cài (transitive qua faster-whisper); thêm khai báo trực tiếp `onnxruntime>=1.17` vì import trực tiếp.
- ONNX I/O (đã kiểm bằng onnxruntime trên máy): inputs `mix [1,257,1,2]`, `conv_cache [2,1,16,16,33]`, `tra_cache [2,3,1,1,16]`, `inter_cache [2,1,33,16]` (float32); outputs `enh` + 3 cache `_out` cùng shape. Đo thử: RTF ≈ 0.043 riêng phần ONNX trên Mac này. Weights trong file là bản train trên DNS3.

## Tasks & Acceptance

**Execution:**
- [ ] `src/audio_translator/denoise.py` -- `GtcrnEnhancer(model_path=None, session=None)`: ring buffer vào, mỗi hop 256 → rfft(win·buf)/32768 → `session.run` → irfft·win OLA ·32768; FIFO ra prefill 224 để mỗi frame 480 luôn đủ mẫu → `latency_samples = 480`; resolve/verify model theo Open Question 1; đăng ký `"gtcrn"` -- enhancer mới
- [ ] `pyproject.toml`, `uv.lock` -- thêm `onnxruntime` trực tiếp -- dependency rõ ràng
- [ ] `src/audio_translator/config.py`, `cli.py` -- comment/help nêu `gtcrn` -- cấu hình
- [ ] `eval/denoise_bench.py` -- ARMS theo enhancer, arm gtcrn (+OA), bỏ qua có thông báo khi thiếu model -- đo đạc
- [ ] `tests/test_denoise.py`, `tests/test_bench.py` -- mọi dòng matrix -- regression
- [ ] `README.md` -- mục "Lọc nhiễu": `--ns gtcrn`, độ trễ 30 ms, cách có file model, nguồn/license, arm mới của bench; cập nhật số test -- docs

**Acceptance Criteria:**
- Given file model hợp lệ và `ns_mode="gtcrn"`, when app chạy một session, then segment vẫn tới Whisper, timestamp đã trừ 30 ms, RTF của cả stage trên Mac này < 0.1.
- Given `uv run pytest` offline không có file model, when chạy, then tất cả pass (test model thật skip).
- Given `uv run --group eval python eval/denoise_bench.py --smoke` với file model, when chạy, then CSV có thêm 4 arm gtcrn × 5 SNR.

## Implementation Notes

## Plan Change Log

## Review Triage Log

## Design Notes

Độ trễ: hop 256 không chia hết 480 (gcd 32). Sau n frame, số mẫu ra hoàn chỉnh = 256·⌊480n/256⌋; thiếu hụt lớn nhất so với 480n là 224 → prefill FIFO 224 zeros. Tổng trễ = (512−256) + 224 = 480 mẫu = 30 ms, cố định. Session có thể inject (`session=`) để test WOLA/căn chỉnh bằng identity mà không cần model.

## Open Questions

1. **Cách có file model (535 KB).** Research ghi license weights GTCRN chưa rõ (repo MIT).
   - (a) Tự tải lần đầu từ raw.githubusercontent ở commit pin về `~/.cache/audio-translator/gtcrn/`, kiểm sha256, ghi atomic — repo không chứa weights; lần đầu cần mạng.
   - (b) Vendor file vào repo (`src/audio_translator/assets/`) — chạy offline ngay, nhưng phân phối weights khi license chưa rõ.
   - (c) Không tải: người dùng tự đặt file, chỉ đường bằng `--ns-model PATH`/mặc định cache dir; thiếu thì báo lỗi kèm URL.

## Verification

**Commands:**
- `uv run pytest -q` -- expected: all pass
- `uv run --group eval python eval/denoise_bench.py --smoke --out /tmp/ns_smoke.csv` -- expected: 9 arms × 5 SNR khi có model, 5 × 5 khi không
- `uv run python -c "from audio_translator.denoise import GtcrnEnhancer; import numpy as np, time; e=GtcrnEnhancer(); x=(np.random.randn(160000)*3000).astype('int16'); t=time.perf_counter(); [e.process(x[i:i+480]) for i in range(0,len(x)-479,480)]; print('rtf',(time.perf_counter()-t)/10)"` -- expected: rtf < 0.1
