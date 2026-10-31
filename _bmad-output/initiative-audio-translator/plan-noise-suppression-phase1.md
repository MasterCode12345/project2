---
title: 'Noise suppression phase 1 — M3a stage (log-MMSE + observation adding + SNR gate) and WER bench'
type: 'feature'
ticket: ''
created: '2026-10-28'
status: 'built'
baseline_revision: '906d52c09f4421dd1e6a3838ac48841325a42fa3'
route: 'full'
route_source: 'auto'
risk: 'medium'
review: 'quick'
review_source: 'pinned'
lenses_ran: ['quick']
review_loop_iteration: 0
context:
  - '{project-root}/_bmad-output/initiative-audio-translator/research-speech-noise-suppression-for-live-asr/research-speech-noise-suppression-for-live-asr.md'
---

<frozen-after-approval reason="human-owned intent — do not modify unless human renegotiates">

## Intent

**Problem:** Ồn nền làm hỏng VAD và ASR, nhưng nghiên cứu (report trong `context`) cho thấy SE dùng ngây thơ thường *tăng* Whisper WER vì artifacts. Project cần một stage lọc nhiễu bật/tắt được, an toàn, và một cách đo xem nó có giúp tiếng Việt không.

**Approach:** Thêm stage M3a trước VAD: `Enhancer` (phase 1: `NoEnhancer`, `LogMmseEnhancer`) → `SnrGate` → `ObservationMixer`. Audio đã lọc ŝ đưa vào webrtcvad, audio trộn OA đưa vào Whisper. Mặc định **off**. Kèm script `eval/denoise_bench.py` đo WER/CER, latency, RTF, VAD false-trigger theo SNR.

## Boundaries & Constraints

**Always:** Mặc định `ns_mode="off"` và khi off thì pipeline giữ đúng hành vi cũ (mọi test hiện có vẫn pass). Stage chạy trong segment worker thread, không bao giờ trong audio callback. Enhancer causal, xử lý frame 480 sample 16 kHz int16; độ trễ thuật toán `latency_samples` được khai báo. Tín hiệu gốc y được delay đúng `latency_samples` trước khi trộn: y_asr = (1−ω_eff)·ŝ + ω_eff·y. Gate bypass bằng cách đặt ω_eff → 1 (ramp ≥ 10 ms, không click), khi bypass VAD nhận raw. Timestamps (`speech_start`/`speech_end`) trừ đi latency để đo end-to-end đúng. State enhancer/gate reset mỗi session. Output clip về int16, không overflow. Enhancer interface đủ chung để phase 2 thêm GTCRN/RNNoise không đổi orchestrator.

**Never:** Không GTCRN, RNNoise, DNSMOS, GUI control trong plan này. Không thêm torch. Không tải dataset trong test; không đưa dataset vào repo. Không đổi default thành bật.

## I/O & Edge-Case Matrix

| Scenario | Input / State | Expected Output / Behavior | Error Handling |
|----------|--------------|---------------------------|----------------|
| Off (default) | `ns_mode="off"` | Hành vi y hệt hiện tại; không thêm latency | none |
| Log-MMSE trên nhiễu | Tone/speech-like + white noise 5 dB SNR | Năng lượng nhiễu dư giảm ≥ 6 dB ở frame không có tiếng nói | none |
| OA ω = 1 | `ns_mix=1.0` | y_asr bằng đúng raw đã delay (sai khác ≤ 1 LSB) | none |
| Gate bypass | Ước lượng SNR > ngưỡng + hysteresis | ω_eff → 1 mượt; VAD nhận raw | none |
| Gate bật lại | SNR < ngưỡng | ω_eff → `ns_mix`; không bật/tắt liên tục khi SNR dao động trong dải hysteresis | none |
| `ns_gate_snr_db=None` | — | Luôn enhance | none |
| Session mới | Stop rồi Start | State enhancer/gate/delay line reset; không lẫn audio session cũ | none |
| Clipping | Input gần full-scale | Output clip int16, không wrap | none |
| Bench smoke | `--smoke` không mạng, không dataset | CSV đủ cột cho mọi arm × SNR | none |
| Bench thiếu data | `--vivos-dir`/`--noise-dir` sai | Thông báo rõ đường dẫn thiếu, exit ≠ 0 | Error message |

</frozen-after-approval>

## Code Map

- `src/audio_translator/segmenter.py` -- `Segmenter.feed(frame, t)` dùng cùng frame cho `self.vad(frame.tobytes(), rate)` và cho audio của utterance. Thêm tham số tùy chọn `vad_frame` (mặc định = frame) để VAD nhận ŝ còn audio là y_asr; giữ nguyên hành vi khi không truyền.
- `src/audio_translator/orchestrator.py` -- `_start_session` tạo `Segmenter(...)` (≈ line 321) — tạo stage M3a cạnh đó, mỗi session một instance. `_segment_worker` (≈ line 581): `sid, frame, t = item` → `seg.feed(frame, t)` → `Segment(... timestamps={"speech_start","speech_end","segmented"})`; chèn stage giữa get frame và feed, trừ latency vào timestamps.
- `src/audio_translator/config.py` -- `Settings` (M3 segmentation block): thêm `ns_mode`, `ns_mix`, `ns_gate_snr_db`, `ns_vad_input`.
- `src/audio_translator/cli.py` -- `build_parser()` (line 18–26) và `Settings(...)` (line 81): thêm `--ns {off,logmmse}`, `--ns-mix`, `--ns-gate-db` (giá trị `none` → None).
- `src/audio_translator/asr.py` -- `WhisperASR.transcribe(audio_int16, 16000)`; bench dùng trực tiếp class này (không qua orchestrator threads).
- `tests/conftest.py` -- `Harness`, `FakeASR`, `energy_vad`, `speech_frame/silence_frame/utterance` (480-sample frames) — tái dùng cho test stage và bench smoke.
- Dữ liệu cho bench (không tải trong test): VIVOS layout `test/prompts.txt` (`<id> <text>`) + `test/waves/<spk>/<id>.wav` 16 kHz [report 32]; noise = thư mục `.wav` bất kỳ (ví dụ tập con MUSAN noise/music) [report 37]. Đọc wav bằng stdlib `wave` + `soxr` resample về 16 kHz mono.
- `jiwer` 4.0 (Apache-2.0, không torch) [report 35] — thêm vào dependency group `eval`.

## Tasks & Acceptance

**Execution:**
- [x] `src/audio_translator/denoise.py` -- `Enhancer` protocol (`latency_samples`, `process(frame_int16)->float32/int16`, `reset()`); `NoEnhancer`; `LogMmseEnhancer` (STFT streaming WOLA trên frame 480, log-MMSE gain + decision-directed a priori SNR, noise PSD tracking causal); `SnrGate` (noise floor/speech level tracking trên raw, hysteresis 3 dB, ramp ω_eff); `ObservationMixer` (delay line + trộn + clip); `NoiseSuppressor` ghép các phần, trả `(vad_frame, asr_frame)` -- stage M3a
- [x] `src/audio_translator/segmenter.py` -- `feed(frame, t, vad_frame=None)` -- tách input VAD
- [x] `src/audio_translator/config.py`, `cli.py` -- settings + flags -- cấu hình
- [x] `src/audio_translator/orchestrator.py` -- tạo `NoiseSuppressor` mỗi session khi `ns_mode != "off"`, chạy trong `_segment_worker`, trừ latency vào timestamps -- tích hợp
- [x] `eval/denoise_bench.py` + `pyproject.toml` (group `eval` = jiwer) -- trộn speech với noise theo SNR {clean,20,10,5,0}, arms {none, logmmse, logmmse+OA ω∈{0.2,0.4,0.6}}, chạy NoiseSuppressor+Segmenter+WhisperASR, normalize text tiếng Việt (lowercase, bỏ dấu câu, giữ dấu thanh), ghi CSV: arm, snr_db, n_utts, wer, cer, added_latency_ms, rtf, vad_false_triggers (số segment trên clip chỉ có noise); `--smoke` dùng tín hiệu tổng hợp + ASR giả để chạy offline -- đo đạc
- [x] `tests/test_denoise.py`, `tests/test_segmenter.py`, `tests/test_orchestrator.py`, `tests/test_bench.py` -- mọi dòng matrix -- regression
- [x] `README.md` -- mục "Lọc nhiễu (thử nghiệm)": flags, cách chạy bench với VIVOS + MUSAN, nhắc mặc định off và lý do -- docs

**Acceptance Criteria:**
- Given `ns_mode="logmmse"`, when the app runs a session, then segments still reach Whisper, end-to-end latency includes the enhancer delay, and CPU RTF of the stage on this Mac is < 0.1.
- Given `uv run --group eval python eval/denoise_bench.py --smoke`, when run offline, then it writes a CSV with every arm × SNR row.
- Given `uv run pytest`, when run offline with no audio devices, then all tests pass.

## Implementation Notes

- `LogMmseEnhancer`: 480-sample periodic sqrt-Hann WOLA, hop 240 (2 hops per frame) → `latency_samples = 240` (15 ms). Gain = log-MMSE with decision-directed ξ (α 0.98, ξ_min −25 dB), floor −15 dB; noise PSD via SPP tracker (Gerkmann & Hendriks 2012), initialised on the first non-silent hop. E1(x) via Abramowitz & Stegun 5.1.53/5.1.56 (no scipy). Measured RTF ≈ 0.004 (enhancer and full stage) on this Mac; noise-only residual −12 dB on the 5 dB tone+white-noise case.
- `SnrGate`: frame energy in dB; noise floor = min over 3 s; speech level = EMA of frames > floor + 6 dB; bypass when SNR > thr + 3 dB, re-enable when SNR < thr. Gate decides on the current raw frame; mixing applies to the delayed frame (15 ms skew, accepted).
- ω_eff ramps per sample (default 20 ms full swing, min 10 ms enforced).
- Settings default `ns_gate_snr_db = 20.0` (plan left the value open); the bench defaults to `--gate-db none` so arms measure the enhancer itself.
- Bench flushes each clip with digital silence so a trailing utterance always closes; `--smoke` uses tone-burst "words", an energy VAD and a toy recognizer (CSV shape/offline check only — its WER numbers mean nothing). WER/CER use jiwer when installed, an equivalent Levenshtein fallback otherwise (so `uv run pytest` without the `eval` group still covers the bench).
- Unknown `ns_mode` is validated in `_start_session`'s try block → start fails with an error event.

## Plan Change Log

## Review Triage Log

Pass 1 (lens: quick) — high 0 / medium 1 / low 1 / false 0 / maybe-false 0.

| # | Finding | Verdict | Route | Evidence / action |
|---|---------|---------|-------|-------------|
| 1 | `SnrGate`: a single loud transient locks bypass under steady noise | medium | patch | Reproduced: after one 25000-peak frame, 2000 noise frames keep bypass=True, snr≈24.4 dB at the default 20 dB gate; `_level` never decays. Patch: decay level toward floor on non-speech frames, smooth init, add test. |
| 2 | Bench `rtf` uses clip duration but times pad/flush frames too | low | patch | Denominator `noisy.size/RATE` excludes ~0.69 s flush per clip. Patch: divide by processed duration. |

## Design Notes

Gate được hiện thực bằng ω_eff thay vì bỏ qua enhancer, để enhancer luôn chạy và giữ state "ấm" (không giật khi bật lại):

```python
w_target = 1.0 if gate.bypass else settings.ns_mix
w_eff += clip(w_target - w_eff, -step, step)     # ramp ≥ 10 ms
asr = (1 - w_eff) * s_hat + w_eff * delayed_raw
vad = s_hat if (not gate.bypass and vad_input == "denoised") else delayed_raw
```

## Verification

**Commands:**
- `uv run pytest -q` -- expected: all pass
- `uv run --group eval python eval/denoise_bench.py --smoke --out /tmp/ns_smoke.csv` -- expected: CSV with 5 arms × 5 SNR rows
- `uv run python -c "from audio_translator.denoise import LogMmseEnhancer; import numpy as np, time; e=LogMmseEnhancer(); x=(np.random.randn(16000*10)*3000).astype('int16'); t=time.perf_counter(); [e.process(x[i:i+480]) for i in range(0,len(x)-479,480)]; print('rtf', (time.perf_counter()-t)/10)"` -- expected: rtf < 0.1
