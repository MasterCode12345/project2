# Audio Translator: dịch giọng nói Việt → Anh qua micro ảo

> **EN summary:** A desktop app (macOS) that listens to Vietnamese speech from the microphone, recognizes it (PhoWhisper via faster-whisper, Silero VAD, hallucination filter), restores casing/punctuation (vibert-capu), translates it to English (Argos, offline), synthesizes English speech (Edge TTS) and plays it into a **virtual microphone (BlackHole)**, so the other side of a Zoom / Google Meet / Discord call hears English. English install guide: [README.en.md](README.en.md).

Đồ án **GR2 (Project 2)**, đề tài 1 **"Công cụ Dịch Audio"**, thuộc định hướng *Ứng dụng đa phương tiện và AI/ML trong hệ thống Edge–Cloud*, Trường CNTT & TT, Đại học Bách khoa Hà Nội (SOICT – HUST).

**Kịch bản:** người dùng nói tiếng Việt vào micro. Ứng dụng dịch sang tiếng Anh rồi phát giọng đọc tiếng Anh vào micro ảo. Trong ứng dụng chat, người dùng chọn micro ảo đó làm micro, nên người bên kia nghe được tiếng Anh.

---

## 1. Luồng hoạt động

Cấu hình mặc định hiện nay (`src/audio_translator/config.py`):

```
🎤 Micro máy tính
   │ khung 30 ms, 16 kHz mono
   ▼
[M2] Thu âm ──▶ (M3a lọc nhiễu, mặc định tắt)
                  ▼
[M3] Silero VAD + cắt câu (600 ms im lặng = hết câu, tối đa 15 s)        mốc speech_end, segmented
                  ▼
[M4] ASR – PhoWhisper-small (faster-whisper, CPU int8)                   mốc asr_start / asr_end
     + bộ lọc câu ma: no_speech_prob / avg_logprob / compression_ratio,
       danh sách đen, viết vòng lặp lặp từ một lần, bỏ token rác "unk"/"ofc" đứng riêng
     (chưa convert PhoWhisper → cảnh báo và dùng tạm Whisper small)
                  ▼
[M4b] Khôi phục chữ hoa + dấu câu – vibert-capu (nếu đã convert)          mốc restored
                  ▼
[M5] Dịch – Argos vi→en offline (câu lỗi → Google)                        mốc mt_start / mt_end
                  ▼
[M6] TTS – Edge TTS (chữ tiếng Anh → giọng nói)                           mốc tts_start / tts_end
                  ▼
[M7] Sắp thứ tự theo segment_id + đổi sang 48 kHz stereo
                  ▼
[M8] Phát vào BlackHole 2ch                                               mốc first_audio_out
                  ▼
Zoom / Meet / Discord chọn "BlackHole 2ch" làm micro ──▶ 🎧 người nghe

[M1] Giao diện PySide6 / CLI + bộ điều phối: Start/Stop, phiên, luồng, hàng đợi, sự kiện,
     ghi độ trễ từng đoạn ra CSV khi có --latency-log,
     đèn báo cho người nói 🟢🟡🔴 (độ dài câu, mức dồn hàng, câu vừa rồi có rõ không)
```

- **Độ trễ đầu cuối** = `first_audio_out − speech_end`, tính từ lúc kết thúc câu nói đến lúc âm thanh đầu tiên vào micro ảo. Con số này gồm cả 600 ms chờ im lặng để biết câu đã kết thúc.
- Mỗi khâu chạy trên **một luồng riêng**, nối với nhau bằng **hàng đợi có giới hạn**, nên các câu được xử lý song song theo kiểu dây chuyền.
- Mỗi câu mang theo `session_id`, `segment_id` và mốc thời gian của từng bước. `--latency-log PATH` ghi các mốc đó ra CSV, mỗi đoạn một dòng (mục 4, "Đo độ trễ trong app thật").

## 2. Kiến trúc 8 module

| Module | File | Cách làm |
|---|---|---|
| M1 Giao diện / điều phối | `gui.py`, `cli.py`, `orchestrator.py`, `latency_log.py`, `health.py` | Start/Stop; chọn micro, đầu ra và model; mỗi lần Start là một phiên mới. Giao diện chỉ giao tiếp với orchestrator qua sự kiện. Log độ trễ (tùy chọn) được ghi từ luồng worker, không bao giờ từ callback âm thanh. Đèn báo (sự kiện `HEALTH`) được tính trên luồng monitor, tối đa 10 lần/giây, chỉ quan sát |
| M2 Thu âm | `capture.py` | sounddevice (PortAudio). Callback chỉ chép khung âm thanh, gắn mốc thời gian rồi đẩy vào hàng đợi |
| M3a Lọc nhiễu (tắt) | `denoise.py` | log-MMSE + observation adding + cổng SNR; chưa bật vì bench chưa thấy lợi (mục 4) |
| M3 VAD / phân đoạn | `segmenter.py` | **Silero VAD** v6 (ONNX có sẵn trong faster-whisper, chạy streaming), hoặc webrtcvad (`--vad webrtc`). Giữ 300 ms trước khi bắt đầu nói, chốt câu sau 600 ms im lặng, cắt câu dài quá 15 s |
| M4 ASR | `asr.py`, `model_manager.py` | faster-whisper chạy CPU int8, ngôn ngữ `vi`. Mặc định **`phowhisper-small`** (convert một lần, dự phòng `small`). Có 8 model trong danh mục, tải ở luồng nền, giữ tối đa 2 model trong RAM. Bộ lọc câu ma theo từng segment Whisper (`--asr-filter`, bật) |
| M4b Khôi phục | `restore.py` | vibert-capu (viBERT gắn nhãn kiểu GECToR), encoder CTranslate2 int8 + lớp nhãn numpy; `--restore auto` dùng nếu đã convert |
| M5 Dịch | `translate.py`, `languages.py` | Model Argos vi→en (~66 MB) chạy trực tiếp bằng CTranslate2 + SentencePiece; dự phòng bằng Google Translate theo từng câu. Tùy chọn NLLB-200 / M2M100 (`--translator`) |
| M6 TTS | `tts.py` | edge-tts, giọng `en-US-AriaNeural`, giải mã MP3 thành PCM |
| M7 Chuẩn hóa / lập lịch phát | `playout.py` | Đổi tần số bằng soxr; sắp câu theo `segment_id`; câu thiếu chờ tối đa 5 s; quá tải thì bỏ câu cũ nhất |
| M8 Kết nối audio ảo | `virtual_out.py`, `devices.py` | Phát vào BlackHole 2ch; watchdog 2 s phát hiện mất thiết bị |

Phần dùng chung: `config.py` (toàn bộ tham số), `models.py` (định dạng dữ liệu và sự kiện).

## 3. Công nghệ, giấy phép và chi phí

| Khâu | Công cụ | Giấy phép | Chạy ở đâu | Chi phí |
|---|---|---|---|---|
| ASR | faster-whisper + CTranslate2 | MIT | Trên máy, CPU | Miễn phí |
| ASR (model mặc định) | [PhoWhisper-small](https://huggingface.co/vinai/PhoWhisper-small) (VinAI), convert sang CTranslate2 | BSD-3-Clause (model card) | Trên máy | Miễn phí |
| ASR (dự phòng, tùy chọn) | Whisper `small` … `large-v3` (OpenAI, bản Systran) | MIT | Trên máy | Miễn phí |
| VAD | Silero VAD v6, file ONNX đi kèm faster-whisper, chạy bằng onnxruntime | MIT | Trên máy | Miễn phí |
| VAD (tùy chọn) | webrtcvad | MIT | Trên máy | Miễn phí |
| Khôi phục chữ hoa, dấu câu | [dragonSwing/vibert-capu](https://huggingface.co/dragonSwing/vibert-capu) | CC-BY-SA-4.0 (`SOURCE.json`) | Trên máy | Miễn phí |
| Dịch (mặc định) | Argos Translate vi→en 1.9, dẫn xuất từ OPUS-MT | CC-BY-4.0 (README của gói) | Trên máy, offline | Miễn phí |
| Dịch (tùy chọn) | NLLB-200-distilled-600M | CC-BY-NC-4.0: chỉ dùng phi thương mại | Trên máy | Miễn phí |
| Dịch (tùy chọn) | M2M100-418M | MIT | Trên máy | Miễn phí |
| Dịch dự phòng | Google Translate qua `deep-translator` (không chính thức) | MIT (thư viện) | Cloud | Miễn phí, bị giới hạn số lần gọi |
| TTS | Edge TTS qua `edge-tts` (không chính thức) | LGPLv3 (thư viện) | Cloud | Miễn phí, cần internet |
| Micro ảo | BlackHole 2ch | GPL-3.0 | macOS | Miễn phí |

Python 3.12 · PySide6 (LGPL) · sounddevice · Silero / webrtcvad · soxr · miniaudio · CTranslate2 · onnxruntime · uv. Không có torch trong app: torch chỉ được cài tạm khi convert model.

## 4. Cài đặt và chạy (macOS)

```sh
brew install uv blackhole-2ch        # có thể cần khởi động lại máy sau khi cài BlackHole
git clone https://github.com/MasterCode12345/project2.git && cd project2
uv sync
uv run audio-translator --cli --list-devices   # phải thấy "BlackHole 2ch"
uv run audio-translator --cli --list-models    # xem các model Whisper đã tải / convert hay chưa
```

**Convert model một lần** (torch và transformers chỉ được cài tạm cho lệnh này):
```sh
uv run --with transformers --with torch python scripts/convert_phowhisper.py small              # ASR mặc định
uv run --with "transformers<5" --with torch python scripts/convert_capu_model.py vibert-capu   # khôi phục dấu câu
```
- Chưa convert PhoWhisper thì app vẫn chạy: nó báo rõ (dòng đỏ trên giao diện, WARNING trong CLI) kèm lệnh convert, rồi dùng tạm Whisper `small` cho phiên đó.
- Chưa convert vibert-capu thì app dịch nguyên chữ ASR như trước, không báo gì, vì `--restore auto`.

**Chạy giao diện:** `uv run audio-translator`, rồi bấm **▶ Bắt đầu**. Đầu ra mặc định là **BlackHole 2ch**, model mặc định là `phowhisper-small`. Muốn đổi micro, đầu ra, model, bộ dịch hay lọc nhiễu thì mở mục **Cài đặt**. Có thể mở sẵn với bộ dịch khác: `uv run audio-translator --translator nllb-600m`.

### Giao diện dùng trong cuộc gọi

Giao diện chỉ có chữ tiếng Việt, gom ở `gui_strings.py` để sau này thêm ngôn ngữ giao diện. Có hai chế độ.

**Chế độ đầy đủ** (cửa sổ chính), từ trên xuống:
- **Băng đỏ báo lỗi**, chỉ hiện khi có lỗi:
  - Hiện tiêu đề tiếng Việt kèm gợi ý cách xử lý, ví dụ "Không tìm thấy BlackHole – cài bằng: brew install blackhole-2ch".
  - Câu gốc tiếng Anh nằm trong tooltip và trong nhật ký kỹ thuật. Nút **Chi tiết** mở nhật ký đó.
- **Băng vàng:** "Đang dịch bằng Google (cần mạng) vì … chưa sẵn sàng", khi bộ dịch offline không dùng được.
- **Hàng điều khiển:** nhãn "Tiếng Việt → Tiếng Anh" (sau này thay bằng ô chọn cặp ngôn ngữ), nút **▶ Bắt đầu / ■ Dừng** cỡ lớn, trạng thái, nút **Thu gọn**.
- **Đèn báo cỡ lớn:** lý do chính, và dòng "Cũng: …" cho các lý do khác.
- **Vùng bản dịch:**
  - Mỗi câu là một khối: tiếng Việt chữ nhỏ màu xám, tiếng Anh chữ lớn, độ trễ ở bên phải.
  - Tự cuộn tới câu mới, trừ khi bạn đang cuộn lên đọc lại. Giữ 200 câu gần nhất.
  - Có nút **Sao chép tất cả** và **Xóa**, cùng dòng thống kê "Câu gần nhất · Trung vị phiên · Số câu".
- **Cài đặt** (đóng sẵn): Micro, Đầu ra (micro ảo), Model nhận dạng, Bộ dịch, Lọc nhiễu.
  - Các ô này bị khóa khi đang chạy, và có hiệu lực từ lần Bắt đầu tiếp theo.
  - VAD, khôi phục dấu câu, bộ lọc ASR và giới hạn token chỉ chỉnh được bằng CLI.
- **Nhật ký kỹ thuật** (ẩn sẵn): các dòng model, info, warning, error.

**Chế độ thu gọn** (khoảng 360×120, luôn nằm trên các cửa sổ khác):
- Gồm đèn, lý do chính, câu tiếng Anh gần nhất trên một dòng (bản đầy đủ ở tooltip), nút ▶ Bắt đầu / ■ Dừng và nút **Mở rộng**.
- Lần đầu mở, cửa sổ nằm ở góc trên bên phải, để không che thanh điều khiển của Meet/Zoom ở phía dưới.
- Đóng cửa sổ thu gọn thì app quay về cửa sổ chính. Cmd+Q vẫn tắt app gọn gàng.
- **Hạn chế trên macOS:** cửa sổ nổi có thể không hiện đè lên ứng dụng đang ở chế độ toàn màn hình (ví dụ Meet trong Chrome toàn màn hình). Khi đó, để Meet ở cửa sổ thường. Xem kịch bản (f) trong `docs/manual_test.md`.

**Chạy dòng lệnh:**
```sh
uv run audio-translator --cli [--model phowhisper-small] [--input "Tên micro"] [--translator google]
```

### Đo độ trễ trong app thật

`--latency-log PATH` (mặc định tắt) ghi thêm vào `PATH` một dòng CSV cho mỗi đoạn. Mỗi dòng gồm:
- `session_id`, `segment_id`, `status` (`played` / `dropped`) và lý do khi bị bỏ, ví dụ `no_speech` (bộ lọc chặn câu ma, `detail` ghi `blacklist: …`), `mt_error`, `tts_error`, `overload`;
- cấu hình đã thật sự chạy: model ASR (ghi `small` nếu đang dùng dự phòng), VAD, bộ lọc, bộ khôi phục, bộ dịch;
- các mốc `speech_end … first_audio_out` và thời gian từng khâu (ms), kể cả thời gian chờ trong hàng đợi;
- `e2e_ms`, số từ, chữ tiếng Việt và bản dịch.

File được ghi từ luồng worker, không bao giờ từ callback âm thanh. Nếu không ghi được file thì app báo một lần, phiên vẫn chạy tiếp. Cờ này dùng được với cả CLI lẫn giao diện:

```sh
LOG=~/gr2_manual/latency_cli_$(date +%Y%m%d_%H%M).csv
uv run audio-translator --cli --latency-log "$LOG"
uv run audio-translator --latency-log "$LOG"
uv run python eval/latency_report.py "$LOG"
```

- Mỗi lần chạy dùng một file mới có ngày giờ trong tên, để không ghi đè log cũ.
- Lệnh thứ hai mở giao diện; lệnh thứ ba ở trên dùng CLI.
- `latency_report.py` in TB / P50 / P95 / max từng khâu, số đoạn bị bỏ và mức đèn; thêm `--markdown` để in dạng bảng.
- Cột `asr_decode` cho biết giới hạn giải mã có được bật không (`base` = tắt).
- **zsh:** các chú thích `# …` trong khối lệnh của README chỉ để đọc. Khi dán vào terminal, hãy bỏ phần đó (hoặc chạy `setopt interactivecomments` một lần), nếu không zsh sẽ báo lỗi.

Kịch bản thử thật (CLI, Google Meet, môi trường ồn; mỗi kịch bản 10 câu chuẩn bị sẵn): [docs/manual_test.md](docs/manual_test.md). File log chứa nguyên văn lời nói, nên để **ngoài repo**.

### Đèn báo cho người nói

Theo góp ý của giảng viên, app có một đèn **xanh / vàng / đỏ** để người nói biết lúc nào nên ngắt nghỉ, nói chậm lại, hoặc nói lại. Mục đích là để cuộc hội thoại không bị lỗi hay mất thông tin.

- Đèn **chỉ quan sát**: không thay đổi cách app cắt câu, nhận dạng, dịch hay phát.
- Đèn hiện mức cao nhất trong ba tín hiệu. Nếu nhiều tín hiệu cùng mức, lý do chính được chọn theo thứ tự dồn hàng nhận dạng > chờ phát > câu dài > chất lượng. Các lý do khác hiện ở dòng nhỏ bên dưới và trong tooltip.

| Tín hiệu | Vàng | Đỏ | Vì sao chọn ngưỡng này |
|---|---|---|---|
| Câu đang nói dài | ≥ 8 s: "Câu dài, hãy ngắt nghỉ" | ≥ 12 s: "Dừng lại một chút" | Câu thật đã đo: P90 6,6 s, dài nhất 7,4 s. Ở 12 s còn 3 s trước khi app tự cắt ở 15 s, và lần cắt đó có thể rơi vào giữa một từ |
| Hệ thống bị dồn | ≥ 1 câu chờ nhận dạng, hoặc nhận dạng > 3 s | ≥ 2 câu chờ, hoặc > 6 s | Nhận dạng thường mất khoảng 1,2 s (P95 khi gọi Meet 1,8 s); mọi lần nhận dạng > 5,7 s trong thử nghiệm đều là bị kẹt |
| Bản dịch phải chờ phát | câu dịch mới phải chờ câu đang phát > 4 s: "Đang phát bản dịch, chờ một chút" | > 8 s | Log thật 14/11: sau các câu dài 10–15 s, câu sau phải chờ phát 4,1 / 6,4 / 7,8 s; các câu khác chờ khoảng 0 |
| Câu vừa xong không rõ (giữ 4 s) | bị bộ lọc chặn; nhận dạng ≥ 1,5 s **và** ≥ 1,5 lần độ dài câu; hoặc phải giải mã lại (temperature > 0) | – | Các đoạn bình thường trong log thật có tỉ lệ < 0,71. Giải mã lại: 0/15 câu VIVOS, so với 11/40 đoạn nhạc/nhiễu |

- Đèn lên mức ngay lập tức, nhưng chỉ hạ mức sau khi mức thấp hơn giữ được 1 s, để không nhấp nháy.
- Giao diện: một đèn tròn có màu, biểu tượng (✓ ⚠ ✋) và dòng chữ, nên không chỉ dựa vào màu. Khi chưa chạy, đèn màu xám.
- CLI: in một dòng `[đèn] VÀNG: …`, chỉ khi đèn đổi mức.
- `--latency-log` ghi thêm các cột:
  - `health_max`, `health_reason`: mức đèn cao nhất trong lúc xử lý đoạn đó, và lý do lần đầu đạt mức đó;
  - `health_reasons`: mọi lý do đèn đã báo trong thời gian đó, theo thứ tự thời gian, không lặp, kèm mức, ví dụ `vàng: Đang xử lý, nói chậm lại | vàng: Đang phát bản dịch, chờ một chút`.
- Mọi ngưỡng nằm trong `Settings` (`indicator_*`). Lý do chọn từng ngưỡng ở `_bmad-output/initiative-audio-translator/plan-speaker-indicator.md`.
- Kịch bản thử: mục (d) của [docs/manual_test.md](docs/manual_test.md).

**Dùng trong ứng dụng chat:** vào phần cài đặt âm thanh của Zoom / Meet / Discord và chọn **BlackHole 2ch** làm micro. Âm thanh đã dịch chỉ đi vào BlackHole, không bao giờ phát ra loa.

### Lọc nhiễu (thử nghiệm)

Có một bước lọc nhiễu tùy chọn (M3a) chạy **trước VAD**, **mặc định tắt**. Lý do: các nghiên cứu tìm được (xem `_bmad-output/initiative-audio-translator/research-speech-noise-suppression-for-live-asr/`) cho thấy lọc nhiễu dùng ngây thơ thường *làm tăng* WER của Whisper vì tạo ra artifacts. Chỉ nên bật mặc định khi đã đo được là nó giúp tiếng Việt.

Cách hoạt động: bộ lọc log-MMSE (Ephraim & Malah) tạo tín hiệu đã lọc ŝ; webrtcvad nhận ŝ, còn Whisper nhận bản trộn "observation adding" `(1−ω)·ŝ + ω·y` với y là tín hiệu gốc. Một cổng SNR đặt ω → 1 (bỏ qua lọc, có ramp mượt) khi đầu vào đã đủ sạch. Bước này thêm 15 ms độ trễ; độ trễ được trừ vào timestamp nên số đo end-to-end vẫn đúng.

```sh
uv run audio-translator --cli --model small --ns logmmse [--ns-mix 0.4] [--ns-gate-db 20|none]
```

| Flag | Ý nghĩa | Mặc định |
|---|---|---|
| `--ns {off,logmmse}` | bật/tắt lọc nhiễu | `off` |
| `--ns-mix ω` | tỷ lệ tín hiệu gốc trộn lại (0 = chỉ dùng bản đã lọc, 1 = như không lọc) | `0.4` |
| `--ns-gate-db X` | bỏ qua lọc khi SNR ước lượng > X + 3 dB, bật lại khi < X; `none` = luôn lọc | `20` |

**Đo bằng bench** (`eval/denoise_bench.py`): trộn giọng sạch với nhiễu ở SNR {clean, 20, 10, 5, 0} dB, chạy các arm {none, logmmse, logmmse+OA ω∈{0.2, 0.4, 0.6}} qua đúng Segmenter + Whisper của app, ghi CSV gồm `arm, snr_db, n_utts, wer, cer, added_latency_ms, rtf, vad_false_triggers, vad_extra_segs`. Script không tự tải dữ liệu.

Cách chấm (mỗi clip mô phỏng một đoạn của luồng âm thanh liên tục):

- Clip = 2 s nhiễu khởi động (không chấm, để VAD và bộ ước lượng nhiễu thích nghi như trong một phiên đang chạy) + 0,5 s nhiễu + câu nói + 1 s nhiễu. Nhiễu fade-in/fade-out 300 ms ở hai đầu clip, nên không có bước nhảy giữa im lặng tuyệt đối và nhiễu làm VAD kích hoạt giả. SNR tính trên đúng khoảng có câu nói. VAD và stage được tạo mới cho mỗi clip.
- WER/CER chỉ lấy text của các đoạn VAD chồng lên khoảng có câu nói (thời gian đã trừ độ trễ của stage). Đoạn nằm ngoài khoảng đó không bị tính là từ thừa, mà được đếm vào `vad_extra_segs`.
- `vad_false_triggers`: số đoạn VAD cắt ra trong phần nhiễu được chấm của clip chỉ có nhiễu, dựng giống hệt clip có câu nói.
- `--limit N` lấy câu luân phiên qua các người nói, nên chạy ít câu vẫn phủ đủ 19 người nói của VIVOS test.

- VIVOS (CC BY-NC-SA 4.0, chỉ dùng cho nghiên cứu): giải nén để có `vivos/test/prompts.txt` và `vivos/test/waves/<speaker>/<id>.wav`.
- Nhiễu: thư mục `.wav` bất kỳ, ví dụ `musan/noise` hoặc `musan/music` của MUSAN (OpenSLR 17).

```sh
uv run --group eval python eval/denoise_bench.py --vivos-dir ~/data/vivos --noise-dir ~/data/musan/noise \
    --model small --limit 100 --out results/denoise.csv
uv run --group eval python eval/denoise_bench.py --smoke --out /tmp/ns_smoke.csv   # chạy thử offline, dữ liệu tổng hợp
```

WER/CER được tính sau khi chuẩn hóa văn bản (chữ thường, bỏ dấu câu, giữ dấu thanh). Không đưa dataset vào repo.

### Model tiếng Việt (PhoWhisper) và câu mồi

[PhoWhisper](https://github.com/VinAIResearch/PhoWhisper) (VinAI, giấy phép BSD-3-Clause) là Whisper được fine-tune trên 844 giờ tiếng Việt nhiều giọng vùng miền. Repo gốc chỉ có bản cho `transformers`, nên phải convert **một lần** sang CTranslate2 để chạy bằng faster-whisper. torch/transformers chỉ được cài tạm cho lệnh này, không thành dependency của project:

```sh
uv run --with transformers --with torch python scripts/convert_phowhisper.py small medium
uv run audio-translator --cli --model phowhisper-small [--asr-prompt "Xin chào, sau đây là đoạn hội thoại bằng tiếng Việt."]
```

Script tải `vinai/PhoWhisper-{small,medium}` ở revision cố định, ghi model vào `~/.cache/audio-translator/models/phowhisper-<size>-ct2/` kèm file `SOURCE.json` ghi rõ nguồn. `--list-models` in lệnh convert cho model nào chưa có. `large-v3-turbo` thì tải thẳng từ Hugging Face khi dùng lần đầu.

`--asr-prompt` truyền câu mồi (`initial_prompt`) cho Whisper. Mặc định không có câu mồi.

**Lưu ý khi so sánh:** PhoWhisper được train trên cả **tập train của VIVOS** (bài báo PhoWhisper, Bảng 1), nên VIVOS test là dữ liệu cùng miền (in-domain) với nó. Vì vậy bench chạy thêm **FLEURS tiếng Việt** (Google, CC-BY-4.0), một bộ không thuộc các nguồn train mà bài báo liệt kê, để kiểm tra ngoài miền (out-of-domain).

**Bench ASR** (`eval/asr_bench.py`): cùng cách dựng clip và cách chấm như bench lọc nhiễu, không có bước lọc nhiễu. So model × câu mồi (tắt/bật) × SNR, ghi CSV `corpus, noise, model, prompt, snr_db, n_utts, wer, cer, asr_ms_mean, asr_ms_p95, asr_rtf, load_s, extra_segs`. Độ trễ là thời gian `transcribe` trên các đoạn được chấm (trung bình, P95, RTF).

```sh
uv run --group eval python eval/asr_bench.py --corpus vivos --vivos-dir ~/data/vivos --noise-dir ~/data/demand_ch01 --out results/asr_vivos_demand.csv
uv run --group eval python eval/asr_bench.py --corpus fleurs --fleurs-dir ~/data/fleurs_vi --noise-dir ~/data/demand_ch01 --out results/asr_fleurs_demand.csv
uv run --group eval python eval/asr_bench.py --smoke --out /tmp/asr_smoke.csv   # chạy thử offline
```

FLEURS: tải `data/vi_vn/test.tsv` và `data/vi_vn/audio/test.tar.gz` từ `google/fleurs` trên Hugging Face, rồi giải nén để có `fleurs_vi/test.tsv` và `fleurs_vi/test/*.wav`.

### Chống câu "ma"

Khi có nhiễu, webrtcvad bật nhầm gần như ở mọi clip chỉ có nhiễu (96–141 lần trên 100 clip), và Whisper "bịa" câu cho các đoạn đó, ví dụ *"Hãy subscribe cho kênh Ghiền Mì Gõ…"*. Có hai lớp chặn, **cả hai bật mặc định** từ khi có kết quả đo ở cuối mục này. Muốn quay lại hành vi cũ:

```sh
uv run audio-translator --cli --model phowhisper-small --vad webrtc --no-asr-filter
```

| Flag | Ý nghĩa | Mặc định |
|---|---|---|
| `--vad {webrtc,silero}` | bộ phát hiện tiếng nói. Silero dùng model `silero_vad_v6.onnx` có sẵn trong faster-whisper (không thêm dependency), chạy streaming theo cửa sổ 32 ms | `silero` |
| `--silero-threshold p` | xác suất tiếng nói để bắt đầu một đoạn; đoạn kết thúc khi xác suất < p − 0,15 | `0.5` |
| `--asr-filter` / `--no-asr-filter` | bỏ từng segment Whisper có `no_speech_prob` > 0,6, **hoặc** `avg_logprob` < −1,0, **hoặc** `compression_ratio` > 2,4, hoặc chứa một câu trong danh sách đen. Trong các segment giữ lại, một cụm bị lặp thành vòng (1 từ ×4, 2–4 từ ×3, ≥ 5 từ ×2) chỉ được viết **một lần**: `asr.collapse_repeats`, không đổi câu nào trong 14.134 câu thật của VIVOS/FLEURS | bật |

faster-whisper tự nó chỉ bỏ segment khi *cả* `no_speech_prob` > 0,6 *và* `avg_logprob` < −1,0, nên câu ma vẫn lọt qua. Danh sách đen nằm ở `src/audio_translator/hallucinations_vi.txt`. Mỗi câu có ít nhất 4 từ, được so khớp nguyên cụm trên văn bản đã chuẩn hóa, và có ghi nguồn. Danh sách này được xây bằng `scripts/collect_hallucinations.py` từ nhiễu **không trùng** với nhiễu dùng để đo (các file MUSAN `noise` không nằm trong mẫu của bench, cùng MUSAN `music`), để không "học tủ" trên tập đo.

Bench ASR có thêm `--vads webrtc,silero --filters off,on` và ba cột: `vad_false_triggers` (đoạn VAD cắt ra trên clip chỉ có nhiễu), `vad_extra_segs` (đoạn thừa nằm ngoài câu nói) và `ghost_texts` (đoạn của clip chỉ có nhiễu vẫn còn ra chữ sau khi lọc, tức là câu sẽ bị dịch và phát ra):

```sh
uv run --group eval python eval/asr_bench.py --corpus vivos --vivos-dir ~/data/vivos --noise-dir ~/data/musan/noise \
    --models small,phowhisper-small --prompt-modes off --vads webrtc,silero --filters off,on --out results/guard_vivos_musan.csv
```

**Kết quả** (`results/guard_vivos_demand.csv`, `results/guard_vivos_musan.csv`: 100 câu VIVOS test, DEMAND 17 cảnh / MUSAN 20 file nhiễu, seed 0, SNR 20/10/5/0 dB). Mỗi ô là số đoạn trên 100 clip chỉ có nhiễu:

| Nhiễu | Model | webrtc, không lọc | webrtc + lọc | Silero, không lọc | Silero + lọc |
|---|---|---|---|---|---|
| DEMAND | `small` | 96/104/108/97 | 6/6/7/5 | 24/24/25/27 | **4/4/5/4** |
| DEMAND | `phowhisper-small` | 96/105/109/101 | 96/104/108/100 | 24/24/26/27 | **24/22/26/26** |
| MUSAN | `small` | 124/102/108/112 | 0/2/4/1 | 0/2/2/8 | **0/0/0/1** |
| MUSAN | `phowhisper-small` | 141/117/115/115 | 140/117/115/115 | 0/3/4/10 | **0/3/4/9** |

(`ghost_texts`, lần lượt 20/10/5/0 dB. Số lần VAD bật nhầm `vad_false_triggers` của webrtc là 96–109 trên DEMAND và 115–141 trên MUSAN; của Silero là 24–27 và 0–10.)

- **Silero giảm số lần bật nhầm khoảng 75% trên DEMAND và 93–100% trên MUSAN.** Khi đo riêng từng file nhiễu, Silero chỉ bật trên 3/17 cảnh DEMAND có tiếng người thật (OMEETING 74%, PCAFETER 96%, PRESTO 88% số frame) và không bật lần nào trên 14 cảnh còn lại. Vậy 24–27 lần còn lại là **tiếng người nói ở nền**, không phải lỗi VAD. Chặn được chúng là một bài toán khác (chỉ nghe người dùng), không thuộc phạm vi chống câu ma.
- **Bộ lọc chỉ có tác dụng với `small`.** Câu ma của `small` là câu kiểu outro YouTube, có `no_speech_prob` khoảng 0,6–0,7 hoặc nằm trong danh sách đen, nên bị bỏ gần hết. PhoWhisper luôn cho `no_speech_prob` = 0,00 trên nhiễu (token `<|nocaptions|>` vẫn có trong vocab; nhiều khả năng do fine-tune chỉ trên dữ liệu có tiếng nói). Nó bịa ra những câu kiểu tin tức, mỗi lần một khác, nên bộ lọc gần như không bỏ được gì. **Với PhoWhisper, Silero là lớp chặn duy nhất có tác dụng.**
- **WER trên câu thật** (%, sạch/20/10/5/0 dB):

| Nhiễu | Model | webrtc, không lọc | Silero + lọc |
|---|---|---|---|
| DEMAND | `small` | 21,9/21,1/24,0/24,3/30,6 | 20,8/21,3/22,4/23,0/27,8 |
| DEMAND | `phowhisper-small` | 6,9/6,0/6,2/7,3/11,3 | 5,6/6,8/7,1/9,4/11,7 |
| MUSAN | `small` | –/23,5/22,6/26,2/36,9 | –/22,1/22,7/24,6/32,4 |
| MUSAN | `phowhisper-small` | –/9,4/7,0/13,7/13,2 | –/5,9/6,7/7,9/17,3 |

  Với `small`, Silero + lọc không làm WER tệ hơn quá 0,2 điểm ở bất kỳ mức SNR nào, và tốt hơn tới 4,5 điểm ở 0 dB vì bộ lọc bỏ được câu ma xen vào câu thật. Với PhoWhisper, có hai ô tệ hơn quá 1 điểm: DEMAND 5 dB (+2,2) và MUSAN 0 dB (+4,1). Đã chẩn đoán từng câu: ở mỗi ô, phần lớn mức tăng đến từ **một câu** mà PhoWhisper rơi vào vòng lặp ("…không cạo a a a a…", thừa 28 từ; một câu lặp ba lần, thừa 17 từ). Bỏ câu đó ra thì Silero ít lỗi hơn webrtc ở cả hai ô. Thêm vào đó, WER của webrtc ở các ô này **không lặp lại được giữa hai lần chạy** (13,2 → 16,0 và 7,3 → 9,0), trong khi Silero ra đúng từng số. Lý do có thể là các đoạn dài, nhiều nhiễu do webrtc cắt ra làm faster-whisper chuyển sang giải mã có lấy mẫu ngẫu nhiên. Vì vậy chênh lệch 1–3 điểm ở SNR thấp chưa đủ để kết luận theo hướng nào.
- **Đo lại với mặc định mới** (`results/guard2_vivos_{demand,musan}.csv`, PhoWhisper, đã cố định seed của CTranslate2 nên chạy lại ra đúng số; bộ lọc gồm cả bước viết vòng lặp một lần). WER, webrtc không lọc → Silero + lọc (%):

| Nhiễu | sạch | 20 dB | 10 dB | 5 dB | 0 dB |
|---|---|---|---|---|---|
| DEMAND | 6,9 → **5,6** | 6,0 → 6,8 | 6,2 → 7,2 | 7,4 → 8,6 | 13,1 → **12,1** |
| MUSAN | – | 10,4 → **5,9** | 6,9 → 6,8 | 13,7 → **7,9** | 14,8 → 14,8 |

  Bước viết vòng lặp một lần, cùng dùng Silero, giảm WER từ 9,4 xuống 8,6 (DEMAND 5 dB) và từ 17,4 xuống 14,8 (MUSAN 0 dB). Hai ô từng tụt mạnh (+2,2 và +4,1) giờ chỉ còn +1,2 và 0. Còn đúng một ô vượt mức "tệ hơn ≤ 1 điểm": DEMAND 5 dB, +1,2 điểm, khoảng 12 từ trên 100 câu. Ba ô khác tốt hơn rõ. Câu ma giữ nguyên như bảng trên: 22–26 (DEMAND, tiếng người ở nền) và 0–10 (MUSAN).
- **Hạn chế:** mới đo 100 câu với một seed; nhiễu chỉ có hai bộ. Danh sách đen (9 cụm) được xây từ 200 file MUSAN (noise và music) không trùng với nhiễu đo, nhưng cùng nguồn MUSAN, nên con số trên MUSAN có thể lạc quan hơn so với nhiễu thật. Ngưỡng lọc giữ mặc định của Whisper, chưa tinh chỉnh.

### Đo chất lượng dịch (M5)

**Bench dịch** (`eval/mt_bench.py`) trả lời hai câu hỏi:
- Khâu dịch tốt đến đâu khi đầu vào là câu nguồn chuẩn? Đo với hai điều kiện:
  - `text`: câu nguồn nguyên bản;
  - `text_norm`: cùng câu đó nhưng viết thường, bỏ dấu câu, giống chữ ASR trả về.
- Lỗi ASR làm bản dịch tệ đi bao nhiêu? Điều kiện `asr`: Whisper `small` và `phowhisper-small` nhận cả câu audio FLEURS sạch, không qua VAD, rồi đưa chữ đó đi dịch.

So `text` → `text_norm` thấy được thiệt hại do mất chữ hoa và dấu câu. So `text_norm` → `asr` thấy được thiệt hại do nhận dạng sai.

Chỉ số và cấu hình:
- Chỉ số: sacrebleu (Apache-2.0, chỉ nằm trong group `eval`). BLEU tính theo corpus, `tok:13a`, giữ nguyên chữ hoa/thường. chrF2 (`nc:6, nw:0`). Một bản tham chiếu. Chữ ký của sacrebleu được in cuối mỗi lần chạy.
- Độ trễ: đo từng lệnh dịch, có một câu khởi động không tính giờ.
- Google bị lỗi câu nào thì câu đó được đếm vào `n_failed` và không được chấm.
- `src_wer`: WER của câu đưa vào dịch so với câu nguồn chuẩn.
- Tập câu: `--limit N` bốc N câu bằng `--seed`, nên chạy lại ra đúng tập cũ.

```sh
uv run --group eval python eval/mt_bench.py --src vi --tgt en \
    --fleurs-src-dir ~/data/fleurs_vi --fleurs-tgt-tsv ~/data/fleurs_en/test.tsv \
    --custom-tsv eval/data/mt_custom_vi_en.tsv --asr-cache results/mt_asr_cache_vi.tsv \
    --out results/mt_vi_en.csv            # thêm --limit 50 để chạy thử
uv run --group eval python eval/mt_bench.py --smoke --out /tmp/mt_smoke.csv   # offline, dữ liệu và bộ dịch giả
```

- **Điều kiện đầu vào (`--conditions`):**
  - `text`: câu chuẩn.
  - `text_norm`: câu chuẩn viết thường, bỏ dấu câu.
  - `asr`: chữ ASR của cả bản ghi.
  - `asr_seg`: **đúng như app chạy thật**. Bản ghi đi qua VAD, Segmenter và bộ lọc ASR của app (lấy từ `Settings`; `--seg-vad`/`--seg-filter` để đổi), rồi mỗi đoạn được nhận dạng và dịch riêng. Độ trễ được tính theo từng đoạn.
- **Khôi phục chữ hoa và dấu câu (`--restore off,sent,vibert-capu`):** có thêm các dòng mà mỗi đầu vào kiểu ASR được khôi phục trước khi dịch. Thời gian khôi phục nằm ở `restore_mean_ms`/`restore_p95_ms`. Xem mục "Khôi phục chữ hoa và dấu câu" bên dưới.
- **CSV:** `corpus, src_lang, tgt_lang, backend, condition, asr_model, restore, n_sents, src_wer, bleu, chrf, bleu_lc, chrf_lc, n_inputs, mean_words, mean_latency_ms, p95_latency_ms, restore_mean_ms, restore_p95_ms, n_failed`. `n_inputs`/`mean_words` cho biết số lần gọi dịch và độ dài trung bình của mỗi đầu vào. `bleu_lc`/`chrf_lc` là cùng điểm đó nhưng không phân biệt hoa thường, vì đầu ra của app là giọng nói, người nghe không thấy chữ hoa. `bleu`/`chrf` (có phân biệt) vẫn là chỉ số chính.
- **File mẫu `<out>_samples.tsv`:** 20 câu FLEURS cố định, kèm toàn bộ tập tự viết. Mỗi dòng có câu nguồn, chữ đưa vào dịch, bản dịch và tham chiếu.

**Dữ liệu:**
- FLEURS (CC-BY-4.0): câu nguồn và câu đích ghép theo id câu FLoRes. Cả 347 câu của `vi_vn` test đều có trong `en_us` test.
- Phía tiếng Anh chỉ cần file chữ: tải `data/en_us/test.tsv` (368 KB) từ `google/fleurs` về `~/data/fleurs_en/test.tsv`.
- `eval/data/mt_custom_vi_en.tsv`: 25 câu tự viết có tên riêng, số, ngày giờ, đơn vị, từ mượn. Bản dịch tham chiếu là bản nháp **cần duyệt**. Bench ghi `corpus=custom_unreviewed` cho tới khi mọi câu có `reviewed=yes`, và kết quả của tập này không được dùng để kết luận.

**Hạn chế:**
- FLoRes vốn được dịch từ tiếng Anh sang các ngôn ngữ khác, nên chiều vi→en dùng câu nguồn kiểu "văn dịch" (translationese).
- Mỗi câu chỉ có một bản tham chiếu, nên BLEU thấp hơn mức khi có nhiều bản.
- Hướng mở rộng: COMET (cần torch nên chạy riêng ngoài app), chrF++.

**Model dịch đa ngôn ngữ offline (thử nghiệm, chưa phải mặc định):** NLLB-200-distilled-600M (200 ngôn ngữ, CC-BY-NC-4.0: dùng được cho đồ án, không dùng thương mại) và M2M100-418M (100 ngôn ngữ, MIT). Mỗi model chạy được mọi cặp ngôn ngữ nó hỗ trợ. Model phải convert **một lần** sang CTranslate2 int8; torch chỉ được cài tạm cho lệnh này. Lệnh convert tải checkpoint gốc: khoảng 2,5 GB với NLLB, 1,9 GB với M2M100.

```sh
uv run --with transformers --with torch --with sentencepiece --with accelerate python scripts/convert_mt_model.py nllb-600m m2m100-418m
uv run audio-translator --cli --model phowhisper-small --translator nllb-600m
uv run --group eval python eval/mt_bench.py ... --backends argos,nllb-600m,m2m100-418m
```

Nếu model chưa convert, app cảnh báo kèm lệnh convert rồi dùng Google. Câu nào dịch lỗi thì câu đó dùng Google, giống Argos. Mặc định vẫn là Argos (`--translator offline`).

**Khôi phục chữ hoa và dấu câu (M4b):** PhoWhisper trả về chữ thường không dấu câu, và cả hai model dịch đều dịch kém hẳn với kiểu đầu vào này. `--restore` khôi phục lại chữ hoa và dấu câu trước khi dịch:
- `auto` (**mặc định**): nếu đã convert model cho ngôn ngữ nguồn thì dùng model đó; nếu chưa thì bỏ qua, không báo gì, và dịch nguyên văn như trước. Ngôn ngữ nguồn hiện chỉ có tiếng Việt.
- `off`: không khôi phục.
- `sent`: viết hoa chữ đầu và thêm dấu chấm cuối.
- `vibert-capu`: dùng model [dragonSwing/vibert-capu](https://huggingface.co/dragonSwing/vibert-capu), là viBERT gắn nhãn kiểu GECToR, chuyên khôi phục `. , ? :` và chữ hoa cho tiếng Việt. Giấy phép CC-BY-SA-4.0.

Về model `vibert-capu`:
- Model được convert **một lần**: encoder sang CTranslate2 int8 (117 MB), lớp gán nhãn sang numpy. Lệnh convert chỉ tải `pytorch_model.bin` (462 MB). Torch chỉ được cài tạm, và không chạy code nào của repo gốc.
- Lúc chạy, app tự cài lại phần suy luận của model card (WordPiece phân biệt hoa thường, token `$START`, 3 vòng lặp). Lệnh convert tự kiểm tra lại với PyTorch fp32: tách từ trùng khớp, nhãn khớp 109/109 vị trí.
- Model tốn khoảng 150 MB RAM và nạp trong 0,12 s.
- Chọn hẳn `--restore vibert-capu` mà model chưa convert thì app cảnh báo rồi dịch nguyên chữ ASR. Với `auto` thì app không cảnh báo. Đoạn nào khôi phục lỗi thì đoạn đó được dịch nguyên văn, không bao giờ bị bỏ.

```sh
uv run --with "transformers<5" --with torch python scripts/convert_capu_model.py vibert-capu
uv run audio-translator --cli --model phowhisper-small      # --restore auto (mặc định) tự dùng model đã convert
uv run --group eval python eval/mt_bench.py ... --conditions text,text_norm,asr,asr_seg --restore off,sent,vibert-capu
uv run --group eval python eval/mt_oracle_ablation.py --fleurs-src-dir ~/data/fleurs_vi --fleurs-tgt-tsv ~/data/fleurs_en/test.tsv
```

**Bảng ngôn ngữ** (`src/audio_translator/languages.py`): mỗi ngôn ngữ một dòng, gồm:
- mã của từng backend dịch;
- ngôn ngữ Whisper và model ASR ưu tiên;
- giọng Edge TTS;
- config FLEURS;
- cách chuẩn hóa văn bản và tokenize BLEU.

Mỗi backend dịch có `supports(src, tgt)`. `plan_route` chọn dịch thẳng; chỉ khi không có model dịch thẳng mới dịch vòng qua tiếng Anh (`pivot:en`), và tên route luôn được ghi ra. App hiện chưa dùng bảng này, nên hành vi không đổi.

## 5. Kiểm thử

```sh
uv run --group eval pytest -q     # 381 passed
```
Các test dùng micro, Whisper, bộ dịch và TTS giả, nên chạy được mà không cần thiết bị âm thanh hay mạng. Phạm vi kiểm thử: luồng chạy bình thường, đèn báo (từng ngưỡng, chuyển mức, Start/Stop), câu dài, câu về sai thứ tự, Stop giữa chừng, quá tải, lỗi mạng, mất thiết bị, chọn và tải model, dịch offline kèm dự phòng, giao diện (hai chế độ, đồng bộ, khóa cài đặt, không còn chữ tiếng Anh) và CLI.

## 6. Kết quả đo

Máy dùng để đo: MacBook Pro (Apple Silicon, 8 GB RAM), chạy bằng CPU (int8). Mục này tách riêng **số đo bằng bench** (cấu hình hiện tại, có file trong `results/`) và **số đo tay cũ** (cấu hình MVP).

### 6.1 Cấu hình hiện tại: số đo bằng bench

| Câu hỏi | Kết quả | File |
|---|---|---|
| ASR: WER trên FLEURS vi sạch (ngoài miền) | `phowhisper-small` **16,7%**, `small` 27,3%, `large-v3-turbo` 14,8%, `phowhisper-medium` 13,0% | `results/asr_fleurs_demand.csv` |
| Câu ma trên 100 clip chỉ có nhiễu (20/10/5/0 dB), PhoWhisper | webrtc, không lọc: 96–109 (DEMAND), 115–141 (MUSAN) → **Silero + lọc: 22–26 (DEMAND, tiếng người nói ở nền), 0–10 (MUSAN)** | `results/guard_vivos_*.csv`, `results/guard2_vivos_*.csv` |
| WER câu thật khi có nhiễu, PhoWhisper, webrtc → Silero + lọc | sạch 6,9 → 5,6; tệ hơn nhiều nhất +1,2 điểm (DEMAND 5 dB); tốt hơn nhiều nhất −5,8 (MUSAN 5 dB) | `results/guard2_vivos_*.csv` |
| Dịch, như app (cắt đoạn, chữ PhoWhisper), Argos, BLEU / chrF | không khôi phục 12,35 / 43,4 → **vibert-capu 15,15 / 45,5** | `results/mt_vi_en_seg_silero.csv` |
| Độ trễ dịch mỗi đoạn, Argos + vibert-capu, TB / P95 | 95 / 215 ms; khôi phục thêm 23 / 47 ms | `results/mt_vi_en_seg_silero.csv` |

| Nhận dạng bị kẹt (cấu hình gốc), PhoWhisper + Silero + lọc | đoạn giọng thật lâu nhất 10–34 s ở mọi bộ (kể cả sạch); đoạn chỉ có nhiễu tới 20–30 s (DEMAND) | `results/stall_*.csv` |
| Thử giới hạn token 15/s + 32 (mặc định **tắt**) | lâu nhất còn 1,5–3 s, P95 ~1,5 s; nhưng WER tệ hơn: sạch +1,12, DEMAND +0,6 đến +1,9; MUSAN 0 dB −0,52, FLEURS −0,59 | `results/stall_*.csv` |
| Chống kẹt theo từng lượt giải mã (mặc định **tắt**) | (C+): lượt T=0 giới hạn ≥ 100 token, các lượt giải mã lại 15/s + 32. ASR lâu nhất còn 2,9 s (A: 35,9 s); hai ô chính không tệ hơn (VIVOS sạch 0,00; FLEURS −0,30), nhưng DEMAND 5/0 dB +1,64/+1,43 vì vòng lặp bị cắt ngắn rồi được chấp nhận (một câu thừa khoảng 11 từ). Chưa cấu hình nào đạt tiêu chí | `results/antistall_*.csv` |

Chi tiết từng bench nằm ở mục 4 ("Chống câu ma") và mục 6.2.

**Độ trễ đầu cuối trong app thật với cấu hình này: chưa đo.** Công cụ đã có (`--latency-log`, `eval/latency_report.py`); kịch bản đo ở [docs/manual_test.md](docs/manual_test.md). Số sẽ được ghi vào đây kèm tên file sau khi chạy.

### 6.2 Chi tiết bench dịch

**Chất lượng dịch (Argos vi→en, FLEURS test, 347 câu, `results/mt_vi_en.csv`):**

| Đầu vào dịch | WER đầu vào | BLEU | chrF | BLEU (bỏ hoa/thường) | chrF (bỏ hoa/thường) | Trễ dịch TB / P95 |
|---|---|---|---|---|---|---|
| Câu nguồn chuẩn (`text`) | 0 | **27,8** | **55,4** | 28,4 | 55,9 | 308 / 551 ms |
| Câu chuẩn viết thường, bỏ dấu câu (`text_norm`) | 0 | 18,1 | 49,7 | 19,9 | 51,5 | 429 / 889 ms |
| Chữ ASR `small` | 21,0% | 12,7 | 41,7 | 13,4 | 42,6 | 416 / 726 ms |
| Chữ ASR `phowhisper-small` | 13,0% | 15,8 | 45,4 | 16,7 | 46,6 | 356 / 599 ms |

Đọc bảng:
- Với PhoWhisper, BLEU giảm 12,1 điểm so với câu chuẩn. **9,7 điểm là do mất chữ hoa và dấu câu**: PhoWhisper trả về chữ thường, không có dấu câu, và Argos dịch kém hẳn với kiểu đầu vào này. Chỉ 2,4 điểm là do nhận sai chữ.
- Bỏ phân biệt hoa thường thì mức giảm do định dạng vẫn còn 8,5 điểm, nên đây là lỗi dịch thật chứ không chỉ do cách chấm.
- PhoWhisper giúp bản dịch tốt hơn `small` 3,1 BLEU và 3,8 chrF.
- Google bị chặn (HTTP 429) suốt lượt đo, nên chưa có số so sánh.
- Độ trễ được đo khi CPU đang chạy song song bench ASR, nên cao hơn thực tế.
- Hai lần chạy cho cùng điểm số.

**So sánh model dịch offline (FLEURS test, 347 câu, BLEU / chrF, `results/mt_vi_en_models.csv`):**

| Đầu vào dịch | Argos | NLLB-200-600M | M2M100-418M |
|---|---|---|---|
| Câu nguồn chuẩn (`text`) | 27,8 / 55,4 | **34,6 / 60,1** | 27,3 / 54,9 |
| Viết thường, bỏ dấu câu (`text_norm`) | 18,1 / 49,7 | **25,0 / 54,2** | 19,2 / 50,6 |
| Chữ ASR `small` | 12,7 / 41,7 | **16,9 / 44,2** | 12,9 / 41,6 |
| Chữ ASR `phowhisper-small` | 15,8 / 45,4 | **21,3 / 49,4** | 11,9 / 44,8 |
| Trễ dịch TB / P95 khi máy rảnh (50 câu, chữ PhoWhisper, beam 4) | **226 / 412 ms** | 1108 / 1560 ms | 863 / 1285 ms |

- NLLB tốt nhất ở cả bốn điều kiện. Với đầu vào thật của app (chữ PhoWhisper), NLLB hơn Argos 5,6 BLEU và 3,9 chrF.
- NLLB chậm gấp khoảng 5 lần Argos và vượt mục tiêu 500 ms/câu. Dùng beam 1 thì nhanh hơn (TB 779 ms, P95 1145 ms, 4 luồng) mà BLEU chỉ giảm 0,8 (20,1 so với 20,8, trên 50 câu). Dùng 8 luồng lại chậm hơn 4 luồng. Câu FLEURS dài, còn đoạn nói thật trong app thường ngắn hơn, nên độ trễ thực tế có thể thấp hơn, nhưng chưa đo.
- M2M100 không hơn Argos. Một điểm bất thường chưa giải thích: BLEU của M2M100 với chữ PhoWhisper thấp hơn với chữ `small`, dù chrF lại cao hơn.
- Ở bảng này, điểm chất lượng đo trên 347 câu. Độ trễ đo lại riêng khi máy rảnh: hàng số trễ dùng 50 câu, các số beam 1 cũng là 50 câu.
- Giấy phép NLLB là CC-BY-NC-4.0, chỉ dùng phi thương mại. Mặc định của app vẫn là Argos, sau khi bench khôi phục bên dưới cho thấy Argos kèm khôi phục vừa nhanh vừa gần bằng NLLB.

**Mất điểm ở đâu (`results/mt_vi_en_oracle.csv`, `eval/mt_oracle_ablation.py`, 347 câu, BLEU):** câu nguồn chuẩn được bớt dần thông tin cho tới khi giống chữ ASR.

| Câu đưa vào dịch | Argos | NLLB-600M |
|---|---|---|
| Viết thường, không dấu câu (`norm`) | 18,1 | 25,0 |
| + chữ hoa đầu câu, dấu chấm cuối (`sent`) | 18,9 | 25,5 |
| + chữ hoa thật, ví dụ tên riêng (`case+sent`) | 21,4 | 28,9 |
| + dấu câu thật (`punct+sent`) | 24,1 | 31,0 |
| Câu chuẩn (`full`) | 27,8 | 34,6 |

Dấu câu giữa câu là phần đáng giá nhất (+5,2 và +5,5 BLEU so với `sent`), sau đó là chữ hoa tên riêng (+2,5 và +3,4). Chỉ viết hoa chữ đầu và thêm dấu chấm cuối thì gần như không giúp được gì.

**Khôi phục chữ hoa và dấu câu (`results/mt_vi_en_restore.csv`, máy rảnh, PhoWhisper-small, 347 câu, BLEU / chrF):**

| Đầu vào | Bộ dịch | `off` | `sent` | `vibert-capu` |
|---|---|---|---|---|
| Viết thường, bỏ dấu (`text_norm`) | Argos | 18,1 / 49,7 | 18,9 / 50,1 | **24,9 / 53,8** |
| | NLLB | 25,0 / 54,2 | 25,5 / 55,5 | **30,4 / 58,3** |
| Chữ ASR cả bản ghi (`asr`) | Argos | 15,8 / 45,4 | 16,3 / 45,9 | **18,8 / 47,8** |
| | NLLB | 21,3 / 49,4 | 21,4 / 49,5 | **23,1 / 51,1** |
| **Như app: cắt đoạn rồi ASR (`asr_seg`)** | Argos | 13,0 / 43,7 | 13,5 / 44,3 | **15,8 / 46,1** |
| | NLLB | 16,6 / 46,5 | 16,7 / 46,6 | **18,2 / 48,0** |

Độ trễ mỗi đoạn trong điều kiện `asr_seg` (761 đoạn, trung bình 14,8 từ), trung bình / P95:

| | Argos | NLLB |
|---|---|---|
| Dịch | 110 / 239 ms (97 / 214 ms khi có `vibert-capu`) | 586 / 1233 ms (552 / 1160 ms khi có `vibert-capu`) |
| Khôi phục `vibert-capu` | 22 / 44 ms | 22 / 44 ms |

- **`vibert-capu` giúp cả hai bộ dịch ở mọi điều kiện.** Trong điều kiện giống app nhất (`asr_seg`), Argos tăng +2,8 BLEU / +2,4 chrF, NLLB tăng +1,6 / +1,5. Chi phí chỉ khoảng 22 ms mỗi đoạn.
- Trên câu chuẩn bỏ dấu, `vibert-capu` cho Argos 24,9 BLEU, cao hơn cả mức "chỉ thêm dấu câu thật" (24,1), vì nó khôi phục được cả chữ hoa tên riêng. Riêng dấu câu thì đo trên chữ FLEURS bỏ dấu: F1 0,81 (dấu chấm 0,88, dấu phẩy 0,70), chữ hoa F1 0,88. Phép đo này chỉ chạy một lần để khám phá.
- **Cắt đoạn như app làm mất điểm:** WER tăng từ 13,0% lên 15,9%, và câu bị cắt giữa chừng nên khó dịch. Argos giảm từ 15,8 xuống 13,0 BLEU, NLLB từ 21,3 xuống 16,6. Đây là khoảng mất mà các số `asr` trước đây không thấy.
- Bảng trên đo với webrtcvad, mặc định khi đó. **Với mặc định mới (Silero + bộ lọc ASR, `results/mt_vi_en_seg_silero.csv`):** Argos đạt 12,35 / 43,4 (`off`) và **15,15 / 45,5** (`vibert-capu`), tức **thấp hơn khoảng 0,7 BLEU** so với webrtc. WER nguồn gần như không đổi (15,9%). Lý do: Silero cắt nhiều đoạn hơn (843 đoạn, trung bình 13,3 từ, so với 761 đoạn, 14,8 từ), nên câu bị chia nhỏ hơn. Đổi lại, trên audio có nhiễu, Silero giảm câu ma 75–100% (mục 4, "Chống câu ma"). Muốn bù khoản mất này thì gom đoạn ngắn trước khi dịch (mục 9).
- **Đạt mục tiêu ≤500 ms/đoạn:** Argos có hoặc không có `vibert-capu`. NLLB **không đạt**: trung bình khoảng 0,57 s, P95 khoảng 1,2 s.
- **Bất thường:** trên bộ 25 câu tự viết (chưa duyệt), `sent` làm *giảm* điểm (Argos từ 31,5 xuống 28,2). Xem mẫu thì thấy câu có chữ hoa đầu và dấu chấm cuối khiến Argos chuyển sang thì quá khứ ("I was", "began"), có lẽ vì giống văn tin tức. Vì vậy `sent` không phải lựa chọn an toàn. Trên bộ này, `vibert-capu` đưa Argos từ 31,5 lên 40,9 và NLLB từ 35,6 lên 48,6, nhưng bộ này chưa được dùng để kết luận.
- `src_wer` của `text_norm` + `vibert-capu` là 0,0002 thay vì 0, vì nhãn `$MERGE_SPACE` có lúc ghép hai từ làm một.

### 6.3 Số đo tay cũ (cấu hình MVP: Whisper `small`/`medium`, webrtcvad, chưa lọc câu ma, chưa khôi phục)

Các số này đo tay trước khi có bench, chỉ để tham khảo lịch sử. Chúng **không** áp dụng cho cấu hình hiện tại.

**Độ trễ khi không gọi Meet (model `small`, 4 câu):** 2,45–2,69 s, **trung bình ≈ 2,57 s**. Trong đó có khoảng 0,6 s là thời gian chờ im lặng để biết câu đã kết thúc.

**So sánh model `small` và `medium` (cùng 5 câu tiếng Việt):**

| Câu gốc | small nhận dạng | medium nhận dạng |
|---|---|---|
| Ngày mai trời nắng hay mưa? | Ngay mai **tới nặng** hay mưa | Ngay mai **tới nặng** hay mưa |
| Tôi muốn đi học võ, đi tập gym. | …học **phỏ**, đi tập **dím** | …học **vỏ**, đi tập **diễn** |
| Bạn ăn tối chưa? | **Bàn** ăn tối chưa | Bạn **ảnh tội** chưa |
| Tôi là sinh viên ĐH Bách Khoa Hà Nội | …**Bạch** Khoa… | ✅ đúng hoàn toàn |
| Cuộc họp sẽ bắt đầu lúc ba giờ chiều | ✅ đúng | ✅ đúng |
| **Tỷ lệ lỗi âm tiết** | **7/37 ≈ 19%** | **7/37 ≈ 19%** |
| **Độ trễ trung bình** | **2,74 s** | **4,61 s** |

**Khi gọi thật qua Google Meet:** người bên kia nghe được tiếng Anh, nhưng độ trễ tăng lên **≈ 4,7–10,5 s** và xuất hiện các câu "ma" (xem mục 8).

## 7. Đã giải quyết được

- ✅ Đi trọn vẹn từ micro đến bên nghe qua micro ảo, theo đúng kiến trúc 8 module. Đã chạy được với Google Meet.
- ✅ **Câu dài và khoảng dừng:** dùng VAD kết hợp cắt câu ở 15 s.
- ✅ **TTS chậm hơn tốc độ nói:** mọi hàng đợi đều có giới hạn; quá tải thì bỏ câu cũ nhất và báo cảnh báo.
- ✅ **Kết quả về sai thứ tự:** bộ đệm sắp xếp lại theo `segment_id`; câu bị lỗi được đánh dấu bỏ qua để không làm kẹt các câu sau.
- ✅ **Stop phải bỏ kết quả cũ:** dùng `session_id`, mọi khâu tự loại kết quả của phiên trước.
- ✅ **Thu âm không bị nghẽn:** callback âm thanh không chạy tác vụ nặng.
- ✅ **Không vòng lặp từ chính app:** âm thanh dịch không bao giờ phát ra loa.
- ✅ **Dịch offline:** Google Translate từng chặn IP vì gọi quá nhiều, nên chuyển sang chạy model Argos trực tiếp bằng CTranslate2, không cần thư viện `argostranslate` vốn kéo theo PyTorch nặng vài GB.
- ✅ **Đổi model ngay trên giao diện:** tải ở luồng nền, giữ model trong RAM để đổi qua lại nhanh.
- ✅ **Mất thiết bị hay lỗi mạng chỉ bị báo lỗi**, ứng dụng không sập.
- ✅ **Nhận dạng tiếng Việt tốt hơn:** mặc định PhoWhisper-small, WER trên FLEURS giảm từ 27,3% (`small`) xuống 16,7% (`results/asr_fleurs_demand.csv`). Chưa convert thì app dùng tạm `small` và báo rõ.
- ✅ **Câu "ma" khi có nhiễu:** Silero VAD và bộ lọc ASR (danh sách đen, ngưỡng độ tin cậy, viết vòng lặp một lần), bật mặc định. Trên 100 clip chỉ có nhiễu, số câu ma giảm từ 96–141 xuống 0–26 (`results/guard*_vivos_*.csv`).
- ✅ **Token rác "unk", "ofc":** PhoWhisper đôi khi viết thêm những từ này. Bộ lọc ASR bỏ chúng khi chúng đứng riêng thành một từ.
  - Danh sách nằm ở `src/audio_translator/junk_tokens_vi.txt`, kèm bằng chứng cho từng từ. Muốn thêm từ mới thì sửa file này, không cần sửa code.
  - "unk": 3 lần trong log thật, 3/1166 đoạn bench.
  - "ofc": 1 lần trong log thật, 1/244 đoạn FLEURS.
  - Tham chiếu VIVOS/FLEURS không chứa hai từ này.
  - Tính lại WER từ transcript đã lưu thì không ô nào tệ hơn: DEMAND 0 dB 11,76 → 11,66; FLEURS 15,65 → 15,62 (`results/stall_*.csv`, cấu hình gốc).
- ✅ **Dịch kém vì chữ ASR không có dấu câu:** khôi phục chữ hoa và dấu câu bằng vibert-capu trước khi dịch, +2,8 BLEU với Argos trong điều kiện giống app (`results/mt_vi_en_restore.csv`).
- ✅ **Bench đo được, chạy lại ra đúng số:** WER/CER, BLEU/chrF, độ trễ từng khâu bench, seed cố định (kể cả seed của bộ giải mã CTranslate2).
- ✅ **Công cụ đo độ trễ trong app thật:** `--latency-log` và `eval/latency_report.py`.

## 8. Vấn đề đang gặp (cần góp ý)

1. **Nhận dạng có lúc bị kẹt khoảng 20 s.**
   - Trong một phiên Meet (12/11), một đoạn tạo ra chữ vô nghĩa ("vơ sơ đê a tha nói.") mất 20 031 ms nhận dạng. Các câu sau phải chờ, e2e lên tới 22 s.
   - Nguồn: số do người dùng chép lại; file log gốc đã bị ghi đè nên mất. Từ đó kịch bản thử đặt tên log theo ngày giờ.
   - Nguyên nhân đã tái hiện được với đoạn nhạc 3 s (22,2 s): faster-whisper giải mã lại tối đa 6 lần (temperature 0 → 1,0), và mỗi lần bị loại lại chạy tới 224 token, khoảng 6–7 s.
   - Bộ lọc câu ma không chặn được vì `no_speech_prob` ≈ 0.
   - Bench (`results/stall_*.csv`) cho thấy kẹt xảy ra cả trên giọng **sạch**: đoạn lâu nhất 10–34 s ở mọi bộ.
   - Giới hạn token 15/s + 32 hết kẹt (lâu nhất 1,5–3 s), nhưng làm đổi kết quả beam search: thêm đuôi "a." hoặc chuỗi dấu chấm, nên WER sạch tệ hơn 1,12 điểm. Chỉ rút dãy temperature thì gần như không giới hạn được.
   - Chưa biến thể nào đạt tiêu chí "WER không tệ hơn quá 0,5 điểm", nên **mặc định vẫn tắt**. Cờ thử nghiệm của app hiện là `--asr-anti-stall`, xem mục bên dưới.
   - Đã thử thêm 3 cách chống kẹt theo từng lượt giải mã (`results/antistall_*.csv`, plan `plan-asr-anti-stall.md`):
     - (B) giới hạn token kèm bỏ đuôi rác;
     - (C) chỉ giới hạn các lượt giải mã lại;
     - (C+)/(C2) giới hạn rộng cho lượt T=0, có hoặc không có ngân sách thời gian.
   - Kết quả: (C+) hết kẹt (ASR lâu nhất 2,9 s) mà hai ô chính không tệ hơn, nhưng vẫn tệ hơn ở DEMAND 5 và 0 dB, nên **mặc định vẫn tắt**. Lần thử thứ hai (C3: lượt T=0 bị cắt hoặc lặp thì giải mã lại) cũng chưa đạt; xem `plan-asr-anti-stall.md`. Thử trong app bằng `--asr-anti-stall c3a` hoặc `c3b`, kịch bản (e) trong `docs/manual_test.md`.
   - Đèn báo cảnh báo người nói khi nhận dạng quá 3 s / 6 s.
2. **Chưa có số độ trễ của app thật với cấu hình mới.** Các số cũ (2,6 s khi không gọi, 4,7–10,5 s qua Meet; mục 6.3) đo với Whisper `small`/`medium` và webrtcvad. Cần chạy [docs/manual_test.md](docs/manual_test.md).
3. **Tiếng người nói ở nền vẫn bị dịch:** Silero nghe đúng là có giọng người (quán cà phê, phòng họp), nên 22–26/100 clip DEMAND vẫn ra chữ (`results/guard2_vivos_demand.csv`). Đây là bài toán chỉ nghe người dùng, chưa có hướng giải.
4. **Cắt đoạn làm mất điểm dịch:** câu bị cắt giữa chừng khó dịch. Argos giảm từ 15,8 BLEU (cả bản ghi) xuống 13,0 (cắt đoạn, webrtc), Silero còn thấp hơn khoảng 0,7 vì cắt đoạn ngắn hơn (`results/mt_vi_en_restore.csv`, `results/mt_vi_en_seg_silero.csv`).
5. **Nhận dạng vẫn còn sai** (WER 16,7% trên FLEURS), nhất là dấu thanh và tên riêng. `phowhisper-medium` tốt hơn (13,0%) nhưng chậm gần gấp đôi.
6. **Nghi có vòng lặp âm thanh khi gọi:** micro thu lại tiếng phát từ loa, làm một câu xuất hiện hai lần. Tạm thời phải đeo tai nghe; chưa kiểm chứng lại.
7. **Mới có một cặp ngôn ngữ Việt → Anh** trên giao diện.
8. **TTS và dịch dự phòng dùng dịch vụ không chính thức, cần internet.**
9. **25 câu tham chiếu dịch tự viết** (`eval/data/mt_custom_vi_en.tsv`) chưa được duyệt, nên kết quả trên bộ này chưa dùng để kết luận.

## 9. Hướng phát triển

Theo định hướng của giảng viên: **ưu tiên làm cho nhận dạng Việt → Anh chính xác**, sau đó mở rộng **càng nhiều ngôn ngữ càng tốt**.

- [x] Thử **PhoWhisper** (VinAI) và **Whisper large-v3-turbo**, convert sang CTranslate2 để dùng với faster-whisper (mục 4, "Model tiếng Việt").
- [x] Truyền **câu mồi tiếng Việt** cho Whisper (`--asr-prompt`).
- [x] Lọc câu theo `no_speech_prob` / `avg_logprob` / danh sách đen, thêm **Silero VAD** (mục 4, "Chống câu ma").
- [x] Đo **WER** trên VIVOS/FLEURS (`eval/asr_bench.py`) và **chất lượng dịch** BLEU/chrF + độ trễ dịch trung bình/P95 (`eval/mt_bench.py`), ghi CSV.
- [x] Công cụ đo độ trễ từng khâu trong app thật (`--latency-log`, `eval/latency_report.py`).
- [ ] Chạy đo thật theo `docs/manual_test.md` (CLI, Meet, môi trường ồn) và ghi số vào mục 6.
- [x] Mặc định cấu hình Việt → Anh đã chọn qua bench: PhoWhisper-small, Silero + bộ lọc, vibert-capu, Argos.
- [x] Thử model dịch tốt hơn cho tiếng Việt: NLLB-600M hơn Argos 3–6 BLEU nhưng chậm gấp khoảng 5 lần (mục 6). Có trong app qua `--translator nllb-600m`.
- [x] Khôi phục chữ hoa và dấu câu trước khi dịch (mặc định `--restore auto`, mục 6): +1,6 đến +2,8 BLEU trong điều kiện giống app.
- [x] Đèn báo cho người nói (xanh / vàng / đỏ), chỉ quan sát (mục 4, "Đèn báo cho người nói").
- [x] Đo chống kẹt nhận dạng: trần token và dãy temperature ngắn (`results/stall_*.csv`). Chưa đạt tiêu chí WER nên mặc định vẫn tắt.
- [ ] Chống kẹt mà không làm đổi chữ: bỏ đuôi "a." / dấu chấm sau trần token, hoặc chỉ giới hạn các lượt giải mã lại.
- [ ] Giảm mất điểm do cắt đoạn: dịch khi đoạn đã đủ câu, hoặc gom đoạn ngắn trước khi dịch.
- [ ] Chỉ nghe người dùng (bỏ tiếng người nói ở nền).
- [ ] **Đa ngôn ngữ:** chọn ngôn ngữ nguồn và đích trên giao diện, Việt ↔ Anh hai chiều rồi tới các ngôn ngữ khác (Whisper hỗ trợ khoảng 99 ngôn ngữ, Argos khoảng 50, Edge TTS rất nhiều giọng). Đổi ngôn ngữ thì tự tạo phiên mới.
- [ ] Giảm độ trễ: phát TTS dần theo từng phần, chỉnh thời gian chờ im lặng, giảm beam size; thêm TTS offline (Piper).
- [ ] Edge–Cloud: cho chọn chạy tại máy hay trên cloud/GPU cho từng khâu và đo so sánh.
- [ ] Hỗ trợ Windows (VB-CABLE).

## 10. Xin góp ý từ chuyên gia

Mình rất mong được góp ý, đặc biệt về các câu hỏi sau:
1. Để nhận dạng tiếng Việt (nhất là dấu thanh) tốt hơn trên **laptop không có GPU**: nên dùng PhoWhisper, Whisper large-v3-turbo, hay một model/dịch vụ khác?
2. Cách lọc câu "ma" của Whisper và chống vòng lặp âm thanh hiệu quả trong ứng dụng thời gian thực?
3. Model dịch Việt → Anh nào chạy offline mà xử lý tên riêng tốt hơn Argos?
4. Nên thiết kế bộ đánh giá (WER, chất lượng dịch, độ trễ P95) thế nào để có ý nghĩa?
5. Kiến trúc pipeline (luồng + hàng đợi có giới hạn + `session_id`/`segment_id`) đã hợp lý chưa, có nên chuyển sang xử lý streaming không?

Vui lòng mở **Issue** trong repo này hoặc liên hệ trực tiếp với mình.

## 11. Cấu trúc thư mục

```
├── src/audio_translator/   mã nguồn
│   ├── __main__.py         điểm vào: giao diện, hoặc CLI khi có --cli
│   ├── gui.py, cli.py      tầng giao diện (gui_widgets.py: đèn, vùng bản dịch, cửa sổ thu gọn; gui_strings.py: mọi chữ hiển thị)
│   ├── orchestrator.py     tầng điều phối (M1); latency_log.py: CSV độ trễ từng đoạn; health.py: đèn báo cho người nói
│   ├── model_manager.py    quản lý model Whisper
│   ├── capture.py → denoise.py → segmenter.py → asr.py → restore.py → translate.py → tts.py → playout.py → virtual_out.py   (M2 → M8)
│   └── config.py, models.py, devices.py, languages.py   dùng chung (languages.py: bảng ngôn ngữ trung tâm)
├── tests/                  381 test (pytest)
├── eval/denoise_bench.py   bench đo lọc nhiễu (WER/CER, độ trễ, RTF)
├── eval/asr_bench.py       bench so model ASR / câu mồi (WER/CER, độ trễ ASR)
├── eval/mt_bench.py        bench dịch: BLEU/chrF, độ trễ dịch, lỗi lan truyền ASR → dịch, cắt đoạn như app, khôi phục dấu câu
├── eval/mt_oracle_ablation.py   mất điểm dịch do chữ hoa hay do dấu câu (biến thể oracle của câu nguồn)
├── eval/latency_report.py   bảng TB / P95 / max từng khâu từ file --latency-log
├── eval/data/              tập câu tự viết cho bench dịch (tham chiếu cần duyệt)
├── scripts/convert_phowhisper.py   convert PhoWhisper sang CTranslate2 (chạy một lần)
├── scripts/convert_mt_model.py     convert NLLB-200 / M2M100 sang CTranslate2 (chạy một lần)
├── scripts/convert_capu_model.py   convert bộ khôi phục chữ hoa/dấu câu vibert-capu sang CTranslate2 (chạy một lần)
├── scripts/collect_hallucinations.py   thu thập câu ma từ nhiễu không trùng với bench
├── results/                CSV kết quả các bench
├── docs/manual_test.md     kịch bản thử thật: CLI, Google Meet, môi trường ồn, đèn báo, chế độ thu gọn
├── _bmad-output/           các bản kế hoạch thiết kế (tiếng Anh): MVP, quản lý model, ô chọn model, dịch offline
├── pyproject.toml, uv.lock, .python-version
├── README.md               (tài liệu này)
└── README.en.md            hướng dẫn cài đặt bằng tiếng Anh
```

> Ghi chú: mã nguồn được viết với sự hỗ trợ của **Claude Code** theo quy trình lập kế hoạch → cài đặt → review (xem `_bmad-output/`). Sinh viên chịu trách nhiệm định hướng, kiểm thử thực tế và đánh giá kết quả.
