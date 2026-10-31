# Audio Translator: dịch giọng nói Việt → Anh qua micro ảo

> **EN summary:** A desktop app (macOS) that listens to Vietnamese speech from the microphone, recognizes it (faster-whisper), translates it to English (Argos, offline), synthesizes English speech (Edge TTS) and plays it into a **virtual microphone (BlackHole)**, so the other side of a Zoom / Google Meet / Discord call hears English. English install guide: [README.en.md](README.en.md).

Đồ án **GR2 (Project 2)**, đề tài 1 **"Công cụ Dịch Audio"**, thuộc định hướng *Ứng dụng đa phương tiện và AI/ML trong hệ thống Edge–Cloud*, Trường CNTT & TT, Đại học Bách khoa Hà Nội (SOICT – HUST).

**Kịch bản:** người dùng nói tiếng Việt vào micro. Ứng dụng dịch sang tiếng Anh rồi phát giọng đọc tiếng Anh vào micro ảo. Trong ứng dụng chat, người dùng chọn micro ảo đó làm micro, nên người bên kia nghe được tiếng Anh.

---

## 1. Luồng hoạt động

```
🎤 Micro máy tính
   │ khung 30 ms, 16 kHz mono
   ▼
[M2] Thu âm ─────────▶ [M3] VAD / cắt câu (600 ms im lặng = hết câu, tối đa 15 s)
                              │ Segment #n  (ghi mốc speech_end)
                              ▼
                       [M4] ASR – faster-whisper (tiếng Việt → chữ)
                              ▼
                       [M5] Dịch – Argos vi→en offline (lỗi → Google)
                              ▼
                       [M6] TTS – Edge TTS (chữ tiếng Anh → giọng nói)
                              ▼
                       [M7] Sắp thứ tự theo segment_id + đổi sang 48 kHz stereo
                              ▼
                       [M8] Phát vào BlackHole 2ch (ghi mốc first_audio_out)
                              ▼
          Zoom / Meet / Discord chọn "BlackHole 2ch" làm micro ──▶ 🎧 người nghe

[M1] Giao diện PySide6 / CLI + bộ điều phối (orchestrator): Start/Stop, phiên, luồng, hàng đợi, sự kiện
```

- **Độ trễ** = `first_audio_out − speech_end`, tính từ lúc kết thúc câu nói đến lúc âm thanh đầu tiên vào micro ảo.
- Mỗi khâu chạy trên **một luồng riêng**, nối với nhau bằng **hàng đợi có giới hạn**, nên các câu được xử lý song song theo kiểu dây chuyền.
- Mỗi câu mang theo `session_id`, `segment_id` và mốc thời gian của từng bước.

## 2. Kiến trúc 8 module

| Module | File | Cách làm |
|---|---|---|
| M1 Giao diện / điều phối | `gui.py`, `cli.py`, `orchestrator.py` | Start/Stop; chọn micro, đầu ra và model; mỗi lần Start là một phiên mới. Giao diện chỉ giao tiếp với orchestrator qua sự kiện |
| M2 Thu âm | `capture.py` | sounddevice (PortAudio). Callback chỉ chép khung âm thanh, gắn mốc thời gian rồi đẩy vào hàng đợi |
| M3 VAD / phân đoạn | `segmenter.py` | webrtcvad, giữ 300 ms trước khi bắt đầu nói, chốt câu sau 600 ms im lặng, cắt câu dài quá 15 s |
| M4 ASR | `asr.py`, `model_manager.py` | faster-whisper chạy CPU int8, ngôn ngữ `vi`. Có 6 model (tiny → large-v3), tải ở luồng nền, giữ tối đa 2 model trong RAM |
| M5 Dịch | `translate.py` | Model Argos vi→en (~66 MB) chạy trực tiếp bằng CTranslate2 + SentencePiece; dự phòng bằng Google Translate |
| M6 TTS | `tts.py` | edge-tts, giọng `en-US-AriaNeural`, giải mã MP3 thành PCM |
| M7 Chuẩn hóa / lập lịch phát | `playout.py` | Đổi tần số bằng soxr; sắp câu theo `segment_id`; câu thiếu chờ tối đa 5 s; quá tải thì bỏ câu cũ nhất |
| M8 Kết nối audio ảo | `virtual_out.py`, `devices.py` | Phát vào BlackHole 2ch; watchdog 2 s phát hiện mất thiết bị |

Phần dùng chung: `config.py` (toàn bộ tham số), `models.py` (định dạng dữ liệu và sự kiện).

## 3. Công nghệ và chi phí

| Khâu | Công cụ | Chạy ở đâu | Chi phí |
|---|---|---|---|
| ASR | faster-whisper (Whisper, mã nguồn mở) | Trên máy (edge), CPU | Miễn phí |
| Dịch | Argos Translate vi→en (mã nguồn mở) | Trên máy, offline | Miễn phí |
| Dịch dự phòng | Google Translate qua `deep-translator` (không chính thức) | Cloud | Miễn phí, bị giới hạn số lần gọi |
| TTS | Edge TTS qua `edge-tts` (không chính thức) | Cloud | Miễn phí, cần internet |
| Micro ảo | BlackHole 2ch | macOS | Miễn phí |

Python 3.12 · PySide6 · sounddevice · webrtcvad · soxr · miniaudio · CTranslate2 · uv.

## 4. Cài đặt và chạy (macOS)

```sh
brew install uv blackhole-2ch        # có thể cần khởi động lại máy sau khi cài BlackHole
git clone https://github.com/MasterCode12345/project2.git && cd project2
uv sync
uv run audio-translator --cli --list-devices   # phải thấy "BlackHole 2ch"
uv run audio-translator --cli --list-models    # xem các model Whisper đã tải hay chưa
```

**Chạy giao diện:** `uv run audio-translator`. Chọn micro, Output = **BlackHole 2ch**, Model = `small`, rồi bấm **Start**.

**Chạy dòng lệnh:**
```sh
uv run audio-translator --cli --model small [--input "Tên micro"] [--translator google]
```

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

## 5. Kiểm thử

```sh
uv run pytest -q     # 137 passed
```
Các test dùng micro, Whisper, bộ dịch và TTS giả, nên chạy được mà không cần thiết bị âm thanh hay mạng. Phạm vi kiểm thử: luồng chạy bình thường, câu dài, câu về sai thứ tự, Stop giữa chừng, quá tải, lỗi mạng, mất thiết bị, chọn và tải model, dịch offline kèm dự phòng, giao diện và CLI.

## 6. Kết quả đo hiện tại

Máy dùng để đo: MacBook Pro (Apple Silicon), Whisper chạy bằng CPU (int8), dịch Argos offline, TTS Edge.

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

## 8. Vấn đề đang gặp (cần góp ý)

1. **Nhận dạng tiếng Việt sai, chủ yếu sai dấu thanh** (nắng→nặng, ngày→ngay, tối→tội, Bách→Bạch). Lỗi này **kéo theo dịch sai** ("Bàn ăn tối chưa" thành "Is the table bad?"). Model `medium` chậm hơn khoảng 70% mà không chính xác hơn `small` trên mẫu thử. **Đây là ưu tiên số 1.**
2. **Whisper "ảo giác" sinh ra câu "ma":** khi có tạp âm, Whisper tạo ra câu người dùng không nói, thường là câu kiểu phụ đề YouTube ("…subscribe kênh… để không bỏ lỡ video"). Nguyên nhân là webrtcvad để lọt tạp âm và code chưa lọc theo độ tin cậy của Whisper.
3. **Nghi có vòng lặp âm thanh khi gọi:** micro thu lại tiếng phát từ loa, làm một câu xuất hiện hai lần.
4. **Độ trễ tăng mạnh khi gọi Meet** (7–10 s): câu "ma" chiếm chỗ trong hàng đợi, và Chrome dùng chung CPU với Whisper.
5. **Model dịch nhỏ yếu với tên riêng:** "Đại học Bách Khoa Hà Nội" bị dịch thành "University of Hanoi".
6. **Mới có một cặp ngôn ngữ Việt → Anh**, chưa chọn được ngôn ngữ trên giao diện.
7. **TTS và dịch dự phòng dùng dịch vụ không chính thức, cần internet.**
8. **Chưa có báo cáo đo đầy đủ:** WER trên bộ câu lớn, chất lượng dịch (BLEU/COMET), P95, độ ổn định. Mốc thời gian của từng bước đã có sẵn, chỉ chưa ghi ra file và tính toán.

## 9. Hướng phát triển

Theo định hướng của giảng viên: **ưu tiên làm cho nhận dạng Việt → Anh chính xác**, sau đó mở rộng **càng nhiều ngôn ngữ càng tốt**.

- [ ] Thử **PhoWhisper** (VinAI; Whisper được huấn luyện thêm trên 844 giờ tiếng Việt, nhiều giọng vùng miền) và **Whisper large-v3-turbo**. Chuyển sang định dạng CTranslate2 để dùng với faster-whisper.
- [ ] Truyền **câu mồi tiếng Việt** cho Whisper (`initial_prompt`), lọc câu theo `no_speech_prob` / `avg_logprob`, thay webrtcvad bằng **Silero VAD**.
- [ ] Đo **WER** trên bộ 30–50 câu tự thu âm hoặc bộ dữ liệu VIVOS; đo chất lượng dịch; ghi CSV; tính trung bình và P95 độ trễ của từng khâu.
- [ ] Thử model dịch tốt hơn cho tiếng Việt.
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
├── src/audio_translator/   mã nguồn (~2.450 dòng)
│   ├── __main__.py         điểm vào: giao diện, hoặc CLI khi có --cli
│   ├── gui.py, cli.py      tầng giao diện
│   ├── orchestrator.py     tầng điều phối (M1)
│   ├── model_manager.py    quản lý model Whisper
│   ├── capture.py → denoise.py → segmenter.py → asr.py → translate.py → tts.py → playout.py → virtual_out.py   (M2 → M8)
│   └── config.py, models.py, devices.py   dùng chung
├── tests/                  129 test (pytest)
├── eval/denoise_bench.py   bench đo lọc nhiễu (WER/CER, độ trễ, RTF)
├── _bmad-output/           các bản kế hoạch thiết kế (tiếng Anh): MVP, quản lý model, ô chọn model, dịch offline
├── pyproject.toml, uv.lock, .python-version
├── README.md               (tài liệu này)
└── README.en.md            hướng dẫn cài đặt bằng tiếng Anh
```

> Ghi chú: mã nguồn được viết với sự hỗ trợ của **Claude Code** theo quy trình lập kế hoạch → cài đặt → review (xem `_bmad-output/`). Sinh viên chịu trách nhiệm định hướng, kiểm thử thực tế và đánh giá kết quả.
