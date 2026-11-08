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
  - `asr_seg`: **đúng như app chạy thật**. Bản ghi đi qua webrtcvad và Segmenter của app (độ nhạy và ngưỡng im lặng lấy từ `Settings`), rồi mỗi đoạn được nhận dạng và dịch riêng. Độ trễ được tính theo từng đoạn.
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

**Khôi phục chữ hoa và dấu câu (thử nghiệm, mặc định tắt):** PhoWhisper trả về chữ thường không dấu câu, và cả hai model dịch đều dịch kém hẳn với kiểu đầu vào này. `--restore` khôi phục lại chữ hoa và dấu câu trước khi dịch:
- `off`: mặc định.
- `sent`: viết hoa chữ đầu và thêm dấu chấm cuối.
- `vibert-capu`: dùng model [dragonSwing/vibert-capu](https://huggingface.co/dragonSwing/vibert-capu), là viBERT gắn nhãn kiểu GECToR, chuyên khôi phục `. , ? :` và chữ hoa cho tiếng Việt. Giấy phép CC-BY-SA-4.0.

Về model `vibert-capu`:
- Model được convert **một lần**: encoder sang CTranslate2 int8 (117 MB), lớp gán nhãn sang numpy. Lệnh convert chỉ tải `pytorch_model.bin` (462 MB). Torch chỉ được cài tạm, và không chạy code nào của repo gốc.
- Lúc chạy, app tự cài lại phần suy luận của model card (WordPiece phân biệt hoa thường, token `$START`, 3 vòng lặp). Lệnh convert tự kiểm tra lại với PyTorch fp32: tách từ trùng khớp, nhãn khớp 109/109 vị trí.
- Model tốn khoảng 150 MB RAM và nạp trong 0,12 s.
- Nếu model chưa convert, app cảnh báo rồi dịch nguyên chữ ASR. Đoạn nào khôi phục lỗi thì đoạn đó được dịch nguyên văn, không bao giờ bị bỏ.

```sh
uv run --with "transformers<5" --with torch python scripts/convert_capu_model.py vibert-capu
uv run audio-translator --cli --model phowhisper-small --restore vibert-capu
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
uv run --group eval pytest -q     # 231 passed
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
- Giấy phép NLLB là CC-BY-NC-4.0, chỉ dùng phi thương mại. Mặc định của app vẫn là Argos.

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
- **Đạt mục tiêu ≤500 ms/đoạn:** Argos có hoặc không có `vibert-capu`. NLLB **không đạt**: trung bình khoảng 0,57 s, P95 khoảng 1,2 s.
- **Bất thường:** trên bộ 25 câu tự viết (chưa duyệt), `sent` làm *giảm* điểm (Argos từ 31,5 xuống 28,2). Xem mẫu thì thấy câu có chữ hoa đầu và dấu chấm cuối khiến Argos chuyển sang thì quá khứ ("I was", "began"), có lẽ vì giống văn tin tức. Vì vậy `sent` không phải lựa chọn an toàn. Trên bộ này, `vibert-capu` đưa Argos từ 31,5 lên 40,9 và NLLB từ 35,6 lên 48,6, nhưng bộ này chưa được dùng để kết luận.
- `src_wer` của `text_norm` + `vibert-capu` là 0,0002 thay vì 0, vì nhãn `$MERGE_SPACE` có lúc ghép hai từ làm một.

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

- [x] Thử **PhoWhisper** (VinAI) và **Whisper large-v3-turbo**, convert sang CTranslate2 để dùng với faster-whisper (mục 4, "Model tiếng Việt").
- [x] Truyền **câu mồi tiếng Việt** cho Whisper (`--asr-prompt`).
- [ ] Lọc câu theo `no_speech_prob` / `avg_logprob`, thay webrtcvad bằng **Silero VAD**.
- [x] Đo **WER** trên VIVOS/FLEURS (`eval/asr_bench.py`) và **chất lượng dịch** BLEU/chrF + độ trễ dịch trung bình/P95 (`eval/mt_bench.py`), ghi CSV.
- [ ] Đo độ trễ trung bình và P95 của từng khâu trong app thật.
- [x] Thử model dịch tốt hơn cho tiếng Việt: NLLB-600M hơn Argos 3–6 BLEU nhưng chậm gấp khoảng 5 lần (mục 6). Có trong app qua `--translator nllb-600m`.
- [x] Khôi phục chữ hoa và dấu câu trước khi dịch (`--restore vibert-capu`, mục 6): +1,6 đến +2,8 BLEU trong điều kiện giống app.
- [ ] Giảm mất điểm do cắt đoạn: dịch khi đoạn đã đủ câu, hoặc gom đoạn ngắn trước khi dịch.
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
│   ├── capture.py → denoise.py → segmenter.py → asr.py → restore.py → translate.py → tts.py → playout.py → virtual_out.py   (M2 → M8)
│   └── config.py, models.py, devices.py, languages.py   dùng chung (languages.py: bảng ngôn ngữ trung tâm)
├── tests/                  231 test (pytest)
├── eval/denoise_bench.py   bench đo lọc nhiễu (WER/CER, độ trễ, RTF)
├── eval/asr_bench.py       bench so model ASR / câu mồi (WER/CER, độ trễ ASR)
├── eval/mt_bench.py        bench dịch: BLEU/chrF, độ trễ dịch, lỗi lan truyền ASR → dịch, cắt đoạn như app, khôi phục dấu câu
├── eval/mt_oracle_ablation.py   mất điểm dịch do chữ hoa hay do dấu câu (biến thể oracle của câu nguồn)
├── eval/data/              tập câu tự viết cho bench dịch (tham chiếu cần duyệt)
├── scripts/convert_phowhisper.py   convert PhoWhisper sang CTranslate2 (chạy một lần)
├── scripts/convert_mt_model.py     convert NLLB-200 / M2M100 sang CTranslate2 (chạy một lần)
├── scripts/convert_capu_model.py   convert bộ khôi phục chữ hoa/dấu câu vibert-capu sang CTranslate2 (chạy một lần)
├── results/                CSV kết quả các bench
├── _bmad-output/           các bản kế hoạch thiết kế (tiếng Anh): MVP, quản lý model, ô chọn model, dịch offline
├── pyproject.toml, uv.lock, .python-version
├── README.md               (tài liệu này)
└── README.en.md            hướng dẫn cài đặt bằng tiếng Anh
```

> Ghi chú: mã nguồn được viết với sự hỗ trợ của **Claude Code** theo quy trình lập kế hoạch → cài đặt → review (xem `_bmad-output/`). Sinh viên chịu trách nhiệm định hướng, kiểm thử thực tế và đánh giá kết quả.
