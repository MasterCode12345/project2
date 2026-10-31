---
title: 'Nghiên cứu academic-lit + technical: speech noise suppression cho live ASR'
type: 'academic-lit+technical'
topic: 'Real-time speech noise suppression làm bước tiền xử lý M3 trước webrtcvad + faster-whisper (app dịch trực tiếp VI→EN, macOS arm64, CPU)'
decision: 'Chọn phương pháp denoising để tích hợp vào M3'
source: 'native run'
status: complete
preset: 'standard'
validation: 'normal'
created: '2026-10-28'
updated: '2026-10-28'
claims_verified: 6
claims_unverified: 9
language: 'vi (thuật ngữ kỹ thuật giữ tiếng Anh)'
---

# Nghiên cứu academic-lit + technical: speech noise suppression cho live ASR

**Quyết định mà nghiên cứu này phục vụ:** chọn phương pháp denoising để đưa vào M3 (bước preprocessing trước VAD và Whisper ASR) của app dịch giọng nói trực tiếp Việt→Anh.

## Executive summary

**Nên làm:** thêm noise suppression vào M3 dưới dạng **một stage bật/tắt được, mặc định tắt**, dùng **GTCRN (streaming ONNX, 16 kHz) bọc trong "observation adding"**: output = (1−ω)·denoised + ω·noisy, với ω≈0.4. Có thể thêm SNR gate tùy chọn. RNNoise và bộ lọc cổ điển log-MMSE được giữ làm ablation arm. Có bật stage này mặc định hay không sẽ do kết quả đo Whisper WER quyết định, đo trên giọng tiếng Việt trộn nhiễu MUSAN ở 20/10/5/0 dB.

**Lý do:**
1. **Denoising thường *làm hại* Whisper.**
   - Ba preprint đều cho thấy speech enhancement (SE) dùng sẵn làm tăng Whisper WER [26][27][28] (medium: cả ba là preprint, một bài có số liệu chưa kiểm chứng).
   - Với các ASR không phải Whisper, nguyên nhân là processing artifacts chứ không phải nhiễu còn sót. Cách khắc phục đã được công bố là trộn lại tín hiệu gốc, gọi là observation adding (OA) [23][24]. Việc cơ chế này cũng đúng với Whisper chỉ là suy luận (medium-low).
   - Dưới khoảng 10 dB pub noise, Whisper xuống cấp chậm hơn các model huấn luyện trên LibriSpeech [25].
2. **GTCRN là learned model phù hợp nhất.**
   - Đây là model duy nhất qua mọi gate, nếu tính "đang được bảo trì" là một gate. NSNet2 ONNX tương tự nhưng repo đã ngừng cập nhật. RNNoise chạy được trên macOS 26.
   - GTCRN chạy native 16 kHz, không cần torch, giấy phép MIT, vẫn được bảo trì, RTF 0.07, 48K params, đăng ở ICASSP 2024 [10][11][15].
   - DeepFilterNet có PESQ tốt nhất nhưng trượt gate: cần torch, không có wheel cho Python 3.12 arm64, bản phát hành cuối năm 2023 [17].
   - Demucs dùng giấy phép non-commercial và repo đã archive [19].
3. **Chưa paper nào đo được từ mức SNR nào thì denoising bắt đầu giúp Whisper.** Con số này là đóng góp riêng của project.

**Lưu ý lớn nhất:** bằng chứng hiện nghiêng về "không enhancement" làm mặc định an toàn. Chọn GTCRN là để có cái *thử nghiệm và trích dẫn*, còn lợi ích của nó với Whisper chưa được đo (medium confidence).


## D1: Canon và state of the art cho real-time

**Các baseline cổ điển (đều causal, gần như không tốn CPU).**
- **Spectral subtraction:** trừ phổ nhiễu ước lượng từ các frame không có tiếng nói. Bài gốc đã định vị phương pháp này là bước preprocessing cho speech recognition [1].
- **MMSE-STSA (Ephraim & Malah):** giả định phổ tiếng nói và phổ nhiễu đều có phân bố Gaussian. Nhiễu dư "colorless", tức ít musical noise hơn [2].
- **Log-MMSE:** biến thể log-spectral của MMSE-STSA, tối thiểu hóa log-spectral MSE [3].
- **Wiener lọc theo decision-directed a priori SNR (Scalart & Vieira Filho):** mốc so sánh Wiener đơn giản thường dùng [4].

Đây là các ablation arm "classical" tự nhiên.

Thư viện `noisereduce` phổ biến dùng spectral gating [5]:
- **Stationary mode:** dùng một ngưỡng duy nhất (trung bình nhiễu + 1.5 std).
- **Non-stationary mode (mặc định):** làm mượt ngưỡng theo cả hai chiều thời gian trong khoảng 2 s. Vì vậy, như đang viết, chế độ này non-causal và chỉ xử lý theo batch.
- **Trích dẫn:** paper mà thư viện đề nghị trích dẫn thuộc lĩnh vực bioacoustics, không phải một paper về SE.

**Các neural model real-time.** Số liệu do từng paper/repo gốc tự báo cáo. Giao thức test khác nhau, nên so sánh giữa các paper chỉ mang tính tương đối.

| Model | Venue | Kích thước / chi phí | Latency | Chất lượng báo cáo | CPU |
|---|---|---|---|---|---|
| RNNoise [6] | arXiv 2017 / MMSP 2018 (theo arXiv comment) | 87.5K weights, tổng ~40 MFLOPS | 10 ms hop (không nêu latency theo ms) | PESQ chỉ có trong hình; > SpeexDSP MMSE | ~1.3% một core i7-4800MQ ở 48 kHz |
| PercepNet [7] | arXiv 2020 (venue chưa xác minh) | không nêu; weights 8-bit | 40 ms lookahead | hạng 2 DNS 2020 RT track, P.808 MOS 3.52 | 5.2% một core i7-8565U |
| Demucs (causal) [8] | Interspeech 2020 | không nêu | 40 ms frame, 16 ms stride | PESQ 2.93 / STOI 95% (VoiceBank-DEMAND) | RTF 0.6–1.05 trên i5 4 nhân |
| DeepFilterNet3 [9] | Interspeech 2023 demo | không nêu | 40 ms | PESQ 3.17 / STOI 0.944 (VoiceBank-DEMAND) | RTF 0.19 single-thread i5-8250U (việc gán con số này cho riêng DFN3: medium) |
| GTCRN [10][11] | ICASSP 2024 | 23.7K params / 39.6 MMACs/s (paper); 48.2K / 33.0 MMACs/s (repo, tính cả ERB) | 16 ms hop (STFT 512/256) | PESQ 2.87, STOI 0.940 (VCTK-DEMAND) | RTF 0.07 (streaming, i5-12400) |
| GRU kiểu NSNet [12] | SPECOM 2020 | 2.8M params | 32 ms window / 16 ms hop, không lookahead | CHiME-2 PESQ 2.29 → ~3.3 | không báo cáo |

Repo GTCRN giải thích chênh lệch số params: con số 23.7K trong paper không tính module ERB, còn tính đủ là 48.2K [10] (đã đối chiếu với repo chính thức; high).
- **Chất lượng và chi phí:** DeepFilterNet3 dẫn đầu về chất lượng báo cáo [9], GTCRN dẫn đầu về chi phí [10].
- **Kết quả ASR:** Demucs là paper duy nhất ở đây có kết quả ASR. Ở 0 dB SNR, nó giảm WER từ 12.0 xuống 6.9 trên LibriSpeech dev-clean mà không cần retrain. Tuy vậy, recognizer là kiểu wav2letter chứ không phải Whisper, và chỉ có một lab (medium) [8].

Chưa tìm được kết luận SOTA mới (≤ 12 tháng) cho dòng low-complexity:
- Abstract tổng kết DNS Challenge ICASSP 2023 không nêu đội thắng hay giới hạn độ phức tạp [13].
- UL-UNAS (2025), model kế nhiệm GTCRN của cùng nhóm tác giả, mới chỉ là lead.

Mảng này đã chạy hết giới hạn 2 vòng mà câu hỏi SOTA vẫn còn mở.

## D3: Thực tế triển khai trên Python 3.12 / macOS arm64 / CPU

Đã đối chiếu với PyPI JSON, GitHub API và file trong repo ngày 2026-10-28. Thông tin phiên bản cũ đi nhanh nhất, cần kiểm tra lại trong vòng một tháng.

| Ứng viên | Sample rate gốc | Py3.12 arm64 | Cần torch? | Bảo trì | License | Streaming | Kết luận cho app |
|---|---|---|---|---|---|---|---|
| GTCRN streaming ONNX [10][14] | **16 kHz** (STFT 512 / hop 256) | qua onnxruntime 1.31.1 có wheel cp312 arm64 [15] | Không (ONNX) | repo cập nhật 2026-08 | MIT (repo) | Có, cached states | **Qua mọi gate**; phải tự viết STFT + ring buffer (hop 256 không chia hết 480) |
| RNNoise / `pyrnnoise` 0.4.5 [6][16] | chỉ 48 kHz | wheel `macosx_26_0_universal2` (cần macOS ≥ 26) | Không | 2026-09 | BSD-3 (lib), Apache-2.0 (wrapper) | Có, `denoise_frame` | Dùng được trên macOS 26; cần resampling 16k↔48k |
| DeepFilterNet [9][17] | 48 kHz | DeepFilterLib **không có wheel cp312** | **Có** (README: bắt buộc) | bản cuối 2023-08 | code MIT/Apache; license của weights còn bỏ ngỏ (issue #697) | chỉ qua Rust/LADSPA | **Trượt gate** (torch, không có wheel, ngừng cập nhật) |
| NSNet2 ONNX (DNS-Challenge) [18] | 16 kHz | qua onnxruntime | Không | repo ngừng cập nhật (2024-07) | CC-BY-4.0 | gain theo từng frame | Dùng được; 2.8M params nếu đúng là model của Braun & Tashev [12] (danh tính medium-low); ít bảo trì |
| Demucs denoiser [19] | — | chỉ có sdist | Có (unverified) | **archived** | **CC-BY-NC-4.0** | — | **Loại** (non-commercial, archived) |
| `noisereduce` 3.0.3 [5][20] | bất kỳ | pure-python | tùy chọn | 2024-10 | MIT | không có stateful API | Chỉ dùng làm baseline offline/batch |
| WebRTC NS (`webrtc-noise-gain`) [21] | 16 kHz int16, 10 ms | **không có wheel macOS** | Không | 2026-05 | không khai báo | Có | Rủi ro phải build từ source |

- **GTCRN so với RNNoise:** tuyên bố "vượt xa RNNoise" là của chính tác giả GTCRN (medium) [10].
- **Tốc độ trên Apple Silicon:**
  - Chưa tìm thấy con số RTF trên Apple Silicon cho GTCRN, RNNoise hay NSNet2.
  - Các bản port DeepFilterNet của cộng đồng báo RTF ~0.04 (MLX, M5 Pro) và 0.34 (CoreML, M2 Max). Đây là low confidence, lấy từ search snippet [22].
- **Ghép kích thước frame:** phương án nào cũng cần code ghép frame. Hop 256 sample của GTCRN thêm khoảng 16–32 ms buffering. RNNoise/DFN cần 1440 sample (3 frame) cho mỗi 30 ms ở 48 kHz.

Mảng này kết thúc sau một vòng vì đã đủ coverage: mọi gate đều đã được đánh giá. Khoảng trống về RTF trên Apple Silicon có thể tự đo trên máy với chi phí thấp.

## D2: Enhancement có giúp Whisper không?

**Cơ chế: artifacts, chứ không phải nhiễu dư, mới làm hại ASR.**
- **Phát hiện gốc:** Iwamoto và cộng sự tách lỗi của single-channel SE thành hai thành phần, noise và "artifact". Họ chỉ ra artifact là nguyên nhân chính làm ASR kém đi [23].
- **Cách khắc phục:** *observation adding* (OA), tức trộn lại input có nhiễu: s̃ = (1−ω)·ŝ + ω·y. Cách này được chứng minh làm tăng signal-to-artifact ratio khi ⟨ŝ, y⟩ > 0 [23][24].
- **Nghiên cứu tiếp theo của NTT** (Kaldi hybrid ASR, WSJ0+CHiME-3) [24] (cùng một nhóm lab; medium, chưa thử với Whisper):
  - SE huấn luyện bằng SNR loss còn *tệ hơn không dùng SE*: WER 16.6% so với 15.9%.
  - ω tốt nhất cho SE huấn luyện theo cách thông thường là ≈ 0.4.

**Whisper vốn đã chịu nhiễu tốt.** Dưới 10 dB SNR với pub noise, Whisper thắng cả 14 model huấn luyện trên LibriSpeech được so sánh. Ở 40 dB thì nhiều model trong số đó lại thắng Whisper [25]. Đây là xếp hạng tương đối cho một loại nhiễu, không phải một ngưỡng chịu nhiễu tuyệt đối.

**Trong mọi nghiên cứu tìm được, SE dùng sẵn đặt trước Whisper đều làm WER tệ hơn.** Đây là ba preprint có kết luận khớp nhau (medium; chưa xác định được các nhóm tác giả có độc lập hay không):
- Front-end MetricGAN+ làm tăng semantic WER ở cả 40 cấu hình (4 ASR model, có Whisper), thêm từ +1.1 đến +46.6 điểm [26] (preprint).
- Separation bằng SAM-Audio làm tăng WER ở mọi kích thước Whisper, từ tiny đến large-v3, với cả tiếng Anh và tiếng Bengali, dù PSNR tăng [27] (preprint).
- CMGAN thuần được ghi nhận làm tăng WER của Whisper-base trên Aurora-4 (≈22.1 → 23.6%). Chỉ khi thêm một "bridge" module thích ứng với ASR thì WER mới hồi lại (≈22.0%) [28] (low-medium; số liệu từ search snippet, chưa kiểm chứng).

**Hướng trung gian, bằng chứng gián tiếp:** denoising *có chọn lọc*, tức chỉ enhance những input dưới một ngưỡng intelligibility, đã thu hẹp chênh lệch WER nam/nữ của Whisper mà không làm hại WER tổng [29] (IEEE ICKG 2024). Tuy nhiên đây là kết quả về fairness, không phải bằng chứng rằng một SNR/DNSMOS gate giảm được tác hại của nhiễu (low, nếu dùng để ủng hộ gating).

Các khoảng trống chưa có bằng chứng:
- Không nghiên cứu nào tìm được báo cáo Whisper WER theo từng mức SNR cho DeepFilterNet, RNNoise hay noisereduce.
- Cũng không nghiên cứu nào thử OA với Whisper.
- Một bài Interspeech 2024 có báo cáo WER của DeepFilterNet3 nhưng không đọc được [38] (lead).

**Mức SNR mà tại đó SE bắt đầu giúp Whisper hiện chưa có bằng chứng. Project này phải tự đo.**

Bối cảnh cho tiếng Việt: Whisper WER trên FLEURS tiếng Việt là 21.2% (small), 12.7% (medium) và 10.3% (large-v2). BLEU cho speech translation vi→en lần lượt là 8.5 / 16.6 / 20.4 [25].

## D4: Đánh giá thế nào

- **Metric quyết định là WER:** đo trên giọng tiếng Việt trộn nhiễu ở các mức SNR có kiểm soát, so sánh có và không có stage lọc nhiễu. Điểm chất lượng không dự đoán được WER: ở [27], PSNR tăng trong khi WER cũng tăng. `jiwer` 4.0 tính được WER/CER, license Apache-2.0, không cần torch [35].
- **Metric chất lượng bổ trợ:**
  - **DNSMOS P.835:** non-intrusive (không cần bản sạch để so), dự đoán SIG/BAK/OVRL với Pearson 0.94–0.98 so với đánh giá của người. Rất hợp cho bản ghi trực tiếp [30].
  - **PESQ P.862.2 wideband:** qua gói `pesq` (MIT; license của code ITU gốc không nêu) [31].
  - **STOI** (Taal và cộng sự 2010/2011): qua `pystoi` (MIT, không cần torch) [34].
  - **SI-SDR** (Le Roux và cộng sự) [33].
  - TorchMetrics có wrapper cho các metric này nhưng cần torch [33].
- **Dữ liệu:**
  - Tiếng Việt sạch: VIVOS (760 câu test, CC BY-NC-SA 4.0, chỉ cho nghiên cứu) [32], hoặc FLEURS vi_vn (CC BY 4.0, 16 kHz; chưa xác nhận kích thước split) [36].
  - Nhiễu: MUSAN (OpenSLR 17, CC BY 4.0, license riêng theo từng thư mục) [37].
  - VoiceBank-DEMAND là tiếng Anh, chỉ hữu ích để tái lập các con số PESQ đã công bố.
- **Giao thức đề xuất** (báo cáo WER, DNSMOS OVRL, latency tăng thêm và CPU RTF):
  - SNR ∈ {clean, 20, 10, 5, 0} dB.
  - Loại nhiễu ∈ {babble, music, ambient}.
  - Arm ∈ {none, classical, RNNoise, GTCRN, GTCRN+OA ω∈{0.2, 0.4, 0.6}}.
  - Whisper ∈ {small, medium}.

## Nhận định liên mảng

1. **Model dẫn đầu về chất lượng và mục tiêu ASR đi hai hướng khác nhau.** D1 xếp DeepFilterNet3 cao nhất về PESQ, nhưng D2 cho thấy cải thiện về cảm nhận không chuyển thành Whisper WER tốt hơn, và artifacts còn gây hại. D3 thì loại DeepFilterNet vì lý do khác hẳn (torch, không có wheel cp312). Không nên để bảng xếp hạng chất lượng quyết định lựa chọn này.
2. **Model nhẹ nhất cũng là model dễ tích hợp nhất.** GTCRN là learned option duy nhất tìm được vừa native 16 kHz, vừa không cần torch, vừa MIT *và* đang được bảo trì tích cực [10][15]. Vì vậy arm *thử* rẻ nhất cũng là model hiện đại có trích dẫn tốt nhất (ICASSP 2024).
3. **OA biến bất kỳ bộ enhancement nào thành một núm vặn chỉnh được và hoàn tác được.** Vì ω = 1 cho lại đúng input chưa xử lý, một stage có bọc OA, nếu chỉnh đúng, không bao giờ tệ hơn "không SE". Khi đó thí nghiệm sẽ chọn giá trị ω thay vì chỉ trả lời có/không.
4. **Giả thuyết (chưa có bằng chứng trong lần chạy này):** enhancement có thể giúp **VAD**, tức ít đoạn bị kích hoạt nhầm do nhiễu, kể cả khi không giúp ASR. Nên dành một arm thí nghiệm để đưa audio đã denoise vào webrtcvad và audio đã trộn OA vào Whisper.

## Decision matrix

Thang điểm 1–5 (5 là tốt nhất); trọng số theo bộ tiêu chí đã thống nhất. DeepFilterNet3, Demucs và WebRTC NS trượt hard gate (lần lượt vì torch/không có wheel cp312, license non-commercial, không có wheel macOS) nên không chấm.

| Phương án | Ảnh hưởng WER (40) | CPU (20) | Trích dẫn được (15) | Tích hợp (15) | Cảm nhận (10) | **Tổng** |
|---|---|---|---|---|---|---|
| Không enhancement (baseline) | 4 — an toàn nhất theo [26][27][28] | 5 | 2 | 5 | 1 | **3.75** |
| **GTCRN + OA + SNR gate** | 3 — chưa đo; OA giảm artifacts với ASR không phải Whisper [23]; gating chỉ có bằng chứng gián tiếp [29] | 5 — RTF 0.07 [10] | 5 — ICASSP 2024 + Interspeech 2022 | 3 — tự viết STFT, ONNX state, ring buffer | 4 | **3.80** |
| RNNoise (`pyrnnoise`) + OA | 3 | 4 — cần resampling 16k↔48k | 4 — MMSP 2018 (theo arXiv comment; unverified) | 4 — có pip wheel cho macOS 26 | 3 | **3.50** |
| Classical log-MMSE / Wiener | 2 — điểm tiên nghiệm, không có trích dẫn (low); canon cho biết MMSE/log-MMSE ít musical noise [2][3] | 5 | 4 — seminal [2][3][4] | 4 — ~100 dòng numpy | 2 | **3.20** |

Cột WER là cột mà bằng chứng chưa phân định được: đây chỉ là điểm tiên nghiệm (prior), thí nghiệm sẽ thay bằng số đo thật. **Với bằng chứng hiện có, GTCRN+OA (3.80) và không enhancement (3.75) coi như hòa.** Khoảng cách 0.05 hoàn toàn dựa vào điểm tiên nghiệm chưa đo đó.

## Khuyến nghị

1. **Xây M3 enhancement thành một stage bật/tắt được, mặc định *tắt* cho tới khi đo xong.** Bằng chứng duy nhất về Whisper cho thấy SE dùng ngây thơ gây hại [26][27][28] (medium confidence: ba preprint có kết luận khớp nhau). Khuyến nghị này là đầu vào cho plan `bmad-build` tiếp theo.
2. **Arm chính: GTCRN streaming ONNX + observation adding (ω chỉnh được, bắt đầu ở 0.4) + SNR/DNSMOS gate tùy chọn** [10][23][24]. Gate là lựa chọn kỹ thuật, chỉ có bằng chứng gián tiếp từ [29]. Phương án dự phòng: RNNoise qua `pyrnnoise`. Nó thắng nếu phần ghép STFT/state của GTCRN quá tốn công, vì trên macOS 26 chỉ cần một lệnh pip install [16]. Medium confidence, vì lợi ích với Whisper chưa đo.
3. **Thêm một arm classical log-MMSE** cho phần ablation trong báo cáo. Arm này rẻ, có tính seminal, và cho thấy vì sao cần đến learned SE [2][3].
4. **Viết script đánh giá kèm stage này.** Dùng VIVOS/FLEURS-vi + MUSAN ở 20/10/5/0 dB, đo WER bằng jiwer, DNSMOS, latency/RTF trên chính máy Mac này. Script này tạo ra kết quả chính của báo cáo và quyết định giá trị mặc định.
5. **Lập luận mạnh nhất chống lại lựa chọn trên:** đáp án rẻ và đúng nhất có thể là "không enhancement". Dưới ~10 dB, Whisper xuống cấp chậm hơn các model cũ [25], và mọi nghiên cứu về Whisper tìm được đều thấy SE gây hại. Phương án phòng hờ chính là khuyến nghị 1: nút bật/tắt cộng với ω = 1 giúp lựa chọn này hoàn tác được hoàn toàn.

## Câu hỏi còn mở

- **Mức SNR mà SE bắt đầu giúp Whisper:** chưa có lời giải. Có hai hướng: thí nghiệm riêng của project (khuyến nghị 4), hoặc một lượt Deepen đọc Mawalim và cộng sự 2024 [38] khi có công cụ đọc PDF.
- **OA áp dụng riêng cho Whisper:** chưa có lời giải; đi theo cùng hai hướng trên.
- **RTF của GTCRN/RNNoise trên Apple Silicon:** tự đo trên máy.
- **License của weights GTCRN** (tách khỏi MIT của repo): hỏi trực tiếp nhóm tác giả.
- **SOTA lightweight SE mới (≤ 6 tháng),** như UL-UNAS: chưa có kết luận; hướng xử lý là Deepen D1.

## Danh sách nguồn

| # | Hỗ trợ cho | Nguồn | Ngày xuất bản | Ngày truy cập | Confidence |
|---|---|---|---|---|---|
| [1] | Spectral subtraction, bước preprocessing cho recognition | [Boll, IEEE TASSP 27(2)](https://bibbase.org/network/publication/boll-suppressionofacousticnoiseinspeechusingspectralsubtraction-1979) | 1979-04 | 2026-10-28 | high |
| [2] | Estimator MMSE-STSA | [Ephraim & Malah, IEEE TASSP 32(6)](https://ieeexplore.ieee.org/abstract/document/1164453) | 1984-12 | 2026-10-28 | high |
| [3] | Estimator log-MMSE | [Ephraim & Malah, IEEE TASSP 33(2)](https://dblp.uni-trier.de/rec/journals/tsp/EphraimM85.html) | 1985-04 | 2026-10-28 | high |
| [4] | Wiener theo decision-directed a priori SNR | [Scalart & Vieira Filho, ICASSP 1996](https://dblp.uni-trier.de/rec/conf/icassp/ScalartF96.html) | 1996-05 | 2026-10-28 | high |
| [5] | noisereduce: spectral gating, mặc định non-causal, MIT | [timsainb/noisereduce (GitHub)](https://github.com/timsainb/noisereduce) | live | 2026-10-28 | high |
| [6] | Thiết kế RNNoise, 87.5K weights, chi phí CPU | [Valin, arXiv 1709.08243 (MMSP 2018)](https://arxiv.org/abs/1709.08243) | 2018-05 | 2026-10-28 | high |
| [7] | PercepNet: latency, CPU, hạng DNS 2020 | [Valin et al., arXiv 2008.04259](https://arxiv.org/abs/2008.04259) | 2020 | 2026-10-28 | medium |
| [8] | Demucs causal: RTF, PESQ, kết quả ASR WER | [Défossez et al., Interspeech 2020](https://arxiv.org/abs/2006.12847) | 2020-09 | 2026-10-28 | high / medium (ASR) |
| [9] | DeepFilterNet3: PESQ 3.17, 40 ms, RTF 0.19 | [Schröter et al., arXiv 2305.08227 (Interspeech 2023 demo)](https://arxiv.org/abs/2305.08227) | 2023-05 | 2026-10-28 | high (PESQ, tự báo cáo) / medium (gán RTF) |
| [10] | GTCRN: params (48.2K), RTF 0.07, MIT, streaming ONNX | [Xiaobin-Rong/gtcrn (GitHub)](https://github.com/Xiaobin-Rong/gtcrn) | 2026-08 | 2026-10-28 | high |
| [11] | Paper GTCRN: 23.7K params, PESQ 2.87 | [Rong et al., ICASSP 2024 (dblp)](https://dblp.org/rec/conf/icassp/RongSZHZL24) | 2024-04 | 2026-10-28 | medium |
| [12] | GRU kiểu NSNet, 2.8M params, không lookahead | [Braun & Tashev, arXiv 2008.06412 (SPECOM 2020)](https://arxiv.org/abs/2008.06412) | 2020-08 | 2026-10-28 | high |
| [13] | Tổng kết DNS Challenge ICASSP 2023 | [Dubey et al., arXiv 2303.11510](https://arxiv.org/abs/2303.11510) | 2023-03 | 2026-10-28 | high (ít hữu ích) |
| [14] | File streaming ONNX của GTCRN | [gtcrn stream/onnx_models](https://github.com/Xiaobin-Rong/gtcrn) | 2026-08 | 2026-10-28 | high |
| [15] | onnxruntime có wheel cp312 macOS arm64 | [PyPI onnxruntime](https://pypi.org/pypi/onnxruntime/json) | 2026-10-26 | 2026-10-28 | high |
| [16] | pyrnnoise: wheel chỉ cho macOS 26, API 48 kHz | [PyPI pyrnnoise](https://pypi.org/pypi/pyrnnoise/json) | 2026-09-23 | 2026-10-28 | high |
| [17] | DeepFilterNet cần torch, không có wheel cp312, license weights còn bỏ ngỏ | [PyPI deepfilterlib](https://pypi.org/pypi/deepfilterlib/json); [issue #697](https://github.com/Rikorose/DeepFilterNet/issues/697); [README](https://github.com/Rikorose/DeepFilterNet/blob/main/README.md) | 2023-08 / 2026-10 | 2026-10-28 | high |
| [18] | NSNet2 ONNX baseline, CC-BY-4.0 | [microsoft/DNS-Challenge](https://github.com/microsoft/DNS-Challenge/blob/master/README-DNS3.md) | 2024-07 | 2026-10-28 | medium |
| [19] | Demucs denoiser: CC-BY-NC-4.0, archived | [PyPI denoiser](https://pypi.org/pypi/denoiser/json) | 2022-06 | 2026-10-28 | high |
| [20] | noisereduce 3.0.3: MIT, API xử lý cả mảng | [PyPI noisereduce](https://pypi.org/pypi/noisereduce/json) | 2024-10 | 2026-10-28 | high |
| [21] | webrtc-noise-gain: không có wheel macOS | [PyPI webrtc-noise-gain](https://pypi.org/pypi/webrtc-noise-gain/json) | 2026-05 | 2026-10-28 | high |
| [22] | RTF trên Apple của các bản port DFN3 cộng đồng | [HF aufklarer/DeepFilterNet3-MLX](https://huggingface.co/aufklarer/DeepFilterNet3-MLX); [HF aufklarer/DeepFilterNet3-CoreML](https://huggingface.co/aufklarer/DeepFilterNet3-CoreML) | 2026 | 2026-10-28 | low |
| [23] | Artifacts làm ASR kém đi; observation adding | [Iwamoto et al., Interspeech 2022](https://www.isca-archive.org/interspeech_2022/iwamoto22_interspeech.html) | 2022 | 2026-10-28 | high |
| [24] | Công thức OA, ω tốt nhất ≈0.4, SE dùng SNR loss tệ hơn không dùng | [Ochiai et al., arXiv 2404.14860](https://arxiv.org/abs/2404.14860) | 2024-04 | 2026-10-28 | medium |
| [25] | Độ chịu nhiễu của Whisper; WER/BLEU tiếng Việt | [Radford et al., arXiv 2212.04356](https://arxiv.org/abs/2212.04356) | 2022-12 | 2026-10-28 | high |
| [26] | MetricGAN+ làm tăng Whisper semWER ở 40/40 cấu hình | [arXiv 2512.17562 (preprint)](https://arxiv.org/abs/2512.17562) | 2025-12 | 2026-10-28 | medium |
| [27] | SAM-Audio làm tăng WER ở mọi kích thước Whisper | [arXiv 2603.04710](https://arxiv.org/abs/2603.04710) | 2026-03 | 2026-10-28 | medium-low |
| [28] | CMGAN làm tăng Whisper WER; bridge module khắc phục | [arXiv 2406.12699](https://arxiv.org/abs/2406.12699) | 2024-06 | 2026-10-28 | low-medium |
| [29] | Denoising có chọn lọc thu hẹp chênh lệch WER theo giới tính (Whisper, NeMo) | [arXiv 2410.16712 (IEEE ICKG 2024)](https://arxiv.org/abs/2410.16712) | 2024-10 | 2026-10-28 | medium |
| [30] | DNSMOS P.835: non-intrusive, Pearson 0.94–0.98 | [Reddy et al., arXiv 2110.01763](https://arxiv.org/abs/2110.01763) | 2021-10 | 2026-10-28 | high |
| [31] | Gói `pesq`: MIT, P.862.2 wb | [PyPI pesq](https://pypi.org/project/pesq/) | n/a | 2026-10-28 | high |
| [32] | VIVOS: 760 câu test, CC BY-NC-SA 4.0 | [HF vivos card](https://huggingface.co/datasets/vivos/blob/main/README.md) | n/a | 2026-10-28 | medium-high |
| [33] | Định nghĩa SI-SDR; TorchMetrics cần torch | [Le Roux et al., arXiv 1811.02508](https://arxiv.org/abs/1811.02508); [TorchMetrics docs](https://lightning.ai/docs/torchmetrics/stable/audio/scale_invariant_signal_distortion_ratio.html) | 2018-11 | 2026-10-28 | high / medium |
| [34] | Trích dẫn STOI; pystoi MIT | [mpariente/pystoi](https://github.com/mpariente/pystoi) | n/a | 2026-10-28 | high |
| [35] | jiwer 4.0: Apache-2.0, không cần torch | [PyPI jiwer](https://pypi.org/project/jiwer/) | 2025-06 | 2026-10-28 | high |
| [36] | FLEURS: CC BY 4.0, 16 kHz, vi_vn | [HF google/fleurs](https://huggingface.co/datasets/google/fleurs) | n/a | 2026-10-28 | high |
| [37] | MUSAN: CC BY 4.0, license riêng theo thư mục | [OpenSLR 17](https://openslr.org/17/) | 2015-10 | 2026-10-28 | medium |
| [38] | Các phương pháp SE ở Interspeech 2024, gồm DFN3 kèm WER (lead) | [Mawalim et al., Interspeech 2024](https://www.isca-archive.org/interspeech_2024/mawalim24_interspeech.pdf) | 2024 | 2026-10-28 | low |

## Staleness map

Tính bằng `recon_kit.py staleness` theo ngày xuất bản. Thời hạn: version 1 tháng, ecosystem 6 tháng, ASR-impact/SOTA 6 tháng (mốc của ML), benchmark number 12 tháng.

| Kết luận | Loại | Xuất bản | Kiểm tra lại trước | Đã cũ theo ngày xuất bản |
|---|---|---|---|---|
| onnxruntime có wheel cp312 arm64 | version | 2026-10 | 2026-11-01 | không |
| pyrnnoise chỉ có wheel macOS 26 | version | 2026-09 | 2026-10-01 | có* |
| DeepFilterLib không có wheel cp312 / cần torch | version | 2023-08 | 2023-09-01 | có* |
| GTCRN được bảo trì, RTF 0.07, MIT | ecosystem | 2026-08 | 2027-02-01 | không |
| SE làm tăng Whisper WER (3 preprint) | asr-impact | 2026-03 | 2026-09-01 | có |
| Chưa có kết luận SOTA cho low-complexity SE | sota | 2025-03 | 2025-09-01 | có |
| DeepFilterNet3 PESQ 3.17 | benchmark-number | 2023-05 | 2024-05-01 | có (tự báo cáo, chưa bị nguồn nào tìm được vượt qua) |
| Observation adding ω≈0.4 | asr-impact | 2024-04 | 2024-10-01 | có |

\* Các kết luận về version/availability đã được kiểm tra trực tiếp trên PyPI ngày 2026-10-28, nên vẫn đúng tính tới ngày đó. Hãy kiểm tra lại trước khi build (≤ 1 tháng, tức trước 2026-11-28). Mốc sớm nhất là trạng thái đóng gói của DeepFilterNet; việc "không có bản phát hành mới từ 2023" tự nó đã là một phát hiện. Các kết luận thuộc nhóm ML nhanh chóng vượt mốc 6 tháng, nên chạy lại bằng Refresh trước khi viết báo cáo cuối nếu viết sau 2027-04.
