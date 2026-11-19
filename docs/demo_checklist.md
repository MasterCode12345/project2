# Checklist buổi demo

Dùng cho buổi demo trước giảng viên: chuẩn bị, kịch bản khoảng 5 phút, và cách xử lý khi có sự cố. Mọi lệnh chạy trong thư mục repo (`~/GR2`) và dán thẳng được vào zsh, vì không có chú thích `#` trong khối lệnh. Cài đặt từ đầu: mục 4 của [README.md](../README.md).

Cách bố trí: Mac chạy app và Google Meet. Một **điện thoại** vào cùng cuộc họp, tắt micro, bật loa, để mọi người nghe được giọng tiếng Anh.

## A. Trước buổi demo

**Hôm trước (hoặc 30 phút trước):**

1. Cập nhật `main` và thư viện:
   ```sh
   cd ~/GR2
   git switch main
   git pull
   uv sync
   ```
2. Chạy test. Kết quả phải không có `failed`; con số cỡ 467 passed.
   ```sh
   uv run --group eval pytest -q
   ```
3. Kiểm tra model: `phowhisper-small` phải ghi `downloaded`. Nếu không, xem mục C4.
   ```sh
   uv run audio-translator --cli --list-models
   ```
4. Kiểm tra BlackHole: phải có **BlackHole 2ch** ở cả Input lẫn Output. Nếu không, khởi động lại máy; vẫn không có thì cài lại bằng `brew install blackhole-2ch`.
   ```sh
   uv run audio-translator --cli --list-devices
   ```
5. Kiểm tra mạng cho giọng đọc. Edge TTS cần mạng ở **mỗi câu**; nhận dạng và dịch thì chạy offline. Lệnh phải in `TTS OK: … s of audio`.
   ```sh
   uv run python -c "from audio_translator.tts import EdgeTTS; a, r = EdgeTTS('en-US-AriaNeural').synthesize('Hello, this is a test.'); print('TTS OK:', round(len(a) / r, 1), 's of audio')"
   ```
6. Chuẩn bị mạng dự phòng: bật sẵn điểm phát Wi-Fi trên điện thoại, và đã thử kết nối Mac vào đó một lần.

**Ngay trước buổi demo:**

7. **Tai nghe:** cắm và nghe thử. Dùng tai nghe để micro không thu lại tiếng người bên kia, nếu không một câu có thể bị dịch hai lần.
8. **macOS:** Cài đặt hệ thống → Âm thanh → Đầu vào phải là micro thật, **không phải BlackHole**.
9. **Đóng ứng dụng nặng**: Docker, Xcode, IDE lớn, trình duyệt nhiều tab; chỉ giữ Chrome với Meet. Cắm sạc và tắt Chế độ nguồn điện thấp: nhận dạng chạy trên CPU, máy hạ xung là chậm ngay.
10. Bật **Không làm phiền** để thông báo không hiện lên giữa buổi.
11. **Mở app với tên log có ngày giờ**, để không ghi đè log cũ:
    ```sh
    mkdir -p ~/gr2_manual
    LOG=~/gr2_manual/demo_$(date +%Y%m%d_%H%M).csv
    echo "$LOG"
    uv run audio-translator --latency-log "$LOG"
    ```
12. Trong app → **Cài đặt**:
    - Micro: micro thật (của Mac hoặc của tai nghe);
    - Đầu ra: **BlackHole 2ch**;
    - Model: `phowhisper-small`;
    - Bộ dịch: offline;
    - ô **Chống kẹt** đang tắt.
13. **Khởi động thử một lần:** bấm **▶ Bắt đầu**, nói "Xin chào", chờ có dòng tiếng Anh, rồi bấm **■ Dừng**. Lần đầu, app tải model dịch Argos (khoảng 66 MB) và nạp model nhận dạng; làm trước thì lúc demo không phải chờ.
14. **Meet** (Chrome, cửa sổ thường, **không** toàn màn hình): ⋮ → Cài đặt → Âm thanh:
    - Micrô = **BlackHole 2ch**;
    - Loa = tai nghe;
    - tắt lọc tạp âm nếu có.
15. **Điện thoại** vào cùng cuộc họp, tắt micro của điện thoại, bật loa vừa đủ nghe.

## B. Kịch bản demo (khoảng 5 phút)

Mỗi câu đọc tự nhiên, nghỉ khoảng 2 giây giữa các câu. Người nghe trên điện thoại phải nghe tiếng Anh sau mỗi câu, chậm khoảng 2–3 giây.

**1. Giới thiệu (0:00–0:30).** Cửa sổ đầy đủ đang mở. Nói luồng xử lý: micro → nhận dạng tiếng Việt → khôi phục dấu câu → dịch → giọng đọc tiếng Anh → BlackHole → Meet. Bấm **▶ Bắt đầu**; đèn xanh hiện "Đang nghe, nói bình thường".

**2. Câu mẫu (0:30–1:30).** Đọc 3 câu:
- "Xin chào, hôm nay bạn có khỏe không?"
- "Cuộc họp sẽ bắt đầu lúc ba giờ chiều thứ Năm."
- "Bạn có thể nói chậm lại một chút được không?"

Chỉ vào vùng **Bản dịch**: mỗi câu có dòng tiếng Việt (chữ nhỏ), dòng tiếng Anh (chữ lớn), và độ trễ ở bên phải.

**3. Tên riêng (1:30–2:15).** Đọc:
- "Tôi là sinh viên năm ba của Đại học Bách khoa Hà Nội." Kỳ vọng: "Hanoi University of Science and Technology".
- "Tôi học ở Trường Công nghệ Thông tin và Truyền thông." Kỳ vọng: "School of Information and Communication Technology".
- "Bách khoa toàn thư rất hữu ích." Câu này **không** được đổi thành tên trường.

Giải thích: bảng tên riêng `glossary_vi_en.tsv` giữ đúng tên chính thức. Không cố đọc câu có "giảng viên hướng dẫn": bảng chỉ dành cho tên riêng, nên cụm này vẫn được dịch là "instructor".

**4. Chế độ thu gọn (2:15–3:15).**
- Bấm **Thu gọn**: cửa sổ nhỏ hiện ở góc trên bên phải. Bấm vào cửa sổ Meet; cửa sổ nhỏ vẫn nằm trên Meet.
- Đọc "Ngày mai trời nắng hay mưa?". Câu tiếng Anh mới hiện trên cửa sổ nhỏ; rê chuột lên để xem đủ câu.
- Bấm **Mở rộng** để quay về cửa sổ đầy đủ.

**5. Đèn báo (3:15–4:15).**
- Nói liền một câu dài khoảng 10 giây, ví dụ đọc liền hai câu mẫu không nghỉ. Ở khoảng 8 giây, đèn chuyển **vàng** "Câu dài, hãy ngắt nghỉ".
- Đọc nhanh 3 câu, mỗi câu chỉ nghỉ khoảng nửa giây. Đèn có thể chuyển **vàng** "Đang xử lý, nói chậm lại". Dừng nói thì đèn về xanh.
- Giải thích: đèn chỉ quan sát, nhắc người nói ngắt nghỉ hoặc nói chậm lại; nó không đổi cách app xử lý.

**6. Ô chống kẹt (4:15–5:00).**
- Bấm **■ Dừng**, mở **Cài đặt**, rê chuột lên ô **Chống kẹt nhận dạng (thử nghiệm)** để hiện tooltip.
- Giải thích đánh đổi: trong bench, lần kẹt lâu nhất giảm từ khoảng 36 s xuống khoảng 7,5 s, nhưng trong nhiễu rất nặng có thể mất vài câu, nên mặc định tắt. Pre-roll 600 ms đã giảm nhiều lần kẹt.
- Đánh dấu ô, bấm **▶ Bắt đầu** (model không phải nạp lại), đọc "Cảm ơn mọi người đã lắng nghe.", rồi bấm **■ Dừng**.

**Kết thúc (sau buổi demo, nếu có thời gian):**
```sh
uv run python eval/latency_report.py "$LOG"
```
Báo cáo in độ trễ từng khâu (`e2e_ms` là từ lúc nói xong tới lúc có tiếng Anh), và dòng `anti-stall per session: 1=off, 2=c4b`. Chạy trong cùng cửa sổ terminal để biến `$LOG` còn giá trị, hoặc gõ thẳng đường dẫn file.

## C. Xử lý khi có sự cố

### C1. Không có tiếng ra (điện thoại không nghe tiếng Anh)

1. **App có hiện dòng tiếng Anh không?**
   - **Có chữ nhưng không có tiếng:**
     - Kiểm tra Đầu ra của app là **BlackHole 2ch**.
     - Kiểm tra trong Meet: Micrô là **BlackHole 2ch**, và micro trong Meet không bị tắt.
     - Mở **Nhật ký kỹ thuật**: dòng `speech synthesis failed` nghĩa là mất mạng tới Edge TTS (xem C3).
   - **Không có cả dòng tiếng Việt:** app không nghe thấy gì.
     - Kiểm tra Micro trong Cài đặt là micro thật, không phải BlackHole.
     - Kiểm tra macOS đã cấp quyền micro: **Cài đặt hệ thống → Quyền riêng tư & Bảo mật → Micrô**, bật cho Terminal hoặc ứng dụng chạy lệnh. Bật xong thì tắt app rồi mở lại.
2. Băng đỏ "Không tìm thấy BlackHole…" hoặc "Không tìm thấy thiết bị…": kiểm tra lại bằng lệnh dưới. Nếu không có BlackHole thì khởi động lại máy, vẫn không có thì `brew install blackhole-2ch`.
   ```sh
   uv run audio-translator --cli --list-devices
   ```
3. Nghe thấy tiếng nhưng bị cắt vụn hoặc mất câu: tắt lọc tạp âm trong Meet (⋮ → Cài đặt → Âm thanh).

### C2. Nhận dạng chậm hoặc kẹt

Dấu hiệu: đèn **vàng/đỏ** "Đang xử lý, nói chậm lại" hoặc "Dừng nói, chờ hệ thống bắt kịp", và tiếng Anh tới chậm hơn 5 giây.

1. Ngừng nói vài giây cho hệ thống bắt kịp, rồi nói chậm hơn, ngắt nghỉ giữa các câu.
2. Đóng ứng dụng nặng và cắm sạc (mục A9).
3. Bật chống kẹt: **■ Dừng** → Cài đặt → đánh dấu **Chống kẹt nhận dạng (thử nghiệm)** → **▶ Bắt đầu**. Hoặc mở lại app với cờ:
   ```sh
   uv run audio-translator --asr-anti-stall c4b --latency-log "$LOG"
   ```
4. Nếu app không phản hồi: bấm **■ Dừng**. Không được thì Cmd+Q rồi mở lại app như ở mục A11, với tên log mới.

### C3. Mất mạng

- Nhận dạng và dịch (Argos) vẫn chạy offline, nhưng **giọng đọc Edge TTS cần mạng**: app vẫn hiện chữ, còn người nghe không nghe được gì. Meet cũng cần mạng.
- Cách xử lý:
  1. Chuyển Mac sang điểm phát của điện thoại (mục A6).
  2. Kiểm tra lại bằng lệnh TTS ở mục A5.
  3. Bấm **■ Dừng** rồi **▶ Bắt đầu** để mở phiên mới.
- Băng vàng "Đang dịch bằng Google (cần mạng)…" nghĩa là model dịch offline chưa sẵn sàng; thường là vì chưa từng bấm Bắt đầu khi có mạng. Có mạng lại thì Dừng → Bắt đầu, app sẽ tải model dịch.

### C4. Thiếu model

Kiểm tra trước:
```sh
uv run audio-translator --cli --list-models
```

- **Băng đỏ "Model 'phowhisper-small' chưa được convert, đang dùng tạm 'small'…":** app vẫn chạy được bằng Whisper `small`, nhưng lần đầu phải tải 485 MB và nhận dạng tiếng Việt kém hơn. Convert PhoWhisper (tải 971 MB, khoảng 1–2 phút), rồi Dừng → Bắt đầu:
  ```sh
  uv run --with transformers --with torch python scripts/convert_phowhisper.py small
  ```
- **Bản dịch không có chữ hoa và dấu câu:** chưa convert vibert-capu; app không báo lỗi gì. Lệnh convert tải 462 MB, mất khoảng 1 phút:
  ```sh
  uv run --with "transformers<5" --with torch python scripts/convert_capu_model.py vibert-capu
  ```
- **Thiếu model dịch Argos và không có mạng:** app chuyển sang Google, mà Google cũng cần mạng. Xem C3.

### C5. Lỗi khác

- **Một câu bị dịch hai lần, hoặc nghe tiếng vọng:** đeo tai nghe; trong Meet, Loa phải là tai nghe.
- **Cửa sổ thu gọn không hiện trên Meet:** Meet đang ở chế độ toàn màn hình. Thoát toàn màn hình bằng Ctrl+Cmd+F.
- **Băng đỏ khác:** bấm **Chi tiết** để mở Nhật ký kỹ thuật, chụp màn hình lại để xem sau. Bấm Dừng → Bắt đầu để thử lại.

Sau buổi demo: file log trong `~/gr2_manual/` chứa nguyên văn lời nói, nên **không đưa vào repo**.
