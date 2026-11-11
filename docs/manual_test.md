# Kịch bản thử thật: Việt → Anh

Ba lần thử với micro thật: (a) CLI không gọi, (b) qua Google Meet với điện thoại, (c) môi trường ồn. Mỗi lần thử ghi một file độ trễ (`--latency-log`) và một bảng kết quả điền tay. Làm xong cả ba, gửi lại các file ở mục [Gửi lại](#gửi-lại).

> ⚠️ **Luôn đeo tai nghe.** Nếu loa Mac phát ra tiếng (của Meet hay của bất kỳ ứng dụng nào), micro sẽ thu lại và app sẽ dịch chính tiếng đó. Đây là vòng lặp âm thanh đã gặp trước đây (README mục 8). Trong kịch bản (b), điện thoại nghe cũng phải để xa Mac hoặc đeo tai nghe.

## Chuẩn bị (một lần)

```sh
cd ~/GR2
git switch main && git pull
uv sync
mkdir -p ~/gr2_manual                                    # file đo để NGOÀI repo (có chữ bạn nói)
uv run audio-translator --cli --list-devices             # phải thấy "BlackHole 2ch" ở output
uv run audio-translator --cli --list-models              # phowhisper-small: "downloaded" (đã convert)
```

- Cấu hình mặc định của app hiện nay:
  - ASR: `phowhisper-small`, dùng tạm `small` nếu chưa convert;
  - VAD: Silero;
  - bộ lọc câu ma: bật;
  - khôi phục dấu câu: `vibert-capu` (`--restore auto`);
  - dịch: Argos offline, dự phòng Google;
  - TTS: Edge (cần internet).
- Nếu app báo *"Model 'phowhisper-small' chưa được convert, đang dùng tạm 'small'"*, hãy dừng lại và convert trước (lệnh nằm ngay trong thông báo). Nếu không, số đo sẽ là của `small`. Cột `asr_model` trong file log luôn ghi đúng model đã thật sự chạy.
- Lần chạy đầu có thể mất thêm vài giây để nạp model. Câu đầu tiên của mỗi lần thử nên là một câu chào ngắn để "khởi động"; câu này không tính vào kết quả.
- Mỗi kịch bản chạy trong **một phiên** (Start một lần, Stop một lần), để `session_id` trong file log không bị lẫn.

## Cách đọc bảng độ trễ

```sh
uv run python eval/latency_report.py ~/gr2_manual/latency_cli.csv
uv run python eval/latency_report.py ~/gr2_manual/latency_cli.csv --markdown   # dạng bảng để dán
```

Script in số đoạn đã phát và đã bỏ, cấu hình đã chạy, rồi in bảng TB / P50 / P95 / Max (ms) cho từng khâu:

| Cột | Nghĩa |
|---|---|
| `segment_ms` | từ lúc hết nói tới lúc cắt đoạn, **gồm cả 600 ms chờ im lặng** để biết câu đã kết thúc |
| `asr_wait_ms`, `mt_wait_ms`, `tts_wait_ms` | thời gian chờ trong hàng đợi (lớn ⇒ khâu trước bị nghẽn) |
| `asr_ms` | nhận dạng (PhoWhisper) |
| `restore_ms` | khôi phục chữ hoa và dấu câu |
| `mt_ms` | dịch |
| `tts_ms` | tổng hợp giọng nói (Edge, qua mạng) |
| `playout_wait_ms` | từ lúc có âm thanh tới lúc âm thanh bắt đầu phát vào BlackHole |
| **`e2e_ms`** | **độ trễ đầu cuối: âm thanh dịch bắt đầu phát − lúc bạn ngừng nói** |

Cuối bảng là các đoạn bị bỏ, kèm lý do:
- `no_speech`: bộ lọc chặn, ví dụ câu ma; cột `detail` của file CSV cho biết bị chặn vì `blacklist`, `no_speech_prob`…
- `mt_error`, `tts_error`: lỗi dịch, lỗi TTS (thường do mạng).
- `overload`, `playout_*`: quá tải hoặc phát trễ.

---

## (a) CLI, không gọi, phòng yên tĩnh

Mục đích: số đo cơ sở, khi không có Chrome hay Meet tranh CPU.

```sh
uv run audio-translator --cli --output "BlackHole 2ch" --latency-log ~/gr2_manual/latency_cli.csv
```

- Muốn **nghe** bản dịch: mở QuickTime Player → *File → New Audio Recording* → bấm mũi tên cạnh nút ghi âm và chọn nguồn **BlackHole 2ch** → bấm ghi. Cuối buổi phát lại bằng tai nghe.
- Terminal in mỗi câu dạng `[n] VI: … | EN: … | latency …`.
- Đọc từng câu dưới đây, ngắt khoảng 2 giây giữa các câu, nói tự nhiên ở khoảng cách bình thường tới micro.

**10 câu (dùng lại cho kịch bản c):**
1. Xin chào, hôm nay bạn có khỏe không?
2. Tôi là sinh viên năm ba của Đại học Bách khoa Hà Nội.
3. Cuộc họp sẽ bắt đầu lúc ba giờ chiều thứ Năm.
4. Ngày mai trời nắng hay mưa?
5. Giá vé máy bay từ Hà Nội vào Thành phố Hồ Chí Minh là một triệu hai trăm nghìn đồng.
6. Em đã gửi báo cáo qua email cho thầy Nguyễn Văn An rồi ạ.
7. Chúng tôi cần thêm hai tuần để hoàn thành phần kiểm thử.
8. Bạn có thể nói chậm lại một chút được không?
9. Công ty Viettel vừa mở văn phòng mới ở Đà Nẵng vào tháng Mười năm hai nghìn hai mươi lăm.
10. Cảm ơn mọi người đã lắng nghe, hẹn gặp lại vào tuần sau.

Bấm `Ctrl+C` để dừng, rồi chạy `uv run python eval/latency_report.py ~/gr2_manual/latency_cli.csv`.

## (b) Google Meet, người nghe là điện thoại

Mục đích: độ trễ và độ dễ hiểu khi gọi thật, có Chrome chạy song song. Chạy **bằng giao diện**, ghi log qua dòng lệnh:

```sh
uv run audio-translator --latency-log ~/gr2_manual/latency_meet.csv
```

1. Trong GUI chọn Input là micro của Mac (hoặc micro tai nghe) và Output là **BlackHole 2ch**, nhưng **chưa bấm Start**.
2. Trên Mac, vào Meet bằng Chrome. Trong *Settings → Audio* của Meet:
   - **Microphone = BlackHole 2ch**;
   - Speakers = tai nghe;
   - tắt *Noise cancellation* nếu có, vì nó có thể cắt mất giọng TTS.
3. Trên điện thoại, vào cùng cuộc họp, **tắt micro** của điện thoại và đeo tai nghe (hoặc để ở phòng khác).
4. Bấm **Start** trong GUI. Nếu thấy dòng đỏ báo *"…chưa được convert…"* thì dừng lại và xem mục Chuẩn bị.
5. Đọc 10 câu dưới đây. Người cầm điện thoại ghi lại câu tiếng Anh nghe được, hoặc bạn tự ghi âm màn hình điện thoại.
6. Bấm **Stop** rồi đóng cửa sổ, sau đó chạy `uv run python eval/latency_report.py ~/gr2_manual/latency_meet.csv`.

**10 câu (kiểu hội thoại khi gọi):**
1. Alo, bạn có nghe rõ tôi nói không?
2. Hôm nay mình họp nhanh khoảng mười lăm phút thôi nhé.
3. Tuần trước nhóm mình đã xong phần nhận dạng giọng nói.
4. Bây giờ tôi sẽ chia sẻ màn hình để mọi người cùng xem.
5. Độ trễ hiện tại khoảng hai đến ba giây mỗi câu.
6. Anh Minh ơi, anh gửi giúp em file thiết kế được không?
7. Hạn nộp báo cáo là ngày hai mươi tháng Mười Hai.
8. Nếu có câu hỏi thì mọi người cứ nhắn vào khung chat.
9. Tôi nghĩ chúng ta nên thử thêm với môi trường ồn.
10. Vậy thôi, cảm ơn mọi người, chào nhé.

## (c) Môi trường ồn

Mục đích: xem bộ chống câu ma (Silero và bộ lọc) có chặn được nhiễu thật không, và nhiễu làm câu nói bị nhận sai đến đâu.

```sh
uv run audio-translator --cli --output "BlackHole 2ch" --latency-log ~/gr2_manual/latency_noisy.csv
```

1. Tạo nhiễu từ một thiết bị khác, đặt cách micro khoảng 1 m:
   - một video "tiếng quán cà phê" (café ambience) trên YouTube ở mức nghe rõ nhưng vẫn nói chuyện được; hoặc
   - quạt hay máy lọc không khí bật số to.

   Ghi lại bạn đã dùng nguồn nhiễu nào.
2. **60 giây đầu: không nói gì.** Đếm xem app có phát ra câu nào không. Mỗi câu phát ra lúc này là một câu ma lọt qua. Các đoạn bị chặn sẽ hiện trong log với lý do `no_speech`.
3. Sau đó đọc lại đúng **10 câu của kịch bản (a)**, giữ nguyên nhiễu.
4. `Ctrl+C`, rồi chạy `uv run python eval/latency_report.py ~/gr2_manual/latency_noisy.csv`.

---

## Bảng kết quả (chép thành `~/gr2_manual/ket_qua.md` và điền)

```markdown
# Kết quả thử thật – <ngày>
Máy: <MacBook ...>, mạng: <wifi ...>, micro: <micro Mac / tai nghe ...>
Nguồn nhiễu ở (c): <...>

## (a) CLI
| # | Câu đã nói | VI app nhận (terminal) | EN nghe được | Đúng nghĩa? (Đ / 1 phần / S) | Ghi chú |
|---|---|---|---|---|---|
| 1 | Xin chào, hôm nay bạn có khỏe không? | | | | |
| … | | | | | |

## (b) Meet
| # | Câu đã nói | EN điện thoại nghe được | Đúng nghĩa? | Nghe có rõ không? | Ghi chú |
|---|---|---|---|---|---|

## (c) Ồn
- 60 giây im lặng: số câu bị phát ra = <n>; nội dung: <...>
| # | Câu đã nói | VI app nhận | EN nghe được | Đúng nghĩa? | Ghi chú |
|---|---|---|---|---|---|

## Nhận xét chung
- Câu bị lặp hai lần (vòng lặp âm thanh)? <có/không>
- Cảm nhận độ trễ khi gọi: <...>
- Lỗi hoặc cảnh báo lạ trên màn hình: <...>
```

## Gửi lại

1. `~/gr2_manual/latency_cli.csv`
2. `~/gr2_manual/latency_meet.csv`
3. `~/gr2_manual/latency_noisy.csv`
4. `~/gr2_manual/ket_qua.md` (bảng đã điền)

Các file CSV chứa nguyên văn chữ bạn nói, nên **không commit chúng vào repo**. Gửi cho tôi đường dẫn là được; tôi sẽ đọc, phân tích và đưa các số đã tổng hợp vào `results/` và README.
