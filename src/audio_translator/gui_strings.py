"""Every string the GUI shows, in one place (Vietnamese), so a UI language can be added later.

The orchestrator's events stay in English (they also feed the CLI and logs); the GUI maps
them to Vietnamese here and keeps the original text for the technical log and tooltips.
"""

from __future__ import annotations

import re

from .models import IDLE, RUNNING, STARTING, STOPPED, STOPPING

VI = {
    "window_title": "Phiên dịch giọng nói",
    "compact_title": "Phiên dịch – thu gọn",
    "lang_pair": "Tiếng Việt → Tiếng Anh",
    "start": "▶ Bắt đầu",
    "stop": "■ Dừng",
    "compact": "Thu gọn",
    "expand": "Mở rộng",
    "details": "Chi tiết",
    "close_banner": "×",
    "close_banner_tip": "Ẩn thông báo",
    # light
    "light_off": "Chưa chạy",
    "light_others": "Cũng: {}",
    "level_green": "XANH",
    "level_yellow": "VÀNG",
    "level_red": "ĐỎ",
    # translation pane
    "transcript_title": "Bản dịch",
    "copy_all": "Sao chép tất cả",
    "clear": "Xóa",
    "no_latency": "–",
    "stats": "Câu gần nhất: {last} · Trung vị phiên: {median} · Số câu: {n}",
    "compact_no_sentence": "Chưa có câu nào",
    # settings
    "settings": "Cài đặt",
    "settings_locked": "Đang chạy: dừng lại để đổi cài đặt (có hiệu lực từ lần Bắt đầu tiếp theo).",
    "mic": "Micro:",
    "output": "Đầu ra (micro ảo):",
    "output_help": "Trong Zoom/Meet, chọn Microphone = BlackHole 2ch",
    "model": "Model nhận dạng:",
    "translator": "Bộ dịch:",
    "noise": "Lọc nhiễu:",
    "default_input": "(Mặc định của hệ thống)",
    "no_output": "(chọn thiết bị đầu ra)",
    "fixed_model": "(cố định)",
    # technical log
    "tech_log": "Nhật ký kỹ thuật",
    # banners
    "no_blackhole": "Không tìm thấy BlackHole – cài bằng: brew install blackhole-2ch",
    "choose_output": "Chưa chọn đầu ra – mở Cài đặt và chọn BlackHole 2ch",
    "model_fallback": ("Model '{model}' chưa được convert, đang dùng tạm '{fallback}'. "
                       "Convert một lần bằng: {command}"),
    "mt_google": "Đang dịch bằng Google (cần mạng) vì {translator} chưa sẵn sàng",
    # status line (model work shown while stopped, and start-up progress)
    "model_loading": "Đang tải model '{model}'…",
    "model_downloading": "Đang tải về model '{model}' ({size})…",
    "model_selected": "Đã chọn model '{model}'",
    "asr_loading": "Đang tải model nhận dạng…",
    "mt_downloading": "Đang tải model dịch (~66 MB)…",
}

STATE_TEXT = {
    IDLE: "Đã dừng",
    STOPPED: "Đã dừng",
    STARTING: "Đang khởi động…",
    RUNNING: "Đang chạy",
    STOPPING: "Đang dừng…",
}

# Translator dropdown: (Settings.translator value, label). Google needs the network.
TRANSLATORS = [
    ("offline", "Offline (Argos)"),
    ("nllb-600m", "NLLB-600M (offline, cần convert một lần)"),
    ("m2m100-418m", "M2M100-418M (offline, cần convert một lần)"),
    ("google", "Google (cần mạng)"),
]
# Short names for messages ("… vì Argos chưa sẵn sàng").
TRANSLATOR_SHORT = {"offline": "Argos", "nllb-600m": "NLLB-600M", "m2m100-418m": "M2M100-418M", "google": "Google"}
# Noise suppression dropdown: (Settings.ns_mode value, label).
NOISE = [("off", "Tắt"), ("logmmse", "logMMSE (thử nghiệm)")]

MODEL_STATUS = {"downloaded": "đã tải", "loading": "đang tải…", "loaded": "trong RAM", "failed": "lỗi"}
MODEL_NOT_CONVERTED = "chưa convert (xem --list-models)"
MODEL_NOT_DOWNLOADED = "chưa tải (~{size})"

# Orchestrator ERROR text -> Vietnamese headline + short hint (first match wins). "{name}" is the
# first quoted name of the original message (a device or model name), when there is one.
ERROR_PATTERNS = [
    (r"not found", "Không tìm thấy thiết bị '{name}' – kiểm tra đã cắm/cài và chọn lại trong Cài đặt"),
    (r"stopped (responding|unexpectedly)|removed",
     "Thiết bị '{name}' ngừng hoạt động – cắm lại rồi bấm Bắt đầu"),
    (r"Cannot open (input|output) device", "Không mở được thiết bị '{name}' – chọn lại trong Cài đặt"),
    (r"Could not load model", "Không tải được model '{name}' – vẫn dùng model đang chọn"),
    (r"Could not start", "Không khởi động được – kiểm tra BlackHole đã cài và chọn lại Đầu ra trong Cài đặt"),
]
ERROR_FALLBACK = "Đã xảy ra lỗi – bấm Chi tiết để xem"


def error_headline(message: str) -> str:
    """Vietnamese banner text for an orchestrator error; never the English original."""
    m = re.search(r"'([^']+)'", message)
    name = m.group(1) if m else ""
    for pattern, text in ERROR_PATTERNS:
        if re.search(pattern, message, re.IGNORECASE):
            if "{name}" in text and not name:
                text = text.replace(" '{name}'", "")
            return text.format(name=name)
    return ERROR_FALLBACK


def is_google_fallback(message: str) -> bool:
    """The orchestrator's WARNING that MT switched to Google (whole session or one segment)."""
    return "using Google Translate" in message


def status_text(state: str | None, message: str) -> str:
    """Status line for a STATUS event: start-up progress, else the state word."""
    if state == STARTING and message.startswith("Loading ASR model"):
        return VI["asr_loading"]
    if state == STARTING and message.startswith("Downloading translation model"):
        return VI["mt_downloading"]
    return STATE_TEXT.get(state or "", STATE_TEXT[STOPPED])


def model_event_text(model: str, detail: str) -> str | None:
    """Status line for a MODEL event shown while stopped (None: nothing to show)."""
    if detail == "selected":
        return VI["model_selected"].format(model=model)
    m = re.match(r"downloading \(([^)]*)\)", detail)
    if m:
        return VI["model_downloading"].format(model=model, size=m.group(1))
    if detail.startswith("loading"):
        return VI["model_loading"].format(model=model)
    return None


def fmt_seconds(x: float | None) -> str:
    """``3,0 s`` (Vietnamese decimal comma); ``–`` when unknown."""
    return VI["no_latency"] if x is None else f"{x:.1f}".replace(".", ",") + " s"
