"""Single source of configuration."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Settings:
    # Devices: substrings matched case-insensitively against device names.
    # input_device=None means the system default input. The output device has
    # no "default" fallback on purpose: translated audio must never go to the
    # user's speakers.
    input_device: str | None = None
    output_device: str = "BlackHole"

    # Languages (MVP: VI -> EN only).
    source_lang: str = "vi"
    target_lang: str = "en"

    # M5 MT: "offline" (Argos vi->en model via CTranslate2, Google as fallback)
    # or "google" (Google Translate only, network).
    translator: str = "offline"
    translation_cache_dir: str = "~/.cache/audio-translator/argos"
    argos_vi_en_url: str = "https://argos-net.com/v1/translate-vi_en-1_9.argosmodel"

    # M2 capture
    capture_rate: int = 16000
    frame_ms: int = 30

    # M3 segmentation
    vad_aggressiveness: int = 2  # 0..3
    pre_roll_ms: int = 300
    silence_ms: int = 600
    max_segment_s: float = 15.0

    # M4 ASR
    model_size: str = "medium"
    asr_device: str = "cpu"
    compute_type: str = "int8"
    beam_size: int = 5
    # Whisper models kept resident (LRU). ~2 fit in 8 GB RAM with int8 CPU models.
    max_loaded_models: int = 2

    # M6 TTS
    tts_voice: str = "en-US-AriaNeural"

    # Queues (every queue is bounded)
    frame_queue_size: int = 1000  # ~30 s of 30 ms frames
    segment_queue_size: int = 8
    transcript_queue_size: int = 8
    translation_queue_size: int = 8
    playout_max_pending: int = 8
    event_queue_size: int = 256

    # M7 playout
    gap_timeout_s: float = 5.0
    output_blocksize: int = 0  # 0 = let PortAudio choose

    # Device liveness: a stream silent for this long is treated as removed.
    device_watchdog_s: float = 2.0

    @property
    def frame_samples(self) -> int:
        return self.capture_rate * self.frame_ms // 1000
