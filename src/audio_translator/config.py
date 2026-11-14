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
    translator: str = "offline"  # offline (Argos) | google | a translate.MT_MODELS name (nllb-600m, m2m100-418m)
    # M4b casing/punctuation restoration of the transcript before MT (restore.APP_RESTORERS):
    # auto (a converted model for the source language if there is one, else nothing) | off |
    # sent (capital + final period) | vibert-capu (converted once, Vietnamese only).
    restore: str = "auto"
    translation_cache_dir: str = "~/.cache/audio-translator/argos"
    argos_vi_en_url: str = "https://argos-net.com/v1/translate-vi_en-1_9.argosmodel"

    # M2 capture
    capture_rate: int = 16000
    frame_ms: int = 30

    # M3 segmentation
    # Silero is the default since the guard bench (results/guard_vivos_*.csv): 75-100% fewer
    # false triggers on noise than webrtcvad. "webrtc" | "silero" (v6 ONNX shipped with faster-whisper)
    vad_backend: str = "silero"
    vad_aggressiveness: int = 2  # 0..3, webrtc only
    silero_threshold: float = 0.5  # speech probability to start; ends below threshold - 0.15
    pre_roll_ms: int = 300
    silence_ms: int = 600
    max_segment_s: float = 15.0
    # M3a noise suppression (experimental, off by default: naive enhancement
    # tends to raise Whisper WER; measure with eval/denoise_bench.py first).
    ns_mode: str = "off"  # "off" | "logmmse"
    ns_mix: float = 0.4  # observation adding ω: 0 = fully denoised, 1 = raw
    ns_gate_snr_db: float | None = 20.0  # bypass above this SNR (+3 dB hysteresis); None = always enhance
    ns_vad_input: str = "denoised"  # "denoised" | "raw": what webrtcvad sees

    # M4 ASR
    # PhoWhisper-small: FLEURS WER 16.7% vs 27.3% for Whisper small (results/asr_fleurs_demand.csv).
    # Converted once locally (scripts/convert_phowhisper.py); when it is missing the
    # session warns and runs asr_fallback_model instead.
    model_size: str = "phowhisper-small"
    asr_fallback_model: str | None = "small"
    asr_device: str = "cpu"
    compute_type: str = "int8"
    beam_size: int = 5
    # Whisper initial_prompt (decoder priming text), e.g. a Vietnamese sentence
    # with full diacritics. None = no prompt; measure with eval/asr_bench.py.
    asr_initial_prompt: str | None = None
    # Hallucination filter per Whisper segment (on by default since the guard bench):
    # drop it when no_speech_prob > max, OR avg_logprob < min, OR
    # compression_ratio > max (Whisper's own defaults; faster-whisper itself
    # only drops when no_speech AND logprob both fail), or when it matches the
    # Vietnamese blacklist (hallucinations_vi.txt). Decoding loops in kept
    # segments are written once (asr.collapse_repeats).
    asr_filter: bool = True
    asr_max_no_speech_prob: float = 0.6
    asr_min_avg_logprob: float = -1.0
    asr_max_compression_ratio: float = 2.4
    # Drop the stand-alone word "unk" PhoWhisper sometimes writes at an utterance start (3 of the
    # real-test segments, 3/1166 bench segments; never in the references; WER never worse).
    asr_drop_unk: bool = True
    # Bounded decoding against stalls (opt-in; None = faster-whisper's own decoding, unchanged).
    # A rejected attempt on unclear audio runs to 224 tokens (~6-7 s) and up to 6 temperatures
    # are tried: one 3 s music clip took 22 s (plan-speaker-indicator.md). Being measured with
    # eval/asr_bench.py (--max-tokens-per-s, --temperatures); default stays off until decided.
    asr_max_tokens_per_s: float | None = None  # max_new_tokens = ceil(audio_s x this) + base (<= 224)
    asr_max_tokens_base: int = 32  # bench choice with 15 tok/s (plan-speaker-indicator.md)
    asr_temperatures: tuple[float, ...] | None = None  # e.g. (0.0, 0.2, 0.4); None = 0.0..1.0 step 0.2
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

    # Speaker indicator (health.py; observation only, the pipeline is unchanged).
    # Thresholds from the real logs and the stall probe (plan-speaker-indicator.md):
    # length: spoken sentences so far P90 6.6 s, max 7.4 s -> yellow at 8 s; red at 12 s
    #   leaves 3 s before the forced cut at max_segment_s (15 s), which can cut a word.
    indicator_long_warn_s: float = 8.0
    indicator_long_stop_s: float = 12.0
    # backlog: segments waiting for ASR, or ASR time of the current segment (real ASR P50
    # ~1.2 s, Meet P95 1.8 s; every decode > 5.7 s in the probe was a stall).
    indicator_queue_warn: int = 1
    indicator_queue_stop: int = 2
    indicator_asr_warn_s: float = 3.0
    indicator_asr_stop_s: float = 6.0
    # playout: how long the newest ready translation waits behind the one playing (real log
    # 2026-11-14 13:58: long sentences made the next ones wait 4.1, 6.4 and 7.8 s; 0 otherwise).
    indicator_playout_warn_s: float = 4.0
    indicator_playout_stop_s: float = 8.0
    # quality: last segment filtered, or ASR >= min_asr_s AND >= ratio x audio length
    # (normal real segments < 0.71), or a temperature fallback was needed; yellow for hold_s.
    indicator_unclear_min_asr_s: float = 1.5
    indicator_unclear_ratio: float = 1.5
    indicator_fallback: bool = True
    indicator_unclear_hold_s: float = 4.0
    indicator_down_hold_s: float = 1.0  # a lower level shows only after holding this long
    indicator_rate_hz: float = 10.0  # max evaluations per second (monitor thread)

    # Per-segment latency CSV (latency_log.py), appended from worker threads;
    # None = off (nothing is opened or written).
    latency_log: str | None = None

    @property
    def frame_samples(self) -> int:
        return self.capture_rate * self.frame_ms // 1000
