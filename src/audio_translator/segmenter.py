"""M3: VAD segmentation. Pure, thread-free; fed fixed 30 ms int16 frames."""

from __future__ import annotations

import collections
import math
from dataclasses import dataclass
from typing import Callable

import numpy as np

# is_speech(frame_bytes, sample_rate) -> bool  (webrtcvad.Vad.is_speech signature)
VadFn = Callable[[bytes, int], bool]


def webrtc_vad(aggressiveness: int) -> VadFn:
    import webrtcvad

    vad = webrtcvad.Vad(aggressiveness)
    return vad.is_speech


class SileroVad:
    """Streaming Silero VAD (v6 ONNX shipped with faster-whisper) behind the ``VadFn`` signature.

    Silero scores 512-sample windows (32 ms at 16 kHz) with 64 samples of
    context and a recurrent state; the segmenter feeds 480-sample frames. Each
    frame is buffered, every complete window is scored exactly once, and the
    frame gets the hysteresis state after the last complete window: speech
    starts at ``p >= threshold`` and ends at ``p < neg_threshold`` (default
    ``threshold - 0.15``, as in faster-whisper). One instance per session.
    """

    WINDOW = 512
    CONTEXT = 64

    def __init__(self, threshold: float = 0.5, neg_threshold: float | None = None, *, session=None) -> None:
        if not 0.0 < threshold < 1.0:
            raise ValueError(f"silero threshold must be in (0, 1), got {threshold}")
        if session is None:
            from faster_whisper.vad import get_vad_model

            session = get_vad_model().session
        self.session = session
        self.threshold = threshold
        self.neg_threshold = max(threshold - 0.15, 0.01) if neg_threshold is None else neg_threshold
        self.reset()

    def reset(self) -> None:
        self._h = np.zeros((1, 1, 128), np.float32)
        self._c = np.zeros((1, 1, 128), np.float32)
        self._context = np.zeros(self.CONTEXT, np.float32)
        self._buf = np.zeros(0, np.float32)
        self.speech = False
        self.last_prob = 0.0

    def __call__(self, frame: bytes, sample_rate: int) -> bool:
        if sample_rate != 16000:
            raise ValueError("Silero VAD expects 16 kHz audio")
        x = np.frombuffer(frame, dtype=np.int16).astype(np.float32) / 32768.0
        self._buf = np.concatenate([self._buf, x])
        while self._buf.size >= self.WINDOW:
            window, self._buf = self._buf[:self.WINDOW], self._buf[self.WINDOW:]
            inp = np.concatenate([self._context, window])[None, :]
            prob, self._h, self._c = self.session.run(None, {"input": inp, "h": self._h, "c": self._c})
            self._context = window[-self.CONTEXT:]
            self.last_prob = p = float(np.ravel(prob)[0])
            if p >= self.threshold:
                self.speech = True
            elif p < self.neg_threshold:
                self.speech = False
        return self.speech


def silero_vad(threshold: float = 0.5) -> VadFn:
    return SileroVad(threshold)


VAD_BACKENDS = ("webrtc", "silero")


def vad_from_settings(settings) -> VadFn:
    """A fresh VAD for one session, by ``settings.vad_backend``."""
    if settings.vad_backend == "webrtc":
        return webrtc_vad(settings.vad_aggressiveness)
    if settings.vad_backend == "silero":
        return silero_vad(settings.silero_threshold)
    raise ValueError(f"Unknown VAD backend {settings.vad_backend!r} (use one of {', '.join(VAD_BACKENDS)})")


@dataclass
class Utterance:
    audio: np.ndarray  # int16 mono
    speech_start: float
    speech_end: float  # time of the last voiced frame (or the cut point)
    forced_cut: bool


class Segmenter:
    """Turn a stream of frames into utterances.

    * Idle: frames go into a pre-roll ring; when at least half of the ring is
      voiced, the ring becomes the start of a new utterance.
    * Triggered: frames accumulate; ``silence_ms`` of unvoiced frames ends the
      utterance; reaching ``max_segment_s`` force-cuts it and the next frame
      starts a new one immediately (no audio is lost across a forced cut).
    """

    def __init__(
        self,
        vad: VadFn,
        *,
        sample_rate: int = 16000,
        frame_ms: int = 30,
        pre_roll_ms: int = 300,
        silence_ms: int = 600,
        max_segment_s: float = 15.0,
    ) -> None:
        self.vad = vad
        self.sample_rate = sample_rate
        self.frame_ms = frame_ms
        self.frame_samples = sample_rate * frame_ms // 1000
        ring_len = max(1, pre_roll_ms // frame_ms)
        self._ring: collections.deque[tuple[np.ndarray, float, bool]] = collections.deque(maxlen=ring_len)
        self._start_votes = max(1, math.ceil(ring_len / 2))
        self._end_frames = max(1, silence_ms // frame_ms)
        self._max_frames = max(1, int(max_segment_s * 1000) // frame_ms)
        self._triggered = False
        self._frames: list[np.ndarray] = []
        self._start_t = 0.0
        self._last_voiced_t = 0.0
        self._silence_run = 0
        self._voiced_in_seg = 0

    @property
    def triggered(self) -> bool:
        return self._triggered

    @property
    def open_s(self) -> float:
        """Audio length of the utterance being collected (0 when idle). Read-only; safe to
        read from another thread (a ``len`` of the frame list)."""
        return len(self._frames) * self.frame_ms / 1000.0 if self._triggered else 0.0

    def _emit(self, speech_end: float, forced: bool) -> Utterance:
        audio = np.concatenate(self._frames) if self._frames else np.zeros(0, np.int16)
        utt = Utterance(audio.astype(np.int16, copy=False), self._start_t, speech_end, forced)
        self._frames = []
        self._voiced_in_seg = 0
        return utt

    def feed(self, frame: np.ndarray, t: float, vad_frame: np.ndarray | None = None) -> list[Utterance]:
        """Feed one frame captured at time ``t``; return finished utterances.

        ``vad_frame`` (default: ``frame``) is what the VAD sees, e.g. the
        denoised signal, while ``frame`` is the audio kept for ASR.
        """
        frame = np.asarray(frame, dtype=np.int16)
        if frame.shape[0] != self.frame_samples:
            raise ValueError(f"expected {self.frame_samples} samples, got {frame.shape[0]}")
        if vad_frame is None:
            vad_frame = frame
        else:
            vad_frame = np.asarray(vad_frame, dtype=np.int16)
            if vad_frame.shape[0] != self.frame_samples:
                raise ValueError(f"expected {self.frame_samples} VAD samples, got {vad_frame.shape[0]}")
        voiced = bool(self.vad(vad_frame.tobytes(), self.sample_rate))
        out: list[Utterance] = []

        if not self._triggered:
            self._ring.append((frame, t, voiced))
            if sum(v for _, _, v in self._ring) >= self._start_votes:
                self._triggered = True
                self._frames = [f for f, _, _ in self._ring]
                self._voiced_in_seg = sum(v for _, _, v in self._ring)
                self._start_t = self._ring[0][1]
                self._last_voiced_t = t
                self._silence_run = 0
                self._ring.clear()
            return out

        if not self._frames:  # continuing after a forced cut
            self._start_t = t
        self._frames.append(frame)
        if voiced:
            self._silence_run = 0
            self._last_voiced_t = t
            self._voiced_in_seg += 1
        else:
            self._silence_run += 1

        if self._silence_run >= self._end_frames:
            # A segment with no voiced frame (e.g. the tail after a forced cut
            # that landed at the end of speech) is discarded: Whisper would
            # hallucinate text for pure silence.
            if self._voiced_in_seg:
                out.append(self._emit(self._last_voiced_t, forced=False))
            self._frames = []
            self._voiced_in_seg = 0
            self._triggered = False
            self._silence_run = 0
        elif len(self._frames) >= self._max_frames:
            if self._voiced_in_seg:
                out.append(self._emit(t, forced=True))
            else:
                self._frames = []
        return out

    def reset(self) -> None:
        self._ring.clear()
        self._frames = []
        self._triggered = False
        self._silence_run = 0
        self._voiced_in_seg = 0
