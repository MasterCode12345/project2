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

    def _emit(self, speech_end: float, forced: bool) -> Utterance:
        audio = np.concatenate(self._frames) if self._frames else np.zeros(0, np.int16)
        utt = Utterance(audio.astype(np.int16, copy=False), self._start_t, speech_end, forced)
        self._frames = []
        self._voiced_in_seg = 0
        return utt

    def feed(self, frame: np.ndarray, t: float) -> list[Utterance]:
        """Feed one frame captured at time ``t``; return finished utterances."""
        frame = np.asarray(frame, dtype=np.int16)
        if frame.shape[0] != self.frame_samples:
            raise ValueError(f"expected {self.frame_samples} samples, got {frame.shape[0]}")
        voiced = bool(self.vad(frame.tobytes(), self.sample_rate))
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
