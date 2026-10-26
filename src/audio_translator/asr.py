"""M4: faster-whisper ASR, Vietnamese, loaded once."""

from __future__ import annotations

import threading
from typing import Protocol

import numpy as np


class Recognizer(Protocol):
    def load(self) -> None: ...

    def transcribe(self, audio: np.ndarray, sample_rate: int) -> str: ...


class WhisperASR:
    def __init__(
        self,
        model_size: str = "medium",
        *,
        language: str = "vi",
        device: str = "cpu",
        compute_type: str = "int8",
        beam_size: int = 5,
    ) -> None:
        self.model_size = model_size
        self.language = language
        self.device = device
        self.compute_type = compute_type
        self.beam_size = beam_size
        self._model = None
        self._lock = threading.Lock()

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def load(self) -> None:
        with self._lock:
            if self._model is None:
                from faster_whisper import WhisperModel

                self._model = WhisperModel(
                    self.model_size, device=self.device, compute_type=self.compute_type
                )

    def transcribe(self, audio: np.ndarray, sample_rate: int) -> str:
        if sample_rate != 16000:
            raise ValueError("WhisperASR expects 16 kHz audio")
        self.load()
        pcm = audio.astype(np.float32) / 32768.0 if audio.dtype == np.int16 else audio.astype(np.float32)
        segments, _info = self._model.transcribe(
            pcm,
            language=self.language,
            beam_size=self.beam_size,
            vad_filter=False,  # M3 already segmented
            condition_on_previous_text=False,
        )
        return " ".join(s.text.strip() for s in segments).strip()
