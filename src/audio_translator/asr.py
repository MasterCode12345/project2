"""M4: faster-whisper ASR, Vietnamese, loaded once."""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Protocol

import numpy as np

# Models that are not on the Hugging Face hub in CTranslate2 format: converted
# once with scripts/convert_phowhisper.py into LOCAL_MODELS_DIR/<name>-ct2/.
# name -> (source repo, pinned revision). PhoWhisper (VinAI, BSD-3-Clause) is
# Whisper fine-tuned on 844 h of Vietnamese, *including the VIVOS train set*.
LOCAL_MODELS: dict[str, tuple[str, str]] = {
    "phowhisper-small": ("vinai/PhoWhisper-small", "a86b604c346caf7148c37512eafe783a16420adb"),
    "phowhisper-medium": ("vinai/PhoWhisper-medium", "55a7e3eb6c906de891f8f06a107754427dd3be79"),
}
LOCAL_MODELS_DIR = Path("~/.cache/audio-translator/models")


def local_model_dir(name: str, root: Path | None = None) -> Path:
    return (root or LOCAL_MODELS_DIR).expanduser() / f"{name}-ct2"


def convert_command(name: str) -> str:
    size = name.removeprefix("phowhisper-")
    return f"uv run --with transformers --with torch python scripts/convert_phowhisper.py {size}"


def resolve_model(name: str, root: Path | None = None) -> str:
    """faster-whisper model argument: a hub name, or the local CT2 dir of a converted model."""
    if name not in LOCAL_MODELS:
        return name
    path = local_model_dir(name, root)
    if not (path / "model.bin").is_file():
        raise RuntimeError(f"Model '{name}' is not converted yet (expected {path}/model.bin). "
                           f"Run: {convert_command(name)}")
    return str(path)


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
        initial_prompt: str | None = None,
    ) -> None:
        self.model_size = model_size
        self.language = language
        self.device = device
        self.compute_type = compute_type
        self.beam_size = beam_size
        self.initial_prompt = initial_prompt or None
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
                    resolve_model(self.model_size), device=self.device, compute_type=self.compute_type
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
            initial_prompt=self.initial_prompt,
        )
        return " ".join(s.text.strip() for s in segments).strip()
