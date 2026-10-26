"""M6: text-to-speech behind an adapter protocol."""

from __future__ import annotations

import asyncio
from typing import Protocol

import numpy as np


class Synthesizer(Protocol):
    def supports(self, lang: str) -> bool: ...

    def synthesize(self, text: str) -> tuple[np.ndarray, int]:
        """Return (float32 mono PCM, sample_rate)."""
        ...


def decode_mp3(data: bytes, sample_rate: int = 24000) -> np.ndarray:
    import miniaudio

    decoded = miniaudio.decode(
        data, output_format=miniaudio.SampleFormat.FLOAT32, nchannels=1, sample_rate=sample_rate
    )
    return np.asarray(decoded.samples, dtype=np.float32)


class EdgeTTS:
    """Microsoft Edge neural voices via ``edge-tts`` (network, MP3 output)."""

    def __init__(self, voice: str = "en-US-AriaNeural", sample_rate: int = 24000) -> None:
        self.voice = voice
        self.sample_rate = sample_rate

    def supports(self, lang: str) -> bool:
        return self.voice.lower().startswith(lang.lower() + "-")

    async def _fetch(self, text: str) -> bytes:
        import edge_tts

        buf = bytearray()
        async for chunk in edge_tts.Communicate(text, self.voice).stream():
            if chunk.get("type") == "audio":
                buf += chunk["data"]
        return bytes(buf)

    def synthesize(self, text: str) -> tuple[np.ndarray, int]:
        data = asyncio.run(self._fetch(text))
        if not data:
            raise RuntimeError("edge-tts returned no audio")
        return decode_mp3(data, self.sample_rate), self.sample_rate
