"""M8: output stream on the virtual device (BlackHole), fed by M7 playout."""

from __future__ import annotations

import time
from typing import Any, Callable

from .models import SpeechClip
from .playout import Playout


def _default_output_factory(**kwargs):
    import sounddevice as sd

    return sd.OutputStream(**kwargs)


class VirtualOutput:
    def __init__(
        self,
        playout: Playout,
        *,
        device: int | None,
        device_name: str,
        blocksize: int = 0,
        on_clip_started: Callable[[SpeechClip], None] | None = None,
        on_error: Callable[[str], None] | None = None,
        stream_factory: Callable[..., Any] | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.playout = playout
        self.device = device
        self.device_name = device_name
        self.blocksize = blocksize
        self.on_clip_started = on_clip_started
        self.on_error = on_error
        self.clock = clock
        self._factory = stream_factory or _default_output_factory
        self._stream = None
        self._stopping = False
        self._failed = False
        self.underflows = 0
        self.last_callback: float | None = None

    def _callback(self, outdata, frames, time_info, status) -> None:
        now = self.clock()
        self.last_callback = now
        if status and getattr(status, "output_underflow", False):
            self.underflows += 1
        try:
            data, started = self.playout.read(frames)
            outdata[:] = data
        except Exception as exc:  # never let an exception kill the audio thread silently
            outdata.fill(0)
            self._report(f"Output stream error on '{self.device_name}': {exc}")
            return
        for clip in started:
            clip.timestamps.setdefault("first_audio_out", now)
            if self.on_clip_started:
                self.on_clip_started(clip)

    def _report(self, msg: str) -> None:
        if not self._failed and not self._stopping:
            self._failed = True
            if self.on_error:
                self.on_error(msg)

    def _finished(self) -> None:
        self._report(f"Output device '{self.device_name}' stopped unexpectedly")

    def start(self) -> None:
        self._stream = self._factory(
            device=self.device,
            samplerate=self.playout.device_rate,
            channels=self.playout.channels,
            dtype="float32",
            blocksize=self.blocksize,
            callback=self._callback,
            finished_callback=self._finished,
        )
        self.last_callback = self.clock()
        self._stream.start()

    @property
    def active(self) -> bool:
        return bool(self._stream is not None and self._stream.active)

    def stop(self) -> None:
        self._stopping = True
        stream, self._stream = self._stream, None
        if stream is None:
            return
        for op in (stream.stop, stream.close):
            try:
                op()
            except Exception:
                pass
