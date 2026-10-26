"""M2: microphone capture -> bounded frame queue.

The PortAudio callback only copies the block, stamps the time and does a
``put_nowait``; a full queue drops the frame and counts an overflow.
"""

from __future__ import annotations

import queue
import time
from typing import Any, Callable

import numpy as np

StreamFactory = Callable[..., Any]


def _default_input_factory(**kwargs):
    import sounddevice as sd

    return sd.InputStream(**kwargs)


class AudioCapture:
    def __init__(
        self,
        session_id: int,
        out_q: queue.Queue,
        *,
        device: int | None,
        device_name: str,
        samplerate: int = 16000,
        blocksize: int = 480,
        on_error: Callable[[str], None] | None = None,
        stream_factory: StreamFactory | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.session_id = session_id
        self.out_q = out_q
        self.device = device
        self.device_name = device_name
        self.samplerate = samplerate
        self.blocksize = blocksize
        self.on_error = on_error
        self.clock = clock
        self._factory = stream_factory or _default_input_factory
        self._stream = None
        self._stopping = False
        self.overflows = 0  # frames dropped because the queue was full
        self.driver_overflows = 0  # PortAudio input_overflow flags
        self.last_callback: float | None = None

    def _callback(self, indata, frames, time_info, status) -> None:
        now = self.clock()
        self.last_callback = now
        if status and getattr(status, "input_overflow", False):
            self.driver_overflows += 1
        frame = np.array(indata[:, 0] if np.ndim(indata) == 2 else indata, dtype=np.int16, copy=True)
        try:
            self.out_q.put_nowait((self.session_id, frame, now))
        except queue.Full:
            self.overflows += 1

    def _finished(self) -> None:
        if not self._stopping and self.on_error:
            self.on_error(f"Input device '{self.device_name}' stopped unexpectedly")

    def start(self) -> None:
        self._stream = self._factory(
            device=self.device,
            samplerate=self.samplerate,
            channels=1,
            dtype="int16",
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
                pass  # device may already be gone
