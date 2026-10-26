"""Fakes for devices, ASR, MT and TTS plus synthetic audio. No network, no audio hardware."""

from __future__ import annotations

import collections
import threading
import time

import numpy as np
import pytest

from audio_translator.config import Settings
from audio_translator.devices import DeviceError, DeviceInfo

FRAME = 480  # 30 ms @ 16 kHz


# ----------------------------------------------------------------- audio
def speech_frame(i: int = 0, amp: int = 8000) -> np.ndarray:
    t = (np.arange(FRAME) + i * FRAME) / 16000
    return (amp * np.sin(2 * np.pi * 220 * t)).astype(np.int16)


def silence_frame() -> np.ndarray:
    return np.zeros(FRAME, dtype=np.int16)


def utterance(n_speech: int, n_silence: int = 30) -> list[np.ndarray]:
    return [speech_frame(i) for i in range(n_speech)] + [silence_frame() for _ in range(n_silence)]


def energy_vad(frame: bytes, rate: int) -> bool:
    x = np.frombuffer(frame, dtype=np.int16).astype(np.float32)
    return float(np.sqrt(np.mean(x * x))) > 500.0


# ----------------------------------------------------------------- streams
class _FakeStream:
    period = 0.002

    def __init__(self, **kw):
        self.kw = kw
        self.callback = kw["callback"]
        self.finished_callback = kw.get("finished_callback")
        self.active = False
        self._run = False
        self._thread: threading.Thread | None = None
        self.closed = False

    def start(self):
        self.active = True
        self._run = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._run = False
        if self._thread and self._thread is not threading.current_thread():
            self._thread.join(timeout=1)
        self.active = False

    def close(self):
        self.closed = True

    def remove_device(self, notify: bool = True):
        """Simulate unplugging: callbacks cease; optionally PortAudio reports it."""
        self._run = False
        self.active = False
        if notify and self.finished_callback:
            self.finished_callback()

    def hang(self):
        """Simulate a device that silently stops calling back (still 'active')."""
        self._run = False


class FakeInputStream(_FakeStream):
    def __init__(self, script: collections.deque, **kw):
        super().__init__(**kw)
        self.script = script

    def _loop(self):
        while self._run:
            frame = self.script.popleft() if self.script else silence_frame()
            self.callback(frame.reshape(-1, 1), FRAME, None, None)
            time.sleep(self.period)


class FakeOutputStream(_FakeStream):
    def __init__(self, **kw):
        super().__init__(**kw)
        self.channels = kw["channels"]
        self.blocks: list[np.ndarray] = []

    def _loop(self):
        n = 256
        while self._run:
            out = np.zeros((n, self.channels), dtype=np.float32)
            self.callback(out, n, None, None)
            if np.any(out):
                self.blocks.append(out.copy())
            time.sleep(self.period)

    @property
    def nonzero_samples(self) -> int:
        return sum(int(np.count_nonzero(b[:, 0])) for b in self.blocks)


# ----------------------------------------------------------------- models
class FakeASR:
    def __init__(self, responses=None, default="Xin chào, bạn khỏe không?"):
        self.responses = collections.deque(responses or [])
        self.default = default
        self.calls: list[np.ndarray] = []
        self.loaded = False

    def load(self):
        self.loaded = True

    def transcribe(self, audio, sample_rate):
        self.calls.append(audio)
        r = self.responses.popleft() if self.responses else self.default
        if isinstance(r, Exception):
            raise r
        return r


class FakeModelFactory:
    """Recognizer factory for ModelManager: per-model gates and failures, counts loads."""

    def __init__(self, gated=(), fail=()):
        self.gates = {n: threading.Event() for n in gated}
        self.fail = set(fail)
        self.load_calls: list[str] = []
        self.started = collections.defaultdict(threading.Event)
        self._lock = threading.Lock()

    def __call__(self, name):
        factory = self

        class _Rec(FakeASR):
            def load(self):
                with factory._lock:
                    factory.load_calls.append(name)
                factory.started[name].set()
                gate = factory.gates.get(name)
                if gate is not None:
                    assert gate.wait(5), f"gate for {name} never opened"
                if name in factory.fail:
                    raise OSError(f"corrupt model {name}")
                super().load()

        rec = _Rec(default=f"text from {name}")
        rec.model_size = name
        return rec

    def release(self, name):
        self.gates[name].set()


class FakeTranslator:
    def __init__(self, fail_on: set[str] | None = None):
        self.fail_on = fail_on or set()
        self.calls: list[str] = []

    def translate(self, text):
        self.calls.append(text)
        if text in self.fail_on:
            raise ConnectionError("network unreachable")
        return f"EN<{text}>"


class FakeTTS:
    def __init__(self, gate: threading.Event | None = None, fail_on: set[str] | None = None):
        self.gate = gate
        self.fail_on = fail_on or set()
        self.calls: list[str] = []
        self.entered = threading.Event()

    def supports(self, lang):
        return lang == "en"

    def synthesize(self, text):
        self.calls.append(text)
        self.entered.set()
        if self.gate is not None:
            self.gate.wait(5)
        if text in self.fail_on:
            raise ConnectionError("tts down")
        return np.full(2400, 0.5, dtype=np.float32), 24000  # 0.1 s


# ----------------------------------------------------------------- events
class EventLog:
    def __init__(self):
        self.events = []
        self._cv = threading.Condition()

    def __call__(self, ev):
        with self._cv:
            self.events.append(ev)
            self._cv.notify_all()

    def of(self, kind):
        return [e for e in list(self.events) if e.kind == kind]

    def wait_for(self, pred, timeout=5.0):
        end = time.monotonic() + timeout
        with self._cv:
            while True:
                for e in self.events:
                    if pred(e):
                        return e
                left = end - time.monotonic()
                if left <= 0:
                    raise AssertionError(
                        "timed out; events: " + "\n".join(f"{e.kind}: {e.message}" for e in self.events))
                self._cv.wait(left)

    def wait_state(self, state, timeout=5.0, after=0):
        return self.wait_for(lambda e: e.kind == "status" and e.data.get("state") == state
                             and self.events.index(e) >= after, timeout)


# ----------------------------------------------------------------- harness
class Harness:
    def __init__(self, settings=None, asr=None, translator=None, tts=None, missing_output=False,
                 model_manager=None, mt=None):
        from audio_translator.orchestrator import Orchestrator

        self.script: collections.deque = collections.deque()
        self.inputs: list[FakeInputStream] = []
        self.outputs: list[FakeOutputStream] = []
        self.models = model_manager
        self.asr = None if model_manager is not None else (asr or FakeASR())
        # mt: Orchestrator MT-backend seams (package_preparer, offline_/google_translator_factory);
        # when given, no translator is injected so the settings-driven wiring runs.
        self.translator = None if mt is not None else (translator or FakeTranslator())
        self.tts = tts or FakeTTS()
        self.log = EventLog()
        self.missing_output = missing_output

        def in_factory(**kw):
            s = FakeInputStream(self.script, **kw)
            self.inputs.append(s)
            return s

        def out_factory(**kw):
            s = FakeOutputStream(**kw)
            self.outputs.append(s)
            return s

        def resolver(name, kind):
            if kind == "input":
                return DeviceInfo(0, "Fake Mic", 16000.0, 1)
            if self.missing_output:
                raise DeviceError(f"Output device '{name}' not found")
            return DeviceInfo(1, "BlackHole 2ch", 48000.0, 2)

        s = settings or Settings()
        s.silence_ms = 300
        s.device_watchdog_s = min(s.device_watchdog_s, 0.5)
        self.orch = Orchestrator(
            s, asr=self.asr, model_manager=model_manager, translator=self.translator, synthesizer=self.tts,
            vad_factory=lambda aggr: energy_vad,
            input_stream_factory=in_factory, output_stream_factory=out_factory,
            device_resolver=resolver, on_event=self.log, **(mt or {}),
        )

    def say(self, *frame_lists):
        for fl in frame_lists:
            self.script.extend(fl)

    def start(self):
        assert self.orch.start(None, "BlackHole")
        self.log.wait_state("running")


@pytest.fixture
def harness():
    made = []

    def make(**kw):
        h = Harness(**kw)
        made.append(h)
        return h

    yield make
    for h in made:
        h.orch.shutdown()
