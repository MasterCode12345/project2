"""M3a: optional noise suppression in front of the VAD (off by default).

Chain per 30 ms int16 frame (runs in the segment worker, never in an audio
callback)::

    raw y ──► Enhancer ──► ŝ ─┐
       │                      ├─► ObservationMixer: y_asr = (1-ω_eff)·ŝ + ω_eff·y(delayed)
       ├──► delay line ───────┘
       └──► SnrGate (on raw) ──► bypass? → ω_eff ramps to 1, VAD gets raw

Research (see ``_bmad-output/.../research-speech-noise-suppression-for-live-asr``)
shows naive enhancement tends to *raise* Whisper WER through processing
artifacts; observation adding (OA) mixes the noisy input back in, and ω = 1
gives back the untouched input. ŝ goes to webrtcvad, the OA mix to Whisper.

Phase 1 enhancers: :class:`NoEnhancer` and :class:`LogMmseEnhancer`. Any new
enhancer only has to satisfy :class:`Enhancer` and be registered in
:data:`ENHANCERS`.
"""

from __future__ import annotations

import collections
import math
from typing import Callable, Protocol

import numpy as np

INT16_MIN, INT16_MAX = -32768, 32767


class Enhancer(Protocol):
    """Causal frame-in/frame-out enhancer.

    ``process`` takes one int16 frame and returns the same number of samples
    (float32 on the int16 scale, or int16) delayed by ``latency_samples``.
    """

    latency_samples: int

    def process(self, frame: np.ndarray) -> np.ndarray: ...

    def reset(self) -> None: ...


class NoEnhancer:
    """Identity enhancer (zero latency); useful as an ablation arm."""

    latency_samples = 0

    def process(self, frame: np.ndarray) -> np.ndarray:
        return np.asarray(frame).astype(np.float32)

    def reset(self) -> None:
        pass


# ------------------------------------------------------------------ log-MMSE


def expint_e1(x: np.ndarray) -> np.ndarray:
    """Exponential integral E1(x) for x > 0 (Abramowitz & Stegun 5.1.53 / 5.1.56)."""
    x = np.maximum(np.asarray(x, dtype=np.float64), 1e-12)
    small = x <= 1.0
    out = np.empty_like(x)
    xs = x[small]
    out[small] = (-np.log(xs) - 0.57721566 + xs * (0.99999193 + xs * (-0.24991055 + xs * (
        0.05519968 + xs * (-0.00976004 + xs * 0.00107857)))))
    xl = x[~small]
    num = xl * xl + 2.334733 * xl + 0.250621
    den = xl * xl + 3.330657 * xl + 1.681534
    out[~small] = num / den * np.exp(-xl) / xl
    return out


class LogMmseEnhancer:
    """Ephraim–Malah log-spectral amplitude estimator, streaming and causal.

    * STFT: 480-sample periodic sqrt-Hann analysis/synthesis window, hop 240
      (two hops per 30 ms frame), weighted overlap-add → perfect
      reconstruction for unit gain, algorithmic delay 240 samples (15 ms).
    * A priori SNR: decision-directed (α = 0.98) with a floor ``xi_min_db``.
    * Noise PSD: speech-presence-probability tracker (Gerkmann & Hendriks
      2012), purely causal; initialised on the first non-silent hop.
    * Gain floored at ``gain_floor_db`` to limit musical noise/artifacts.
    """

    def __init__(
        self,
        *,
        frame_samples: int = 480,
        win_len: int = 480,
        hop: int = 240,
        alpha: float = 0.98,
        xi_min_db: float = -25.0,
        gain_floor_db: float = -15.0,
    ) -> None:
        if frame_samples % hop:
            raise ValueError("frame_samples must be a multiple of hop")
        if win_len != 2 * hop:
            raise ValueError("win_len must be 2 * hop (50% overlap sqrt-Hann WOLA)")
        self.frame_samples = frame_samples
        self.win_len = win_len
        self.hop = hop
        self.alpha = alpha
        self.xi_min = 10.0 ** (xi_min_db / 10.0)
        self.gain_floor = 10.0 ** (gain_floor_db / 20.0)
        n = np.arange(win_len)
        self._win = np.sqrt(0.5 - 0.5 * np.cos(2 * np.pi * n / win_len))  # periodic: w² sums to 1
        self.latency_samples = win_len - hop
        # SPP noise tracker constants (Gerkmann & Hendriks 2012)
        self._xi_h1 = 10.0 ** (15.0 / 10.0)
        self.reset()

    def reset(self) -> None:
        self._inbuf = np.zeros(self.win_len)
        self._ola = np.zeros(self.win_len)
        self._noise: np.ndarray | None = None
        self._prev_snr: np.ndarray | None = None  # G² · γ of the previous hop
        self._spp_bar = np.zeros(self.win_len // 2 + 1)

    def _gain(self, p: np.ndarray) -> np.ndarray:
        eps = 1e-6
        if self._noise is None:
            if float(p.mean()) < 1.0:  # digital silence: nothing to learn yet
                return np.ones_like(p)
            self._noise = np.maximum(p, eps)
        lam = self._noise
        # --- noise PSD update (speech presence probability, fixed priors)
        g_post = p / lam
        r = self._xi_h1 / (1.0 + self._xi_h1)
        spp = 1.0 / (1.0 + (1.0 + self._xi_h1) * np.exp(-np.minimum(g_post * r, 500.0)))
        self._spp_bar = 0.9 * self._spp_bar + 0.1 * spp
        spp = np.where(self._spp_bar > 0.99, np.minimum(spp, 0.99), spp)
        lam = 0.8 * lam + 0.2 * ((1.0 - spp) * p + spp * lam)
        self._noise = lam = np.maximum(lam, eps)
        # --- log-MMSE gain with decision-directed a priori SNR
        gamma = np.minimum(p / lam, 1e4)
        ml = np.maximum(gamma - 1.0, 0.0)
        xi = ml if self._prev_snr is None else self.alpha * self._prev_snr + (1.0 - self.alpha) * ml
        xi = np.maximum(xi, self.xi_min)
        ratio = xi / (1.0 + xi)
        v = ratio * gamma
        gain = ratio * np.exp(0.5 * expint_e1(v))
        gain = np.clip(gain, self.gain_floor, 1.0)
        self._prev_snr = gain * gain * gamma
        return gain

    def process(self, frame: np.ndarray) -> np.ndarray:
        x = np.asarray(frame, dtype=np.float64)
        if x.shape[0] != self.frame_samples:
            raise ValueError(f"expected {self.frame_samples} samples, got {x.shape[0]}")
        out = np.empty(self.frame_samples, dtype=np.float32)
        h, n = self.hop, self.win_len
        for k in range(self.frame_samples // h):
            self._inbuf[:-h] = self._inbuf[h:]
            self._inbuf[-h:] = x[k * h:(k + 1) * h]
            spec = np.fft.rfft(self._inbuf * self._win)
            gain = self._gain(spec.real ** 2 + spec.imag ** 2)
            self._ola += np.fft.irfft(spec * gain, n=n) * self._win
            out[k * h:(k + 1) * h] = self._ola[:h]
            self._ola[:-h] = self._ola[h:]
            self._ola[-h:] = 0.0
        return out


ENHANCERS: dict[str, Callable[[], Enhancer]] = {
    "logmmse": LogMmseEnhancer,
}
NS_MODES = ("off", *ENHANCERS)


# ---------------------------------------------------------------------- gate
class SnrGate:
    """Decide from the *raw* input whether enhancement should be bypassed.

    Noise floor = minimum frame energy over the last ``window_s``; speech
    level = smoothed energy of frames ``speech_margin_db`` above the floor.
    Bypass when SNR > threshold + hysteresis, re-enable when SNR < threshold.
    ``threshold_db=None`` never bypasses.
    """

    def __init__(self, threshold_db: float | None, *, hysteresis_db: float = 3.0,
                 frame_ms: int = 30, window_s: float = 3.0, speech_margin_db: float = 6.0) -> None:
        self.threshold_db = threshold_db
        self.hysteresis_db = hysteresis_db
        self.speech_margin_db = speech_margin_db
        self._window = max(1, int(window_s * 1000) // frame_ms)
        # Speech level relaxes toward the floor on non-speech frames (time
        # constant = window_s), so one transient cannot hold the gate open.
        self._decay = math.exp(-frame_ms / (window_s * 1000.0))
        self.reset()

    def reset(self) -> None:
        self._energies: collections.deque[float] = collections.deque(maxlen=self._window)
        self._level: float | None = None
        self.snr_db = 0.0
        self.bypass = False

    def decide(self, snr_db: float) -> bool:
        """Hysteresis step on an SNR estimate; returns the new bypass state."""
        self.snr_db = snr_db
        if self.threshold_db is None:
            self.bypass = False
        elif not self.bypass and snr_db > self.threshold_db + self.hysteresis_db:
            self.bypass = True
        elif self.bypass and snr_db < self.threshold_db:
            self.bypass = False
        return self.bypass

    def update(self, frame: np.ndarray) -> bool:
        x = np.asarray(frame, dtype=np.float64)
        e = 10.0 * math.log10(float(np.mean(x * x)) + 1.0)
        self._energies.append(e)
        floor = min(self._energies)
        if e > floor + self.speech_margin_db:
            level = floor if self._level is None else max(self._level, floor)
            self._level = level + 0.1 * (e - level)  # smoothed, also when starting
        elif self._level is not None:
            self._level = floor + max(self._level - floor, 0.0) * self._decay
        snr = 0.0 if self._level is None else self._level - floor
        return self.decide(snr)


# --------------------------------------------------------------------- mixer
class ObservationMixer:
    """Delay line for the raw input + observation adding + int16 clip."""

    def __init__(self, latency_samples: int) -> None:
        self.latency_samples = latency_samples
        self.reset()

    def reset(self) -> None:
        self._buf = np.zeros(self.latency_samples, dtype=np.int16)

    def delay(self, frame: np.ndarray) -> np.ndarray:
        """Raw frame delayed by ``latency_samples`` (same length as ``frame``)."""
        frame = np.asarray(frame, dtype=np.int16)
        if self.latency_samples == 0:
            return frame.copy()
        joined = np.concatenate([self._buf, frame])
        self._buf = joined[frame.shape[0]:].copy()
        return joined[:frame.shape[0]]

    @staticmethod
    def to_int16(x: np.ndarray) -> np.ndarray:
        return np.clip(np.rint(x), INT16_MIN, INT16_MAX).astype(np.int16)

    @classmethod
    def mix(cls, s_hat: np.ndarray, delayed_raw: np.ndarray, w) -> np.ndarray:
        """(1−ω)·ŝ + ω·y, ω scalar or per-sample; clipped to int16 (no wrap)."""
        s = np.asarray(s_hat, dtype=np.float64)
        y = np.asarray(delayed_raw, dtype=np.float64)
        return cls.to_int16((1.0 - w) * s + w * y)


# ---------------------------------------------------------------- suppressor
class NoiseSuppressor:
    """Enhancer → SnrGate → ObservationMixer; one instance per session.

    ``process(frame)`` returns ``(vad_frame, asr_frame)`` (both int16, delayed
    by ``latency_samples``). The enhancer always runs so its state stays warm;
    the gate only moves ω_eff (ramped over ``ramp_ms``, never a jump).
    """

    def __init__(self, enhancer: Enhancer, *, mix: float = 0.4, gate_snr_db: float | None = None,
                 vad_input: str = "denoised", sample_rate: int = 16000, frame_ms: int = 30,
                 ramp_ms: float = 20.0) -> None:
        if not 0.0 <= mix <= 1.0:
            raise ValueError(f"ns_mix must be in [0, 1], got {mix}")
        if vad_input not in ("denoised", "raw"):
            raise ValueError(f"ns_vad_input must be 'denoised' or 'raw', got {vad_input!r}")
        if ramp_ms < 10.0:
            raise ValueError("ramp_ms must be >= 10 ms")
        self.enhancer = enhancer
        self.mix = mix
        self.vad_input = vad_input
        self.sample_rate = sample_rate
        self.gate = SnrGate(gate_snr_db, frame_ms=frame_ms)
        self.mixer = ObservationMixer(enhancer.latency_samples)
        self._step = 1.0 / (ramp_ms * sample_rate / 1000.0)  # max ω change per sample
        self.reset()

    @property
    def latency_samples(self) -> int:
        return self.enhancer.latency_samples

    @property
    def latency_s(self) -> float:
        return self.latency_samples / self.sample_rate

    @property
    def w_eff(self) -> float:
        return self._w

    def reset(self) -> None:
        self.enhancer.reset()
        self.gate.reset()
        self.mixer.reset()
        self._w = self.mix

    def _ramp(self, target: float, n: int) -> np.ndarray | float:
        if self._w == target:
            return target
        delta = target - self._w
        ramp = np.minimum(self._step * np.arange(1, n + 1), abs(delta))
        w = self._w + math.copysign(1.0, delta) * ramp
        self._w = float(w[-1]) if abs(float(w[-1]) - target) > 1e-12 else target
        return w

    def process(self, frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        frame = np.asarray(frame, dtype=np.int16)
        s_hat = self.enhancer.process(frame)
        delayed = self.mixer.delay(frame)
        bypass = self.gate.update(frame)
        w = self._ramp(1.0 if bypass else self.mix, frame.shape[0])
        asr = self.mixer.mix(s_hat, delayed, w)
        if bypass or self.vad_input == "raw":
            vad = delayed
        else:
            vad = ObservationMixer.to_int16(s_hat)
        return vad, asr


def make_noise_suppressor(mode: str, *, mix: float = 0.4, gate_snr_db: float | None = None,
                          vad_input: str = "denoised", sample_rate: int = 16000,
                          frame_ms: int = 30) -> NoiseSuppressor | None:
    """Build the stage for ``mode``; ``"off"`` returns None (no stage, no latency)."""
    if mode not in NS_MODES:
        raise ValueError(f"Unknown noise suppression mode {mode!r} (use one of {', '.join(NS_MODES)})")
    if mode == "off":
        return None
    if sample_rate != 16000 or frame_ms != 30:
        raise ValueError("noise suppression needs 16 kHz audio in 30 ms frames")
    return NoiseSuppressor(ENHANCERS[mode](), mix=mix, gate_snr_db=gate_snr_db, vad_input=vad_input,
                           sample_rate=sample_rate, frame_ms=frame_ms)


def from_settings(settings) -> NoiseSuppressor | None:
    s = settings
    return make_noise_suppressor(s.ns_mode, mix=s.ns_mix, gate_snr_db=s.ns_gate_snr_db,
                                 vad_input=s.ns_vad_input, sample_rate=s.capture_rate,
                                 frame_ms=s.frame_ms)
