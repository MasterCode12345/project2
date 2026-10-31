"""M3a noise-suppression stage: one test per row of the plan's I/O matrix."""

import numpy as np
import pytest

from audio_translator import cli
from audio_translator.config import Settings
from audio_translator.denoise import (
    LogMmseEnhancer,
    NoEnhancer,
    NoiseSuppressor,
    ObservationMixer,
    SnrGate,
    expint_e1,
    from_settings,
    make_noise_suppressor,
)
from conftest import FRAME

RATE = 16000


def frames_of(x):
    x = np.asarray(x)
    return [x[i:i + FRAME] for i in range(0, x.size - FRAME + 1, FRAME)]


def run(stage, x):
    vads, asrs = zip(*(stage.process(f) for f in frames_of(x)))
    return np.concatenate(vads), np.concatenate(asrs)


def tone_in_noise(seconds=6.0, snr_db=5.0, seed=0, on=((2.0, 3.0), (4.0, 5.0))):
    rng = np.random.default_rng(seed)
    n = int(seconds * RATE)
    t = np.arange(n) / RATE
    tone = np.zeros(n)
    for a, b in on:
        m = (t >= a) & (t < b)
        tone[m] = 8000 * np.sin(2 * np.pi * 440 * t[m])
    p_tone = 8000.0 ** 2 / 2  # power of the tone bursts
    noise = rng.normal(0, np.sqrt(p_tone / 10 ** (snr_db / 10)), n)
    return np.clip(np.rint(tone + noise), -32768, 32767).astype(np.int16), t


def test_off_is_default_and_builds_no_stage():
    s = Settings()
    assert s.ns_mode == "off"
    assert from_settings(s) is None
    assert make_noise_suppressor("off") is None


def test_unknown_mode_and_bad_params_are_rejected():
    with pytest.raises(ValueError, match="Unknown noise suppression mode"):
        make_noise_suppressor("gtcrn")
    with pytest.raises(ValueError):
        NoiseSuppressor(NoEnhancer(), mix=1.5)
    with pytest.raises(ValueError):
        NoiseSuppressor(NoEnhancer(), vad_input="both")
    with pytest.raises(ValueError):
        NoiseSuppressor(NoEnhancer(), ramp_ms=5)


def test_expint_matches_reference_values():
    ref = {0.01: 4.037929576538, 0.5: 0.559773594776, 1.0: 0.219383934396, 2.0: 0.048900510708,
           10.0: 4.15696892968e-06}
    got = expint_e1(np.array(list(ref)))
    assert np.allclose(got, list(ref.values()), rtol=1e-3)


def test_logmmse_unit_gain_reconstructs_input_delayed_by_declared_latency():
    e = LogMmseEnhancer()
    e._gain = lambda p: np.ones_like(p)
    x, _ = tone_in_noise(1.0)
    out = np.concatenate([e.process(f) for f in frames_of(x)])
    L = e.latency_samples
    assert L == 240
    assert np.max(np.abs(out[L:] - x[: out.size - L])) < 1e-6


def test_logmmse_reduces_noise_by_at_least_6_db_without_speech():
    x, t = tone_in_noise(6.0, snr_db=5.0)
    e = LogMmseEnhancer()
    out = np.concatenate([e.process(f) for f in frames_of(x)])
    L = e.latency_samples
    y, xin, tt = out[L:], x[:-L].astype(float)[: out.size - L], t[: out.size - L]
    noise_only = (tt >= 1.0) & (tt < 2.0)  # after convergence, before the first tone
    reduction = 10 * np.log10(np.mean(xin[noise_only] ** 2) / np.mean(y[noise_only] ** 2))
    assert reduction >= 6.0
    tone = (tt >= 2.2) & (tt < 2.8)
    kept = 10 * np.log10(np.mean(y[tone] ** 2) / np.mean(xin[tone] ** 2))
    assert kept > -3.0  # the "speech" itself survives


def test_logmmse_rtf_is_far_below_real_time():
    import time

    e = LogMmseEnhancer()
    x = (np.random.default_rng(1).normal(0, 3000, RATE * 3)).astype(np.int16)
    t0 = time.perf_counter()
    for f in frames_of(x):
        e.process(f)
    assert (time.perf_counter() - t0) / 3.0 < 0.1


def test_oa_with_mix_one_returns_delayed_raw():
    x, _ = tone_in_noise(2.0)
    ns = NoiseSuppressor(LogMmseEnhancer(), mix=1.0, gate_snr_db=None)
    _, asr = run(ns, x)
    L = ns.latency_samples
    assert np.all(asr[:L] == 0)
    assert np.max(np.abs(asr[L:].astype(int) - x[: asr.size - L].astype(int))) <= 1


def test_mix_zero_gives_denoised_and_vad_gets_denoised():
    x, _ = tone_in_noise(2.0)
    ns = NoiseSuppressor(LogMmseEnhancer(), mix=0.0, gate_snr_db=None)
    ref = LogMmseEnhancer()
    for f in frames_of(x):
        vad, asr = ns.process(f)
        s_hat = ObservationMixer.to_int16(ref.process(f))
        assert np.array_equal(asr, s_hat) and np.array_equal(vad, s_hat)


def test_vad_input_raw_sends_delayed_raw_to_vad():
    x, _ = tone_in_noise(1.0)
    ns = NoiseSuppressor(LogMmseEnhancer(), mix=0.4, gate_snr_db=None, vad_input="raw")
    vad, _ = run(ns, x)
    L = ns.latency_samples
    assert np.array_equal(vad[L:], x[: vad.size - L])


def _clean_speech_like(seconds=4.0):
    """Quiet noise floor with loud tone bursts: SNR far above any threshold."""
    x, t = tone_in_noise(seconds, snr_db=40.0, on=((0.5, 1.5), (2.0, 3.0), (3.3, 3.9)))
    return x


def test_gate_bypasses_on_high_snr_with_smooth_ramp_and_raw_vad():
    x = _clean_speech_like()
    ns = NoiseSuppressor(LogMmseEnhancer(), mix=0.4, gate_snr_db=20.0, ramp_ms=20.0)
    assert ns.w_eff == pytest.approx(0.4)
    max_step = 1.0 / (10 * RATE / 1000)  # the ramp must take >= 10 ms
    frames = frames_of(x)
    for f in frames:
        before = ns.w_eff
        vad, asr = ns.process(f)
        assert abs(ns.w_eff - before) <= max_step * FRAME + 1e-9
    assert ns.gate.bypass and ns.w_eff == 1.0
    # bypassed with ω_eff = 1: VAD and ASR both get the raw input, delayed
    k, L = len(frames) - 1, ns.latency_samples
    expected = x[k * FRAME - L:(k + 1) * FRAME - L]
    assert np.array_equal(vad, expected) and np.array_equal(asr, expected)


def test_ramp_is_per_sample_and_never_jumps():
    ns = NoiseSuppressor(NoEnhancer(), mix=0.0, ramp_ms=10.0)
    w = ns._ramp(1.0, FRAME)
    assert np.max(np.abs(np.diff(np.concatenate([[0.0], w])))) <= 1.0 / 160 + 1e-12
    assert w[-1] == 1.0 and ns.w_eff == 1.0  # 10 ms = 160 samples < 480


def test_gate_reenables_when_snr_drops():
    quiet = _clean_speech_like()
    rng = np.random.default_rng(3)
    loud = (rng.normal(0, 4000, RATE * 5)).astype(np.int16)  # noise floor jumps: SNR ~ 0 dB
    ns = NoiseSuppressor(LogMmseEnhancer(), mix=0.4, gate_snr_db=20.0)
    run(ns, quiet)
    assert ns.gate.bypass
    run(ns, loud)
    assert not ns.gate.bypass
    assert ns.w_eff == pytest.approx(0.4)


def test_gate_hysteresis_prevents_toggling():
    g = SnrGate(20.0, hysteresis_db=3.0)
    toggles, state = 0, g.bypass
    for snr in [21, 22.9, 20.5, 22, 21.5, 22.8] * 5:  # inside the band, from "enhance"
        if g.decide(snr) != state:
            toggles, state = toggles + 1, g.bypass
    assert toggles == 0 and not g.bypass
    assert g.decide(23.5)  # above threshold + hysteresis: bypass
    for snr in [20.1, 22.9, 20.5, 22, 21.5, 23.5] * 5:  # inside the band, from "bypass"
        assert g.decide(snr)
    assert not g.decide(19.9)


def test_gate_none_always_enhances():
    x = _clean_speech_like()
    ns = NoiseSuppressor(LogMmseEnhancer(), mix=0.4, gate_snr_db=None)
    for f in frames_of(x):
        ns.process(f)
        assert not ns.gate.bypass and ns.w_eff == 0.4


def test_reset_clears_enhancer_gate_and_delay_line():
    x, _ = tone_in_noise(3.0)
    used = NoiseSuppressor(LogMmseEnhancer(), mix=0.4, gate_snr_db=20.0)
    run(used, _clean_speech_like())
    used.reset()
    fresh = NoiseSuppressor(LogMmseEnhancer(), mix=0.4, gate_snr_db=20.0)
    for f in frames_of(x):
        a, b = used.process(f), fresh.process(f)
        assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])


def test_near_full_scale_input_is_clipped_not_wrapped():
    t = np.arange(RATE) / RATE
    x = np.clip(np.rint(33000 * np.sin(2 * np.pi * 300 * t)), -32768, 32767).astype(np.int16)
    for mix in (0.0, 0.4, 1.0):
        ns = NoiseSuppressor(LogMmseEnhancer(), mix=mix, gate_snr_db=None)
        vad, asr = run(ns, x)
        L = ns.latency_samples
        ref = x[: asr.size - L].astype(int)
        for out in (vad, asr):
            assert out.dtype == np.int16
            # no wrap-around: where the input is at a strong positive peak the
            # output is never negative (a wrapped overflow would flip the sign)
            peaks = ref > 30000
            assert np.all(out[L:][peaks] >= 0)
    assert np.array_equal(ObservationMixer.mix(np.array([40000.0, -40000.0]), np.array([0, 0]), 0.0),
                          np.array([32767, -32768], np.int16))


def test_cli_flags_map_to_settings():
    p = cli.build_parser()
    a = p.parse_args([])
    assert (a.ns, a.ns_mix, a.ns_gate_db) == ("off", Settings().ns_mix, Settings().ns_gate_snr_db)
    a = p.parse_args(["--ns", "logmmse", "--ns-mix", "0.6", "--ns-gate-db", "none"])
    assert (a.ns, a.ns_mix, a.ns_gate_db) == ("logmmse", 0.6, None)
    assert p.parse_args(["--ns-gate-db", "15"]).ns_gate_db == 15.0
    for bad in (["--ns", "rnnoise"], ["--ns-mix", "2"], ["--ns-gate-db", "loud"]):
        with pytest.raises(SystemExit):
            p.parse_args(bad)


def test_cli_passes_ns_settings_to_orchestrator():
    seen = {}

    class Orch:
        def __init__(self, settings, on_event):
            seen["s"] = settings
            self.on_event = on_event

        def start(self):
            from audio_translator.models import STATUS, STOPPED, Event

            self.on_event(Event(STATUS, "Stopped", data={"state": STOPPED}))

        def stop(self):
            pass

        def shutdown(self):
            pass

    assert cli.main(["--ns", "logmmse", "--ns-mix", "0.2", "--ns-gate-db", "none"],
                    orchestrator_factory=Orch) == 0
    s = seen["s"]
    assert (s.ns_mode, s.ns_mix, s.ns_gate_snr_db) == ("logmmse", 0.2, None)


def _noise_frames(n, rms_level, rng):
    return [rng.normal(0, rms_level, FRAME).astype(np.int16) for _ in range(n)]


def test_gate_single_transient_does_not_lock_bypass_under_steady_noise():
    rng = np.random.default_rng(5)
    g = SnrGate(20.0)
    for f in _noise_frames(50, 1000, rng):
        g.update(f)
    click = rng.normal(0, 1000, FRAME).astype(np.int16)
    click[:160] = 25000
    g.update(click)
    for f in _noise_frames(2000, 1000, rng):
        g.update(f)
    assert not g.bypass and g.snr_db < 1.0


def test_gate_reenables_after_loud_burst_then_steady_noise():
    rng = np.random.default_rng(6)
    g = SnrGate(20.0)
    for f in _noise_frames(50, 100, rng):
        g.update(f)
    for _ in range(40):  # ~1.2 s of loud signal, 40 dB above the floor
        g.update(rng.normal(0, 10000, FRAME).astype(np.int16))
    assert g.bypass
    for f in _noise_frames(300, 100, rng):  # steady noise only, ~9 s
        g.update(f)
    assert not g.bypass
