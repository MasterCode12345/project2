import numpy as np

from audio_translator.segmenter import Segmenter
from conftest import FRAME, energy_vad, silence_frame, speech_frame, utterance


def run(seg, frames, t0=0.0):
    out = []
    for i, f in enumerate(frames):
        out += seg.feed(f, t0 + (i + 1) * 0.03)
    return out


def test_single_utterance_ends_on_silence():
    seg = Segmenter(energy_vad, silence_ms=300, pre_roll_ms=300)
    frames = [silence_frame()] * 20 + utterance(40, 20)
    utts = run(seg, frames)
    assert len(utts) == 1
    u = utts[0]
    assert not u.forced_cut
    # speech_end is the time of the last voiced frame (frame index 20+40-1)
    assert abs(u.speech_end - (60 * 0.03)) < 1e-9
    assert u.audio.dtype == np.int16
    # pre-roll + speech + trailing silence, nothing more
    assert 40 * FRAME <= u.audio.size <= (40 + 10 + 10) * FRAME


def test_noise_only_produces_nothing():
    seg = Segmenter(energy_vad)
    rng = np.random.default_rng(0)
    frames = [(rng.normal(0, 50, FRAME)).astype(np.int16) for _ in range(200)]
    assert run(seg, frames) == []


def test_long_utterance_is_force_cut_without_losing_audio():
    seg = Segmenter(energy_vad, silence_ms=300, pre_roll_ms=300, max_segment_s=1.5)  # 50 frames
    n_speech = 170  # > 3 x max length
    frames = utterance(n_speech, 20)
    utts = run(seg, frames)
    assert len(utts) >= 4
    assert all(u.forced_cut for u in utts[:-1]) and not utts[-1].forced_cut
    assert all(u.audio.size <= 50 * FRAME for u in utts)
    joined = np.concatenate([u.audio for u in utts])
    # Trigger fires once half the 10-frame pre-roll ring is voiced (frame 5);
    # from there every fed frame up to the closing silence is kept exactly once.
    consumed = np.concatenate(frames)[: joined.size]
    assert np.array_equal(joined, consumed)
    assert joined.size == (n_speech + 10) * FRAME  # speech + 300 ms closing silence


def test_two_utterances():
    seg = Segmenter(energy_vad, silence_ms=300)
    utts = run(seg, utterance(20, 15) + utterance(20, 15))
    assert len(utts) == 2
    assert utts[0].speech_end < utts[1].speech_start


def test_rejects_wrong_frame_size():
    seg = Segmenter(energy_vad)
    try:
        seg.feed(np.zeros(100, np.int16), 0.0)
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def test_silence_only_tail_after_forced_cut_is_discarded():
    # Trigger fires at frame 4 with frames 0..4 kept; the 50-frame cut lands
    # exactly on the last speech frame, leaving only silence afterwards.
    seg = Segmenter(energy_vad, silence_ms=300, pre_roll_ms=300, max_segment_s=1.5)
    frames = utterance(50, 30)
    utts = run(seg, frames)
    assert len(utts) == 1 and utts[0].forced_cut
    speech = np.concatenate(frames[:50])
    assert np.array_equal(utts[0].audio, speech)  # no speech audio lost at the cut
    assert not seg.triggered
    # segmenter still works for the next utterance
    assert len(run(seg, utterance(20, 15))) == 1


def test_vad_frame_drives_detection_while_audio_comes_from_frame():
    # VAD sees speech-like frames, but the kept audio is a different signal:
    # segmentation follows vad_frame, the utterance carries frame.
    seg = Segmenter(energy_vad, silence_ms=300, pre_roll_ms=300)
    vad_frames = [silence_frame()] * 5 + utterance(20, 15)
    audio_frames = [np.full(FRAME, 7, np.int16) for _ in vad_frames]
    out = []
    for i, (a, v) in enumerate(zip(audio_frames, vad_frames)):
        out += seg.feed(a, (i + 1) * 0.03, vad_frame=v)
    assert len(out) == 1
    assert np.all(out[0].audio == 7)
    # without vad_frame the constant (quiet) audio itself never triggers
    seg2 = Segmenter(energy_vad, silence_ms=300, pre_roll_ms=300)
    assert run(seg2, audio_frames) == []


def test_rejects_wrong_vad_frame_size():
    seg = Segmenter(energy_vad)
    try:
        seg.feed(np.zeros(FRAME, np.int16), 0.0, vad_frame=np.zeros(100, np.int16))
    except ValueError:
        return
    raise AssertionError("expected ValueError")


# ------------------------------------------------------------------ Silero VAD
class FakeSilero:
    """Stands in for the ONNX session: probability = window RMS-based; records every call."""

    def __init__(self):
        self.calls = []

    def run(self, _outputs, feeds):
        x = feeds["input"]
        self.calls.append((x.copy(), feeds["h"].copy()))
        rms = float(np.sqrt(np.mean(x[0, 64:] ** 2)))
        p = np.array([[min(1.0, rms * 10)]], np.float32)
        return p, feeds["h"] + 1.0, feeds["c"] + 1.0  # state advances on every window


def _silero(session=None, **kw):
    from audio_translator.segmenter import SileroVad
    return SileroVad(session=session or FakeSilero(), **kw)


def test_silero_scores_each_512_window_once_with_context_and_state():
    vad = _silero()
    frames = [speech_frame(i) for i in range(16)]  # 16 x 480 = 7680 samples = 15 windows of 512
    for f in frames:
        vad(f.tobytes(), 16000)
    calls = vad.session.calls
    assert len(calls) == 15
    audio = np.concatenate(frames).astype(np.float32) / 32768.0
    for k, (inp, h) in enumerate(calls):
        assert inp.shape == (1, 576)
        np.testing.assert_allclose(inp[0, 64:], audio[k * 512:(k + 1) * 512])  # each window exactly once
        ctx = np.zeros(64, np.float32) if k == 0 else audio[k * 512 - 64:k * 512]
        np.testing.assert_allclose(inp[0, :64], ctx)  # previous 64 samples as context
        assert float(h.ravel()[0]) == k  # recurrent state carried across frames


def test_silero_frame_before_a_full_window_keeps_the_previous_state():
    vad = _silero()
    assert vad(speech_frame().tobytes(), 16000) is False  # 480 < 512: nothing scored yet
    assert vad.session.calls == []
    assert vad(speech_frame(1).tobytes(), 16000) is True  # first window complete


def test_silero_hysteresis():
    vad = _silero(threshold=0.5)  # neg = 0.35; fake p = 10 * rms
    loud, mid, quiet = (np.full(512, a, np.int16).tobytes() for a in (3000, 1300, 300))  # p 0.92 / 0.40 / 0.09
    assert vad(mid, 16000) is False  # between the thresholds while off: stays off
    assert vad(loud, 16000) is True
    assert vad(mid, 16000) is True  # between the thresholds while on: stays on
    assert vad(quiet, 16000) is False


def test_silero_reset_and_validation():
    vad = _silero()
    for i in range(4):
        vad(speech_frame(i).tobytes(), 16000)
    vad.reset()
    assert vad.speech is False and vad._buf.size == 0 and not vad._h.any() and not vad._context.any()
    for bad in (0.0, 1.0, 1.5):
        try:
            _silero(threshold=bad)
        except ValueError:
            continue
        raise AssertionError(f"threshold {bad} accepted")
    try:
        vad(speech_frame().tobytes(), 8000)
    except ValueError:
        pass
    else:
        raise AssertionError("8 kHz accepted")


def test_silero_drives_the_segmenter_like_any_vad():
    seg = Segmenter(_silero())
    utts = run(seg, utterance(40, 30))
    assert len(utts) == 1 and utts[0].audio.size >= 40 * FRAME


def test_real_silero_model_ignores_silence_and_weak_steady_noise():
    """Bundled faster-whisper ONNX (offline): digital silence and low white noise never trigger."""
    from audio_translator.segmenter import silero_vad

    rng = np.random.default_rng(0)
    for frames in ([silence_frame()] * 100,
                   [(rng.normal(0, 60, FRAME)).astype(np.int16) for _ in range(100)]):
        seg = Segmenter(silero_vad(0.5))
        assert run(seg, frames + [silence_frame()] * 30) == []


def test_vad_from_settings_picks_a_fresh_backend():
    from audio_translator.config import Settings
    from audio_translator.segmenter import SileroVad, vad_from_settings

    a, b = vad_from_settings(Settings(vad_backend="silero")), vad_from_settings(Settings(vad_backend="silero"))
    assert isinstance(a, SileroVad) and a is not b
    assert callable(vad_from_settings(Settings()))  # webrtc default
    try:
        vad_from_settings(Settings(vad_backend="energy"))
    except ValueError as exc:
        assert "webrtc, silero" in str(exc)
    else:
        raise AssertionError("unknown backend accepted")


# ------------------------------------------------ pre-roll separate from the start vote
def idle(k: int) -> np.ndarray:
    """A quiet frame (never voiced) whose constant value identifies it."""
    return np.full(FRAME, k + 1, np.int16)


def trigger_index(seg, frames):
    for i, f in enumerate(frames):
        seg.feed(f, (i + 1) * 0.03)
        if seg.triggered:
            return i
    return None


def test_pre_roll_starts_exactly_pre_roll_ms_before_the_end_of_the_trigger_frame():
    for pr in (300, 450, 600, 900):
        seg = Segmenter(energy_vad, silence_ms=300, pre_roll_ms=pr)
        frames = [idle(k) for k in range(40)] + utterance(20, 15)
        utts = run(seg, frames)
        assert len(utts) == 1
        trig = 40 + 4  # 5th voiced frame of the 10-frame vote window
        first = trig + 1 - pr // 30
        u = utts[0]
        assert abs(u.speech_start - (first + 1) * 0.03) < 1e-9  # times are frame end times
        assert abs((trig + 1) * 0.03 - (u.speech_start - 0.03) - pr / 1000) < 1e-9
        kept = np.concatenate(frames[first:first + u.audio.size // FRAME])
        assert np.array_equal(u.audio, kept)


def test_start_vote_is_the_same_for_every_pre_roll():
    rng = np.random.default_rng(3)
    patterns = [
        [idle(k) for k in range(30)] + [speech_frame(i) for i in range(10)],  # plain onset
        [speech_frame() if k % 2 else idle(k) for k in range(40)],  # every other frame voiced
        [speech_frame() if k in (3, 7, 12, 15) else idle(k) for k in range(30)],  # 4 of 10: never
        [speech_frame() if k in (20, 22, 25, 27, 29) else idle(k) for k in range(40)],  # 5 scattered
        [speech_frame() if rng.random() < 0.4 else idle(k) for k in range(200)],
    ]
    for frames in patterns:
        got = {pr: trigger_index(Segmenter(energy_vad, pre_roll_ms=pr), frames) for pr in (300, 450, 600, 900)}
        assert len(set(got.values())) == 1, got
    assert trigger_index(Segmenter(energy_vad, pre_roll_ms=900), patterns[2]) is None
    assert trigger_index(Segmenter(energy_vad, pre_roll_ms=900), patterns[3]) == 29


def test_start_of_stream_keeps_only_the_frames_there_are():
    seg = Segmenter(energy_vad, silence_ms=300, pre_roll_ms=600)
    frames = [idle(0)] + utterance(20, 15)  # triggers on frame 5: only 6 frames exist
    utts = run(seg, frames)
    assert len(utts) == 1
    assert abs(utts[0].speech_start - 0.03) < 1e-9
    n = utts[0].audio.size // FRAME
    assert np.array_equal(utts[0].audio, np.concatenate(frames[:n]))  # no padding added


def test_pre_roll_right_after_an_utterance_holds_nothing_of_it():
    seg = Segmenter(energy_vad, silence_ms=300, pre_roll_ms=900)
    first = utterance(20, 10)  # ends on the 10th silent frame
    gap = [idle(100 + k) for k in range(4)]
    frames = [idle(k) for k in range(40)] + first + gap + utterance(20, 15)
    utts = run(seg, frames)
    assert len(utts) == 2
    assert np.array_equal(utts[1].audio[:4 * FRAME], np.concatenate(gap))
    joined = np.concatenate([u.audio for u in utts])
    start = 40 + 5 - 30  # first kept frame of the first utterance (900 ms back from frame 44)
    assert np.array_equal(joined, np.concatenate(frames[start:start + joined.size // FRAME]))  # no frame twice


def test_forced_cut_continuation_has_no_pre_roll_and_loses_nothing():
    seg = Segmenter(energy_vad, silence_ms=300, pre_roll_ms=600, max_segment_s=1.5)  # 50 frames
    frames = [idle(k) for k in range(40)] + utterance(170, 20)
    utts = run(seg, frames)
    assert len(utts) >= 4 and all(u.forced_cut for u in utts[:-1])
    assert all(u.audio.size <= 50 * FRAME for u in utts)  # the cap counts the pre-roll
    start = 44 + 1 - 20
    joined = np.concatenate([u.audio for u in utts])
    assert np.array_equal(joined, np.concatenate(frames[start:start + joined.size // FRAME]))
    assert np.array_equal(utts[0].audio, np.concatenate(frames[start:start + 50]))


def test_open_s_does_not_depend_on_the_pre_roll():
    frames = [idle(k) for k in range(40)] + utterance(60, 15) + [idle(k) for k in range(3)] + utterance(30, 15)
    segs = {pr: Segmenter(energy_vad, silence_ms=300, pre_roll_ms=pr) for pr in (300, 600, 900)}
    for i, f in enumerate(frames):
        vals = {}
        for pr, seg in segs.items():
            seg.feed(f, (i + 1) * 0.03)
            vals[pr] = seg.open_s
        assert vals[600] == vals[300] == vals[900], (i, vals)
    seg = Segmenter(energy_vad, pre_roll_ms=600)
    run(seg, [idle(k) for k in range(40)] + [speech_frame(i) for i in range(5)])
    assert seg.triggered and abs(seg.open_s - 0.3) < 1e-9 and abs(seg.preroll_s - 0.3) < 1e-9
    seg.reset()
    assert seg.open_s == 0.0 and seg.preroll_s == 0.0


def test_cli_pre_roll_flag_reaches_settings_and_is_cli_only():
    import pytest

    from audio_translator import cli
    from audio_translator.config import Settings

    assert Settings().pre_roll_ms == 600  # since the pre-roll bench
    assert cli.build_parser().parse_args([]).pre_roll_ms == Settings().pre_roll_ms
    assert cli.build_parser().parse_args(["--pre-roll-ms", "600"]).pre_roll_ms == 600
    for bad in ("0", "10", "5000", "abc"):
        with pytest.raises(SystemExit):
            cli.build_parser().parse_args(["--pre-roll-ms", bad])
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

    cli.main(["--pre-roll-ms", "450"], orchestrator_factory=Orch)
    assert seen["s"].pre_roll_ms == 450
    pytest.importorskip("PySide6")
    from audio_translator.gui import parse_args

    with pytest.raises(SystemExit):
        parse_args(["--pre-roll-ms", "600"])  # CLI-only: an error, not silently ignored by Qt
