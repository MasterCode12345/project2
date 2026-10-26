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
