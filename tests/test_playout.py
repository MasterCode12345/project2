import numpy as np

from audio_translator.models import SpeechClip
from audio_translator.playout import DROPPED_OVERLOAD, GAP_SKIPPED, Playout, to_device_format


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def clip(seg_id, sid=1, value=None, n=100):
    v = value if value is not None else seg_id / 10
    return SpeechClip(sid, seg_id, np.full(n, v, np.float32), 1000, f"vi{seg_id}", f"en{seg_id}")


def make(**kw):
    notes = []
    clock = kw.pop("clock", Clock())
    p = Playout(1000, 2, clock=clock, on_notice=lambda *a: notes.append(a), **kw)
    p.start_session(1)
    return p, notes, clock


def drain(p, frames=1000):
    """Read until silence; return list of segment ids in play order."""
    order = []
    for _ in range(50):
        _, started = p.read(frames)
        order += [c.segment_id for c in started]
    return order


def test_resample_and_channels():
    out = to_device_format(np.zeros(24000, np.float32), 24000, 48000, 2)
    assert out.shape == (48000, 2) and out.dtype == np.float32


def test_in_order_playback_and_silence_when_empty():
    p, _, _ = make()
    data, started = p.read(64)
    assert not data.any() and started == []
    p.submit(clip(1))
    p.submit(clip(2))
    data, started = p.read(150)
    assert [c.segment_id for c in started] == [1, 2]
    assert np.allclose(data[:100, 0], 0.1) and np.allclose(data[100:150, 1], 0.2)


def test_out_of_order_completion_plays_in_order():
    p, _, _ = make()
    p.submit(clip(2))
    data, started = p.read(50)
    assert started == [] and not data.any()  # waits for 1
    p.submit(clip(1))
    assert drain(p) == [1, 2]


def test_gap_timeout_skips_missing_id_and_logs():
    p, notes, clock = make(gap_timeout_s=5.0)
    p.submit(clip(2))
    p.read(10)
    clock.t = 4.9
    assert p.read(10)[1] == []
    clock.t = 5.1
    _, started = p.read(10)
    assert [c.segment_id for c in started] == [2]
    assert notes and notes[0][0] == GAP_SKIPPED and notes[0][2] == 1
    assert p.gaps_skipped == 1
    # the late segment 1 is dropped, never played out of order
    assert p.submit(clip(1)) is False
    assert drain(p) == []


def test_skip_releases_next_without_waiting():
    p, notes, _ = make()
    p.submit(clip(2))
    p.skip(1, 1)
    assert drain(p) == [2]
    assert notes == []


def test_overload_drops_oldest_pending():
    p, notes, _ = make(max_pending=2)
    for i in (1, 2, 3):
        p.submit(clip(i))
    assert p.dropped_overload == 1
    assert notes[0][0] == DROPPED_OVERLOAD and notes[0][2] == 1
    assert drain(p) == [2, 3]


def test_overload_never_cuts_the_clip_currently_playing():
    p, _, _ = make(max_pending=1)
    p.submit(clip(1, n=1000))
    p.read(10)  # 1 is now playing
    p.submit(clip(2))
    p.submit(clip(3))  # 2 is the oldest *pending* one
    assert drain(p) == [3]  # 1 had already started (not reported again)


def test_stale_session_is_discarded():
    p, _, _ = make()
    assert p.submit(clip(1, sid=99)) is False
    p.submit(clip(1))
    assert p.clear() == 1
    data, started = p.read(500)
    assert not data.any() and started == []
    assert p.submit(clip(2)) is False  # no session -> stale
    assert p.dropped_stale == 3
