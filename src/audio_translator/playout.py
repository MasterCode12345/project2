"""M7: resample to the device format and release clips in segment_id order.

``read()`` runs on the PortAudio output callback, so it never blocks for long
(one short lock) and never allocates more than one output block.
"""

from __future__ import annotations

import collections
import threading
import time
from typing import Callable

import numpy as np

from .models import SpeechClip

# on_notice(kind, session_id, segment_id, message) — must be non-blocking.
NoticeFn = Callable[[str, int, int, str], None]

GAP_SKIPPED = "gap_skipped"
DROPPED_OVERLOAD = "dropped_overload"
DROPPED_LATE = "dropped_late"


def to_device_format(audio: np.ndarray, in_rate: int, out_rate: int, channels: int) -> np.ndarray:
    """Mono float32 at ``in_rate`` -> (n, channels) float32 at ``out_rate``."""
    mono = np.asarray(audio, dtype=np.float32).reshape(-1)
    if in_rate != out_rate and mono.size:
        import soxr

        mono = soxr.resample(mono, in_rate, out_rate).astype(np.float32, copy=False)
    mono = np.clip(mono, -1.0, 1.0)
    return np.repeat(mono[:, None], channels, axis=1)


class _Playing:
    __slots__ = ("clip", "pcm", "pos")

    def __init__(self, clip: SpeechClip, pcm: np.ndarray) -> None:
        self.clip = clip
        self.pcm = pcm
        self.pos = 0


class Playout:
    def __init__(
        self,
        device_rate: int,
        channels: int,
        *,
        max_pending: int = 8,
        gap_timeout_s: float = 5.0,
        clock: Callable[[], float] = time.monotonic,
        on_notice: NoticeFn | None = None,
    ) -> None:
        self.device_rate = int(device_rate)
        self.channels = int(channels)
        self.max_pending = max(1, max_pending)
        self.gap_timeout_s = gap_timeout_s
        self.clock = clock
        self.on_notice = on_notice
        self._lock = threading.Lock()
        self._session: int | None = None
        self._next_id = 1
        self._pending: dict[int, _Playing] = {}
        self._skipped: set[int] = set()
        self._ready: collections.deque[_Playing] = collections.deque()
        self._blocked_since: float | None = None
        self.dropped_stale = 0
        self.dropped_overload = 0
        self.gaps_skipped = 0

    # -- session control -------------------------------------------------
    def start_session(self, session_id: int, first_id: int = 1) -> None:
        with self._lock:
            self._reset(session_id, first_id)

    def clear(self) -> int:
        """End the session: discard everything. Returns number discarded."""
        with self._lock:
            n = len(self._pending) + len(self._ready)
            self.dropped_stale += n
            self._reset(None, 1)
            return n

    def _reset(self, session_id: int | None, first_id: int) -> None:
        self._session = session_id
        self._next_id = first_id
        self._pending.clear()
        self._skipped.clear()
        self._ready.clear()
        self._blocked_since = None

    # -- producer side (M6 worker thread) -----------------------------------
    def submit(self, clip: SpeechClip) -> bool:
        pcm = to_device_format(clip.audio, clip.sample_rate, self.device_rate, self.channels)
        notices = []
        with self._lock:
            if clip.session_id != self._session:
                self.dropped_stale += 1
                return False
            if clip.segment_id < self._next_id or clip.segment_id in self._skipped:
                notices.append((DROPPED_LATE, clip.session_id, clip.segment_id,
                                f"segment {clip.segment_id} arrived after it was skipped; dropped"))
                accepted = False
            else:
                self._pending[clip.segment_id] = _Playing(clip, pcm)
                accepted = True
                self._release(notices)
                self._enforce_bound(notices)
        self._notify(notices)
        return accepted

    def skip(self, session_id: int, segment_id: int) -> None:
        """Declare that ``segment_id`` will never arrive (empty/failed/dropped)."""
        notices = []
        with self._lock:
            if session_id != self._session or segment_id < self._next_id:
                return
            self._skipped.add(segment_id)
            self._release(notices)
        self._notify(notices)

    # -- consumer side (audio callback) ---------------------------------------
    def read(self, frames: int) -> tuple[np.ndarray, list[SpeechClip]]:
        """Return ``frames`` of audio (silence-padded) and clips that just started."""
        out = np.zeros((frames, self.channels), dtype=np.float32)
        started: list[SpeechClip] = []
        notices = []
        with self._lock:
            self._release(notices)
            filled = 0
            while filled < frames and self._ready:
                cur = self._ready[0]
                if cur.pos == 0:
                    started.append(cur.clip)
                n = min(frames - filled, cur.pcm.shape[0] - cur.pos)
                out[filled:filled + n] = cur.pcm[cur.pos:cur.pos + n]
                cur.pos += n
                filled += n
                if cur.pos >= cur.pcm.shape[0]:
                    self._ready.popleft()
        self._notify(notices)
        return out, started

    # -- internals (lock held) -------------------------------------------------
    def _release(self, notices: list) -> None:
        while True:
            if self._next_id in self._pending:
                self._ready.append(self._pending.pop(self._next_id))
                self._next_id += 1
                self._blocked_since = None
            elif self._next_id in self._skipped:
                self._skipped.discard(self._next_id)
                self._next_id += 1
                self._blocked_since = None
            elif self._pending:
                now = self.clock()
                if self._blocked_since is None:
                    self._blocked_since = now
                    return
                if now - self._blocked_since < self.gap_timeout_s:
                    return
                missing = self._next_id
                self.gaps_skipped += 1
                notices.append((GAP_SKIPPED, self._session, missing,
                                f"segment {missing} missing after {self.gap_timeout_s:.1f}s; skipped"))
                self._skipped.add(missing)
                self._blocked_since = None
            else:
                self._blocked_since = None
                return

    def _enforce_bound(self, notices: list) -> None:
        def waiting() -> int:
            playing = 1 if self._ready and self._ready[0].pos > 0 else 0
            return len(self._ready) - playing + len(self._pending)

        while waiting() > self.max_pending:
            playing = 1 if self._ready and self._ready[0].pos > 0 else 0
            if len(self._ready) > playing:
                victim = self._ready[playing]
                del self._ready[playing]
            else:
                sid = min(self._pending)
                victim = self._pending.pop(sid)
                self._skipped.add(sid)
            self.dropped_overload += 1
            notices.append((DROPPED_OVERLOAD, victim.clip.session_id, victim.clip.segment_id,
                            f"playout full; dropped oldest pending segment {victim.clip.segment_id}"))

    def _notify(self, notices: list) -> None:
        if self.on_notice:
            for n in notices:
                self.on_notice(*n)

    # -- introspection ---------------------------------------------------------
    @property
    def pending_count(self) -> int:
        with self._lock:
            return len(self._pending) + len(self._ready)
