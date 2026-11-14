"""Speaker indicator: green / yellow / red, from three signals (observation only).

* ``length``: how long the utterance being spoken is (open segment, audio time);
* ``backlog``: segments waiting for ASR and how long the current ASR has run;
* ``quality``: the last finished segment was filtered, took abnormally long
  to recognise for its length, or needed a decoder temperature fallback.

The level shown is the highest one; on a tie the reason follows
``PRIORITY`` (backlog > length > quality). Nothing here changes how the
pipeline processes audio. ``assess`` is pure; ``HealthTracker`` (owned by the
orchestrator's monitor thread) adds the quality hold, the down-hold smoothing
and a short history for the latency log.
"""

from __future__ import annotations

import collections
import threading
from dataclasses import dataclass

GREEN, YELLOW, RED = "green", "yellow", "red"
RANK = {GREEN: 0, YELLOW: 1, RED: 2}
LEVELS = (GREEN, YELLOW, RED)
BACKLOG, PLAYOUT, LENGTH, QUALITY = "backlog", "playout", "length", "quality"
PRIORITY = (BACKLOG, PLAYOUT, LENGTH, QUALITY)  # playout is part of the backlog, ranked after ASR

MESSAGES = {
    (LENGTH, YELLOW): "Câu dài, hãy ngắt nghỉ",
    (LENGTH, RED): "Dừng lại một chút",
    (BACKLOG, YELLOW): "Đang xử lý, nói chậm lại",
    (BACKLOG, RED): "Dừng nói, chờ hệ thống bắt kịp",
    (PLAYOUT, YELLOW): "Đang phát bản dịch, chờ một chút",
    (PLAYOUT, RED): "Đang phát bản dịch, chờ một chút",
    (QUALITY, YELLOW): "Câu vừa rồi nghe không rõ, hãy nói lại",
}
OK_MESSAGE = "Đang nghe, nói bình thường"
LEVEL_WORDS = {GREEN: "xanh", YELLOW: "vàng", RED: "đỏ"}


@dataclass
class Metrics:
    open_s: float = 0.0  # length of the utterance being spoken (0 when none is open)
    asr_waiting: int = 0  # segments queued for ASR
    asr_busy_s: float = 0.0  # how long the segment in ASR has been processed (0 when idle)
    unclear: str = ""  # why the last segment looked unclear, while its hold lasts ("" = no)
    playout_wait_s: float = 0.0  # how long the newest ready translation still waits to be played


@dataclass(frozen=True)
class Reason:
    signal: str
    level: str
    message: str
    detail: str = ""

    @property
    def text(self) -> str:
        return f"{self.message} ({self.detail})" if self.detail else self.message


def _fmt_s(x: float) -> str:
    return f"{x:.1f}".replace(".", ",") + " s"


def assess(m: Metrics, s) -> list[Reason]:
    """Active (non-green) reasons, most important first: level, then ``PRIORITY``."""
    out: list[Reason] = []
    if m.asr_waiting >= s.indicator_queue_stop or m.asr_busy_s > s.indicator_asr_stop_s:
        lvl = RED
    elif m.asr_waiting >= s.indicator_queue_warn or m.asr_busy_s > s.indicator_asr_warn_s:
        lvl = YELLOW
    else:
        lvl = GREEN
    if lvl != GREEN:
        parts = []
        if m.asr_waiting:
            parts.append(f"{m.asr_waiting} câu đang chờ")
        if m.asr_busy_s > s.indicator_asr_warn_s:
            parts.append(f"nhận dạng {_fmt_s(m.asr_busy_s)}")
        out.append(Reason(BACKLOG, lvl, MESSAGES[(BACKLOG, lvl)], ", ".join(parts)))
    if m.playout_wait_s > s.indicator_playout_stop_s:
        lvl = RED
    elif m.playout_wait_s > s.indicator_playout_warn_s:
        lvl = YELLOW
    else:
        lvl = GREEN
    if lvl != GREEN:
        out.append(Reason(PLAYOUT, lvl, MESSAGES[(PLAYOUT, lvl)], f"chờ {_fmt_s(m.playout_wait_s)}"))
    if m.open_s >= s.indicator_long_stop_s:
        out.append(Reason(LENGTH, RED, MESSAGES[(LENGTH, RED)], f"câu {_fmt_s(m.open_s)}"))
    elif m.open_s >= s.indicator_long_warn_s:
        out.append(Reason(LENGTH, YELLOW, MESSAGES[(LENGTH, YELLOW)], f"câu {_fmt_s(m.open_s)}"))
    if m.unclear:
        out.append(Reason(QUALITY, YELLOW, MESSAGES[(QUALITY, YELLOW)], m.unclear))
    out.sort(key=lambda r: (-RANK[r.level], PRIORITY.index(r.signal)))
    return out


def unclear_reason(s, *, filtered: bool, asr_s: float, audio_s: float, fallback: bool) -> str:
    """Why a finished segment looks unclear ("" = it does not)."""
    why = []
    if filtered:
        why.append("bộ lọc chặn")
    if audio_s > 0 and asr_s >= s.indicator_unclear_min_asr_s and asr_s / audio_s >= s.indicator_unclear_ratio:
        why.append(f"nhận dạng {_fmt_s(asr_s)} cho {_fmt_s(audio_s)} audio")
    if fallback and s.indicator_fallback:
        why.append("phải giải mã lại")
    return ", ".join(why)


@dataclass
class Health:
    level: str
    reasons: tuple[Reason, ...]  # active reasons, main one first (empty when green)
    metrics: Metrics

    @property
    def main(self) -> Reason | None:
        return self.reasons[0] if self.reasons else None

    @property
    def key(self) -> tuple:
        """What a change of the light means: level, main signal and message, other active messages."""
        return (self.level, *((r.signal, r.message) for r in self.reasons))

    def event_data(self) -> dict:
        m = self.main
        return {
            "level": self.level,
            "signal": m.signal if m else "",
            "reason": m.text if m else OK_MESSAGE,
            "others": [r.text for r in self.reasons[1:]],
            "metrics": {"open_s": round(self.metrics.open_s, 2), "asr_waiting": self.metrics.asr_waiting,
                        "asr_busy_s": round(self.metrics.asr_busy_s, 2), "unclear": self.metrics.unclear,
                        "playout_wait_s": round(self.metrics.playout_wait_s, 2)},
        }


class HealthTracker:
    """Quality hold + down-hold smoothing + history (60 s) of displayed levels.

    ``note_unclear``/``update``/``reset`` run on one thread (the monitor);
    ``max_between`` may be called from any worker (short lock on the history).
    """

    HISTORY_S = 60.0

    def __init__(self, settings) -> None:
        self.s = settings
        self._lock = threading.Lock()
        self._hist: collections.deque[tuple[float, str, str]] = collections.deque()
        self.reset(0.0)

    def reset(self, now: float) -> Health:
        self._unclear, self._unclear_until = "", 0.0
        self._lower_since: float | None = None
        self.current = Health(GREEN, (), Metrics())
        with self._lock:
            self._hist.clear()
            self._hist.append((now, GREEN, ""))
        return self.current

    def note_unclear(self, now: float, why: str) -> None:
        if why:
            self._unclear, self._unclear_until = why, now + self.s.indicator_unclear_hold_s

    def update(self, now: float, m: Metrics) -> Health:
        if self._unclear and now >= self._unclear_until:
            self._unclear = ""
        m.unclear = self._unclear
        reasons = tuple(assess(m, self.s))
        cand = Health(reasons[0].level if reasons else GREEN, reasons, m)
        cur = self.current
        if RANK[cand.level] >= RANK[cur.level]:
            self._lower_since = None
            self._apply(now, cand)
        else:
            if self._lower_since is None:
                self._lower_since = now
            if now - self._lower_since >= self.s.indicator_down_hold_s:
                self._lower_since = None
                self._apply(now, cand)
            else:  # hold the higher level; keep its reasons, refresh the numbers
                self.current = Health(cur.level, cur.reasons, m)
        return self.current

    def _apply(self, now: float, h: Health) -> None:
        prev = self.current
        self.current = h
        main = (h.main.signal, h.main.message) if h.main else None
        prev_main = (prev.main.signal, prev.main.message) if prev.main else None
        if (h.level, main) != (prev.level, prev_main):
            with self._lock:
                self._hist.append((now, h.level, h.main.message if h.main else ""))
                while len(self._hist) > 1 and self._hist[1][0] < now - self.HISTORY_S:
                    self._hist.popleft()

    def max_between(self, t0: float, t1: float) -> tuple[str, str]:
        """Highest level shown during [t0, t1] and its reason (the first time it was reached)."""
        with self._lock:
            hist = list(self._hist)
        best: tuple[str, str] | None = None
        for i, (t, lvl, why) in enumerate(hist):
            end = hist[i + 1][0] if i + 1 < len(hist) else float("inf")
            if end < t0 or t > t1:
                continue
            if best is None or RANK[lvl] > RANK[best[0]]:
                best = (lvl, why)
        return best or ("", "")

    def reasons_between(self, t0: float, t1: float) -> list[tuple[str, str]]:
        """Every (level, reason) the light showed as its main reason during [t0, t1], in time
        order, each once (green is left out: it has no reason)."""
        with self._lock:
            hist = list(self._hist)
        out: list[tuple[str, str]] = []
        for i, (t, lvl, why) in enumerate(hist):
            end = hist[i + 1][0] if i + 1 < len(hist) else float("inf")
            if end < t0 or t > t1 or lvl == GREEN or (lvl, why) in out:
                continue
            out.append((lvl, why))
        return out


def format_reasons(reasons: list[tuple[str, str]]) -> str:
    """``"vàng: Đang xử lý, nói chậm lại | vàng: Đang phát bản dịch, chờ một chút"``."""
    return " | ".join(f"{LEVEL_WORDS.get(lvl, lvl)}: {why}" for lvl, why in reasons)
