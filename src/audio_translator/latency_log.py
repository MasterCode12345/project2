"""Per-segment latency CSV of a real session (``--latency-log PATH``, off by default).

One row per segment: ``played`` when its first audio reached the output, or
``dropped`` with the reason it never did. Rows are written by orchestrator
worker threads only (the audio callbacks just stamp ``first_audio_out`` and
post to the monitor queue). Times are ``time.monotonic()`` seconds of the
pipeline timestamps; durations are milliseconds. ``eval/latency_report.py``
summarises the file.
"""

from __future__ import annotations

import csv
import threading
from datetime import datetime
from pathlib import Path

PLAYED = "played"
DROPPED = "dropped"

STAMPS = ["speech_start", "speech_end", "segmented", "asr_start", "asr_end", "mt_start", "restored",
          "mt_end", "tts_start", "tts_end", "first_audio_out"]
SESSION_FIELDS = ["asr_model", "vad", "asr_filter", "restore", "translator"]
# (column, end stamp, start stamps: the first present one is used)
DURATIONS = [
    ("segment_ms", "segmented", ("speech_end",)),
    ("asr_wait_ms", "asr_start", ("segmented",)),
    ("asr_ms", "asr_end", ("asr_start",)),
    ("mt_wait_ms", "mt_start", ("asr_end",)),
    ("restore_ms", "restored", ("mt_start",)),
    ("mt_ms", "mt_end", ("restored", "mt_start")),
    ("tts_wait_ms", "tts_start", ("mt_end",)),
    ("tts_ms", "tts_end", ("tts_start",)),
    ("playout_wait_ms", "first_audio_out", ("tts_end",)),
    ("e2e_ms", "first_audio_out", ("speech_end",)),
]
COLUMNS = (["wall_time", "session_id", "segment_id", "status", "reason", "detail", *SESSION_FIELDS,
            "forced_cut", "audio_s", "n_words_vi", "n_words_en", *STAMPS, *(d[0] for d in DURATIONS),
            "text_vi", "text_en"])


def durations(ts: dict[str, float]) -> dict[str, float]:
    """Stage durations (ms) that both of their timestamps allow."""
    out = {}
    for col, end, starts in DURATIONS:
        start = next((ts[k] for k in starts if k in ts), None)
        if end in ts and start is not None:
            out[col] = (ts[end] - start) * 1000.0
    return out


class LatencyLog:
    """Appends rows to ``path`` (header written when the file is new or empty)."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).expanduser()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        new = not self.path.exists() or self.path.stat().st_size == 0
        self._f = open(self.path, "a", newline="", encoding="utf-8")
        self._w = csv.writer(self._f)
        self._lock = threading.Lock()
        if new:
            self._w.writerow(COLUMNS)
            self._f.flush()

    def write(self, *, session_id: int, segment_id: int, status: str, timestamps: dict[str, float],
              session: dict, reason: str = "", detail: str = "", text_vi: str = "", text_en: str = "",
              audio_s: float | None = None, forced_cut: bool | None = None) -> None:
        ts = timestamps or {}
        row = {"wall_time": datetime.now().isoformat(timespec="milliseconds"), "session_id": session_id,
               "segment_id": segment_id, "status": status, "reason": reason, "detail": detail,
               **{k: session.get(k, "") for k in SESSION_FIELDS},
               "forced_cut": "" if forced_cut is None else int(bool(forced_cut)),
               "audio_s": "" if audio_s is None else f"{audio_s:.3f}",
               "n_words_vi": len(text_vi.split()) if text_vi else 0,
               "n_words_en": len(text_en.split()) if text_en else 0,
               **{k: f"{ts[k]:.6f}" for k in STAMPS if k in ts},
               **{k: f"{v:.1f}" for k, v in durations(ts).items()},
               "text_vi": text_vi, "text_en": text_en}
        with self._lock:
            if self._f.closed:
                return
            self._w.writerow([row.get(c, "") for c in COLUMNS])
            self._f.flush()

    def close(self) -> None:
        with self._lock:
            if not self._f.closed:
                self._f.close()
