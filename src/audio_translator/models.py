"""Shared data contract between pipeline stages.

Every item carries ``session_id``, ``segment_id`` and a ``timestamps`` dict of
stage name -> ``time.monotonic()`` value so latency can be computed end to end.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class Segment:
    """M3 output: one utterance of 16 kHz mono int16 audio."""

    session_id: int
    segment_id: int
    audio: np.ndarray  # int16, mono
    sample_rate: int
    forced_cut: bool = False
    timestamps: dict[str, float] = field(default_factory=dict)


@dataclass
class Transcript:
    """M4 output."""

    session_id: int
    segment_id: int
    text: str
    timestamps: dict[str, float] = field(default_factory=dict)


@dataclass
class Translation:
    """M5 output."""

    session_id: int
    segment_id: int
    source_text: str
    text: str
    timestamps: dict[str, float] = field(default_factory=dict)


@dataclass
class SpeechClip:
    """M6 output: synthesized speech as float32 mono PCM."""

    session_id: int
    segment_id: int
    audio: np.ndarray  # float32, mono, [-1, 1]
    sample_rate: int
    source_text: str = ""
    text: str = ""
    timestamps: dict[str, float] = field(default_factory=dict)

    @property
    def latency(self) -> float | None:
        """End-to-end latency: first audio out minus end of speech."""
        ts = self.timestamps
        if "first_audio_out" in ts and "speech_end" in ts:
            return ts["first_audio_out"] - ts["speech_end"]
        return None


# Event kinds emitted by the orchestrator.
STATUS = "status"  # state change / informational; data["state"] holds the state
RESULT = "result"  # a segment started playing; carries texts + latency
WARNING = "warning"  # a segment was skipped/dropped; session continues
ERROR = "error"  # fatal for the session; session has been stopped
INFO = "info"  # per-segment progress (e.g. empty transcript skipped)
MODEL = "model"  # ASR model status change; data: model, status, detail, selected

# Orchestrator states (data["state"] on STATUS events).
IDLE = "idle"
STARTING = "starting"
RUNNING = "running"
STOPPING = "stopping"
STOPPED = "stopped"


@dataclass
class Event:
    kind: str
    message: str
    session_id: int | None = None
    segment_id: int | None = None
    data: dict[str, Any] = field(default_factory=dict)
