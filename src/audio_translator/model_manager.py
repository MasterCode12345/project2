"""Whisper model manager: catalog, download status, background loading, bounded LRU.

Loading (and the download it may trigger) always runs on a background thread
owned by the manager. ``get`` blocks and is meant for worker threads only.
Pinned models (the one a session is using) are never evicted.
"""

from __future__ import annotations

import collections
import threading
from dataclasses import dataclass
from typing import Callable

from .asr import LOCAL_MODELS, Recognizer, local_model_dir

# Multilingual faster-whisper models (``.en`` / ``distil-*`` are English-only),
# then Vietnamese fine-tunes converted locally (see asr.LOCAL_MODELS).
# Approximate download/disk size in MB (int8 CPU memory use is of the same order).
CATALOG: dict[str, int] = {
    "tiny": 75,
    "base": 145,
    "small": 485,
    "medium": 1530,
    "large-v3-turbo": 1620,
    "large-v3": 3090,
    "phowhisper-small": 485,
    "phowhisper-medium": 1530,
}

NOT_DOWNLOADED = "not-downloaded"
DOWNLOADED = "downloaded"
LOADING = "loading"
LOADED = "loaded"
FAILED = "failed"

RecognizerFactory = Callable[[str], Recognizer]
ChangeFn = Callable[[str, str, str], None]  # (name, status, detail)
DoneFn = Callable[[str, "BaseException | None"], None]


@dataclass(frozen=True)
class ModelInfo:
    name: str
    size_mb: int
    status: str
    selected: bool = False

    @property
    def size_label(self) -> str:
        return f"{self.size_mb / 1000:.1f} GB" if self.size_mb >= 1000 else f"{self.size_mb} MB"


def hf_is_downloaded(name: str) -> bool:
    """True when the model is in the local Hugging Face cache, or converted locally (no network)."""
    if name in LOCAL_MODELS:
        return (local_model_dir(name) / "model.bin").is_file()
    try:
        from faster_whisper.utils import download_model

        download_model(name, local_files_only=True)
        return True
    except Exception:
        return False


class _Job:
    def __init__(self) -> None:
        self.done = threading.Event()
        self.error: BaseException | None = None
        self.callbacks: list[DoneFn] = []


class ModelManager:
    def __init__(
        self,
        factory: RecognizerFactory,
        *,
        max_loaded: int = 2,
        is_downloaded: Callable[[str], bool] = hf_is_downloaded,
        on_change: ChangeFn | None = None,
    ) -> None:
        self._factory = factory
        self.max_loaded = max(1, max_loaded)
        self._is_downloaded = is_downloaded
        self.on_change = on_change
        self._lock = threading.Lock()
        self._loaded: collections.OrderedDict[str, Recognizer] = collections.OrderedDict()  # LRU first
        self._jobs: dict[str, _Job] = {}
        self._failed: dict[str, str] = {}
        self._pins: collections.Counter[str] = collections.Counter()
        self._downloaded: set[str] = set()

    # ------------------------------------------------------------ queries
    @staticmethod
    def names() -> list[str]:
        return list(CATALOG)

    def _check(self, name: str) -> None:
        if name not in CATALOG:
            raise ValueError(f"Unknown model '{name}'. Choose one of: {', '.join(CATALOG)}")

    def status(self, name: str) -> str:
        self._check(name)
        with self._lock:
            if name in self._loaded:
                return LOADED
            if name in self._jobs:
                return LOADING
            if name in self._failed:
                return FAILED
        return DOWNLOADED if self.is_downloaded(name) else NOT_DOWNLOADED

    def models(self) -> list[ModelInfo]:
        return [ModelInfo(n, CATALOG[n], self.status(n)) for n in CATALOG]

    def is_loaded(self, name: str) -> bool:
        with self._lock:
            return name in self._loaded

    def loaded_names(self) -> list[str]:
        with self._lock:
            return list(self._loaded)

    # ------------------------------------------------------------ loading
    def request(self, name: str, on_done: DoneFn | None = None) -> None:
        """Ensure ``name`` is (being) loaded in the background; returns at once.

        ``on_done(name, error)`` runs once the model is resident (error None)
        or the load failed; immediately in this thread if already loaded.
        """
        self._check(name)
        with self._lock:
            if name in self._loaded:
                self._loaded.move_to_end(name)
                job = None
            else:
                job, started = self._job_locked(name)
                if on_done is not None:
                    job.callbacks.append(on_done)
        if job is None:
            if on_done is not None:
                on_done(name, None)
            return
        if started:
            self._start(name, job)

    def get(self, name: str, *, pin: bool = False) -> Recognizer:
        """Return the loaded recognizer, loading it if needed. Blocks: worker threads only.

        With ``pin=True`` the model is pinned atomically before returning.
        """
        self._check(name)
        while True:
            with self._lock:
                rec = self._loaded.get(name)
                if rec is not None:
                    self._loaded.move_to_end(name)
                    if pin:
                        self._pins[name] += 1
                    return rec
                job, started = self._job_locked(name)
            if started:
                self._start(name, job)
            job.done.wait()
            if job.error is not None:
                raise RuntimeError(f"Could not load model '{name}': {job.error}") from job.error
            # Loaded (or already evicted again by a racing load): loop and re-check.

    def pin(self, name: str) -> None:
        with self._lock:
            self._pins[name] += 1

    def unpin(self, name: str) -> None:
        evicted: list[str] = []
        with self._lock:
            if self._pins[name] > 0:
                self._pins[name] -= 1
            if self._pins[name] <= 0:
                del self._pins[name]
            evicted = self._evict_locked(keep=None)
        for n in evicted:
            self._notify(n, DOWNLOADED, "released from memory")

    def _job_locked(self, name: str) -> tuple[_Job, bool]:
        job = self._jobs.get(name)
        if job is not None:
            return job, False
        job = self._jobs[name] = _Job()
        self._failed.pop(name, None)
        return job, True

    def _start(self, name: str, job: _Job) -> None:
        size = ModelInfo(name, CATALOG[name], "").size_label
        # Local (converted) models are never downloaded: a missing one fails at load with the convert command.
        fetch = not self.is_downloaded(name) and name not in LOCAL_MODELS
        detail = f"downloading (~{size}) and loading" if fetch else "loading"
        self._notify(name, LOADING, detail)
        threading.Thread(target=self._load, args=(name, job), name=f"at-model-{name}", daemon=True).start()

    def is_downloaded(self, name: str) -> bool:
        """On-disk check (cached once true); never downloads."""
        with self._lock:
            if name in self._downloaded:
                return True
        ok = self._is_downloaded(name)
        if ok:
            with self._lock:
                self._downloaded.add(name)
        return ok

    def _load(self, name: str, job: _Job) -> None:
        # Make room first so at most max_loaded models are resident during the load.
        with self._lock:
            freed = self._evict_locked(keep=name, limit=self.max_loaded - 1)
        for n in freed:
            self._notify(n, DOWNLOADED, "released from memory")
        evicted: list[str] = []
        try:
            rec = self._factory(name)
            rec.load()
        except BaseException as exc:  # noqa: BLE001 - reported via callbacks/events
            with self._lock:
                job.error = exc
                self._jobs.pop(name, None)
                self._failed[name] = str(exc)
        else:
            with self._lock:
                self._loaded[name] = rec
                self._loaded.move_to_end(name)
                self._downloaded.add(name)
                self._jobs.pop(name, None)
                evicted = self._evict_locked(keep=name)
        # Callbacks first (e.g. apply a selection), so the status events reflect it.
        for cb in job.callbacks:
            try:
                cb(name, job.error)
            except Exception:
                pass
        job.done.set()
        if job.error is not None:
            self._notify(name, FAILED, str(job.error))
        else:
            self._notify(name, LOADED, "ready")
        for n in evicted:
            self._notify(n, DOWNLOADED, "released from memory")

    def _evict_locked(self, keep: str | None, limit: int | None = None) -> list[str]:
        limit = self.max_loaded if limit is None else limit
        evicted = []
        for n in list(self._loaded):  # least recently used first
            if len(self._loaded) <= limit:
                break
            if n == keep or self._pins.get(n, 0) > 0:
                continue
            del self._loaded[n]  # dropping the last reference frees the model
            evicted.append(n)
        return evicted

    def _notify(self, name: str, status: str, detail: str) -> None:
        cb = self.on_change
        if cb is None:
            return
        try:
            cb(name, status, detail)
        except Exception:
            pass
