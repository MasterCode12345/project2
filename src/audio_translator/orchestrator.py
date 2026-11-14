"""M1 core: GUI-independent session control and stage workers.

Threads + bounded ``queue.Queue`` (see plan Design Notes). Workers live for the
lifetime of the orchestrator; every item carries its ``session_id`` and any
item from a session that is no longer current is dropped and counted.

Audio callbacks (M2 capture, M8 output) never call into this class beyond a
``put_nowait`` on the monitor queue; events reach the user via ``on_event`` from
worker threads only.
"""

from __future__ import annotations

import queue
import threading
import time
from typing import Any, Callable

from .capture import AudioCapture
from .config import Settings
from .denoise import NoiseSuppressor, from_settings as noise_suppressor_from_settings
from .devices import DeviceInfo, resolve_device
from .health import GREEN, HealthTracker, Metrics, format_reasons, unclear_reason
from .latency_log import DROPPED, PLAYED, LatencyLog
from .model_manager import ModelInfo, ModelManager
from .models import (
    ERROR,
    HEALTH,
    IDLE,
    INFO,
    MODEL,
    RESULT,
    RUNNING,
    STARTING,
    STATUS,
    STOPPED,
    STOPPING,
    WARNING,
    Event,
    Segment,
    SpeechClip,
    Transcript,
    Translation,
)
from .playout import Playout
from .segmenter import Segmenter, VadFn, vad_from_settings
from .virtual_out import VirtualOutput

EventFn = Callable[[Event], None]
_POLL = 0.05


class Orchestrator:
    def __init__(
        self,
        settings: Settings | None = None,
        *,
        asr=None,
        model_manager: ModelManager | None = None,
        translator=None,
        synthesizer=None,
        vad_factory: Callable[[int], VadFn] | None = None,
        input_stream_factory: Callable[..., Any] | None = None,
        output_stream_factory: Callable[..., Any] | None = None,
        device_resolver: Callable[[str | None, str], DeviceInfo] = resolve_device,
        on_event: EventFn | None = None,
        clock: Callable[[], float] = time.monotonic,
        package_preparer: Callable[..., Any] | None = None,
        offline_translator_factory: Callable[[Any], Any] | None = None,
        google_translator_factory: Callable[[str, str], Any] | None = None,
        mt_model_factory: Callable[[str, str, str], Any] | None = None,
        restorer_factory: Callable[[str], Any] | None = None,
    ) -> None:
        self.settings = settings or Settings()
        self._asr = asr
        # MT: an injected translator is used as-is; otherwise the backend is
        # chosen from settings at each session start (see _prepare_translator).
        self._translator = translator
        self._translator_fixed = translator is not None
        self._offline_tr = None  # loaded OfflineTranslator, reused across sessions
        self._package_preparer = package_preparer
        self._offline_factory = offline_translator_factory
        self._google_factory = google_translator_factory
        self._mt_model_factory = mt_model_factory  # (name, source, target) -> translator; None: CT2 model
        self._mt_model = None  # ((name, source, target), loaded multilingual model), reused across sessions
        self._restorer_factory = restorer_factory  # name -> object with restore(text, lang); None: restore.py
        self._restorer_model = None  # (name, loaded restore model), reused across sessions
        self._restore = None  # (text) -> text for the current session, None = off
        self._synth = synthesizer
        self._vad_factory = vad_factory  # None: backend from settings (webrtc | silero), fresh per session
        self._in_factory = input_stream_factory
        self._out_factory = output_stream_factory
        self._resolve = device_resolver
        self.on_event = on_event
        self.clock = clock

        s = self.settings
        # ASR: an injected recognizer is used as-is; otherwise a ModelManager
        # owns cached Whisper models and the selection API is available.
        self._models = model_manager
        if asr is None and model_manager is None:
            self._models = ModelManager(self._make_recognizer, max_loaded=s.max_loaded_models)
        if self._models is not None:
            self._models.on_change = self._on_model_change
        self._selected = s.model_size  # model used by the next session
        self._pending: str | None = None  # selection waiting for its background load
        self._select_token: object | None = None
        self._pinned: str | None = None  # model pinned by the current session
        self._frames_q: queue.Queue = queue.Queue(s.frame_queue_size)
        self._seg_q: queue.Queue = queue.Queue(s.segment_queue_size)
        self._tr_q: queue.Queue = queue.Queue(s.transcript_queue_size)
        self._tl_q: queue.Queue = queue.Queue(s.translation_queue_size)
        self._monitor_q: queue.Queue = queue.Queue(s.event_queue_size)

        self._lock = threading.RLock()
        self._state = IDLE
        self._session: int | None = None
        self._session_counter = 0
        self._seg_counter = 0
        self._abort_start = False
        self._fatal: tuple[int, str] | None = None
        self._capture: AudioCapture | None = None
        self._output: VirtualOutput | None = None
        self._playout: Playout | None = None
        self._segmenter: Segmenter | None = None
        self._ns: NoiseSuppressor | None = None  # M3a stage of the current session (None = off)
        self._in_dev: DeviceInfo | None = None
        self._out_dev: DeviceInfo | None = None
        # --latency-log: per-session CSV writer (None = off), the session's configuration
        # written on every row, per-segment audio facts, and segments already logged.
        self._latlog: LatencyLog | None = None
        self._lat_session: dict = {}
        self._lat_seg: dict[tuple[int, int], tuple[float, bool]] = {}
        self._lat_done: set[tuple[int, int]] = set()
        self._lat_lock = threading.Lock()
        # Speaker indicator (health.py): the tracker lives on the monitor thread; the ASR
        # worker only stamps _asr_busy = (session, asr_start) and posts "unclear" notes.
        self._health = HealthTracker(s)
        self._health_sid: int | None = None  # session the tracker was reset for
        self._health_tick = float("-inf")
        self._health_emit_t = float("-inf")
        self._health_key: tuple | None = None
        self._asr_busy: tuple[int, float] | None = None

        self._shutdown = threading.Event()
        self._workers: list[threading.Thread] = []

        # counters (cumulative over the orchestrator lifetime)
        self.dropped_stale = 0
        self.dropped_overload = 0
        self.segments_skipped = 0

    # ------------------------------------------------------------------ API
    @property
    def state(self) -> str:
        return self._state

    @property
    def current_session(self) -> int | None:
        return self._session

    @property
    def selected_model(self) -> str:
        return self._selected

    def list_models(self) -> list[ModelInfo]:
        """Catalog with status (not-downloaded/downloaded/loading/loaded/failed)."""
        if self._models is None:
            return []
        return [ModelInfo(m.name, m.size_mb, m.status, m.name == self._selected)
                for m in self._models.models()]

    def select_model(self, name: str) -> bool:
        """Choose the ASR model for the next session; never blocks on loading.

        A loaded model becomes selected at once; otherwise it is loaded in the
        background and selected when ready (latest call wins). Refused (False)
        while a session is starting/running or for an unknown model.
        """
        with self._lock:
            if self._models is None:
                self._emit(Event(WARNING, "Model selection is unavailable (fixed ASR injected)"))
                return False
            if self._state not in (IDLE, STOPPED):
                self._emit(Event(WARNING, f"Cannot change the model while a session is {self._state}; "
                                          "stop it first", data={"model": name}))
                return False
            if name not in self._models.names():
                self._emit(Event(ERROR, f"Unknown model '{name}'; keeping '{self._selected}'",
                                 data={"model": name}))
                return False
            token = self._select_token = object()
            self._pending = name
        self._models.request(name, on_done=lambda n, err, _t=token: self._on_selected_loaded(_t, n, err))
        return True

    def _on_selected_loaded(self, token: object, name: str, error: BaseException | None) -> None:
        with self._lock:
            if self._select_token is not token:
                return  # superseded by a later selection: the model just stays cached
            self._pending = None
            if error is None:
                self._selected = name
        if error is None:
            self._emit(Event(MODEL, f"Model '{name}' selected", data={
                "model": name, "status": "loaded", "detail": "selected", "selected": True}))
        else:
            self._emit(Event(ERROR, f"Could not load model '{name}': {error}; keeping '{self._selected}'",
                             data={"model": name}))

    def _on_model_change(self, name: str, status: str, detail: str) -> None:
        self._emit(Event(MODEL, f"Model '{name}': {detail}", data={
            "model": name, "status": status, "detail": detail, "selected": name == self._selected}))

    def _make_recognizer(self, name: str):
        from .asr import WhisperASR, filter_from_settings

        s = self.settings
        return WhisperASR(name, language=s.source_lang, device=s.asr_device,
                          compute_type=s.compute_type, beam_size=s.beam_size,
                          initial_prompt=s.asr_initial_prompt, segment_filter=filter_from_settings(s),
                          max_tokens_per_s=s.asr_max_tokens_per_s, max_tokens_base=s.asr_max_tokens_base,
                          temperatures=s.asr_temperatures)

    def _release_model(self) -> None:
        name, self._pinned = self._pinned, None
        if name is not None and self._models is not None:
            self._asr = None
            self._models.unpin(name)

    def start(self, input_device: str | None = None, output_device: str | None = None) -> bool:
        """Begin a session asynchronously (model load + device open off-thread).

        Returns False if a session is already starting or running.
        """
        with self._lock:
            if self._state not in (IDLE, STOPPED):
                return False
            self._abort_start = False
            self._set_state(STARTING, "Starting…")
        self._ensure_workers()
        threading.Thread(
            target=self._start_session,
            args=(input_device if input_device is not None else self.settings.input_device,
                  output_device or self.settings.output_device),
            name="at-start",
            daemon=True,
        ).start()
        return True

    def stop(self) -> None:
        """Stop the current session; nothing from it reaches the device afterwards."""
        with self._lock:
            if self._state == STARTING:
                self._abort_start = True
                return
        self._stop_session()

    def shutdown(self) -> None:
        self.stop()
        self._shutdown.set()
        # Workers are daemons and may be mid ASR/MT/TTS call: share one short
        # deadline so closing the window never freezes the UI for long.
        deadline = time.monotonic() + 0.5
        for t in self._workers:
            t.join(timeout=max(0.0, deadline - time.monotonic()))

    def wait_until_stopped(self, timeout: float | None = None) -> bool:
        end = None if timeout is None else time.monotonic() + timeout
        while self._state not in (IDLE, STOPPED):
            if end is not None and time.monotonic() > end:
                return False
            time.sleep(_POLL)
        return True

    def stats(self) -> dict[str, int]:
        p = self._playout
        c = self._capture
        return {
            "dropped_stale": self.dropped_stale + (p.dropped_stale if p else 0),
            "dropped_overload": self.dropped_overload + (p.dropped_overload if p else 0),
            "segments_skipped": self.segments_skipped,
            "gaps_skipped": p.gaps_skipped if p else 0,
            "capture_overflows": c.overflows if c else 0,
        }

    # ------------------------------------------------------------- lifecycle
    def _components(self):
        s = self.settings
        if self._synth is None:
            from .tts import EdgeTTS

            self._synth = EdgeTTS(s.tts_voice)

    def _start_session(self, input_name: str | None, output_name: str) -> None:
        s = self.settings
        try:
            self._components()
            if not self._synth.supports(s.target_lang):
                raise RuntimeError(f"TTS voice does not support target language '{s.target_lang}'")
            # Resolve devices first: a missing BlackHole must not wait for a model download.
            in_dev = self._resolve(input_name, "input")
            out_dev = self._resolve(output_name, "output")
            # Fresh M3a stage per session (validates the ns_* settings early):
            # no enhancer/gate/delay-line state leaks from a previous session.
            ns = noise_suppressor_from_settings(s)
            vad = self._vad_factory(s.vad_aggressiveness) if self._vad_factory else vad_from_settings(s)
            self._prepare_translator()
            self._prepare_restorer()
            with self._lock:
                if self._abort_start:  # Stop during the translation-model download
                    self._set_state(STOPPED, "Stopped")
                    return
            if self._models is None:
                self._emit(Event(STATUS, "Loading ASR model…", data={"state": STARTING}))
                self._asr.load()
            else:
                # A selection still loading in the background is waited for (same load).
                name = self._pending or self._selected
                usable = self._usable_model(name)
                fell_back = usable != name
                name = usable
                if not self._models.is_loaded(name):
                    self._emit(Event(STATUS, f"Loading ASR model '{name}'…", data={"state": STARTING}))
                try:
                    rec = self._models.get(name, pin=True)
                except Exception:
                    if name == self._selected or fell_back:
                        raise
                    # Failed pending selection (its callback reports the error):
                    # fall back to the previous, still-selected model.
                    with self._lock:
                        if self._pending == name:
                            self._pending = None
                    name = self._selected
                    rec = self._models.get(name, pin=True)
                with self._lock:
                    self._pinned = name
                    self._asr = rec
                    if self._pending == name:
                        self._pending = None
                        self._selected = name
        except Exception as exc:
            with self._lock:
                self._release_model()
            self._fail_start(f"Could not start: {exc}")
            return

        with self._lock:
            if self._abort_start:
                self._release_model()
                self._set_state(STOPPED, "Stopped")
                return
            self._session_counter += 1
            sid = self._session_counter
            self._seg_counter = 0
            self._fatal = None
            self._in_dev, self._out_dev = in_dev, out_dev
            self._segmenter = Segmenter(
                vad,
                sample_rate=s.capture_rate,
                frame_ms=s.frame_ms,
                pre_roll_ms=s.pre_roll_ms,
                silence_ms=s.silence_ms,
                max_segment_s=s.max_segment_s,
            )
            self._ns = ns
            self._playout = Playout(
                int(out_dev.samplerate),
                min(2, max(1, out_dev.max_channels)),
                max_pending=s.playout_max_pending,
                gap_timeout_s=s.gap_timeout_s,
                clock=self.clock,
                on_notice=lambda kind, ssid, seg, msg: self._post(("notice", kind, ssid, seg, msg)),
            )
            self._playout.start_session(sid, first_id=1)
            self._output = VirtualOutput(
                self._playout,
                device=out_dev.index,
                device_name=out_dev.name,
                blocksize=s.output_blocksize,
                on_clip_started=lambda clip: self._post(("started", clip)),
                on_error=lambda msg, _sid=sid: self._post_fatal(_sid, msg),
                stream_factory=self._out_factory,
                clock=self.clock,
            )
            self._capture = AudioCapture(
                sid,
                self._frames_q,
                device=in_dev.index,
                device_name=in_dev.name,
                samplerate=s.capture_rate,
                blocksize=s.frame_samples,
                on_error=lambda msg, _sid=sid: self._post_fatal(_sid, msg),
                stream_factory=self._in_factory,
                clock=self.clock,
            )
            self._open_latency_log()
            self._session = sid
            try:
                self._output.start()
            except Exception as exc:
                self._abort_session(f"Cannot open output device '{out_dev.name}': {exc}")
                return
            try:
                self._capture.start()
            except Exception as exc:
                self._abort_session(f"Cannot open input device '{in_dev.name}': {exc}")
                return
            self._set_state(
                RUNNING,
                f"Listening on '{in_dev.name}' → speaking into '{out_dev.name}'",
                session_id=sid,
            )

    def _usable_model(self, name: str) -> str:
        """``name``, or ``Settings.asr_fallback_model`` with a warning when ``name`` is a local
        model (PhoWhisper) that was never converted: the session runs instead of failing."""
        from .asr import LOCAL_MODELS, convert_command, local_model_dir
        from .model_manager import NOT_DOWNLOADED

        fallback = self.settings.asr_fallback_model
        if (name not in LOCAL_MODELS or not fallback or fallback == name
                or self._models.status(name) != NOT_DOWNLOADED):
            return name
        self._emit(Event(WARNING, f"Model '{name}' is not converted ({local_model_dir(name)}); "
                                  f"using '{fallback}' for this session. Convert it once with: "
                                  f"{convert_command(name)}",
                         data={"model_fallback": True, "model": name, "fallback": fallback,
                               "command": convert_command(name)}))
        return fallback

    def _prepare_translator(self) -> None:
        """Pick the MT backend for this session (runs in the session-start thread).

        Offline mode downloads/loads the Argos package once and wraps it with a
        per-segment Google fallback; if the package cannot be obtained or loaded
        the session uses Google with a warning (and offline is retried next start).
        A multilingual model (``translate.MT_MODELS``, converted once by
        ``scripts/convert_mt_model.py``) is loaded the same way, never downloaded.
        """
        if self._translator_fixed:
            return
        s = self.settings
        from .translate import MT_MODELS

        choices = ("offline", "google", *MT_MODELS)
        if s.translator not in choices:
            raise ValueError(f"Unknown translator '{s.translator}' (use one of {', '.join(choices)})")

        def google():
            if self._google_factory is not None:
                return self._google_factory(s.source_lang, s.target_lang)
            from .translate import GoogleTranslator

            return GoogleTranslator(s.source_lang, s.target_lang)

        if s.translator == "google":
            self._translator = google()
            return

        from .translate import CT2MultilingualTranslator, FallbackTranslator, OfflineTranslator, ensure_argos_package

        def on_fallback(exc: BaseException) -> None:
            self._emit(Event(WARNING, f"Offline translation failed ({exc}); "
                                      "using Google Translate for this segment"))

        if s.translator in MT_MODELS:
            key = (s.translator, s.source_lang, s.target_lang)
            if self._mt_model is None or self._mt_model[0] != key:
                try:
                    model = (self._mt_model_factory or CT2MultilingualTranslator)(*key)
                    load = getattr(model, "load", None)
                    if load is not None:
                        load()
                    self._mt_model = (key, model)
                except Exception as exc:
                    self._emit(Event(WARNING, f"Offline translation model {s.translator} unavailable ({exc}); "
                                              "using Google Translate"))
                    self._translator = google()
                    return
            self._translator = FallbackTranslator(self._mt_model[1], google, on_fallback)
            return

        if self._offline_tr is None:
            def on_download():
                self._emit(Event(STATUS, "Downloading translation model (~66 MB)…",
                                 data={"state": STARTING}))

            try:
                prepare = self._package_preparer or ensure_argos_package
                pkg = prepare(s.translation_cache_dir, s.argos_vi_en_url, on_download=on_download,
                              source=s.source_lang, target=s.target_lang)
                offline = (self._offline_factory or OfflineTranslator)(pkg)
                load = getattr(offline, "load", None)
                if load is not None:
                    load()
                self._offline_tr = offline
            except Exception as exc:
                self._emit(Event(WARNING, f"Offline translation unavailable ({exc}); "
                                          "using Google Translate"))
                self._translator = google()
                return

        self._translator = FallbackTranslator(self._offline_tr, google, on_fallback)

    def _prepare_restorer(self) -> None:
        """Casing/punctuation restoration for this session (``Settings.restore``), never downloaded.

        ``auto`` (the default) uses a converted model for the source language and
        silently does nothing when there is none. A model named explicitly that
        is not converted, or does not support the source language, leaves the
        session without restoration with a warning; it is retried next session.
        """
        s = self.settings
        from .restore import APP_RESTORERS, RESTORE_MODELS, auto_restorer_name, make_restorer

        if s.restore not in APP_RESTORERS:
            raise ValueError(f"Unknown restorer '{s.restore}' (use one of {', '.join(APP_RESTORERS)})")
        self._restore = None
        lang = s.source_lang
        name = auto_restorer_name(lang) if s.restore == "auto" else s.restore
        if name is None or name == "off":
            return
        if name not in RESTORE_MODELS:
            fn = make_restorer(name)
            self._restore = lambda text: fn(text, lang)
            return
        if self._restorer_model is None or self._restorer_model[0] != name:
            try:
                if self._restorer_factory is not None:
                    model = self._restorer_factory(name)
                else:
                    from .restore import CapuRestorer

                    model = CapuRestorer(name)
                if not model.supports(lang):
                    self._emit(Event(WARNING, f"Restorer {name} does not support '{lang}'; "
                                              "translating the transcript as recognised"))
                    return
                load = getattr(model, "load", None)
                if load is not None:
                    load()
                self._restorer_model = (name, model)
            except Exception as exc:
                self._emit(Event(WARNING, f"Restorer {name} unavailable ({exc}); "
                                          "translating the transcript as recognised"))
                return
        model = self._restorer_model[1]
        self._restore = lambda text: model.restore(text, lang)

    def _fail_start(self, msg: str) -> None:
        with self._lock:
            self._emit(Event(ERROR, msg))
            self._set_state(STOPPED, "Stopped")

    def _abort_session(self, msg: str) -> None:
        # lock held by caller
        self._teardown_streams()
        self._session = None
        if self._playout:
            self._playout.clear()
        self._release_model()
        self._close_latency_log()
        self._emit(Event(ERROR, msg))
        self._set_state(STOPPED, "Stopped")

    def _teardown_streams(self) -> None:
        cap, out = self._capture, self._output
        if cap:
            cap.stop()
        if out:
            out.stop()

    def _stop_session(self, error: str | None = None, expected_sid: int | None = None) -> None:
        with self._lock:
            sid = self._session
            if sid is None or (expected_sid is not None and sid != expected_sid):
                return
            self._set_state(STOPPING, "Stopping…", session_id=sid)
            self._session = None  # from here on every in-flight item is stale
            self._teardown_streams()
            discarded = self._playout.clear() if self._playout else 0
            discarded += self._drain_stale()
            self._segmenter = None
            self._ns = None
            self._release_model()
            self._close_latency_log()
            if error:
                self._emit(Event(ERROR, error, session_id=sid))
            stats = self.stats()
            self._set_state(
                STOPPED,
                f"Stopped (discarded {discarded} in-flight item(s))",
                session_id=sid,
                extra={"discarded": discarded, **stats},
            )

    def _drain_stale(self) -> int:
        n = 0
        for q in (self._frames_q, self._seg_q, self._tr_q, self._tl_q):
            while True:
                try:
                    q.get_nowait()
                except queue.Empty:
                    break
                if q is not self._frames_q:
                    n += 1
                    self.dropped_stale += 1
        return n

    # ----------------------------------------------------------- latency log
    def _open_latency_log(self) -> None:
        """Open ``Settings.latency_log`` for this session (lock held); a file error only warns."""
        self._close_latency_log()
        s = self.settings
        if not s.latency_log:
            return
        asr = self._asr
        model = self._pinned or getattr(asr, "model_size", None) or type(asr).__name__
        filt = getattr(asr, "segment_filter", None)
        restore = "off" if self._restore is None else (
            self._restorer_model[0] if self._restorer_model else s.restore)
        self._lat_session = {
            "asr_model": model,  # the model that really runs (a fallback, not the selection)
            "asr_decode": getattr(asr, "decode", ""),  # bounded decoding in use (base = off)
            "vad": s.vad_backend if self._vad_factory is None else "injected",
            "asr_filter": "on" if filt is not None else ("off" if hasattr(asr, "segment_filter") else ""),
            "restore": restore,
            "translator": s.translator if not self._translator_fixed else type(self._translator).__name__,
        }
        self._lat_seg.clear()
        self._lat_done.clear()
        try:
            self._latlog = LatencyLog(s.latency_log)
        except OSError as exc:
            self._emit(Event(WARNING, f"Latency log disabled: cannot open {s.latency_log}: {exc}"))
            return
        self._emit(Event(INFO, f"Latency log: {self._latlog.path}", data={"latency_log": str(self._latlog.path)}))

    def _close_latency_log(self) -> None:
        log, self._latlog = self._latlog, None
        if log is not None:
            log.close()

    def _log_latency(self, sid: int, seg_id: int, status: str, ts: dict, *, reason: str = "",
                     detail: str = "", text_vi: str = "", text_en: str = "") -> None:
        """One row per segment (first outcome wins); called from worker threads only."""
        log = self._latlog
        if log is None or sid != self._session:
            return
        with self._lat_lock:
            if (sid, seg_id) in self._lat_done:
                return
            self._lat_done.add((sid, seg_id))
            audio_s, forced = self._lat_seg.pop((sid, seg_id), (None, None))
        start = ts.get("speech_start", ts.get("segmented"))
        if start is None or sid != self._health_sid:
            hmax, hwhy, hall = "", "", ""
        else:
            now = self.clock()
            hmax, hwhy = self._health.max_between(start, now)
            hall = format_reasons(self._health.reasons_between(start, now))
        try:
            log.write(session_id=sid, segment_id=seg_id, status=status, timestamps=ts,
                      session=self._lat_session, reason=reason, detail=detail, text_vi=text_vi,
                      text_en=text_en, audio_s=audio_s, forced_cut=forced, health_max=hmax,
                      health_reason=hwhy, health_reasons=hall)
        except (OSError, ValueError) as exc:  # disk full, closed file: never stop the session
            self._latlog = None
            log.close()
            self._emit(Event(WARNING, f"Latency log disabled: write failed: {exc}"))

    def _log_drop(self, item, reason: str, detail: str = "") -> None:
        text_vi = getattr(item, "source_text", None) or getattr(item, "text", "") or ""
        text_en = getattr(item, "text", "") if hasattr(item, "source_text") else ""
        self._log_latency(item.session_id, item.segment_id, DROPPED, dict(getattr(item, "timestamps", {})),
                          reason=reason, detail=detail, text_vi=text_vi if isinstance(text_vi, str) else "",
                          text_en=text_en if isinstance(text_en, str) else "")

    # --------------------------------------------------------------- helpers
    def _set_state(self, state: str, message: str, session_id: int | None = None,
                   extra: dict | None = None) -> None:
        self._state = state
        self._emit(Event(STATUS, message, session_id=session_id, data={"state": state, **(extra or {})}))

    def _emit(self, ev: Event) -> None:
        cb = self.on_event
        if cb is None:
            return
        try:
            cb(ev)
        except Exception:
            pass  # a broken listener must not take the pipeline down

    def _post(self, item: tuple) -> None:
        """Called from audio callbacks: never blocks."""
        try:
            self._monitor_q.put_nowait(item)
        except queue.Full:
            pass

    def _post_fatal(self, sid: int, msg: str) -> None:
        if self._fatal is None:
            self._fatal = (sid, msg)

    def _is_current(self, item) -> bool:
        if item.session_id != self._session:
            self.dropped_stale += 1
            return False
        return True

    def _skip_report(self, item, kind: str, message: str, data: dict | None = None, *,
                     reason: str = "", detail: str = "") -> None:
        """Skip a failed/empty segment and report it — only if its session is current."""
        if not self._is_current(item):
            return
        self._skip(item.session_id, item.segment_id)
        if reason:
            self._log_drop(item, reason, detail)
        self._emit(Event(kind, message, item.session_id, item.segment_id, data=data or {}))

    def _skip(self, sid: int, seg_id: int) -> None:
        self.segments_skipped += 1
        p = self._playout
        if p is not None:
            p.skip(sid, seg_id)

    def _put(self, q: queue.Queue, item) -> None:
        """Bounded put; on overload drop the oldest pending item and warn."""
        while True:
            try:
                q.put_nowait(item)
                return
            except queue.Full:
                try:
                    old = q.get_nowait()
                except queue.Empty:
                    continue
                if old.session_id != self._session:
                    self.dropped_stale += 1
                    continue
                self.dropped_overload += 1
                self._skip(old.session_id, old.segment_id)
                self._log_drop(old, "overload", "pipeline queue full; oldest pending segment dropped")
                self._emit(Event(WARNING, f"Overload: dropped oldest pending segment {old.segment_id}",
                                 session_id=old.session_id, segment_id=old.segment_id))

    # --------------------------------------------------------------- workers
    def _ensure_workers(self) -> None:
        with self._lock:
            if self._workers:
                return
            for name, fn in (
                ("at-m3-segment", self._segment_worker),
                ("at-m4-asr", self._asr_worker),
                ("at-m5-translate", self._mt_worker),
                ("at-m6-tts", self._tts_worker),
                ("at-monitor", self._monitor_worker),
            ):
                t = threading.Thread(target=self._guard(fn), name=name, daemon=True)
                t.start()
                self._workers.append(t)

    def _guard(self, fn):
        def run():
            while not self._shutdown.is_set():
                try:
                    fn()
                    return
                except Exception as exc:  # keep the worker alive, report it
                    self._emit(Event(WARNING, f"{threading.current_thread().name} error: {exc}"))
        return run

    def _get(self, q: queue.Queue):
        try:
            return q.get(timeout=_POLL)
        except queue.Empty:
            return None

    def _segment_worker(self) -> None:
        while not self._shutdown.is_set():
            item = self._get(self._frames_q)
            if item is None:
                continue
            sid, frame, t = item
            seg = self._segmenter
            ns = self._ns
            if sid != self._session or seg is None:
                continue  # stale raw frames are discarded silently
            vad_frame, lat = None, 0.0
            if ns is not None:
                # M3a: VAD sees the denoised signal, ASR the OA mix; both lag
                # the capture time by the enhancer delay, so timestamps are
                # shifted back to when the sound was actually captured.
                vad_frame, frame = ns.process(frame)
                lat = ns.latency_s
            for utt in seg.feed(frame, t, vad_frame=vad_frame):
                with self._lock:
                    if sid != self._session:
                        break
                    self._seg_counter += 1
                    seg_id = self._seg_counter
                segment = Segment(
                    sid, seg_id, utt.audio, self.settings.capture_rate, forced_cut=utt.forced_cut,
                    timestamps={"speech_start": utt.speech_start - lat, "speech_end": utt.speech_end - lat,
                                "segmented": self.clock()},
                )
                if self._latlog is not None:
                    self._lat_seg[(sid, seg_id)] = (len(utt.audio) / self.settings.capture_rate, utt.forced_cut)
                self._put(self._seg_q, segment)

    def _asr_worker(self) -> None:
        while not self._shutdown.is_set():
            seg: Segment | None = self._get(self._seg_q)
            if seg is None or not self._is_current(seg):
                continue
            ts = dict(seg.timestamps)
            ts["asr_start"] = self.clock()
            asr = self._asr
            if asr is None:
                continue  # session ended; the segment is stale
            self._asr_busy = (seg.session_id, ts["asr_start"])
            try:
                text = asr.transcribe(seg.audio, seg.sample_rate).strip()
            except Exception as exc:
                self._asr_busy = None
                self._skip_report(seg, WARNING, f"Segment {seg.segment_id}: ASR failed: {exc}",
                                  reason="asr_error", detail=str(exc))
                continue
            ts["asr_end"] = self.clock()
            self._asr_busy = None
            # ASR filter reasons (one ASR worker: read right after transcribe), e.g. a ghost sentence.
            filtered = [d for d in getattr(asr, "dropped", None) or [] if d[0] not in ("repetition", "unk")]
            why = unclear_reason(self.settings, filtered=bool(filtered), asr_s=ts["asr_end"] - ts["asr_start"],
                                 audio_s=len(seg.audio) / seg.sample_rate if seg.sample_rate else 0.0,
                                 fallback=(getattr(asr, "last_temperature", 0.0) or 0.0) > 0)
            if why:
                self._post(("unclear", seg.session_id, seg.segment_id, why))
            if not text:
                seg.timestamps.update(ts)
                self._skip_report(seg, INFO, f"Segment {seg.segment_id}: no speech recognised; skipped",
                                  reason="no_speech",
                                  detail="; ".join(f"{why}: {txt}" for why, txt in filtered))
                continue
            if not self._is_current(seg):
                continue
            self._put(self._tr_q, Transcript(seg.session_id, seg.segment_id, text, ts))

    def _mt_worker(self) -> None:
        while not self._shutdown.is_set():
            tr: Transcript | None = self._get(self._tr_q)
            if tr is None or not self._is_current(tr):
                continue
            ts = dict(tr.timestamps)
            ts["mt_start"] = self.clock()
            text = tr.text
            restore = self._restore
            if restore is not None:
                try:
                    text = restore(text) or text
                except Exception as exc:  # never lose a segment to restoration
                    self._emit(Event(WARNING, f"Segment {tr.segment_id}: restoration failed ({exc}); "
                                              "translating as recognised"))
                ts["restored"] = self.clock()
            try:
                en = (self._translator.translate(text) or "").strip()
            except Exception as exc:
                tr.timestamps.update(ts)
                self._skip_report(tr, WARNING, f"Segment {tr.segment_id}: translation failed: {exc}",
                                  {"source_text": tr.text}, reason="mt_error", detail=str(exc))
                continue
            ts["mt_end"] = self.clock()
            if not en:
                tr.timestamps.update(ts)
                self._skip_report(tr, INFO, f"Segment {tr.segment_id}: empty translation; skipped",
                                  reason="empty_translation")
                continue
            if not self._is_current(tr):
                continue
            self._put(self._tl_q, Translation(tr.session_id, tr.segment_id, tr.text, en, ts))

    def _tts_worker(self) -> None:
        while not self._shutdown.is_set():
            tl: Translation | None = self._get(self._tl_q)
            if tl is None or not self._is_current(tl):
                continue
            ts = dict(tl.timestamps)
            ts["tts_start"] = self.clock()
            try:
                audio, rate = self._synth.synthesize(tl.text)
            except Exception as exc:
                tl.timestamps.update(ts)
                self._skip_report(tl, WARNING, f"Segment {tl.segment_id}: speech synthesis failed: {exc}",
                                  {"source_text": tl.source_text, "text": tl.text}, reason="tts_error",
                                  detail=str(exc))
                continue
            ts["tts_end"] = self.clock()
            clip = SpeechClip(tl.session_id, tl.segment_id, audio, rate, tl.source_text, tl.text, ts)
            playout = self._playout
            if not self._is_current(clip) or playout is None:
                continue
            playout.submit(clip)  # stale clips are rejected (and counted) inside

    def _monitor_worker(self) -> None:
        while not self._shutdown.is_set():
            item = self._get(self._monitor_q)
            if item is not None:
                self._handle_monitor(item)
            self._tick_health()
            fatal = self._fatal
            if fatal is not None:
                self._fatal = None
                self._stop_session(error=fatal[1], expected_sid=fatal[0])
                continue
            self._watchdog()

    def _handle_monitor(self, item: tuple) -> None:
        kind = item[0]
        if kind == "started":
            clip: SpeechClip = item[1]
            if clip.session_id != self._session:
                return
            lat = clip.latency
            self._log_latency(clip.session_id, clip.segment_id, PLAYED, dict(clip.timestamps),
                              text_vi=clip.source_text, text_en=clip.text)
            lat_s = f"{lat:.2f}s" if lat is not None else "n/a"
            self._emit(Event(
                RESULT,
                f"[{clip.segment_id}] VI: {clip.source_text} | EN: {clip.text} | latency {lat_s}",
                clip.session_id, clip.segment_id,
                data={"source_text": clip.source_text, "text": clip.text, "latency": lat,
                      "timestamps": dict(clip.timestamps)},
            ))
        elif kind == "unclear":
            _, sid, _seg_id, why = item
            if sid == self._session:
                now = self.clock()
                self._health_session(sid, now)
                self._health.note_unclear(now, why)
        elif kind == "notice":
            _, nkind, sid, seg_id, msg = item
            if sid == self._session:
                self._log_latency(sid, seg_id, DROPPED, {}, reason=f"playout_{nkind}", detail=msg)
                self._emit(Event(WARNING, msg, sid, seg_id, data={"notice": nkind}))

    # ------------------------------------------------------ speaker indicator
    def _health_session(self, sid: int, now: float) -> None:
        """Monitor thread: a new session starts from green, nothing carried over."""
        if sid != self._health_sid:
            self._health_sid = sid
            self._health_key = None
            self._health.reset(now)

    def _tick_health(self) -> None:
        """Monitor thread: evaluate the indicator (at most indicator_rate_hz) and emit HEALTH
        when the light changes, or once a second while it is not green (live numbers)."""
        now = self.clock()
        if now - self._health_tick < 1.0 / max(self.settings.indicator_rate_hz, 0.1):
            return
        self._health_tick = now
        sid = self._session
        if sid is None or self._state != RUNNING:
            return
        self._health_session(sid, now)
        seg = self._segmenter
        busy = self._asr_busy
        playout = self._playout
        busy_now = busy is not None and busy[0] == sid
        queued = self._seg_q.qsize()
        # A segment queued while the ASR worker is idle is picked up at once: it does not wait.
        waiting = queued if busy_now else max(0, queued - 1)
        m = Metrics(open_s=seg.open_s if seg is not None else 0.0, asr_waiting=waiting,
                    asr_busy_s=now - busy[1] if busy_now else 0.0,
                    playout_wait_s=playout.wait_s if playout is not None else 0.0)
        h = self._health.update(now, m)
        key = h.key
        if key == self._health_key and (h.level == GREEN or now - self._health_emit_t < 1.0):
            return
        with self._lock:  # short: never emit for a session that has just stopped
            if self._session != sid:
                return
            data = {**h.event_data(), "changed": key != self._health_key}  # False: numbers refresh only
            self._health_key, self._health_emit_t = key, now
            self._emit(Event(HEALTH, f"{data['level']}: {data['reason']}", session_id=sid, data=data))

    def _watchdog(self) -> None:
        with self._lock:
            sid = self._session
            if sid is None or self._state != RUNNING:
                return
            now = self.clock()
            limit = self.settings.device_watchdog_s
            for stream, dev, label in ((self._capture, self._in_dev, "Input"),
                                       (self._output, self._out_dev, "Output")):
                if stream is None:
                    continue
                silent = stream.last_callback is not None and now - stream.last_callback > limit
                if silent or not stream.active:
                    name = dev.name if dev else "?"
                    self._post_fatal(sid, f"{label} device '{name}' stopped responding (removed?)")
                    return
