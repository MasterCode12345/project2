"""Anti-stall on a model that is already loaded (plan-gui-anti-stall.md): WhisperASR.set_decoding,
an ongoing decode keeps its rules, the orchestrator gives the model a session really uses the
current settings, and the latency log names what really ran."""

import sys
import threading
import time
import types

import pytest

from audio_translator import asr as asr_mod
from audio_translator.asr import DECODING_OPTIONS, DecodeLimiter, Decoding, WhisperASR, anti_stall_label
from audio_translator.config import (ANTI_STALL_FIELDS, ANTI_STALL_PRESETS, Settings, anti_stall_preset,
                                     apply_anti_stall_preset)
from audio_translator.model_manager import ModelManager
from audio_translator.models import MODEL, RESULT, STOPPED, WARNING
from conftest import utterance

C4B_LABEL = "cap20+32,budget4s,reject:trunc+loop"


def preset_options(name):
    s = Settings()
    apply_anti_stall_preset(s, name)
    return asr_mod.decoding_from_settings(s)


class CT2:
    """Stands in for CTranslate2's Whisper: a short real result for every attempt."""

    def generate(self, features, prompts, **kw):
        return [types.SimpleNamespace(sequences_ids=[[7, 8]], scores=[-0.1])]


class SlowWhisperModel:
    """faster-whisper stand-in: two attempts per call (temperature 0, then 0.2), reading ``self.model``
    for each, as faster-whisper does. With ``gate`` set, the call waits between the two attempts."""

    instances: list = []

    def __init__(self, path, device=None, compute_type=None):
        self.path = path
        self.model = CT2()
        self.attempts = []  # per call: (object of attempt 1, object of attempt 2)
        self.entered = threading.Event()
        self.gate = None
        self.fail = False
        SlowWhisperModel.instances.append(self)

    def transcribe(self, audio, **kw):
        first = self.model
        first.generate(None, [[1, 2, 3]], max_length=448)
        gate = self.gate
        if gate is not None:
            self.entered.set()
            assert gate.wait(10), "gate never opened"
        second = self.model
        second.generate(None, [[1, 2, 3]], max_length=448, sampling_temperature=0.2)
        self.attempts.append((first, second))
        if self.fail:
            raise RuntimeError("decoder blew up on the old segment")
        return iter([types.SimpleNamespace(text=" xin chào ", no_speech_prob=0.1, avg_logprob=-0.2,
                                           compression_ratio=1.0, temperature=0.0)]), None


@pytest.fixture
def slow_fw(monkeypatch):
    SlowWhisperModel.instances = []
    monkeypatch.setitem(sys.modules, "faster_whisper", types.SimpleNamespace(WhisperModel=SlowWhisperModel))
    return SlowWhisperModel


# ------------------------------------------------------------------ settings helpers
def test_presets_fields_and_options_line_up():
    assert DECODING_OPTIONS == tuple(f[len("asr_"):] for f in ANTI_STALL_FIELDS)
    assert all(set(p) <= set(ANTI_STALL_FIELDS) for p in ANTI_STALL_PRESETS.values())
    for name in ("off", "c4a", "c4b"):
        s = Settings()
        apply_anti_stall_preset(s, name)
        assert anti_stall_preset(s) == name
    s = Settings(**ANTI_STALL_PRESETS["c4b"], asr_temperatures=(0.0, 0.4))
    assert anti_stall_preset(s) == "custom"
    apply_anti_stall_preset(s, "off")  # every decoding field back to its default, not only the preset's
    assert anti_stall_preset(s) == "off" and s.asr_temperatures is None and s.asr_max_tokens_per_s is None
    assert anti_stall_preset(Settings()) == "off"


# ------------------------------------------------------------------ WhisperASR.set_decoding
def test_set_decoding_attaches_replaces_and_detaches_on_a_loaded_model(slow_fw):
    a = WhisperASR("small")
    assert a.set_decoding(**preset_options("c4b"))  # before load: only the options change
    assert a.limiter is None and a.decode == C4B_LABEL
    a.load()
    raw = slow_fw.instances[0].model if not isinstance(slow_fw.instances[0].model, DecodeLimiter) \
        else slow_fw.instances[0].model._inner
    lim_b = a.limiter
    assert isinstance(lim_b, DecodeLimiter) and lim_b._inner is raw and a.anti_stall == "c4b"
    assert (lim_b.reject_loop_t0, lim_b.reject_truncated_all) == (True, True)

    assert not a.set_decoding(**preset_options("c4b"))  # nothing changed: the same limiter stays
    assert a.limiter is lim_b

    assert a.set_decoding(**preset_options("c4a"))
    lim_a = a.limiter
    assert lim_a is not lim_b and lim_a._inner is raw and a.anti_stall == "c4a"
    assert (lim_a.reject_loop_t0, lim_b.reject_loop_t0) == (False, True)  # the old limiter is never changed

    assert a.set_decoding(**preset_options("off"))
    assert a.limiter is None and a._model.model is raw and a.anti_stall == "off" and a.decode == "base"

    a.reject_loop_t0 = True  # attribute assignment goes through set_decoding too
    assert isinstance(a.limiter, DecodeLimiter) and a.limiter.reject_loop_t0 and a.anti_stall == "custom"
    with pytest.raises(TypeError, match="beam_size"):
        a.set_decoding(beam_size=1)


def test_a_preset_name_only_with_its_limiter_attached():
    c4b = Decoding(**preset_options("c4b"))
    assert anti_stall_label(c4b, True) == "c4b" and anti_stall_label(c4b, False) == "off"
    assert anti_stall_label(Decoding(), False) == "off"
    assert anti_stall_label(Decoding(temperatures=(0.0, 0.2)), False) == "custom"
    assert WhisperASR("small", **preset_options("c4b")).anti_stall == "off"  # not loaded: nothing attached


@pytest.mark.parametrize("before,after", [("off", "c4b"), ("c4b", "off")])
def test_a_decode_in_progress_keeps_its_rules_and_the_next_call_gets_the_new_ones(slow_fw, before, after):
    a = WhisperASR("small", **preset_options(before))
    a.load()
    fw = slow_fw.instances[0]
    old_obj = a._model.model
    fw.gate = threading.Event()
    out = {}

    def decode():
        try:
            out["segments"] = a.transcribe_segments(utterance_audio(), 16000)
        except BaseException as exc:  # noqa: BLE001 - reported below
            out["error"] = exc

    t = threading.Thread(target=decode)
    t.start()
    assert fw.entered.wait(5)  # attempt 1 done, the call waits before attempt 2
    t0 = time.monotonic()
    assert a.set_decoding(**preset_options(after))
    assert time.monotonic() - t0 < 0.5  # never waits for the running call
    assert a.anti_stall == after and (a.limiter is not None) == (after == "c4b")  # attached for the next calls
    fw.gate.set()
    t.join(5)
    assert "error" not in out and out["segments"]
    first, second = fw.attempts[0]
    assert first is second is old_obj  # both attempts of the running call: its own object
    assert a.last_anti_stall == before
    if isinstance(old_obj, DecodeLimiter):
        assert len(old_obj.calls) == 2 and not a.last_hit_limit  # read from its own limiter, after the swap
    fw.gate = None
    a.transcribe_segments(utterance_audio(), 16000)
    first, second = fw.attempts[1]
    assert first is second is a._model.model and a.last_anti_stall == after
    if after == "c4b":
        assert isinstance(first, DecodeLimiter) and len(first.calls) == 2


def utterance_audio():
    import numpy as np

    return np.concatenate(utterance(15, 5))


# ------------------------------------------------------------------ through the orchestrator
def wait_rows(path, n):
    from test_latency_log import wait_rows as _wait

    return _wait(path, n)


def make_session_harness(harness, settings, is_downloaded=lambda n: True):
    holder = {}
    mgr = ModelManager(lambda n: holder["h"].orch._make_recognizer(n), is_downloaded=is_downloaded)
    h = harness(settings=settings, model_manager=mgr)
    holder["h"] = h
    return h


def run_session(h, preset, *, speak=True):
    apply_anti_stall_preset(h.orch.settings, preset)  # what the GUI does right before start()
    n0 = len(h.log.events)
    assert h.orch.start(None, "BlackHole")
    h.log.wait_state("running", after=n0, timeout=10)
    sid, rec = h.orch.current_session, h.orch._asr
    if speak:
        h.say(utterance(15))
        h.log.wait_for(lambda e: e.kind == RESULT and e.session_id == sid, timeout=10)
    return sid, rec


def stop(h):
    n0 = len(h.log.events)
    h.orch.stop()
    h.log.wait_state(STOPPED, after=n0)


def test_three_sessions_off_c4b_off_on_one_loaded_model(harness, slow_fw, tmp_path):
    import importlib.util
    from pathlib import Path

    p = tmp_path / "lat.csv"
    h = make_session_harness(harness, Settings(model_size="small", latency_log=str(p)))
    seen = []
    for preset in ("off", "c4b", "off"):
        sid, rec = run_session(h, preset)
        fw = slow_fw.instances[-1]
        seen.append((sid, preset, rec.limiter is not None, rec.last_anti_stall, rec.decode,
                     isinstance(fw.attempts[-1][0], DecodeLimiter)))
        stop(h)
    assert [s[2:] for s in seen] == [(False, "off", "base", False), (True, "c4b", C4B_LABEL, True),
                                    (False, "off", "base", False)]
    assert len(slow_fw.instances) == 1  # the model was loaded once; only its limiter changed
    rows = wait_rows(p, 3)
    assert [(r["session_id"], r["asr_anti_stall"], r["asr_decode"]) for r in rows] == [
        ("1", "off", "base"), ("2", "c4b", C4B_LABEL), ("3", "off", "base")]

    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("latency_report_t", root / "eval" / "latency_report.py")
    lr = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(lr)
    text = lr.report(lr.load([p]))
    assert "anti-stall per session: 1=off, 2=c4b, 3=off" in text
    assert f"asr_decode={C4B_LABEL} anti_stall=c4b" in text


def test_the_fallback_model_gets_the_new_settings(harness, slow_fw, tmp_path, monkeypatch):
    monkeypatch.setattr(asr_mod, "LOCAL_MODELS_DIR", tmp_path / "models")  # phowhisper-small not converted
    p = tmp_path / "lat.csv"
    h = make_session_harness(harness, Settings(latency_log=str(p)), is_downloaded=lambda n: n == "small")
    run_session(h, "off")
    stop(h)
    sid, rec = run_session(h, "c4b")
    assert any(e.kind == WARNING and e.data.get("model_fallback") for e in h.log.events)
    assert rec.model_size == "small" and rec.anti_stall == "c4b" and rec.last_anti_stall == "c4b"
    stop(h)
    rows = wait_rows(p, 2)
    assert [(r["asr_model"], r["asr_anti_stall"]) for r in rows] == [("small", "off"), ("small", "c4b")]


def test_with_two_models_loaded_the_one_used_next_gets_the_settings(harness, slow_fw):
    h = make_session_harness(harness, Settings(model_size="small"))
    _, small = run_session(h, "off")
    stop(h)
    n0 = len(h.log.events)
    assert h.orch.select_model("base")  # loaded in the background with the settings of now (off)
    h.log.wait_for(lambda e: e.kind == MODEL and e.data.get("model") == "base" and e.data.get("detail") == "selected"
                   and h.log.events.index(e) >= n0)
    assert h.orch._models.loaded_names() == ["small", "base"]
    _, base = run_session(h, "c4b")
    assert base.model_size == "base" and base.anti_stall == "c4b" and base.last_anti_stall == "c4b"
    assert small.limiter is None  # not used in that session: untouched until it is used
    stop(h)
    n0 = len(h.log.events)
    assert h.orch.select_model("small")
    h.log.wait_for(lambda e: e.kind == MODEL and e.data.get("model") == "small" and e.data.get("detail") == "selected"
                   and h.log.events.index(e) >= n0)
    _, again = run_session(h, "c4b")
    assert again is small and small.anti_stall == "c4b" and small.last_anti_stall == "c4b"
    stop(h)
    _, again = run_session(h, "off")
    assert again is small and small.limiter is None and small.last_anti_stall == "off"
    stop(h)
    assert sorted(m.path for m in slow_fw.instances) == ["base", "small"]  # no reload


@pytest.mark.parametrize("old_fails", [False, True])
def test_settings_changed_while_an_old_segment_is_decoding(harness, slow_fw, tmp_path, old_fails):
    p = tmp_path / "lat.csv"
    h = make_session_harness(harness, Settings(model_size="small", latency_log=str(p)))
    sid1, rec = run_session(h, "off", speak=False)
    fw = slow_fw.instances[0]
    fw.gate, fw.fail = threading.Event(), old_fails
    h.say(utterance(15))
    assert fw.entered.wait(10)  # the session-1 segment is being decoded (slowly)
    stop(h)
    sid2, rec2 = run_session(h, "c4b", speak=False)  # starts while the old decode still runs
    assert rec2 is rec and rec.anti_stall == "c4b" and isinstance(rec.limiter, DecodeLimiter)
    assert not fw.gate.is_set() and fw.attempts == []  # the old call is still inside the model
    fw.fail = False
    fw.gate.set()
    h.say(utterance(15))
    res = h.log.wait_for(lambda e: e.kind == RESULT and e.session_id == sid2, timeout=10)
    assert res.data["source_text"]  # the ASR worker is alive after the old call (even a failed one)
    old_call, new_call = fw.attempts[0], fw.attempts[-1]
    assert not isinstance(old_call[0], DecodeLimiter) and old_call[0] is old_call[1]  # old rules to the end
    assert isinstance(new_call[0], DecodeLimiter) and new_call[0] is new_call[1]
    assert rec.last_anti_stall == "c4b"
    assert not [e for e in h.log.events if e.session_id == sid1 and e.kind in (RESULT, WARNING)
                and e.segment_id is not None]  # the old segment is dropped, never reported
    stop(h)
    rows = wait_rows(p, 1)
    assert [(r["session_id"], r["asr_anti_stall"]) for r in rows] == [(str(sid2), "c4b")]
