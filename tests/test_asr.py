"""WhisperASR: local (converted) models, initial prompt; wiring through settings/CLI/orchestrator."""

import sys
import types

import numpy as np
import pytest

from audio_translator import asr as asr_mod
from audio_translator import cli, model_manager
from audio_translator.asr import LOCAL_MODELS, WhisperASR, convert_command, local_model_dir, resolve_model
from audio_translator.config import Settings


class FakeWhisperModel:
    instances: list = []

    def __init__(self, model, device, compute_type):
        self.model, self.calls = model, []
        FakeWhisperModel.instances.append(self)

    def transcribe(self, audio, **kw):
        self.calls.append(kw)
        return iter([types.SimpleNamespace(text=" xin chào ")]), None


@pytest.fixture
def fake_fw(monkeypatch):
    FakeWhisperModel.instances = []
    monkeypatch.setitem(sys.modules, "faster_whisper", types.SimpleNamespace(WhisperModel=FakeWhisperModel))
    return FakeWhisperModel


@pytest.fixture
def models_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(asr_mod, "LOCAL_MODELS_DIR", tmp_path)
    return tmp_path


def _convert(root, name):
    d = local_model_dir(name, root)
    d.mkdir(parents=True)
    (d / "model.bin").write_bytes(b"x")
    return d


def test_hub_model_names_pass_through():
    assert resolve_model("small") == "small"
    assert resolve_model("large-v3-turbo") == "large-v3-turbo"


def test_converted_local_model_loads_from_its_directory(models_dir, fake_fw):
    d = _convert(models_dir, "phowhisper-small")
    a = WhisperASR("phowhisper-small")
    a.load()
    assert fake_fw.instances[0].model == str(d)
    assert model_manager.hf_is_downloaded("phowhisper-small")
    assert model_manager.ModelManager(lambda n: None).status("phowhisper-small") == model_manager.DOWNLOADED


def test_missing_local_model_reports_path_and_convert_command(models_dir, fake_fw):
    assert not model_manager.hf_is_downloaded("phowhisper-medium")
    assert model_manager.ModelManager(lambda n: None).status("phowhisper-medium") == model_manager.NOT_DOWNLOADED
    with pytest.raises(RuntimeError) as exc:
        WhisperASR("phowhisper-medium").load()
    msg = str(exc.value)
    assert str(local_model_dir("phowhisper-medium", models_dir)) in msg
    assert "scripts/convert_phowhisper.py medium" in msg and msg.endswith(convert_command("phowhisper-medium"))
    assert fake_fw.instances == []


def test_local_models_pin_their_source_revision():
    assert set(LOCAL_MODELS) == {"phowhisper-small", "phowhisper-medium"}
    for repo, rev in LOCAL_MODELS.values():
        assert repo.startswith("vinai/PhoWhisper-") and len(rev) == 40


def test_initial_prompt_is_passed_to_transcribe(fake_fw):
    a = WhisperASR("small", initial_prompt="Xin chào, sau đây là đoạn hội thoại bằng tiếng Việt.")
    assert a.transcribe(np.zeros(1600, np.int16), 16000) == "xin chào"
    kw = fake_fw.instances[0].calls[0]
    assert kw["initial_prompt"] == "Xin chào, sau đây là đoạn hội thoại bằng tiếng Việt."
    assert kw["language"] == "vi" and kw["vad_filter"] is False and kw["condition_on_previous_text"] is False


@pytest.mark.parametrize("prompt", [None, ""])
def test_no_prompt_keeps_the_previous_call(fake_fw, prompt):
    WhisperASR("small", initial_prompt=prompt).transcribe(np.zeros(1600, np.int16), 16000)
    assert fake_fw.instances[0].calls[0]["initial_prompt"] is None  # faster-whisper's default


def test_settings_default_has_no_prompt():
    assert Settings().asr_initial_prompt is None


def test_cli_asr_prompt_reaches_settings():
    seen = {}

    class Orch:
        def __init__(self, settings, on_event):
            seen["settings"] = settings
            self.on_event = on_event

        def start(self):
            from audio_translator.models import ERROR, Event
            self.on_event(Event(ERROR, "stop here"))
            from audio_translator.models import STATUS, STOPPED
            self.on_event(Event(STATUS, "Stopped", data={"state": STOPPED}))

        def stop(self):
            pass

        def shutdown(self):
            pass

    cli.main(["--model", "phowhisper-small", "--asr-prompt", "Xin chào."], orchestrator_factory=Orch)
    assert seen["settings"].asr_initial_prompt == "Xin chào."
    assert seen["settings"].model_size == "phowhisper-small"
    assert cli.build_parser().parse_args([]).asr_prompt is None


def test_orchestrator_recognizer_gets_prompt_from_settings():
    from audio_translator.orchestrator import Orchestrator

    orch = Orchestrator.__new__(Orchestrator)  # only _make_recognizer is exercised
    orch.settings = Settings(asr_initial_prompt="Xin chào.", beam_size=3)
    rec = orch._make_recognizer("phowhisper-small")
    assert isinstance(rec, WhisperASR)
    assert rec.initial_prompt == "Xin chào." and rec.model_size == "phowhisper-small" and rec.beam_size == 3


def test_loading_an_unconverted_local_model_is_not_announced_as_a_download(models_dir):
    seen = []
    mgr = model_manager.ModelManager(lambda n: WhisperASR(n), on_change=lambda n, st, d: seen.append((n, st, d)))
    with pytest.raises(RuntimeError, match="convert_phowhisper.py small"):
        mgr.get("phowhisper-small")
    assert ("phowhisper-small", model_manager.LOADING, "loading") in seen
    assert not any("downloading" in d for _, _, d in seen)
    assert mgr.status("phowhisper-small") == model_manager.FAILED


# ---------------------------------------------------------- hallucination filter
from audio_translator.asr import SegmentFilter, filter_from_settings, load_blacklist, normalize_text  # noqa: E402


def seg(text, nsp=0.1, lp=-0.3, cr=1.5):
    return types.SimpleNamespace(text=text, no_speech_prob=nsp, avg_logprob=lp, compression_ratio=cr)


BL = ("hãy subscribe cho kênh ghiền mì gõ", "hẹn gặp lại các bạn trong video")


@pytest.mark.parametrize("s, why", [
    (seg("Câu thật.", nsp=0.61), "no_speech_prob"),
    (seg("Câu thật.", lp=-1.01), "avg_logprob"),
    (seg("Câu thật.", cr=2.41), "compression_ratio"),
    (seg("Hãy subscribe cho kênh Ghiền Mì Gõ để không bỏ lỡ"), "blacklist"),
    (seg("HẸN GẶP LẠI CÁC BẠN TRONG VIDEO tiếp theo!"), "blacklist"),
    (seg("Câu thật.", nsp=0.6, lp=-1.0, cr=2.4), None),  # thresholds are strict
])
def test_filter_reasons(s, why):
    assert SegmentFilter(blacklist=BL).reason(s) == why


def test_blacklist_matches_whole_words_only():
    f = SegmentFilter(blacklist=BL)
    assert f.keep(seg("Tôi sẽ hẹn gặp lại các bạn sau"))  # shares words, not the phrase
    assert f.keep(seg("hẹn gặp lại các bạn trong videos"))  # 'video' must be a whole word
    assert f.keep(seg("Kênh này hay"))


def test_blacklist_file_loads_and_rejects_short_phrases(tmp_path):
    p = tmp_path / "bl.txt"
    p.write_text("# comment\n\nHãy Subscribe, cho kênh Ghiền Mì Gõ!\t# MUSAN noise/free-sound x3\n", encoding="utf-8")
    assert load_blacklist(p) == ("hãy subscribe cho kênh ghiền mì gõ",)
    p.write_text("cảm ơn nhé\n", encoding="utf-8")  # 3 words
    with pytest.raises(ValueError, match="needs >= 4 words"):
        load_blacklist(p)
    assert all(len(x.split()) >= 4 for x in load_blacklist())  # the shipped list


def test_normalize_text_keeps_tone_marks():
    assert normalize_text("  Xin CHÀO, các bạn!!  ") == "xin chào các bạn"


class SegModel(FakeWhisperModel):
    segments: list = []

    def transcribe(self, audio, **kw):
        self.calls.append(kw)
        return iter(self.segments), None


@pytest.fixture
def seg_fw(monkeypatch):
    SegModel.instances = []
    monkeypatch.setitem(sys.modules, "faster_whisper", types.SimpleNamespace(WhisperModel=SegModel))
    return SegModel


def test_filter_drops_only_the_bad_segment(seg_fw):
    seg_fw.segments = [seg(" Xin chào. "), seg("Hãy subscribe cho kênh Ghiền Mì Gõ"), seg(" Tạm biệt. ", lp=-0.5)]
    a = WhisperASR("small", segment_filter=SegmentFilter(blacklist=BL))
    assert a.transcribe(np.zeros(1600, np.int16), 16000) == "Xin chào. Tạm biệt."
    assert a.dropped == [("blacklist", "Hãy subscribe cho kênh Ghiền Mì Gõ")]


def test_filter_off_keeps_everything(seg_fw):
    seg_fw.segments = [seg("Xin chào.", nsp=0.99, lp=-3.0), seg("Hãy subscribe cho kênh Ghiền Mì Gõ")]
    a = WhisperASR("small")
    assert a.transcribe(np.zeros(1600, np.int16), 16000) == "Xin chào. Hãy subscribe cho kênh Ghiền Mì Gõ"
    assert a.dropped == []


def test_all_segments_filtered_gives_empty_text(seg_fw):
    seg_fw.segments = [seg("Hẹn gặp lại các bạn trong video", nsp=0.9)]
    a = WhisperASR("small", segment_filter=SegmentFilter(blacklist=BL))
    assert a.transcribe(np.zeros(1600, np.int16), 16000) == ""
    assert a.transcribe_segments(np.zeros(1600, np.int16), 16000)[0].text.startswith("Hẹn")  # raw access


def test_filter_from_settings():
    assert filter_from_settings(Settings()) is None
    f = filter_from_settings(Settings(asr_filter=True, asr_max_no_speech_prob=0.5, asr_min_avg_logprob=-0.8,
                                      asr_max_compression_ratio=2.0))
    assert (f.max_no_speech_prob, f.min_avg_logprob, f.max_compression_ratio) == (0.5, -0.8, 2.0)


def test_orchestrator_recognizer_gets_filter_and_session_vad_from_settings(monkeypatch):
    from audio_translator.orchestrator import Orchestrator

    orch = Orchestrator.__new__(Orchestrator)
    orch.settings = Settings(asr_filter=True)
    assert isinstance(orch._make_recognizer("small").segment_filter, SegmentFilter)
    orch.settings = Settings()
    assert orch._make_recognizer("small").segment_filter is None


def test_cli_vad_and_filter_flags_reach_settings():
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

    cli.main(["--vad", "silero", "--silero-threshold", "0.7", "--asr-filter"], orchestrator_factory=Orch)
    assert (seen["s"].vad_backend, seen["s"].silero_threshold, seen["s"].asr_filter) == ("silero", 0.7, True)
    d = cli.build_parser().parse_args([])
    assert (d.vad, d.asr_filter) == ("webrtc", False)
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(["--vad", "energy"])
