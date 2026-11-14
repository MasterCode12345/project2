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
from audio_translator.asr import (  # noqa: E402
    SegmentFilter, collapse_repeats, filter_from_settings, load_blacklist, normalize_text)


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
    assert filter_from_settings(Settings(asr_filter=False)) is None
    assert isinstance(filter_from_settings(Settings()), SegmentFilter)  # on by default
    f = filter_from_settings(Settings(asr_filter=True, asr_max_no_speech_prob=0.5, asr_min_avg_logprob=-0.8,
                                      asr_max_compression_ratio=2.0))
    assert (f.max_no_speech_prob, f.min_avg_logprob, f.max_compression_ratio) == (0.5, -0.8, 2.0)


def test_orchestrator_recognizer_gets_filter_and_session_vad_from_settings(monkeypatch):
    from audio_translator.orchestrator import Orchestrator

    orch = Orchestrator.__new__(Orchestrator)
    orch.settings = Settings(asr_filter=True)
    assert isinstance(orch._make_recognizer("small").segment_filter, SegmentFilter)
    orch.settings = Settings(asr_filter=False)
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

    cli.main(["--vad", "webrtc", "--silero-threshold", "0.7", "--no-asr-filter"], orchestrator_factory=Orch)
    assert (seen["s"].vad_backend, seen["s"].silero_threshold, seen["s"].asr_filter) == ("webrtc", 0.7, False)
    d = cli.build_parser().parse_args([])
    assert (d.vad, d.asr_filter) == ("silero", True)  # defaults since the guard bench
    assert cli.build_parser().parse_args(["--asr-filter"]).asr_filter is True
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(["--vad", "energy"])


@pytest.mark.parametrize("text, want", [
    ("râu ria đã ba ngày không cạo a a a a a a a a d n", "râu ria đã ba ngày không cạo a d n"),  # 1-word loop
    ("nga thi ấy càng nhìn nốt nga a nga thi ấy càng nhìn nốt nga a nga thi ấy càng nhìn nốt nguy cơ",
     "nga thi ấy càng nhìn nốt nga a nga thi ấy càng nhìn nốt nguy cơ"),  # 8-word unit twice -> once
    ("cảm ơn, cảm ơn, cảm ơn, cảm ơn.", "cảm ơn."),  # compared normalised; the last copy is kept
    ("rất rất rất đẹp", "rất rất rất đẹp"),  # 3 x one word: speech
    ("alo alo anh nghe không", "alo alo anh nghe không"),
    ("bác sĩ phẫu thuật bác sĩ phẫu thuật và bác sĩ gây mê", "bác sĩ phẫu thuật bác sĩ phẫu thuật và bác sĩ gây mê"),
    ("", ""),
])
def test_collapse_repeats(text, want):
    assert collapse_repeats(text) == want


def test_filter_writes_a_decoding_loop_once_and_records_it(seg_fw):
    loop = "Xin chào " + "a " * 10
    seg_fw.segments = [seg(loop)]
    a = WhisperASR("small", segment_filter=SegmentFilter(blacklist=BL))
    assert a.transcribe(np.zeros(1600, np.int16), 16000) == "Xin chào a"
    assert a.dropped == [("repetition", loop.strip())]
    b = WhisperASR("small", segment_filter=SegmentFilter(blacklist=BL, collapse_loops=False))
    assert b.transcribe(np.zeros(1600, np.int16), 16000) == loop.strip()


def test_last_temperature_records_a_fallback_observation_only(seg_fw):
    s1, s2 = seg("Xin chào."), seg("Tạm biệt.")
    s1.temperature, s2.temperature = 0.0, 0.4
    seg_fw.segments = [s1, s2]
    a = WhisperASR("small")
    assert a.last_temperature == 0.0
    assert a.transcribe(np.zeros(1600, np.int16), 16000) == "Xin chào. Tạm biệt."
    assert a.last_temperature == 0.4
    assert "temperature" not in a._model.calls[0]  # decoding options unchanged
    seg_fw.segments = [seg("Một.")]  # no temperature attribute: no fallback
    a.transcribe(np.zeros(1600, np.int16), 16000)
    assert a.last_temperature == 0.0


def test_bounded_decoding_is_off_by_default_and_passed_only_when_set(fake_fw):
    s = Settings()
    assert (s.asr_max_tokens_per_s, s.asr_temperatures) == (None, None)
    a = WhisperASR("small")
    a.transcribe(np.zeros(16000 * 3, np.int16), 16000)
    kw = a._model.calls[0]
    assert "max_new_tokens" not in kw and "temperature" not in kw  # faster-whisper's own decoding
    b = WhisperASR("small", max_tokens_per_s=12, max_tokens_base=16, temperatures=(0.0, 0.2, 0.4))
    b.transcribe(np.zeros(16000 * 3, np.int16), 16000)
    kw = b._model.calls[0]
    assert kw["max_new_tokens"] == 52 and kw["temperature"] == [0.0, 0.2, 0.4]
    assert b.max_new_tokens(15.0) == 196 and b.max_new_tokens(30.0) == 224  # never above Whisper's limit
    assert WhisperASR("small", temperatures=(0.0,)).decode_options(1.0) == {"temperature": [0.0]}


def test_bounded_decoding_flags_reach_the_recognizer():
    args = cli.build_parser().parse_args(["--asr-max-tokens-per-s", "12", "--asr-temperatures", "0,0.2,0.4"])
    assert args.asr_max_tokens_per_s == 12 and args.asr_temperatures == (0.0, 0.2, 0.4)
    args = cli.build_parser().parse_args([])
    assert args.asr_max_tokens_per_s is None and args.asr_temperatures is None
    from audio_translator.orchestrator import Orchestrator

    rec = Orchestrator(Settings(asr_max_tokens_per_s=12, asr_temperatures=(0.0, 0.4)))._make_recognizer("small")
    assert rec.max_tokens_per_s == 12 and rec.temperatures == (0.0, 0.4) and rec.max_tokens_base == 32


@pytest.mark.parametrize("text,want", [
    ("unk họp này có ý nghĩa rất quan trọng.", "họp này có ý nghĩa rất quan trọng."),
    ("UNK muốn biết", "muốn biết"),
    ("ở louis unk.", "ở louis."),
    ("tin nhắn. unk bao giờ", "tin nhắn. bao giờ"),
    ("unk", ""), ("unk.", ""), ("unk unk", ""),
    ("junk food unknown chunk", "junk food unknown chunk"),  # whole word only
    ("xin chào", "xin chào"),
])
def test_strip_unk(text, want):
    assert asr_mod.strip_unk(text) == want


def test_filter_drops_unk_and_reports_it_without_a_ghost_reason(seg_fw):
    seg_fw.segments = [seg("unk yêu bạn nhất trên đời."), seg("unk")]
    a = WhisperASR("small", segment_filter=SegmentFilter(blacklist=BL))
    assert a.transcribe(np.zeros(1600, np.int16), 16000) == "yêu bạn nhất trên đời."
    assert [w for w, _ in a.dropped] == ["unk", "unk"]
    assert filter_from_settings(Settings()).drop_unk is True
    assert filter_from_settings(Settings(asr_drop_unk=False)).drop_unk is False
    b = WhisperASR("small", segment_filter=SegmentFilter(blacklist=BL, drop_unk=False))
    assert b.transcribe(np.zeros(1600, np.int16), 16000) == "unk yêu bạn nhất trên đời. unk"
    assert WhisperASR("small").transcribe(np.zeros(1600, np.int16), 16000).startswith("unk")  # filter off


def test_an_unk_only_segment_is_not_a_ghost_on_the_light(harness):
    from audio_translator.health import QUALITY
    from audio_translator.models import HEALTH, INFO
    from conftest import FakeASR, utterance

    class UnkASR(FakeASR):
        def transcribe(self, audio, sample_rate):
            super().transcribe(audio, sample_rate)
            self.dropped = [("unk", "unk")]
            return ""

    h = harness(settings=Settings(indicator_rate_hz=100.0), asr=UnkASR())
    h.start()
    h.say(utterance(10))
    ev = h.log.wait_for(lambda e: e.kind == INFO and "no speech" in e.message)
    import time
    time.sleep(0.2)
    assert not [e for e in h.log.of(HEALTH) if e.data["signal"] == QUALITY] and ev
