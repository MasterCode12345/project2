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
    # the cap is applied in real tokens through DecodeLimiter, never as faster-whisper's max_new_tokens
    # (CTranslate2 would give about half of it: plan-asr-anti-stall.md, C4)
    assert "max_new_tokens" not in kw and kw["temperature"] == [0.0, 0.2, 0.4]
    assert b.uses_limiter and b.attempt_limits(3.0) == (52, 52)
    assert b.max_new_tokens(15.0) == 196 and b.max_new_tokens(30.0) == 224  # never above Whisper's limit
    assert WhisperASR("small", temperatures=(0.0,)).decode_options(1.0) == {"temperature": [0.0]}


def test_bounded_decoding_flags_reach_the_recognizer():
    # the rejected global cap / temperature ladder are Settings-only now (bench reproducibility)
    for flag in ("--asr-max-tokens-per-s", "--asr-temperatures"):
        with pytest.raises(SystemExit):
            cli.build_parser().parse_args([flag, "1"])
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
def test_strip_junk_unk(text, want):
    assert asr_mod.strip_junk(text) == want


def test_filter_drops_unk_and_reports_it_without_a_ghost_reason(seg_fw):
    seg_fw.segments = [seg("unk yêu bạn nhất trên đời."), seg("unk")]
    a = WhisperASR("small", segment_filter=SegmentFilter(blacklist=BL))
    assert a.transcribe(np.zeros(1600, np.int16), 16000) == "yêu bạn nhất trên đời."
    assert [w for w, _ in a.dropped] == ["junk", "junk"]
    assert filter_from_settings(Settings()).drop_junk is True
    assert filter_from_settings(Settings(asr_drop_junk=False)).drop_junk is False
    b = WhisperASR("small", segment_filter=SegmentFilter(blacklist=BL, drop_junk=False))
    assert b.transcribe(np.zeros(1600, np.int16), 16000) == "unk yêu bạn nhất trên đời. unk"
    assert WhisperASR("small").transcribe(np.zeros(1600, np.int16), 16000).startswith("unk")  # filter off


def test_an_unk_only_segment_is_not_a_ghost_on_the_light(harness):
    from audio_translator.health import QUALITY
    from audio_translator.models import HEALTH, INFO
    from conftest import FakeASR, utterance

    class UnkASR(FakeASR):
        def transcribe(self, audio, sample_rate):
            super().transcribe(audio, sample_rate)
            self.dropped = [("junk", "unk")]
            return ""

    h = harness(settings=Settings(indicator_rate_hz=100.0), asr=UnkASR())
    h.start()
    h.say(utterance(10))
    ev = h.log.wait_for(lambda e: e.kind == INFO and "no speech" in e.message)
    import time
    time.sleep(0.2)
    assert not [e for e in h.log.of(HEALTH) if e.data["signal"] == QUALITY] and ev


@pytest.mark.parametrize("text,want", [
    ("cảm ơn bạn. ofc thật sự thông thoáng", "cảm ơn bạn. thật sự thông thoáng"),  # real test 2026-11-14
    ("ba mươi mốt dặm. OFC ba mươi mốt dặm.", "ba mươi mốt dặm. ba mươi mốt dặm."),  # FLEURS bench
    ("unk bao giờ phẩy. ofc đâu phẩy.", "bao giờ phẩy. đâu phẩy."),
    ("ofc", ""), ("office ofcourse", "office ofcourse"),
])
def test_strip_junk_ofc(text, want):
    assert asr_mod.strip_junk(text) == want


def test_junk_tokens_come_from_one_list(tmp_path):
    assert asr_mod.JUNK_TOKENS == ("unk", "ofc")
    p = tmp_path / "junk.txt"
    p.write_text("# comment\nunk\t# why\nXyz\n\n", encoding="utf-8")
    assert asr_mod.load_junk_tokens(p) == ("unk", "xyz")
    assert asr_mod.strip_junk("xyz a unk b", ("unk", "xyz")) == "a b"
    p.write_text("two words\n", encoding="utf-8")
    with pytest.raises(ValueError):
        asr_mod.load_junk_tokens(p)
    assert SegmentFilter(blacklist=BL, junk_tokens=("abc",)).clean("abc xin chào") == "xin chào"


# ------------------------------------------------------------ anti-stall (plan-asr-anti-stall.md)
class _Res:
    def __init__(self, n):
        self.sequences_ids = [list(range(n))]


class _CT2:
    """Fake CTranslate2 Whisper: records each call's max_length; output length per call from a list,
    at most max_length // 2 tokens like the real one (plan-asr-anti-stall.md, step 0)."""

    def __init__(self, lengths):
        self.lengths = list(lengths)
        self.max_lengths = []
        self.other = "kept"

    def generate(self, features, prompts, **kw):
        self.max_lengths.append(kw.get("max_length"))
        return [_Res(min(self.lengths.pop(0), kw.get("max_length", 448) // 2))]


def test_limiter_caps_only_the_attempts_it_is_told_to():
    from audio_translator.asr import DecodeLimiter

    inner = _CT2([224, 30, 20])
    lim = DecodeLimiter(inner)
    assert lim.other == "kept"
    lim.begin(None, 52, None)  # (C): temperature 0 untouched, fallbacks capped
    lim.generate(None, [[1, 2, 3]], max_length=448)
    lim.generate(None, [[1, 2, 3]], max_length=448, sampling_temperature=0.2)
    assert inner.max_lengths == [448, 105]  # 52 real tokens -> max_length 2 * 52 + 1 (3-token prompt)
    assert [(t, n, cap) for _s, t, n, cap in lim.calls] == [(0.0, 224, 224), (0.2, 30, 52)]
    lim.begin(100, 52, None)  # (C+): temperature 0 gets a wide limit
    lim.generate(None, [[1, 2, 3]], max_length=448)
    assert inner.max_lengths[-1] == 201 and lim.calls[0][2:] == (20, 100)
    # measured on phowhisper-small (plan-asr-anti-stall.md C4): //2 with the 3-token prompt,
    # max_length - prompt + 1 with an initial-prompt context of 28 tokens
    assert asr_mod.real_token_limit(23) == 11 and asr_mod.real_token_limit(448) == 224
    assert asr_mod.real_token_limit(36, 28) == 9 and asr_mod.real_token_limit(48, 28) == 21
    for n in (1, 10, 52, 100, 224):
        assert asr_mod.real_token_limit(asr_mod.max_length_for(n)) == n
        assert asr_mod.real_token_limit(asr_mod.max_length_for(n, 28), 28) == n


def test_budget_replays_the_last_real_result_and_never_invents_one():
    from audio_translator.asr import DecodeLimiter

    now = [0.0]
    inner = _CT2([100, 90, 80])
    lim = DecodeLimiter(inner, clock=lambda: now[0])
    lim.begin(None, None, 4.0)
    first = lim.generate(None, [[1]], max_length=448)  # t=0
    now[0] = 3.0
    second = lim.generate(None, [[1]], sampling_temperature=0.2)  # within budget: real decode
    now[0] = 4.5
    third = lim.generate(None, [[1]], sampling_temperature=0.4)  # over budget: replay the last real one
    fourth = lim.generate(None, [[1]], sampling_temperature=0.6)
    assert third is second and fourth is second and second is not first
    assert len(inner.max_lengths) == 2 and lim.budget_skips == 2 and len(lim.calls) == 2
    lim.begin(None, None, 4.0)  # a new segment: the budget restarts
    now[0] = 100.0
    lim.generate(None, [[1]], max_length=448)
    assert len(inner.max_lengths) == 3 and lim.budget_skips == 0


def test_budget_never_skips_the_first_attempt():
    from audio_translator.asr import DecodeLimiter

    now = [50.0]
    inner = _CT2([10])
    lim = DecodeLimiter(inner, clock=lambda: now[0])
    lim.begin(None, None, 0.0)
    assert lim.generate(None, [[1]], sampling_temperature=0.2) is not None  # nothing to replay: decodes
    assert len(inner.max_lengths) == 1 and lim.budget_skips == 0


@pytest.mark.parametrize("text,want", [
    ("đó chứng tỏ rằng chúng ta đã già. a.......................", "đó chứng tỏ rằng chúng ta đã già."),
    ("vì đó mới chính là thứ bạn cần. a.", "vì đó mới chính là thứ bạn cần."),
    ("nhạc sĩ la hối với ca khúc xuân và tuổi trẻ........", "nhạc sĩ la hối với ca khúc xuân và tuổi trẻ."),
    ("bạn đi đâu a", "bạn đi đâu"),  # only ever applied when the limit was hit
    ("a.", "a."), ("xin chào.", "xin chào."),
])
def test_strip_cap_tail(text, want):
    assert asr_mod.strip_cap_tail(text) == want


def test_tail_strip_only_when_the_limit_was_hit(monkeypatch):
    calls = {}

    class Model:
        def __init__(self, *a, **k):
            self.ct2 = self.model = _CT2([])  # WhisperASR wraps .model in a DecodeLimiter

        def transcribe(self, audio, **kw):
            n = calls["n"]
            self.ct2.lengths = [n]
            self.model.generate(None, [[1, 2, 3]], max_length=3 + kw.get("max_new_tokens", 445))
            return iter([types.SimpleNamespace(text="đã già. a....", no_speech_prob=0.1, avg_logprob=-0.2,
                                               compression_ratio=1.0, temperature=0.0)]), None

    monkeypatch.setitem(sys.modules, "faster_whisper", types.SimpleNamespace(WhisperModel=Model))
    a = WhisperASR("small", max_tokens_per_s=15, cap_tail_strip=True)
    calls["n"] = 10  # well below the 77-token limit of 3 s: untouched
    assert a.transcribe(np.zeros(48000, np.int16), 16000) == "đã già. a...." and not a.last_hit_limit
    calls["n"] = 500  # reaches the limit: the tail goes
    assert a.transcribe(np.zeros(48000, np.int16), 16000) == "đã già." and a.last_hit_limit


def test_anti_stall_is_off_by_default_and_wired_through(fake_fw):
    s = Settings()
    assert (s.asr_fallback_max_tokens_per_s, s.asr_first_min_tokens, s.asr_decode_budget_s,
            s.asr_cap_tail_strip) == (None, None, None, False)
    a = WhisperASR("small")
    a.load()
    assert a.limiter is None and not a.uses_limiter and a.decode == "base"
    c2 = WhisperASR("small", fallback_max_tokens_per_s=15, first_min_tokens=100, decode_budget_s=4.0)
    assert c2.decode == "fb15+32,t0min100,budget4s"
    assert c2.attempt_limits(3.0) == (100, 77) and c2.attempt_limits(12.0) == (212, 212)
    assert c2.attempt_limits(30.0) == (224, 224)
    assert WhisperASR("small", fallback_max_tokens_per_s=15).attempt_limits(3.0) == (None, 77)
    from audio_translator.config import ANTI_STALL_PRESETS

    assert cli.build_parser().parse_args([]).asr_anti_stall == "off" and ANTI_STALL_PRESETS["off"] == {}
    args = cli.build_parser().parse_args(["--asr-anti-stall", "c4b"])
    s4 = Settings(**ANTI_STALL_PRESETS[args.asr_anti_stall])
    assert (s4.asr_max_tokens_per_s, s4.asr_max_tokens_base, s4.asr_decode_budget_s, s4.asr_reject_truncated_all,
            s4.asr_reject_loop_t0, s4.asr_cap_tail_strip) == (20, 32, 4, True, True, False)
    assert not Settings(**ANTI_STALL_PRESETS["c4a"]).asr_reject_loop_t0
    assert set(ANTI_STALL_PRESETS) == {"off", "c4a", "c4b"}  # C3 (rejected) has no app preset
    for flag in ("--asr-fallback-max-tokens-per-s", "--asr-first-min-tokens", "--asr-decode-budget-s",
                 "--asr-cap-tail-strip"):  # rejected configurations: no app flag
        with pytest.raises(SystemExit):
            cli.build_parser().parse_args([flag] + ([] if flag.endswith("strip") else ["1"]))
    from audio_translator.orchestrator import Orchestrator

    rec = Orchestrator(Settings(asr_fallback_max_tokens_per_s=15, asr_first_min_tokens=100,
                                asr_decode_budget_s=4.0, asr_cap_tail_strip=True, asr_reject_truncated_t0=True,
                                asr_reject_loop_t0=True))._make_recognizer("small")
    assert (rec.fallback_max_tokens_per_s, rec.first_min_tokens, rec.decode_budget_s, rec.cap_tail_strip,
            rec.reject_truncated_t0, rec.reject_loop_t0) == (15, 100, 4.0, True, True, True)
    assert rec.decode == "fb15+32,t0min100,budget4s,tail,reject:t0trunc+loop"
    pytest.importorskip("PySide6")
    from audio_translator.gui import parse_args

    g, _ = parse_args(["--asr-anti-stall", "c4a"])
    assert g.asr_anti_stall == "c4a"


class _ScoredRes:
    def __init__(self, ids, score=-0.2, nsp=0.01):
        self.sequences_ids, self.scores, self.no_speech_prob = [ids], [score], nsp


class _ScoredCT2:
    def __init__(self, outs):
        self.outs = list(outs)

    def generate(self, features, prompts, **kw):
        return [self.outs.pop(0)]


@pytest.mark.parametrize("text,loop", [
    ("nhạc sĩ la hối với ca khúc xuân và tuổi trẻ. a sĩ la hối với ca khúc xuân và tuổi trẻ. a sĩ la hối với ca", True),
    ("đó chứng tỏ rằng chúng ta đã già....", True),
    ("đó chứng tỏ rằng chúng ta đã già.", False),
    ("tôi đi học, tôi đi chơi, tôi đi ngủ", False),  # 3-word repeats only
    ("một hai ba bốn năm một hai ba bốn", True),
    ("một hai ba bốn một hai ba", False),  # the second copy is shorter than 4 words
])
def test_has_loop_tail(text, loop):
    assert asr_mod.has_loop_tail(text) is loop


def test_t0_rules_reject_without_touching_the_tokens():
    from audio_translator.asr import DecodeLimiter

    trunc = _ScoredRes(list(range(100)))  # stopped at its real limit (100 tokens)
    lim = DecodeLimiter(_ScoredCT2([trunc, _ScoredRes([1, 2, 3])]))
    lim.reject_truncated_t0 = True
    lim.begin(100, 52, None)
    out = lim.generate(None, [[1, 2, 3]], max_length=448)
    assert lim.t0_rejected == "truncated" and out[0].sequences_ids is trunc.sequences_ids
    assert out[0].scores == [-1e9] and out[0].no_speech_prob == trunc.no_speech_prob
    retry = lim.generate(None, [[1, 2, 3]], max_length=448, sampling_temperature=0.2)  # retries are never marked
    assert retry[0].scores == [-0.2]
    lim2 = DecodeLimiter(_ScoredCT2([_ScoredRes([5, 6]), _ScoredRes([7])]))
    lim2.reject_loop_t0, lim2.decode_text = True, lambda ids: "a b c d. a b c d" if ids == [5, 6] else "ok"
    lim2.begin(None, None, None)
    assert lim2.generate(None, [[1]], max_length=448)[0].scores == [-1e9] and lim2.t0_rejected == "loop"
    lim2.begin(None, None, None)
    assert lim2.generate(None, [[1]], max_length=448)[0].scores == [-0.2] and lim2.t0_rejected == ""
    lim3 = DecodeLimiter(_ScoredCT2([_ScoredRes(list(range(10)))]))  # rules off: untouched
    lim3.begin(100, 52, None)
    assert lim3.generate(None, [[1, 2, 3]], max_length=448)[0].scores == [-0.2]


def test_a_rejected_t0_is_never_chosen_over_a_real_retry():
    """faster-whisper's 'all failed' choice: max average log-prob (score * len / (len + 1))."""
    from audio_translator.asr import _mark_failed

    rej = _mark_failed(_ScoredRes(list(range(51)), score=-0.1))
    bad_retry = _ScoredRes(list(range(30)), score=-3.0)
    avg = lambda r: r.scores[0] * len(r.sequences_ids[0]) / (len(r.sequences_ids[0]) + 1)  # noqa: E731
    assert avg(rej) < -1.0 and max([rej, bad_retry], key=avg) is bad_retry


def test_c4_rejects_a_truncated_attempt_of_any_temperature_and_keeps_order():
    from audio_translator.asr import DecodeLimiter

    outs = [_ScoredRes(list(range(52)), score=-0.3), _ScoredRes(list(range(52)), score=-0.1),
            _ScoredRes(list(range(20)), score=-0.2)]
    lim = DecodeLimiter(_ScoredCT2(outs))
    lim.reject_truncated_all = True
    lim.begin(52, 52, None)
    a = lim.generate(None, [[1, 2, 3]], max_length=448)  # T=0 truncated
    b = lim.generate(None, [[1, 2, 3]], max_length=448, sampling_temperature=0.2)  # retry truncated too
    c = lim.generate(None, [[1, 2, 3]], max_length=448, sampling_temperature=0.4)  # ended by itself
    avg = lambda r: r.scores[0] * len(r.sequences_ids[0]) / (len(r.sequences_ids[0]) + 1)  # noqa: E731
    assert avg(a[0]) < -999 and avg(b[0]) < -999 and avg(c[0]) == pytest.approx(-0.2 * 20 / 21)
    assert avg(b[0]) > avg(a[0])  # order between marked attempts kept: the best real one stays best
    assert avg(b[0]) + 1000 == pytest.approx(-0.1 * 52 / 53)  # reversible
    assert lim.truncated == [True, True, False] and not lim.all_truncated and lim.rejected == 2
    assert max([a[0], b[0], c[0]], key=avg) is c[0]  # a truncated attempt is never chosen over another


@pytest.mark.parametrize("text,want", [
    ("nhạc sĩ la hối với ca khúc xuân và tuổi trẻ. a sĩ la hối với ca khúc xuân và tuổi trẻ. a sĩ la",
     "nhạc sĩ la hối với ca khúc xuân và tuổi trẻ."),
    ("trừ khi bạn biết tôn trọng bản thân mình. a...................", "trừ khi bạn biết tôn trọng bản thân mình."),
    ("xin chào bạn", "xin chào bạn"),
])
def test_trim_truncated_tail(text, want):
    assert asr_mod.trim_truncated_tail(text) == want


def test_all_truncated_restores_the_log_prob_and_trims(monkeypatch):
    class Model:
        def __init__(self, *a, **k):
            self.ct2 = self.model = _ScoredCT2([])

        def transcribe(self, audio, **kw):
            self.ct2.outs = [_ScoredRes(list(range(77)), score=-0.2), _ScoredRes(list(range(77)), score=-0.3)]
            r0 = self.model.generate(None, [[1, 2, 3]], max_length=448)
            r1 = self.model.generate(None, [[1, 2, 3]], max_length=448, sampling_temperature=0.2)
            best = max([r0[0], r1[0]], key=lambda r: r.scores[0])
            avg = best.scores[0] * 77 / 78
            return iter([types.SimpleNamespace(text="đã già. a sĩ la hối với đã già. a sĩ la hối với", avg_logprob=avg,
                                               no_speech_prob=0.1, compression_ratio=1.0, temperature=0.2)]), None

    monkeypatch.setitem(sys.modules, "faster_whisper", types.SimpleNamespace(WhisperModel=Model))
    a = WhisperASR("small", max_tokens_per_s=20, max_tokens_base=17, reject_truncated_all=True)  # 3 s -> 77 tokens
    segs = a.transcribe_segments(np.zeros(48000, np.int16), 16000)
    assert a.last_all_truncated and segs[0].avg_logprob == pytest.approx(-0.2 * 77 / 78)
    assert segs[0].text == "đã già. a sĩ la hối với"
    assert a.decode == "cap20+17,reject:trunc"


def test_real_model_generates_exactly_the_configured_tokens():
    """The configured limit is the number of tokens the model really generates (phowhisper-small)."""
    import glob
    import wave
    from pathlib import Path

    if not (asr_mod.local_model_dir("phowhisper-small") / "model.bin").exists():
        pytest.skip("phowhisper-small not converted on this machine")
    waves = sorted(glob.glob(str(Path("~/data/vivos/test/waves/*/*.wav").expanduser())))
    if not waves:
        pytest.skip("VIVOS test set not present")
    pytest.importorskip("faster_whisper")
    path = max(waves[:200], key=lambda p: wave.open(p).getnframes())
    with wave.open(path) as w:
        audio = np.frombuffer(w.readframes(w.getnframes()), np.int16)
    a = WhisperASR("phowhisper-small", reject_truncated_all=False, decode_budget_s=999.0)
    a.load()
    for n in (6, 12, 20):
        for prompt in (None, "Xin chào, sau đây là đoạn hội thoại bằng tiếng Việt."):
            a.initial_prompt = prompt
            a.first_min_tokens = None
            a.attempt_limits = lambda audio_s, _n=n: (_n, _n)
            a.transcribe_segments(audio, 16000)
            t, generated, limit = a.limiter.calls[0][1:]
            assert (t, generated, limit) == (0.0, n, n), (prompt, n)
