"""Glossary: files, matching, placeholder protection and the safe fallback (plan-glossary.md)."""

import unicodedata

import pytest

from audio_translator import glossary as g
from audio_translator.glossary import Entry, Glossary, GlossaryError, GlossaryTranslator, glossary_path, parse


@pytest.fixture
def gl():
    return Glossary.load("vi", "en")[0]


def targets(gl, text):
    return [e.target for _a, _b, e in gl.find(text)]


# ------------------------------------------------------------ files
def test_shipped_file_is_valid_and_has_the_requested_entries():
    entries, problems = parse(glossary_path("vi", "en"), strict=True)  # a format error fails here
    assert problems == []
    got = {e.source: e.target for e in entries}
    assert got["Đại học Bách khoa Hà Nội"] == "Hanoi University of Science and Technology"
    assert got["Trường Công nghệ Thông tin và Truyền thông"] == "School of Information and Communication Technology"
    assert got["Trần Hải Anh"] == "Tran Hai Anh" and got["Trần Trung Nghĩa"] == "Tran Trung Nghia"
    assert got["Đại học Quốc gia Hà Nội"] == "Vietnam National University, Hanoi"
    assert "đồ án tốt nghiệp" not in got and "giảng viên hướng dẫn" not in got  # proper names only
    assert "Đại học Bách khoa" not in got  # several universities share it
    assert all(len(e.source.split()) >= 3 for e in entries)


def test_short_terms_are_refused_unless_flagged(tmp_path):
    p = tmp_path / "g.tsv"
    p.write_text("bách khoa\tpolytechnic\nHà Nội\tHanoi\nViệt Nam\tVietnam\tshort_ok\n", encoding="utf-8")
    with pytest.raises(GlossaryError, match="g.tsv:1"):
        parse(p, strict=True)
    entries, problems = parse(p, strict=False)
    assert [e.source for e in entries] == ["Việt Nam"]
    assert len(problems) == 2 and problems[0].endswith("(generic words are refused; add 'short_ok' in column 3 to keep it)")
    assert ":1:" in problems[0] and ":2:" in problems[1]


def test_user_file_adds_overrides_and_skips_bad_lines(tmp_path):
    p = tmp_path / "mine.tsv"
    p.write_text("# my terms\nĐại học Bách khoa Hà Nội\tHUST\nNguyễn Văn An\tNguyen Van An\nchỉ một cột\n"
                 "hai\tcột\n", encoding="utf-8")
    gl, problems = Glossary.load("vi", "en", p)
    assert targets(gl, "Tôi học ở Đại học Bách khoa Hà Nội với Nguyễn Văn An.") == ["HUST", "Nguyen Van An"]
    assert targets(gl, "Thầy Trần Hải Anh") == ["Tran Hai Anh"]  # shipped entries stay
    assert [x.split(":")[-2] for x in problems] == ["4", "5"]
    with pytest.raises(FileNotFoundError, match="Glossary file not found"):
        Glossary.load("vi", "en", tmp_path / "typo.tsv")


# ------------------------------------------------------------ matching
@pytest.mark.parametrize("text", [
    "Bách khoa toàn thư là nguồn tham khảo.", "Em học bách khoa vì thích kỹ thuật.",
    "Hà Nội có nhiều trường đại học.", "Đại học Bách khoa Đà Nẵng rất đẹp.", "Bách khoa, Hà Nội", "Tran Hai Anh",
    "Trần Hai Anh",  # tone marks matter
])
def test_generic_or_different_words_do_not_match(gl, text):
    assert gl.find(text) == []


def test_case_nfc_whole_words_and_longest_first(gl):
    hust = "Hanoi University of Science and Technology"
    assert targets(gl, "tôi học ở đại học bách khoa hà nội.") == [hust]
    decomposed = unicodedata.normalize("NFD", "Trường Đại học Bách khoa Hà Nội")
    found = gl.find(decomposed)
    assert [e.source for *_x, e in found] == ["Trường Đại học Bách khoa Hà Nội"]  # longest wins
    assert [e.source for *_x, e in gl.find("học ở Bách khoa Hà Nội")] == ["Bách khoa Hà Nội"]
    assert targets(gl, "TRẦN TRUNG NGHĨA và Trần Hải Anh") == ["Tran Trung Nghia", "Tran Hai Anh"]
    assert gl.find("Trần Trung Nghĩacó") == []  # whole words only


# ------------------------------------------------------------ translator
class Echo:
    """Fake MT: returns the input with Vietnamese markers mapped; records what it received."""

    def __init__(self, transform=lambda s: s):
        self.calls = []
        self.transform = transform

    def translate(self, text):
        self.calls.append(text)
        return self.transform(text)


def test_sentence_without_terms_is_passed_unchanged(gl):
    inner = Echo(lambda s: f"EN<{s}>")
    t = GlossaryTranslator(inner, gl, "x")
    for s in ("Xin chào, bạn khỏe không?", "Bách khoa toàn thư.", ""):
        assert t.translate(s) == f"EN<{s}>" and inner.calls[-1] == s and t.last == ""
    assert t.stats == {"calls": 3, "with_terms": 0, "fallbacks": 0}


@pytest.mark.parametrize("form", ["x", "zqx", "bracket", "under", "inline"])
def test_placeholders_come_back_as_target_terms(gl, form):
    inner = Echo(lambda s: s.replace("Tôi là sinh viên của", "I am a student of").replace("và", "and"))
    t = GlossaryTranslator(inner, gl, form)
    out = t.translate("Tôi là sinh viên của Đại học Bách khoa Hà Nội và Trần Hải Anh")
    assert out == "I am a student of Hanoi University of Science and Technology and Tran Hai Anh"
    assert "Bách khoa" not in inner.calls[0] and len(inner.calls) == 1
    assert t.last == "Đại học Bách khoa Hà Nội;Trần Hải Anh"


@pytest.mark.parametrize("form,broken", [
    ("x", lambda s: s.replace("X1", "")),  # lost
    ("x", lambda s: s + " X1"),  # repeated
    ("x", lambda s: s.replace("X1", "X 1")),  # deformed
    ("zqx", lambda s: s.replace("ZQX1", "Zqx 1")),
    ("bracket", lambda s: s.replace("[1]", "[ 1")),
    ("under", lambda s: s.replace("__T1__", "_T1_")),
    ("inline", lambda s: s.replace("Hanoi University of Science and Technology", "Hanoi Polytechnic")),
])
def test_a_damaged_placeholder_falls_back_to_the_plain_translation(gl, form, broken):
    plain = "PLAIN"
    calls = []

    def mt(s):
        calls.append(s)
        return broken(s) if len(calls) == 1 else plain

    t = GlossaryTranslator(Echo(mt), gl, form)
    assert t.translate("Tôi học ở Đại học Bách khoa Hà Nội") == plain
    assert t.last == "fallback" and t.stats["fallbacks"] == 1 and calls[1] == "Tôi học ở Đại học Bách khoa Hà Nội"


def test_no_placeholder_residue_ever_leaks(gl):
    for form in ("x", "zqx", "bracket", "under"):
        t = GlossaryTranslator(Echo(lambda s: s + " ZQX X9 [7] __T3__"), gl, form)
        out = t.translate("Thầy Trần Hải Anh")
        assert out == "Thầy Trần Hải Anh ZQX X9 [7] __T3__" and t.last == "fallback"  # the plain call (echo)


def test_wrapper_passes_other_attributes_through(gl):
    class Inner(Echo):
        loaded = False

        def load(self):
            self.loaded = True

    inner = Inner()
    t = GlossaryTranslator(inner, gl)
    t.load()
    assert inner.loaded
    with pytest.raises(ValueError):
        GlossaryTranslator(inner, gl, "nope")


def test_entry_order_user_overrides_shipped():
    gl = Glossary([Entry("Trần Hải Anh", "A", "x:1"), Entry("trần hải anh", "B", "y:1")])
    assert [e.target for *_x, e in gl.find("Trần Hải Anh")] == ["B"] and len(gl) == 1


def test_empty_glossary_matches_nothing():
    assert Glossary([]).find("Trần Hải Anh") == [] and g.MIN_SYLLABLES == 3


# ------------------------------------------------------------ app wiring
def test_glossary_in_a_session_and_its_log_column(harness, tmp_path):
    from audio_translator.config import Settings
    from audio_translator.models import INFO, RESULT
    from conftest import FakeASR, FakeTranslator, utterance
    from test_latency_log import wait_rows

    class MT(FakeTranslator):
        def translate(self, text):
            self.calls.append(text)
            return text.replace("Tôi học ở", "I study at")

    p = tmp_path / "lat.csv"
    mt = MT()
    h = harness(settings=Settings(glossary=True, latency_log=str(p)), translator=mt,
                asr=FakeASR(responses=["Tôi học ở Đại học Bách khoa Hà Nội", "Xin chào"]))
    h.start()
    h.log.wait_for(lambda e: e.kind == INFO and e.data.get("glossary_terms"))
    h.say(utterance(10), utterance(10))
    res = h.log.wait_for(lambda e: e.kind == RESULT and e.segment_id == 1)
    assert res.data["text"] == "I study at Hanoi University of Science and Technology"
    assert mt.calls[0] == "Tôi học ở ZQX1"
    rows = {r["segment_id"]: r for r in wait_rows(p, 2)}
    assert rows["1"]["glossary"] == "Đại học Bách khoa Hà Nội" and rows["2"]["glossary"] == ""


def test_glossary_on_by_default_and_off_means_no_wrapper(harness):
    from audio_translator.config import Settings
    from audio_translator.models import RESULT
    from conftest import FakeASR, utterance

    assert Settings().glossary is True and Settings().glossary_placeholder == "zqx"  # bench decision
    h = harness(settings=Settings(glossary=False), asr=FakeASR(responses=["Đại học Bách khoa Hà Nội"]))
    h.start()
    h.say(utterance(10))
    assert h.log.wait_for(lambda e: e.kind == RESULT).data["text"] == "EN<Đại học Bách khoa Hà Nội>"
    assert h.orch._mt is None


def test_missing_user_glossary_refuses_the_start_and_bad_lines_only_warn(harness, tmp_path):
    from audio_translator.config import Settings
    from audio_translator.models import ERROR, RUNNING, WARNING

    h = harness(settings=Settings(glossary=True, glossary_path=str(tmp_path / "typo.tsv")))
    assert h.orch.start(None, "BlackHole")
    err = h.log.wait_for(lambda e: e.kind == ERROR)
    assert "Glossary file not found" in err.message and "typo.tsv" in err.message
    good = tmp_path / "mine.tsv"
    good.write_text("Nguyễn Văn An\tNguyen Van An\nhỏng\n", encoding="utf-8")
    h2 = harness(settings=Settings(glossary=True, glossary_path=str(good)))
    h2.start()
    w = h2.log.wait_for(lambda e: e.kind == WARNING and e.data.get("glossary_problem"))
    assert "mine.tsv:2" in w.message and h2.orch.state == RUNNING


def test_glossary_flags():
    from audio_translator import cli
    from audio_translator.config import Settings, glossary_settings

    p = cli.build_parser()
    assert glossary_settings(p.parse_args([])) == {}
    assert glossary_settings(p.parse_args(["--glossary"])) == {"glossary": True}
    assert glossary_settings(p.parse_args(["--glossary", "my.tsv"])) == {"glossary": True, "glossary_path": "my.tsv"}
    assert glossary_settings(p.parse_args(["--no-glossary"])) == {"glossary": False}
    assert Settings(**glossary_settings(p.parse_args(["--glossary", "x.tsv"]))).glossary_path == "x.tsv"
    pytest.importorskip("PySide6")
    from audio_translator.gui import parse_args

    args, _ = parse_args(["--glossary", "my.tsv"])
    assert glossary_settings(args) == {"glossary": True, "glossary_path": "my.tsv"}


def test_gui_shows_glossary_problems_and_a_missing_file(qapp_for_glossary):
    import os

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from test_frontends import _window

    from audio_translator.models import ERROR, WARNING, Event

    w, orch = _window(qapp_for_glossary)
    w.on_event(Event(WARNING, "Glossary line skipped: /x/mine.tsv:4: needs 'source<TAB>target'",
                     data={"glossary_problem": "/x/mine.tsv:4: needs 'source<TAB>target'"}))
    assert w.warn_banner.text() == "File thuật ngữ có dòng lỗi, đã bỏ qua: /x/mine.tsv:4"
    w.on_event(Event(ERROR, "Could not start: Glossary file not found: '/x/typo.tsv'"))
    assert w.error_label.text() == "Không tìm thấy file thuật ngữ '/x/typo.tsv' – kiểm tra đường dẫn --glossary"


@pytest.fixture
def qapp_for_glossary():
    import os

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])
