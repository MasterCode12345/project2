"""scripts/convert_phowhisper.py with a fake converter/hub (offline)."""

import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest

_spec = importlib.util.spec_from_file_location(
    "convert_phowhisper", Path(__file__).resolve().parents[1] / "scripts" / "convert_phowhisper.py")
cv = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cv)


@pytest.fixture
def fake_tools(monkeypatch, tmp_path):
    calls = {"download": [], "convert": []}
    behaviour = {"fail": False, "files": ["model.bin", "tokenizer.json", "preprocessor_config.json"]}

    def snapshot_download(repo, revision):
        calls["download"].append((repo, revision))
        return str(tmp_path / "hf-src")

    class TransformersConverter:
        def __init__(self, src, copy_files):
            self.src, self.copy_files = src, copy_files

        def convert(self, out, quantization, force):
            calls["convert"].append((self.src, self.copy_files, quantization))
            for f in behaviour["files"]:
                (Path(out) / f).write_text("x")
            if behaviour["fail"]:
                raise RuntimeError("conversion blew up")

    monkeypatch.setitem(sys.modules, "huggingface_hub", types.SimpleNamespace(snapshot_download=snapshot_download))
    monkeypatch.setitem(sys.modules, "ctranslate2.converters",
                        types.SimpleNamespace(TransformersConverter=TransformersConverter))
    return calls, behaviour


def test_converts_pinned_revision_and_records_source(tmp_path, fake_tools):
    calls, _ = fake_tools
    out = cv.convert("small", tmp_path / "models", log=lambda *_: None)
    assert out == tmp_path / "models" / "phowhisper-small-ct2"
    assert calls["download"] == [cv.LOCAL_MODELS["phowhisper-small"]]
    assert calls["convert"][0][1:] == (["tokenizer.json", "preprocessor_config.json"], "float16")
    src = json.loads((out / "SOURCE.json").read_text())
    assert (src["repo"], src["revision"]) == cv.LOCAL_MODELS["phowhisper-small"]
    assert sorted(p.name for p in (tmp_path / "models").iterdir()) == ["phowhisper-small-ct2"]  # no temp left


def test_existing_output_is_kept_unless_forced(tmp_path, fake_tools):
    calls, _ = fake_tools
    cv.convert("small", tmp_path, log=lambda *_: None)
    msgs = []
    cv.convert("small", tmp_path, log=msgs.append)
    assert len(calls["convert"]) == 1 and "already exists" in msgs[0]
    cv.convert("small", tmp_path, force=True, log=lambda *_: None)
    assert len(calls["convert"]) == 2


@pytest.mark.parametrize("mode", ["raises", "incomplete"])
def test_failed_conversion_leaves_no_output(tmp_path, fake_tools, mode):
    _, behaviour = fake_tools
    if mode == "raises":
        behaviour["fail"] = True
    else:
        behaviour["files"] = ["model.bin"]  # tokenizer/preprocessor missing
    with pytest.raises(RuntimeError):
        cv.convert("medium", tmp_path, log=lambda *_: None)
    assert list(tmp_path.iterdir()) == []


def test_main_writes_where_the_app_looks(tmp_path, fake_tools, monkeypatch, capsys):
    _, behaviour = fake_tools
    monkeypatch.setattr(cv, "LOCAL_MODELS_DIR", tmp_path)
    assert cv.main(["small"]) == 0
    from audio_translator.asr import resolve_model
    assert resolve_model("phowhisper-small", tmp_path) == str(tmp_path / "phowhisper-small-ct2")
    behaviour["fail"] = True
    assert cv.main(["medium"]) == 1
    assert "conversion blew up" in capsys.readouterr().err
    for bad in (["large"], ["small", "--out-dir", str(tmp_path)]):  # no other output root
        with pytest.raises(SystemExit):
            cv.main(bad)


def test_missing_conversion_tools_explain_how_to_run(tmp_path, monkeypatch, capsys):
    monkeypatch.setitem(sys.modules, "ctranslate2.converters", None)  # import fails
    monkeypatch.setitem(sys.modules, "huggingface_hub", None)
    monkeypatch.setattr(cv, "LOCAL_MODELS_DIR", tmp_path)
    assert cv.main(["small"]) == 2
    assert "uv run --with transformers --with torch" in capsys.readouterr().err
