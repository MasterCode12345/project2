"""scripts/convert_mt_model.py with a fake converter/hub (offline)."""

import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest

_spec = importlib.util.spec_from_file_location(
    "convert_mt_model", Path(__file__).resolve().parents[1] / "scripts" / "convert_mt_model.py")
cv = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cv)


@pytest.fixture
def fake_tools(monkeypatch, tmp_path):
    calls = {"download": [], "convert": []}
    behaviour = {"fail": False, "files": ["model.bin", "sentencepiece.bpe.model", "shared_vocabulary.json"]}

    def snapshot_download(repo, revision, allow_patterns):
        calls["download"].append((repo, revision, allow_patterns))
        return str(tmp_path / "hf-src")

    class TransformersConverter:
        def __init__(self, src, copy_files, low_cpu_mem_usage=False):
            self.src, self.copy_files, self.low_mem = src, copy_files, low_cpu_mem_usage

        def convert(self, out, quantization, force):
            calls["convert"].append((self.copy_files, quantization, self.low_mem))
            for f in behaviour["files"]:
                (Path(out) / f).write_text("x")
            if behaviour["fail"]:
                raise RuntimeError("conversion blew up")

    monkeypatch.setitem(sys.modules, "huggingface_hub", types.SimpleNamespace(snapshot_download=snapshot_download))
    monkeypatch.setitem(sys.modules, "ctranslate2.converters",
                        types.SimpleNamespace(TransformersConverter=TransformersConverter))
    return calls, behaviour


@pytest.mark.parametrize("name", ["nllb-600m", "m2m100-418m"])
def test_converts_pinned_revision_and_only_needed_files(tmp_path, fake_tools, name):
    calls, _ = fake_tools
    out = cv.convert(name, tmp_path / "models", log=lambda *_: None)
    assert out == tmp_path / "models" / f"{name}-ct2"
    repo, revision, patterns = calls["download"][0]
    assert (repo, revision) == cv.MT_MODELS[name][:2] and len(revision) == 40
    assert "pytorch_model.bin" in patterns and not any("rust" in p or p == "*" for p in patterns)
    assert calls["convert"][0] == (["sentencepiece.bpe.model"], "int8", True)
    src = json.loads((out / "SOURCE.json").read_text())
    assert (src["repo"], src["revision"], src["quantization"]) == (repo, revision, "int8")
    assert sorted(p.name for p in (tmp_path / "models").iterdir()) == [f"{name}-ct2"]  # no temp left


def test_existing_output_is_kept_unless_forced(tmp_path, fake_tools):
    calls, _ = fake_tools
    cv.convert("nllb-600m", tmp_path, log=lambda *_: None)
    msgs = []
    cv.convert("nllb-600m", tmp_path, log=msgs.append)
    assert len(calls["convert"]) == 1 and "already exists" in msgs[0]
    cv.convert("nllb-600m", tmp_path, force=True, log=lambda *_: None)
    assert len(calls["convert"]) == 2


@pytest.mark.parametrize("mode", ["raises", "incomplete"])
def test_failed_conversion_leaves_no_output(tmp_path, fake_tools, mode):
    _, behaviour = fake_tools
    if mode == "raises":
        behaviour["fail"] = True
    else:
        behaviour["files"] = ["model.bin"]  # sentencepiece model missing
    with pytest.raises(RuntimeError):
        cv.convert("m2m100-418m", tmp_path, log=lambda *_: None)
    assert list(tmp_path.iterdir()) == []


def test_main_writes_where_the_app_looks(tmp_path, fake_tools, monkeypatch, capsys):
    _, behaviour = fake_tools
    monkeypatch.setattr(cv, "MT_MODELS_DIR", tmp_path)
    assert cv.main(["nllb-600m"]) == 0
    from audio_translator.translate import mt_model_dir

    assert (mt_model_dir("nllb-600m", tmp_path) / "model.bin").is_file()
    behaviour["fail"] = True
    assert cv.main(["m2m100-418m"]) == 1
    assert "conversion blew up" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        cv.main(["madlad"])


def test_missing_conversion_tools_explain_how_to_run(tmp_path, monkeypatch, capsys):
    monkeypatch.setitem(sys.modules, "ctranslate2.converters", None)
    monkeypatch.setitem(sys.modules, "huggingface_hub", None)
    monkeypatch.setattr(cv, "MT_MODELS_DIR", tmp_path)
    assert cv.main(["nllb-600m"]) == 2
    assert "uv run --with transformers --with torch --with sentencepiece" in capsys.readouterr().err
