"""eval/mt_oracle_ablation.py: the five source variants."""

import importlib.util
import sys
from pathlib import Path

import pytest

pytest.importorskip("sacrebleu")  # dependency group `eval`

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "eval"))
_spec = importlib.util.spec_from_file_location("mt_oracle_ablation", _ROOT / "eval" / "mt_oracle_ablation.py")
ab = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ab)


def test_variants_go_from_asr_like_to_reference():
    v = ab.variants("vi")
    src = "Tôi ở Hà Nội, còn bạn?"
    assert v["norm"](src) == "tôi ở hà nội còn bạn"
    assert v["sent"](src) == "Tôi ở hà nội còn bạn."
    assert v["case+sent"](src) == "Tôi ở Hà Nội còn bạn."
    assert v["punct+sent"](src) == "Tôi ở hà nội, còn bạn?"
    assert v["full"](src) == src
