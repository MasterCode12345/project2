"""ModelManager: catalog/status, background loading, LRU with pins, failures."""

import threading
import time

import pytest

from audio_translator.model_manager import (
    CATALOG, DOWNLOADED, FAILED, LOADED, LOADING, NOT_DOWNLOADED, ModelManager,
)
from conftest import FakeModelFactory


class Changes:
    def __init__(self):
        self.items = []
        self.cv = threading.Condition()

    def __call__(self, name, status, detail):
        with self.cv:
            self.items.append((name, status, detail))
            self.cv.notify_all()

    def wait(self, name, status, timeout=5):
        with self.cv:
            assert self.cv.wait_for(lambda: (name, status) in [(n, s) for n, s, _ in self.items], timeout), \
                self.items


def make(gated=(), fail=(), downloaded=("small", "medium"), cap=2):
    f = FakeModelFactory(gated=gated, fail=fail)
    ch = Changes()
    m = ModelManager(f, max_loaded=cap, is_downloaded=lambda n: n in downloaded, on_change=ch)
    return m, f, ch


def test_catalog_is_multilingual_hub_models_then_local_vietnamese_models():
    assert list(CATALOG) == ["tiny", "base", "small", "medium", "large-v3-turbo", "large-v3",
                             "phowhisper-small", "phowhisper-medium"]
    assert not any(n.endswith(".en") or n.startswith("distil") for n in CATALOG)


def test_status_reports_download_state_and_size():
    m, _, _ = make()
    info = {i.name: i for i in m.models()}
    assert info["small"].status == DOWNLOADED and info["medium"].status == DOWNLOADED
    assert info["large-v3"].status == NOT_DOWNLOADED and info["large-v3"].size_label == "3.1 GB"
    assert info["tiny"].size_label == "75 MB"


def test_request_returns_at_once_and_loads_in_background():
    m, f, ch = make(gated=["small"])
    t0 = time.monotonic()
    m.request("small")
    assert time.monotonic() - t0 < 0.2
    assert f.started["small"].wait(2)
    assert m.status("small") == LOADING
    assert f.load_calls == ["small"]
    f.release("small")
    ch.wait("small", LOADED)
    assert m.status("small") == LOADED
    assert [s for n, s, _ in ch.items if n == "small"] == [LOADING, LOADED]


def test_not_downloaded_model_is_downloaded_then_loaded():
    m, f, ch = make(gated=["large-v3-turbo"])
    m.request("large-v3-turbo")
    ch.wait("large-v3-turbo", LOADING)
    assert "downloading" in ch.items[0][2] and "1.6 GB" in ch.items[0][2]
    f.release("large-v3-turbo")
    ch.wait("large-v3-turbo", LOADED)
    assert m.status("large-v3-turbo") == LOADED


def test_cached_request_calls_back_immediately_without_loading():
    m, f, _ = make()
    m.get("small")
    done = []
    m.request("small", on_done=lambda n, e: done.append((n, e)))
    assert done == [("small", None)] and f.load_calls == ["small"]


def test_get_waits_for_inflight_load_without_second_load():
    m, f, _ = make(gated=["medium"])
    m.request("medium")
    got = []
    t = threading.Thread(target=lambda: got.append(m.get("medium")))
    t.start()
    time.sleep(0.1)
    assert not got
    f.release("medium")
    t.join(2)
    assert got and got[0].model_size == "medium" and f.load_calls == ["medium"]


def test_lru_eviction_spares_pinned_model():
    m, f, ch = make()
    m.get("small", pin=True)
    m.get("medium")
    m.get("tiny")  # cap 2: LRU is "small" but it is pinned -> "medium" goes
    assert m.loaded_names() == ["small", "tiny"]
    assert ("medium", DOWNLOADED, "released from memory") in ch.items
    m.unpin("small")
    m.get("base")
    assert m.loaded_names() == ["tiny", "base"]


def test_load_failure_reports_and_get_raises():
    m, f, ch = make(fail=["small"])
    errs = []
    m.request("small", on_done=lambda n, e: errs.append(e))
    ch.wait("small", FAILED)
    assert m.status("small") == FAILED and isinstance(errs[0], OSError)
    with pytest.raises(RuntimeError, match="corrupt model small"):
        m.get("small")


def test_unknown_model_rejected():
    m, _, _ = make()
    with pytest.raises(ValueError):
        m.request("large-v2")


def test_lru_released_before_new_load_completes():
    m, f, ch = make(gated=["tiny"])
    m.get("small")
    m.get("medium")
    m.request("tiny")
    assert f.started["tiny"].wait(2)
    assert m.loaded_names() == ["medium"]  # room made before loading: never cap + 1 resident
    assert ("small", DOWNLOADED, "released from memory") in ch.items
    f.release("tiny")
    ch.wait("tiny", LOADED)
    assert m.loaded_names() == ["medium", "tiny"]
