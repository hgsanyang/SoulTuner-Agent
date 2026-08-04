"""The loader's memory behaviour has to be enforced, not remembered.

The real measurement lives in a report. What CI has to prevent is someone
deleting ``mmap=True`` or ``assign=True`` — either one silently restores the
sequence that held two full fp32 copies of a 663M-parameter model and was
OOM-killed in a 4.7 GB container.

Nothing here loads a real model: the 663M weights are not present in CI and
there is no GPU. Every collaborator is a stand-in, so what is pinned is the call
contract — which is exactly the thing that would regress.
"""

from __future__ import annotations

import sys
import types

import pytest

import retrieval.muq_embedder as muq


class FakeTensor:
    def __init__(self, name: str = "w"):
        self.name = name


class FakeModel:
    """Records how it was loaded and moved."""

    def __init__(self, config=None):
        self.config = config
        self.load_calls: list[dict] = []
        self.to_calls: list[dict] = []
        self.half_called = False
        self.eval_called = False

    def load_state_dict(self, state, strict=True, assign=False):
        self.load_calls.append({"state": state, "strict": strict, "assign": assign})
        return [], []

    def to(self, *args, **kwargs):
        self.to_calls.append({"args": args, "kwargs": kwargs})
        return self

    def half(self):
        self.half_called = True
        return self

    def eval(self):
        self.eval_called = True
        return self


@pytest.fixture
def harness(monkeypatch, tmp_path):
    """Replace everything that touches disk, network or a GPU."""
    monkeypatch.setattr(muq, "_MUQ_MODEL", None, raising=False)
    monkeypatch.setattr(muq, "_get_device", lambda: "cuda")
    monkeypatch.setattr(muq, "_use_fp16", lambda device: True)
    monkeypatch.setattr(muq, "_local_files_only", lambda: False)
    monkeypatch.setattr(muq, "_rewrite_config_to_local_snapshots", lambda cfg: cfg)

    config_path = tmp_path / "config.json"
    config_path.write_text("{}", encoding="utf-8")
    weights = tmp_path / "pytorch_model.bin"
    weights.write_bytes(b"not-a-real-checkpoint")

    monkeypatch.setattr(
        muq, "_download",
        lambda repo, name: str(config_path if name.endswith(".json") else weights),
    )

    built: list[FakeModel] = []

    def factory(config=None):
        model = FakeModel(config)
        built.append(model)
        return model

    monkeypatch.setitem(sys.modules, "muq", types.SimpleNamespace(MuQMuLan=factory))

    calls: list[dict] = []
    sentinel = {"weight": FakeTensor()}

    def fake_load(path, **kwargs):
        calls.append(kwargs)
        if kwargs.get("mmap"):
            return sentinel
        return {"weight": FakeTensor("full-read")}

    monkeypatch.setattr(muq.torch, "load", fake_load)
    monkeypatch.delenv("MUSIC_MUQ_LEGACY_LOADER", raising=False)
    return types.SimpleNamespace(calls=calls, built=built, sentinel=sentinel)


# ---- the two flags that must not disappear ---------------------------------

def test_default_path_memory_maps_the_checkpoint(harness):
    """Without mmap the whole state dict materialises in RAM — one of the two
    copies that caused the OOM."""
    muq.get_muq_model()
    assert harness.calls, "torch.load was never called"
    assert harness.calls[0].get("mmap") is True
    assert harness.calls[0].get("weights_only") is True


def test_default_path_assigns_rather_than_copies(harness):
    """assign=False copies into the freshly allocated parameters, which is the
    second full copy. With assign=True the originals are freed instead."""
    muq.get_muq_model()
    load = harness.built[0].load_calls[0]
    assert load["assign"] is True
    assert load["state"] is harness.sentinel


def test_the_move_casts_during_transfer(harness):
    """`.to(device)` then `.half()` puts a full fp32 copy on the GPU first.
    Passing dtype to `.to()` casts in one step."""
    muq.get_muq_model()
    model = harness.built[0]
    assert model.to_calls, "model was never moved"
    kwargs = model.to_calls[0]["kwargs"]
    assert kwargs.get("device") == "cuda"
    assert kwargs.get("dtype") is muq.torch.float16
    assert model.half_called is False, "half() after to() defeats the point"


def test_the_model_survives_dropping_the_state_dict(harness):
    """del state + gc must not disturb the returned model."""
    model = muq.get_muq_model()
    assert model is harness.built[0]
    assert model.eval_called is True


# ---- fallback contract ------------------------------------------------------

def test_a_failed_mmap_load_raises_instead_of_reading_the_whole_file(harness, monkeypatch):
    """The automatic full-read fallback was the bug: it turns a clear failure
    here into an OOM kill later, elsewhere, with no trace of the cause."""
    def only_mmap_fails(path, **kwargs):
        harness.calls.append(kwargs)
        if kwargs.get("mmap"):
            raise ValueError("cannot memory-map this checkpoint")
        pytest.fail("fell back to a full read — that is the OOM path")

    monkeypatch.setattr(muq.torch, "load", only_mmap_fails)
    with pytest.raises(RuntimeError) as excinfo:
        muq.get_muq_model()

    message = str(excinfo.value)
    assert "ValueError" in message                      # what went wrong
    assert "MUSIC_MUQ_LEGACY_LOADER" in message         # how to proceed anyway
    assert "pytorch_model.bin" in message               # which file


def test_the_error_does_not_leak_the_full_path(harness, monkeypatch):
    """Only the file name, not the directory it sits in."""
    def boom(path, **kwargs):
        raise ValueError("nope")

    monkeypatch.setattr(muq.torch, "load", boom)
    with pytest.raises(RuntimeError) as excinfo:
        muq.get_muq_model()
    message = str(excinfo.value)
    assert "/" not in message.replace("MuQ low-memory", "")
    assert "\\" not in message


# ---- the escape hatch is opt-in --------------------------------------------

def test_the_legacy_loader_is_off_by_default(harness):
    muq.get_muq_model()
    assert harness.calls[0].get("mmap") is True


@pytest.mark.parametrize("value", ["1", "true", "yes", "TRUE"])
def test_the_legacy_loader_runs_only_when_explicitly_requested(harness, monkeypatch, value):
    monkeypatch.setenv("MUSIC_MUQ_LEGACY_LOADER", value)
    muq.get_muq_model()
    assert harness.calls[0].get("mmap") is None
    assert harness.calls[0].get("weights_only") is False
    load = harness.built[0].load_calls[0]
    assert load["assign"] is False
    assert harness.built[0].half_called is True


@pytest.mark.parametrize("value", ["", "0", "false", "no"])
def test_falsey_values_do_not_enable_the_legacy_loader(harness, monkeypatch, value):
    monkeypatch.setenv("MUSIC_MUQ_LEGACY_LOADER", value)
    muq.get_muq_model()
    assert harness.calls[0].get("mmap") is True


# ---- singleton behaviour must not regress ----------------------------------

def test_the_model_is_built_once(harness):
    first = muq.get_muq_model()
    second = muq.get_muq_model()
    assert first is second
    assert len(harness.built) == 1
    assert len(harness.calls) == 1


def test_a_failed_load_does_not_cache_a_broken_singleton(harness, monkeypatch):
    """A raise must leave the singleton unset, so a later call can retry rather
    than returning None forever."""
    def boom(path, **kwargs):
        raise ValueError("nope")

    monkeypatch.setattr(muq.torch, "load", boom)
    with pytest.raises(RuntimeError):
        muq.get_muq_model()
    assert muq._MUQ_MODEL is None
