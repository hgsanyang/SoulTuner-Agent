from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace


MODULE_PATH = Path(__file__).resolve().parents[2] / "scripts" / "assert_cuda.py"
SPEC = importlib.util.spec_from_file_location("soultuner_accelerator_probe", MODULE_PATH)
assert SPEC and SPEC.loader
probe = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(probe)


def test_rocm_report_uses_torch_cuda_namespace(monkeypatch) -> None:
    fake_torch = SimpleNamespace(
        __version__="2.10.0+rocm7.2.4",
        version=SimpleNamespace(cuda=None, hip="7.2.4"),
        cuda=SimpleNamespace(
            is_available=lambda: True,
            device_count=lambda: 1,
            get_device_name=lambda _index: "AMD Instinct MI308X",
        ),
    )
    monkeypatch.setitem(__import__("sys").modules, "torch", fake_torch)
    report = probe.cuda_report()
    assert report["ok"] is True
    assert report["hip_build"] == "7.2.4"
    assert report["cuda_build"] is None
    assert report["devices"] == ["AMD Instinct MI308X"]


def test_accelerator_requirement_accepts_legacy_alias(monkeypatch) -> None:
    monkeypatch.setenv("MUSIC_REQUIRE_CUDA", "1")
    monkeypatch.delenv("MUSIC_REQUIRE_ACCELERATOR", raising=False)
    monkeypatch.setattr(
        probe,
        "cuda_report",
        lambda: {
            "ok": False,
            "reason": "no device",
            "torch": "2.10.0",
            "cuda_build": None,
            "hip_build": None,
            "device_count": 0,
            "devices": [],
        },
    )
    monkeypatch.setattr("sys.argv", ["assert_cuda.py"])
    assert probe.main() == 1
