"""A GPU guard is only worth having if it fails.

The failure it prevents has no symptom: a container built with CUDA wheels but
started without a device reservation imports torch fine, reports
``cuda.is_available() == False``, and extracts every vector on CPU. From outside
that is indistinguishable from "ingestion is slow today".

So the tests that matter here are the ones asserting a non-zero exit, not the
ones asserting it prints something.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from scripts.assert_cuda import cuda_report

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "assert_cuda.py"


def run(env_extra: dict, *args: str) -> subprocess.CompletedProcess:
    import os

    env = {**os.environ, "PYTHONUTF8": "1", **env_extra}
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        capture_output=True, text=True, encoding="utf-8", errors="replace", env=env,
        cwd=str(SCRIPT.parent.parent), timeout=120,
    )


def test_report_describes_what_torch_actually_sees():
    report = cuda_report()
    assert set(report) >= {"ok", "reason", "torch"}
    assert isinstance(report["ok"], bool)


def test_a_cpu_host_without_the_flag_is_not_an_error():
    """A CPU deployment is a legitimate configuration, not a failure."""
    result = run({"MUSIC_REQUIRE_CUDA": ""})
    assert result.returncode == 0


def test_report_mode_never_fails_even_when_gpu_was_required():
    result = run({"MUSIC_REQUIRE_CUDA": "1"}, "--report")
    assert result.returncode == 0


# The failure branch is forced rather than waited for. This machine has a 4070,
# so a hardware-dependent test skips here and the guard's whole reason for
# existing goes unexercised — green suite, untested gate.
CPU_ONLY = {"ok": False, "reason": "当前是 CPU 版 PyTorch（2.5.1+cpu）。"
                                   "镜像需要用 GPU overlay 构建："
                                   "docker compose -f docker-compose.yml "
                                   "-f docker-compose.gpu.yml build",
            "torch": "2.5.1+cpu", "cuda_build": None, "devices": [], "device_count": 0}
NO_DEVICE = {"ok": False, "reason": "PyTorch 带 CUDA 12.4 但看不到设备。"
                                    "多半是 compose 服务缺 "
                                    "deploy.resources.reservations.devices，"
                                    "或宿主机没装 NVIDIA Container Toolkit。",
             "torch": "2.5.1+cu124", "cuda_build": "12.4", "devices": [], "device_count": 0}


def _main_with(monkeypatch, report, *, required, argv=("assert_cuda.py",)):
    import scripts.assert_cuda as mod

    monkeypatch.setattr(mod, "cuda_report", lambda: report)
    monkeypatch.setenv("MUSIC_REQUIRE_CUDA", "1" if required else "")
    monkeypatch.setattr(sys, "argv", list(argv))
    return mod.main()


@pytest.mark.parametrize("report", [CPU_ONLY, NO_DEVICE])
def test_requiring_cuda_without_a_gpu_exits_non_zero(monkeypatch, report):
    """The whole point. Forced, so it runs on GPU hosts too."""
    assert _main_with(monkeypatch, report, required=True) == 1


@pytest.mark.parametrize("report", [CPU_ONLY, NO_DEVICE])
def test_the_same_state_is_fine_when_nobody_asked_for_gpu(monkeypatch, report):
    assert _main_with(monkeypatch, report, required=False) == 0


@pytest.mark.parametrize("report", [CPU_ONLY, NO_DEVICE])
def test_report_mode_never_fails_however_bad_the_state(monkeypatch, report):
    assert _main_with(monkeypatch, report, required=True,
                      argv=("assert_cuda.py", "--report")) == 0


def test_a_working_gpu_passes_even_when_required(monkeypatch):
    good = {"ok": True, "reason": "", "torch": "2.5.1+cu124", "cuda_build": "12.4",
            "devices": ["NVIDIA GeForce RTX 4070 Laptop GPU"], "device_count": 1}
    assert _main_with(monkeypatch, good, required=True) == 0


def test_the_failure_says_how_to_fix_it():
    """"CUDA unavailable" alone sends someone hunting through three files.
    The two causes need different fixes, so they must read differently."""
    assert "docker-compose.gpu.yml" in CPU_ONLY["reason"]      # rebuild
    assert "devices" in NO_DEVICE["reason"]                     # fix compose


@pytest.mark.parametrize("value", ["1", "true", "yes", "TRUE", "Yes"])
def test_the_flag_accepts_the_usual_truthy_spellings(monkeypatch, value):
    import scripts.assert_cuda as mod

    monkeypatch.setattr(mod, "cuda_report", lambda: CPU_ONLY)
    monkeypatch.setenv("MUSIC_REQUIRE_CUDA", value)
    monkeypatch.setattr(sys, "argv", ["assert_cuda.py"])
    assert mod.main() == 1


@pytest.mark.parametrize("value", ["", "0", "false", "no"])
def test_a_falsey_flag_leaves_cpu_alone(value):
    assert run({"MUSIC_REQUIRE_CUDA": value}).returncode == 0


def test_a_cpu_build_is_named_as_the_cause_rather_than_the_symptom():
    """Distinguishing "CPU wheels installed" from "CUDA wheels, no device" is
    the difference between rebuilding and fixing compose."""
    assert "CPU 版" in CPU_ONLY["reason"]
    assert "看不到设备" in NO_DEVICE["reason"]


# ---- the compose contract this guard backs up -------------------------------

def test_each_gpu_service_reserves_exactly_one_device():
    """Compose appends list entries, so a reservation repeated in the overlay
    and the base asks for two GPUs on a one-GPU machine. ingest-worker's lives
    in docker-compose.yml; only backend's belongs in the overlay."""
    root = SCRIPT.parent.parent
    base = (root / "docker-compose.yml").read_text(encoding="utf-8")
    overlay = (root / "docker-compose.gpu.yml").read_text(encoding="utf-8")
    assert base.count("capabilities: [gpu]") == 1      # ingest-worker
    assert overlay.count("capabilities: [gpu]") == 1   # backend
    assert (base + overlay).count("driver: nvidia") == 2


def test_both_gpu_services_require_cuda_at_runtime():
    overlay = (SCRIPT.parent.parent / "docker-compose.gpu.yml").read_text(encoding="utf-8")
    assert overlay.count("MUSIC_REQUIRE_CUDA") == 2


def test_the_launcher_builds_before_starting_a_gpu_profile():
    """`up` reuses an existing image, so without an explicit build the overlay's
    cu124 arg is never applied and `up gpu` yields a CPU container."""
    launcher = (SCRIPT.parent.parent / "soultuner.ps1").read_text(encoding="utf-8")
    assert "build backend ingest-worker" in launcher


def test_the_ingest_action_loads_the_gpu_overlay():
    """A profile selects services; the overlay sets build args and devices.
    Passing only --profile gpu runs a CPU-built worker."""
    launcher = (SCRIPT.parent.parent / "soultuner.ps1").read_text(encoding="utf-8")
    ingest = launcher[launcher.index('"ingest" {'):]
    assert "docker-compose.gpu.yml" in ingest[:1500]


def test_up_gpu_self_checks_both_services():
    """Checking only the backend leaves the case that costs most: the
    long-running worker quietly extracting every vector on CPU."""
    launcher = (SCRIPT.parent.parent / "soultuner.ps1").read_text(encoding="utf-8")
    up_block = launcher[launcher.index('"up" {'):launcher.index('"down" {')]
    assert "exec -T backend python scripts/assert_cuda.py" in up_block
    assert "exec -T ingest-worker python scripts/assert_cuda.py" in up_block


def test_the_ingest_action_also_self_checks():
    launcher = (SCRIPT.parent.parent / "soultuner.ps1").read_text(encoding="utf-8")
    ingest = launcher[launcher.index('"ingest" {'):]
    assert "assert_cuda.py" in ingest[:2000]
