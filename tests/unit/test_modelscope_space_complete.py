from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


SPACE = Path(__file__).resolve().parents[2] / "deploy" / "modelscope_space"
sys.path.insert(0, str(SPACE))

RUNTIME_SPEC = importlib.util.spec_from_file_location("space_planner_runtime", SPACE / "planner_runtime.py")
assert RUNTIME_SPEC and RUNTIME_SPEC.loader
runtime = importlib.util.module_from_spec(RUNTIME_SPEC)
RUNTIME_SPEC.loader.exec_module(runtime)

RETRIEVAL_SPEC = importlib.util.spec_from_file_location("space_retrieval_demo", SPACE / "retrieval_demo.py")
assert RETRIEVAL_SPEC and RETRIEVAL_SPEC.loader
retrieval = importlib.util.module_from_spec(RETRIEVAL_SPEC)
RETRIEVAL_SPEC.loader.exec_module(retrieval)

READINESS_SPEC = importlib.util.spec_from_file_location("space_amd_readiness", SPACE / "amd_readiness.py")
assert READINESS_SPEC and READINESS_SPEC.loader
readiness = importlib.util.module_from_spec(READINESS_SPEC)
READINESS_SPEC.loader.exec_module(readiness)

BENCHMARK_SPEC = importlib.util.spec_from_file_location("space_benchmark_endpoint", SPACE / "benchmark_endpoint.py")
assert BENCHMARK_SPEC and BENCHMARK_SPEC.loader
benchmark = importlib.util.module_from_spec(BENCHMARK_SPEC)
BENCHMARK_SPEC.loader.exec_module(benchmark)


def test_space_card_text_has_theme_independent_contrast() -> None:
    source = (SPACE / "app.py").read_text(encoding="utf-8")
    assert ".st-card h3" in source
    assert "color: #102a20 !important" in source
    assert "color: #315647 !important" in source


def test_public_catalog_and_hybrid_retrieval() -> None:
    assert len(retrieval.load_catalog()) == 120
    query = "90年代英文摇滚，但整体要温暖一点"
    plan = runtime.safe_plan(query)
    route = runtime.compile_route(plan)
    rows = retrieval.retrieve(query, plan, route, top_k=5)
    assert len(rows) == 5
    assert route["graph_weight"] > 0
    assert route["dense_weight"] > 0


def test_subjective_acoustics_are_dense_only() -> None:
    plan = runtime.safe_plan("我希望 bass 更重、鼓声更大一些")
    assert plan["lane_policy"] == {"graph": "off", "dense": "required", "web": "off"}


def test_amd_readiness_fails_closed_without_rocm(monkeypatch) -> None:
    monkeypatch.setattr(
        readiness,
        "runtime_capabilities",
        lambda: {"device": "CPU", "accelerator": "none"},
    )
    report = readiness.readiness_report(require_rocm=True, probe_adapter=False, probe_endpoint=False)
    assert report["ready"] is False
    assert report["findings"] == ["ROCm/HIP GPU is required but was not detected"]


def test_amd_readiness_accepts_rocm(monkeypatch) -> None:
    monkeypatch.setattr(
        readiness,
        "runtime_capabilities",
        lambda: {"device": "AMD GPU", "accelerator": "ROCm/HIP"},
    )
    report = readiness.readiness_report(require_rocm=True, probe_adapter=False, probe_endpoint=False)
    assert report["ready"] is True
    assert report["findings"] == []


def test_amd_readiness_checks_adapter_files(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        readiness,
        "runtime_capabilities",
        lambda: {"device": "AMD GPU", "accelerator": "ROCm/HIP"},
    )
    monkeypatch.setenv("SOULTUNER_ADAPTER_DIR", str(tmp_path))
    report = readiness.readiness_report(require_rocm=True, probe_adapter=True, probe_endpoint=False)
    assert report["ready"] is False
    assert report["findings"] == [
        "missing adapter file: adapter_config.json",
        "missing adapter file: adapter_model.safetensors",
    ]


def test_public_endpoint_summary_does_not_store_prompt_or_response_text() -> None:
    report = benchmark.summarise(
        [
            {"case_id": "mood", "repeat": 0, "ok": True, "latency_ms": 100.0},
            {"case_id": "hybrid", "repeat": 0, "ok": True, "latency_ms": 200.0},
        ]
    )
    assert report["contract_valid_rate"] == 1.0
    assert report["latency_ms"]["p50"] == 100.0
    assert "query" not in report
    assert "raw_response" not in report
