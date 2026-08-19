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

BOOTSTRAP_SPEC = importlib.util.spec_from_file_location("space_bootstrap", SPACE / "space_bootstrap.py")
assert BOOTSTRAP_SPEC and BOOTSTRAP_SPEC.loader
bootstrap = importlib.util.module_from_spec(BOOTSTRAP_SPEC)
BOOTSTRAP_SPEC.loader.exec_module(bootstrap)

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


def test_space_endpoint_is_private_by_default_and_requires_public_auth() -> None:
    script = (SPACE / "start_amd_35b.sh").read_text(encoding="utf-8")
    assert 'SOULTUNER_PLANNER_HOST:-127.0.0.1' in script
    assert '"${HOST}" == "0.0.0.0"' in script
    assert "SOULTUNER_SERVE_API_KEY" in script


def test_cpu_profile_does_not_start_local_planner(monkeypatch) -> None:
    monkeypatch.setenv("SOULTUNER_MODEL_PROFILE", "demo-heuristic")
    status = bootstrap.launch_local_planner_if_requested()
    assert status == {
        "requested": False,
        "state": "disabled",
        "profile": "demo-heuristic",
    }


def test_external_35b_endpoint_does_not_spawn_local_process(monkeypatch) -> None:
    monkeypatch.setenv("SOULTUNER_MODEL_PROFILE", "soultuner-v4.2-35b")
    monkeypatch.setenv("SOULTUNER_PLANNER_BASE_URL", "https://planner.example/v1")
    status = bootstrap.launch_local_planner_if_requested()
    assert status["state"] == "external-endpoint"
    assert status["base_url"] == "https://planner.example/v1"


def test_local_35b_status_reports_ready_endpoint(monkeypatch) -> None:
    monkeypatch.setenv("SOULTUNER_MODEL_PROFILE", "soultuner-v4.2-35b")
    monkeypatch.setenv("SOULTUNER_PLANNER_BASE_URL", "http://127.0.0.1:8000/v1")
    monkeypatch.setattr(bootstrap, "_endpoint_ready", lambda _base_url: True)
    monkeypatch.setattr(bootstrap, "_planner_process", None)
    status = bootstrap.planner_runtime_status()
    assert status["state"] == "ready"
    assert "已就绪" in bootstrap.startup_markdown(status)


def test_local_35b_status_reports_failed_process(monkeypatch) -> None:
    class FailedProcess:
        pid = 42

        @staticmethod
        def poll() -> int:
            return 7

    monkeypatch.setenv("SOULTUNER_MODEL_PROFILE", "soultuner-v4.2-35b")
    monkeypatch.setenv("SOULTUNER_PLANNER_BASE_URL", "http://127.0.0.1:8000/v1")
    monkeypatch.setattr(bootstrap, "_endpoint_ready", lambda _base_url: False)
    monkeypatch.setattr(bootstrap, "_planner_process", FailedProcess())
    status = bootstrap.planner_runtime_status()
    assert status["state"] == "failed"
    assert status["returncode"] == 7
    assert "退出码 `7`" in bootstrap.startup_markdown(status)


def test_gradio_entrypoint_bootstraps_requested_35b_profile() -> None:
    source = (SPACE / "app.py").read_text(encoding="utf-8")
    script = (SPACE / "start_amd_35b.sh").read_text(encoding="utf-8")
    assert "launch_local_planner_if_requested()" in source
    assert "live_startup_markdown" in source
    assert "gr.Timer" in source
    assert "requirements-amd.txt" in script
    assert '("modelscope", "swift", "vllm")' in script
    assert script.index("amd_readiness.py --skip-adapter --skip-endpoint") < script.index(
        "pip install"
    )
    assert script.index("amd_readiness.py --skip-adapter --skip-endpoint") < script.index(
        "modelscope download"
    )


def test_amd_requirements_cover_qwen36_runtime() -> None:
    requirements = (SPACE / "requirements-amd.txt").read_text(encoding="utf-8")
    assert "transformers>=5.2.0" in requirements
    assert "qwen-vl-utils>=0.0.14" in requirements
    assert "decord>=0.6.0" in requirements


def test_released_35b_legacy_payload_is_adapted_without_fallback() -> None:
    query = "外面下暴雨，窝在家里想听氛围感强、安静但不压抑的音乐"
    fallback = runtime.safe_plan(query)
    legacy = {
        "task_mode": "music_search",
        "dialogue_mode": "single_turn",
        "response_mode": "direct",
        "evidence": "用户想在暴雨天居家听氛围感强、安静但不压抑的音乐。",
        "lane_policy": {"graph": "off", "web": "off", "dense": "required"},
        "hard": {"mood": "calm, cozy, atmospheric", "tempo": "slow", "energy": "low"},
        "soft": ["rainy day ambiance", "warm texture", "spacious", "gentle"],
        "hints": ["氛围感强", "安静但不压抑", "居家音乐"],
        "metadata": {"genre": ["ambient", "downtempo", "chill"], "language": "any"},
        "acoustic_queries": [
            "atmospheric ambient music with warm textures",
            "slow tempo downtempo with gentle piano and soft pads",
        ],
        "clarification": [],
    }

    normalized, adapted = runtime.normalize_legacy_candidate(legacy, fallback, query)
    accepted, status = runtime.validate_plan(normalized, fallback)

    assert adapted is True
    assert status == "模型候选通过结构与策略守卫"
    assert accepted is normalized
    assert accepted["lane_policy"] == {"graph": "off", "web": "off", "dense": "required"}
    assert accepted["evidence"]["brief_reason"] == legacy["evidence"]
    assert accepted["hints"]["genre"] == ["ambient", "downtempo", "chill"]
    assert "rainy day ambiance" in accepted["soft"]["vibe"]
    assert accepted["acoustic_queries"] == legacy["acoustic_queries"]


def test_released_35b_mixed_payload_gets_bounded_field_projection() -> None:
    query = "外面下暴雨，窝在家里想听氛围感强、安静但不压抑的音乐"
    fallback = runtime.safe_plan(query)
    mixed = {
        "task_mode": "music_search",
        "dialogue_mode": "single_turn",
        "response_mode": "direct",
        "evidence": {
            "decision_phase": "final",
            "failed_lanes": [],
            "reason_codes": ["mood", "atmosphere"],
            "reference_songs": [],
            "brief_reason": "暴雨天居家氛围，安静但不压抑",
        },
        "lane_policy": {"graph": "off", "web": "off", "dense": "required"},
        "hard": {"mood": "平静", "atmosphere": "氛围感强", "energy": "low"},
        "soft": {"scene": "居家暴雨", "emotion": "平静放松", "texture": "温暖"},
        "hints": ["氛围感", "安静", "不压抑"],
        "metadata": {"genre": [], "instrument": [], "vocal_style": []},
        "acoustic_queries": [],
        "clarification": [],
    }

    normalized, adapted = runtime.normalize_legacy_candidate(mixed, fallback, query)
    accepted, status = runtime.validate_plan(normalized, fallback)

    assert adapted is True
    assert status == "模型候选通过结构与策略守卫"
    assert accepted is normalized
    assert accepted["lane_policy"] == {"graph": "off", "web": "off", "dense": "required"}
    assert accepted["evidence"]["brief_reason"] == "暴雨天居家氛围，安静但不压抑"
    assert accepted["acoustic_queries"] == [query]
    assert {"平静", "氛围感强", "居家暴雨", "安静"} <= set(accepted["soft"]["vibe"])
