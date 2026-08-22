from __future__ import annotations

import importlib.util
import json
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

FULL_BOOTSTRAP_SPEC = importlib.util.spec_from_file_location(
    "full_space_bootstrap_open_audio",
    SPACE.parent / "modelscope_full" / "bootstrap_open_audio.py",
)
assert FULL_BOOTSTRAP_SPEC and FULL_BOOTSTRAP_SPEC.loader
full_bootstrap = importlib.util.module_from_spec(FULL_BOOTSTRAP_SPEC)
FULL_BOOTSTRAP_SPEC.loader.exec_module(full_bootstrap)


def test_space_card_text_has_theme_independent_contrast() -> None:
    source = (SPACE / "app.py").read_text(encoding="utf-8")
    renderer = (SPACE / "ui_render.py").read_text(encoding="utf-8")
    assert ".st-track-heading h3" in source
    assert "color: #f3f9f6" in source
    assert "background: rgba(11,20,29,.88)" in source
    assert ".st-card.is-current" in source
    assert 'class="st-cover"' in renderer
    assert 'class="st-play-state' in renderer


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


def test_dialogue_plan_compiles_to_no_retrieval_route() -> None:
    plan = runtime.safe_plan("你好")
    route = runtime.compile_route(plan)

    assert route["profile"] == "no_retrieval"
    assert route["graph_weight"] == 0.0
    assert route["dense_weight"] == 0.0


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
    assert "SOULTUNER_PLANNER_HOST:-127.0.0.1" in script
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


def test_local_endpoint_requires_planner_and_chat_model_ids(monkeypatch) -> None:
    models = ["soultuner-v4.2-35b"]

    class Response:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        @staticmethod
        def read() -> bytes:
            return json.dumps({"data": [{"id": model} for model in models]}).encode()

    monkeypatch.setenv("SOULTUNER_DUAL_ROLE_MODELS", "1")
    monkeypatch.setenv("SOULTUNER_PLANNER_MODEL", "soultuner-v4.2-35b")
    monkeypatch.setenv("SOULTUNER_CHAT_MODEL", "qwen3.6-35b-a3b")
    monkeypatch.setattr(bootstrap, "urlopen", lambda *_args, **_kwargs: Response())

    assert bootstrap._endpoint_ready("http://127.0.0.1:8000/v1") is False
    models.append("qwen3.6-35b-a3b")
    assert bootstrap._endpoint_ready("http://127.0.0.1:8000/v1") is True


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
    assert script.index("amd_readiness.py --skip-adapter --skip-endpoint") < script.index("pip install")
    assert script.index("amd_readiness.py --skip-adapter --skip-endpoint") < script.index("modelscope download")


def test_amd_requirements_cover_qwen36_runtime() -> None:
    requirements = (SPACE / "requirements-amd.txt").read_text(encoding="utf-8")
    assert "transformers>=5.2.0" in requirements
    assert "qwen-vl-utils>=0.0.14" in requirements
    assert "decord>=0.6.0" in requirements


def test_amd_launcher_has_dual_role_models_and_single_lora_compatibility() -> None:
    script = (SPACE / "start_amd_35b.sh").read_text(encoding="utf-8")
    bootstrap_source = (SPACE / "space_bootstrap.py").read_text(encoding="utf-8")
    assert 'default_cache_dir="/mnt/workspace/soultuner/model_cache"' in script
    assert "SOULTUNER_DUAL_ROLE_MODELS:-0" in script
    assert '--adapters "${SERVED_MODEL_NAME}=${ADAPTER_MODEL_DIR}"' in script
    assert '--served_model_name "${CHAT_MODEL_NAME}"' in script
    assert '--adapters "${ADAPTER_MODEL_DIR}"' in script
    assert '--served_model_name "${SERVED_MODEL_NAME}"' in script
    assert 'setdefault("SOULTUNER_DUAL_ROLE_MODELS", "1")' in bootstrap_source
    assert 'setdefault("SOULTUNER_CHAT_MODEL", DEFAULT_CHAT_MODEL)' in bootstrap_source
    assert "required <= models" in bootstrap_source


def test_full_space_requires_both_vllm_roles() -> None:
    full_script = (SPACE.parent / "modelscope_full" / "start_full_space.sh").read_text(encoding="utf-8")
    assert "SOULTUNER_DUAL_ROLE_MODELS:-1" in full_script
    assert 'required.add(os.environ["SOULTUNER_CHAT_MODEL"])' in full_script
    assert "if required <= models" in full_script
    assert 'CONVERSATION_LLM_MODEL="${CONVERSATION_LLM_MODEL:-${SOULTUNER_CHAT_MODEL}}"' in full_script
    assert 'INTENT_LLM_MODEL="${INTENT_LLM_MODEL:-${SOULTUNER_PLANNER_MODEL}}"' in full_script
    assert "hgsanyang/SoulTuner-Open-Audio-Demo" in full_script
    assert "deploy.modelscope_full.bootstrap_open_audio" in full_script


def test_full_space_open_audio_requires_all_three_embedding_families() -> None:
    ready = {
        "enrichment_status": "ready",
        "muq_dim": 512,
        "m2d_dim": 768,
        "omar_dim": 1024,
    }
    assert full_bootstrap._is_ready(ready) is True
    assert full_bootstrap._is_ready({**ready, "muq_dim": 0}) is False
    assert full_bootstrap._is_ready({**ready, "enrichment_status": "queued"}) is False


def test_neo4j_schema_matches_current_omar_rq_vector_width() -> None:
    source = (SPACE.parents[1] / "data" / "pipeline" / "neo4j_schema_v2.py").read_text(encoding="utf-8")
    assert "OMAR_EMBEDDING_DIM = 1024" in source


def test_amd_space_materialises_and_verifies_public_open_audio_in_parallel() -> None:
    script = (SPACE / "start_space_amd.sh").read_text(encoding="utf-8")
    assert "hgsanyang/SoulTuner-Open-Audio-Demo" in script
    assert "--repo-type dataset" in script
    assert 'prepare_open_audio >"${open_audio_log}" 2>&1 &' in script
    assert "audio_sha256" in script
    assert "candidate.relative_to(root)" in script
    assert 'SOULTUNER_DUAL_ROLE_MODELS="${SOULTUNER_DUAL_ROLE_MODELS:-1}"' in script
    assert 'SOULTUNER_CHAT_MODEL="${SOULTUNER_CHAT_MODEL:-qwen3.6-35b-a3b}"' in script
    assert 'required.add(os.environ["SOULTUNER_CHAT_MODEL"])' in script
    assert "required <= models" in script
    assert "SOULTUNER_OPEN_AUDIO_ALREADY_VERIFIED=1" in script


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


def test_remote_planner_uses_frozen_v42_prompt_and_context_format(monkeypatch) -> None:
    captured = {}
    candidate = runtime.safe_plan("公路旅行摇滚")

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        @staticmethod
        def read() -> bytes:
            return json.dumps(
                {"choices": [{"message": {"content": json.dumps(candidate, ensure_ascii=False)}}]},
                ensure_ascii=False,
            ).encode("utf-8")

    def fake_urlopen(request, timeout):
        captured["payload"] = json.loads(request.data.decode("utf-8"))
        captured["timeout"] = timeout
        return Response()

    monkeypatch.setattr(
        runtime,
        "_endpoint",
        lambda _profile: ("http://127.0.0.1:8000/v1/chat/completions", "soultuner-v4.2-35b", ""),
    )
    monkeypatch.setattr(runtime.urllib.request, "urlopen", fake_urlopen)

    result = runtime._remote_plan(
        runtime.PROFILE_SOULTUNER,
        "公路旅行摇滚",
        {
            "profile_snapshot": "偏好标签：rock",
            "retrieved_memories": ["喜欢《Open Road》"],
            "chat_history": "用户：再摇滚一点",
            "previous_plan": "{}",
            "reference_title": "Open Road",
            "reference_artist": "Open Artist",
        },
    )

    payload = captured["payload"]
    assert result["task_mode"] == "recommendation"
    assert payload["messages"][0]["content"] == runtime.SYSTEM_PROMPT
    user_content = payload["messages"][1]["content"]
    assert "[用户画像] 偏好标签：rock" in user_content
    assert "[长期记忆] 喜欢《Open Road》" in user_content
    assert "[上轮推荐结果] 1. Open Road — Open Artist" in user_content
    assert user_content.endswith("[当前输入] 公路旅行摇滚")
    assert payload["response_format"] == {"type": "json_object"}
    assert payload["enable_thinking"] is False
