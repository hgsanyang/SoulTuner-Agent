from __future__ import annotations

import importlib.util
import json
import urllib.request
from pathlib import Path

from deploy.self_hosted_35b import benchmark_endpoint
from deploy.self_hosted_35b.runtime_readiness import readiness_report


ROOT = Path(__file__).resolve().parents[2]
DEPLOY = ROOT / "deploy" / "self_hosted_35b"


def _minimal_plan() -> dict[str, object]:
    return {
        "task_mode": "recommendation",
        "dialogue_mode": None,
        "response_mode": "answer",
        "evidence": {
            "decision_phase": "initial",
            "failed_lanes": [],
            "reason_codes": ["metaphorical_vibe"],
            "reference_songs": [],
            "brief_reason": "请求描述听感，以向量召回为主",
        },
        "lane_policy": {"graph": "off", "dense": "required", "web": "off"},
        "hard": {
            "artist": [],
            "song": [],
            "language": None,
            "region": None,
            "instrumental": False,
        },
        "soft": {"goal": "轻松音乐", "trajectory": "", "vibe": [], "avoid": []},
        "hints": {"mood": [], "scenario": [], "genre": []},
        "metadata": {
            "era": None,
            "release_year_from": None,
            "release_year_to": None,
            "recency_required": False,
            "external_knowledge_required": False,
        },
        "acoustic_queries": ["relaxed music"],
        "clarification": None,
    }


def test_benchmark_payload_uses_frozen_prompt_and_context() -> None:
    case = benchmark_endpoint.PUBLIC_CASES[3]
    payload = json.loads(
        benchmark_endpoint._request_payload("soultuner-planner-v4.2-35b", case)
    )
    assert payload["messages"][0]["content"] == benchmark_endpoint.STUDENT_SYSTEM_PROMPT_V4_2
    user = payload["messages"][1]["content"]
    assert user.endswith(f"[当前输入] {case.query}")
    assert "[对话历史]\n[上轮推荐结果] 1. 公开参考曲 — 公开演示艺人" in user
    assert "[已解析参考歌曲]" not in user
    assert payload["enable_thinking"] is False
    assert payload["response_format"] == {"type": "json_object"}
    assert payload["chat_template_kwargs"] == {"enable_thinking": False}


def test_benchmark_summary_separates_schema_guard_and_safe_plan() -> None:
    records = [
        {
            "case_id": "one",
            "repeat": 0,
            "ok": True,
            "schema_valid": True,
            "guard_accepted": False,
            "safe_plan_available": True,
            "guard_findings": ["lane mismatch"],
            "latency_ms": 100.0,
            "prompt_tokens": 10,
            "completion_tokens": 5,
        }
    ]
    report = benchmark_endpoint.summarise(records, wall_seconds=0.1)
    assert report["schema_valid_rate"] == 1.0
    assert report["guard_accept_rate"] == 0.0
    assert report["safe_plan_rate"] == 1.0
    assert report["guard_findings"] == {"lane mismatch": 1}
    assert "query" not in report
    assert "raw_response" not in report


def test_readiness_fails_closed_when_assets_are_not_configured(monkeypatch) -> None:
    monkeypatch.delenv("SOULTUNER_BASE_MODEL", raising=False)
    monkeypatch.delenv("SOULTUNER_ADAPTER", raising=False)
    report = readiness_report(probe_endpoint=False)
    assert report["ready"] is False
    assert report["findings"] == [
        "base model path is not configured",
        "adapter path is not configured",
    ]


def test_readiness_accepts_complete_local_assets(monkeypatch, tmp_path: Path) -> None:
    base = tmp_path / "base"
    adapter = tmp_path / "adapter"
    base.mkdir()
    adapter.mkdir()
    (base / "config.json").write_text("{}", encoding="utf-8")
    (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
    (adapter / "adapter_model.safetensors").write_bytes(b"adapter")
    monkeypatch.setenv("SOULTUNER_BASE_MODEL", str(base))
    monkeypatch.setenv("SOULTUNER_ADAPTER", str(adapter))
    report = readiness_report(probe_endpoint=False)
    assert report["ready"] is True
    assert report["findings"] == []


def test_endpoint_probe_accepts_application_api_key(monkeypatch) -> None:
    captured: dict[str, str] = {}

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self) -> bytes:
            return b'{"data": [{"id": "soultuner-v4.2-35b"}]}'

    def fake_urlopen(request: urllib.request.Request, timeout: float):
        captured["authorization"] = request.get_header("Authorization") or ""
        captured["timeout"] = str(timeout)
        return Response()

    monkeypatch.setenv("SOULTUNER_PLANNER_API_KEY", "application-secret")
    monkeypatch.delenv("SOULTUNER_SERVE_API_KEY", raising=False)
    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    report = importlib.import_module(
        "deploy.self_hosted_35b.runtime_readiness"
    ).probe_models("http://127.0.0.1:8000/v1")
    assert report["ok"] is True
    assert captured["authorization"] == "Bearer application-secret"


def test_server_script_is_private_by_default_and_requires_public_auth() -> None:
    script = (DEPLOY / "start_35b_endpoint.sh").read_text(encoding="utf-8")
    assert 'SOULTUNER_SERVE_HOST:-127.0.0.1' in script
    assert '"$host" == "0.0.0.0"' in script
    assert "SOULTUNER_SERVE_API_KEY" in script
