import asyncio
from dataclasses import dataclass

from agent import music_tool_registry as registry_module
from agent.tool_orchestrator import BoundedToolOrchestrator
from schemas.tool_plan import ToolPlan


@dataclass
class _Gap:
    needs_online: bool = False
    target_web_count: int = 0

    def model_dump(self):
        return {"action": "none", "target_web_count": self.target_web_count}


def test_production_registry_executes_local_tools_and_gap_in_dependency_order(monkeypatch):
    monkeypatch.setattr(
        registry_module,
        "graph_candidate_recall",
        lambda hard, hints, limit: '[{"music_id":"g1","title":"Graph"}]',
    )

    class _Semantic:
        @staticmethod
        def invoke(payload):
            return '[{"music_id":"a1","title":"Audio"}]'

    monkeypatch.setattr(registry_module, "semantic_search", _Semantic())
    monkeypatch.setattr(registry_module, "analyze_catalog_gap", lambda *args, **kwargs: _Gap())

    registry = registry_module.build_music_tool_registry(
        user_id="user-a",
        query="quiet Japanese music",
        retrieval_plan={},
    )
    plan = ToolPlan.model_validate(
        {
            "request_mode": "recommendation",
            "tool_calls": [
                {
                    "id": "graph",
                    "name": "search_graph",
                    "arguments": {"language": "Japanese"},
                },
                {
                    "id": "audio",
                    "name": "search_audio",
                    "arguments": {"acoustic_queries": ["quiet sparse music"]},
                },
                {
                    "id": "gap",
                    "name": "inspect_catalog_gap",
                    "arguments": {"requirements": {}},
                    "depends_on": ["graph", "audio"],
                },
            ],
        }
    )
    result = asyncio.run(BoundedToolOrchestrator(registry).run(plan))
    assert [item.status for item in result.observations] == ["success", "success", "success"]
    assert result.by_call_id["graph"].data["songs"][0]["music_id"] == "g1"
    assert result.by_call_id["audio"].data["songs"][0]["music_id"] == "a1"
