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


def test_library_tool_binds_user_server_side_and_only_reads(monkeypatch):
    calls = []

    class _Client:
        @staticmethod
        def execute_query(query, params):
            calls.append((query, params))
            return [
                {
                    "music_id": "s1",
                    "title": "Rain",
                    "artist": "Artist",
                    "album": "Album",
                    "audio_url": "/static/audio/rain.mp3",
                    "cover_url": "",
                    "interaction_at": 123,
                }
            ]

    monkeypatch.setattr(
        "retrieval.neo4j_client.get_neo4j_client",
        lambda: _Client(),
    )
    registry = registry_module.build_music_tool_registry(
        user_id="trusted-user",
        query="查看我的喜欢",
    )
    plan = ToolPlan.model_validate(
        {
            "request_mode": "library",
            "tool_calls": [
                {
                    "id": "library",
                    "name": "read_library",
                    "arguments": {"collection": "liked", "query": "Rain"},
                }
            ],
        }
    )
    result = asyncio.run(BoundedToolOrchestrator(registry).run(plan))

    assert result.by_call_id["library"].data["songs"][0]["music_id"] == "s1"
    assert calls[0][1]["user_id"] == "trusted-user"
    assert "DELETE" not in calls[0][0].upper()
    assert result.by_call_id["library"].metadata["read_only"] is True


def test_stage_ingest_executor_is_a_non_mutating_shadow_preview(monkeypatch):
    monkeypatch.setattr(
        registry_module,
        "execute_search_online_music",
        lambda _query: None,
    )
    registry = registry_module.build_music_tool_registry(
        user_id="trusted-user",
        query="暂存这首歌",
    )
    plan = ToolPlan.model_validate(
        {
            "request_mode": "acquisition",
            "tool_calls": [
                {
                    "id": "candidate",
                    "name": "search_graph",
                    "arguments": {},
                },
                {
                    "id": "stage",
                    "name": "stage_ingest",
                    "arguments": {"mode": "preview", "preserve_audio": False},
                    "depends_on": ["candidate"],
                },
            ],
        }
    )
    registry.register(
        registry_module.ToolName.SEARCH_GRAPH,
        lambda _args, _deps: {
            "songs": [
                {
                    "music_id": "s1",
                    "title": "Song",
                    "artist": "Artist",
                    "audio_url": "/preview/song.mp3",
                },
                {"music_id": "s2", "title": "Broken", "artist": "Artist"},
            ]
        },
    )
    result = asyncio.run(BoundedToolOrchestrator(registry).run(plan))
    preview = result.by_call_id["stage"].data

    assert preview["shadow"] is True
    assert preview["side_effects_applied"] is False
    assert [item["source_id"] for item in preview["would_stage"]] == ["s1"]
    assert preview["rejected"] == [
        {"source_id": "s2", "reason": "missing audio_pointer"}
    ]
    assert result.by_call_id["stage"].metadata["needs_confirmation"] is True
