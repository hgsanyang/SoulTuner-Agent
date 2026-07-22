import asyncio

from agent.music_graph import (
    MusicRecommendationGraph,
    _schedule_recommended_knowledge_backfill,
    _state_user_id,
)
from config.settings import settings
from retrieval.hybrid_retrieval import (
    USER_EXPOSURE_UPDATE_QUERY,
    MusicHybridRetrieval,
)
from services.graphzep_client import group_id_for_user
from services.memory_gateway import GraphZepAdapter


def test_state_user_id_prefers_top_level_and_supports_legacy_metadata():
    assert _state_user_id({"user_id": "u-top", "metadata": {"user_id": "u-old"}}) == "u-top"
    assert _state_user_id({"metadata": {"user_id": "u-old"}}) == "u-old"


def test_planner_cache_key_is_user_scoped():
    from agent.intent.planner import PlannerResultCache

    base = {
        "user_input": "quiet music",
        "user_preferences": "",
        "chat_history": "",
        "previous_plan": "",
        "graphzep_facts": "",
        "provider": "dashscope",
        "model_name": "teacher",
        "current_date": "2026-07-13",
    }
    assert PlannerResultCache.make_key(**base, user_id="alice") != PlannerResultCache.make_key(
        **base,
        user_id="bob",
    )


def test_exposure_query_is_user_scoped_not_song_global():
    assert "MERGE (u)-[e:EXPOSED]->(s)" in USER_EXPOSURE_UPDATE_QUERY
    assert "e.last_exposed_at" in USER_EXPOSURE_UPDATE_QUERY
    assert "SET s.ts_beta" not in USER_EXPOSURE_UPDATE_QUERY
    assert "$user_id" in USER_EXPOSURE_UPDATE_QUERY


def test_post_recall_metadata_reads_user_exposure(monkeypatch):
    captured = {}

    class FakeNeo4j:
        driver = object()

        def execute_query(self, query, params):
            captured["query"] = query
            captured["params"] = params
            return [
                {
                    "title": "Song A",
                    "updated_at": 0,
                    "ts_alpha": 1,
                    "ts_beta": 2,
                    "ts_last_exposed_at": 3,
                }
            ]

    monkeypatch.setattr("retrieval.neo4j_client.get_neo4j_client", lambda: FakeNeo4j())

    metadata = MusicHybridRetrieval._fetch_post_recall_metadata(
        [{"song": {"title": "Song A", "artist": "Artist A"}}],
        user_id="user-b",
    )

    assert metadata["Song A"]["ts_beta"] == 2
    assert captured["params"]["user_id"] == "user-b"
    assert "[e:EXPOSED]" in captured["query"]
    assert "coalesce(s.ts_beta" not in captured["query"]


def test_dislike_cache_is_isolated_per_user(monkeypatch):
    calls = []

    class FakeNeo4j:
        def execute_query(self, query, params):
            calls.append(params["uid"])
            return [{"titles": [f"blocked-{params['uid']}"]}]

    monkeypatch.setattr("retrieval.neo4j_client.get_neo4j_client", lambda: FakeNeo4j())
    retriever = MusicHybridRetrieval()

    assert retriever._get_disliked_titles("alice") == {"blocked-alice"}
    assert retriever._get_disliked_titles("bob") == {"blocked-bob"}
    assert retriever._get_disliked_titles("alice") == {"blocked-alice"}
    assert calls == ["alice", "bob"]


def test_eval_mode_does_not_schedule_knowledge_backfill(monkeypatch):
    called = []
    monkeypatch.setattr(settings, "eval_disable_side_effects", True)
    monkeypatch.setattr(
        "services.recommendation_knowledge_backfill.schedule_recommendation_knowledge_backfill",
        lambda recommendations: called.append(recommendations) or {"scheduled": 1},
    )

    _schedule_recommended_knowledge_backfill([{"song": {"title": "Song A"}}], context="eval")

    assert called == []


def test_create_playlist_does_not_convert_recommendations_into_likes(monkeypatch):
    class ForbiddenMemoryManager:
        def __init__(self, *args, **kwargs):
            raise AssertionError("playlist creation must not write LIKE relationships")

    monkeypatch.setattr("agent.music_graph.UserMemoryManager", ForbiddenMemoryManager)
    graph = MusicRecommendationGraph.__new__(MusicRecommendationGraph)
    result = asyncio.run(
        graph.create_playlist_node(
            {
                "user_id": "user-a",
                "recommendations": [
                    {"song": {"title": "Song A", "artist": "Artist A", "audio_url": "/a.mp3"}}
                ],
                "intent_type": "create_playlist",
                "intent_parameters": {},
                "step_count": 0,
                "error_log": [],
            }
        )
    )

    assert result["playlist"]["track_count"] == 1


def test_graphzep_adapter_uses_distinct_user_groups(monkeypatch):
    calls = []

    class FakeClient:
        async def add_user_event(self, **kwargs):
            calls.append(("write", kwargs))
            return True

        async def search_facts(self, **kwargs):
            calls.append(("read", kwargs))
            return "fact"

    monkeypatch.setattr("services.graphzep_client.get_graphzep_client", lambda: FakeClient())
    adapter = GraphZepAdapter()

    assert asyncio.run(adapter.remember_text("hello", user_id="alice")) is True
    assert asyncio.run(adapter.retrieve_context("music", user_id="bob")) == "fact"

    assert calls[0][1]["group_id"] == group_id_for_user("alice")
    assert calls[1][1]["group_ids"] == [group_id_for_user("bob")]
    assert calls[0][1]["group_id"] != calls[1][1]["group_ids"][0]
