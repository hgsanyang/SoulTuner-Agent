"""Execute production node bodies without model/database startup hooks."""
import ast
import asyncio
import logging
import os
from pathlib import Path
from types import SimpleNamespace

import pytest


def node(name, **injected):
    path = Path(__file__).resolve().parents[2] / "agent/music_graph.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    owner = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "MusicRecommendationGraph")
    method = next(n for n in owner.body if isinstance(n, ast.AsyncFunctionDef) and n.name == name)
    module = ast.fix_missing_locations(ast.Module(body=[
        ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), method,
    ], type_ignores=[]))
    scope = {"asyncio": asyncio, "os": os, "logger": logging.getLogger(__name__),
             "settings": SimpleNamespace(graphzep_total_timeout_seconds=.5, user_preference_limit=10),
             "_state_user_id": lambda s: "trusted", "_record_timing": lambda *a: {},
             "safe_labels": lambda v: str(v), **injected}
    exec(compile(module, str(path), "exec"), scope)
    return scope[name]


@pytest.mark.parametrize("failure", ["error", "timeout", "degraded"])
def test_recall_failure_overwrites_stale_context(monkeypatch, failure):
    monkeypatch.delenv("MUSIC_MOCK_MODE", raising=False)
    async def recall(**kwargs):
        if failure == "error":
            raise RuntimeError("private-host")
        if failure == "timeout":
            await asyncio.sleep(2)
        return {"memory_trace": {"status": "degraded"}}
    monkeypatch.setattr("services.memory_gateway.get_memory_gateway",
                        lambda: SimpleNamespace(retrieve_context=recall))
    result = asyncio.run(node("recall_graphzep_memory")(None, {"input": "rain", "memory_context": {"old": True}}))
    assert result["memory_context"]["memory_trace"]["status"] == "degraded"
    assert "old" not in result["memory_context"]
    assert "请勿据此推断用户没有偏好" in result["graphzep_facts"]
    assert "private-host" not in str(result)


def test_empty_user_does_not_acquire_demonstration_favorites():
    manager = SimpleNamespace(get_user_preferences=lambda *a, **k: {})
    result = asyncio.run(node("analyze_user_preferences_node", UserMemoryManager=lambda: manager)(None, {}))
    assert result["favorite_songs"] == []
    assert result["user_preferences"]["favorite_artists"] == []
    assert result["user_preferences"]["favorite_genres"] == []
    assert result["user_preferences"]["favorite_decades"] == []


def test_preference_node_keeps_actual_avoid_preferences():
    manager = SimpleNamespace(get_user_preferences=lambda *a, **k: {"avoid_genres": ["metal"], "add_moods": ["warm"]})
    result = asyncio.run(node("analyze_user_preferences_node", UserMemoryManager=lambda: manager)(None, {}))
    assert result["user_preferences"]["avoid_genres"] == ["metal"]
    assert result["user_preferences"]["mood_preferences"] == ["warm"]
