"""Exercise real API/Agent stream bodies with injected, network-free graph work.

AST loading avoids importing API startup hooks, secrets, and heavyweight audio
dependencies. The function bodies executed here are the production source.
"""

import ast
import asyncio
from contextlib import suppress
import json
import logging
from pathlib import Path
from types import MethodType, SimpleNamespace

import pytest

from agent.session_identity import checkpoint_thread_id, resolve_conversation_id
from schemas.runtime_context import build_runtime_context
from services.recommendation_execution import (
    RecommendationExecution, RecommendationModels, current_execution, web_search_allowed,
)
from services.runtime_context import current_runtime_context


ROOT = Path(__file__).resolve().parents[2]


def test_playlist_disconnect_cleans_context_and_pending_work(monkeypatch):
    import api.recommendation_execution as factory

    models = RecommendationModels(*([object()] * 4))
    monkeypatch.setattr(factory, "build_recommendation_execution", lambda **kw:
                        RecommendationExecution(models=models, **kw))

    async def run():
        entered, stopped, disconnected = asyncio.Event(), asyncio.Event(), asyncio.Event()
        previous_context = current_runtime_context()
        context = build_runtime_context(profile_id="playlist-user", session_id="playlist-session")

        async def recommend(**kwargs):
            assert current_execution().models is models
            assert current_runtime_context() == context
            assert kwargs["user_id"] == context.effective_user_id
            entered.set()
            try:
                await asyncio.Event().wait()
            finally:
                stopped.set()
            yield {"type": "complete", "success": True}

        async def probe():
            return disconnected.is_set()

        scope = {"asyncio": asyncio, "json": json, "logger": logging.getLogger("test"),
                 "get_agent": lambda: SimpleNamespace(stream_recommendations=recommend)}
        load_bodies("api/server.py", {"stream_playlist", "_stream_playlist"}, scope)

        async def consume():
            return [event async for event in scope["stream_playlist"](
                "rain", user_id="untrusted", runtime_context=context, is_disconnected=probe,
            )]

        task = asyncio.create_task(consume())
        await entered.wait()
        disconnected.set()
        events = await task
        assert stopped.is_set()
        assert not any('"type": "complete"' in event for event in events)
        assert current_execution() is None
        assert current_runtime_context() == previous_context

    asyncio.run(asyncio.wait_for(run(), 3))


def load_bodies(relative_path, names, scope, owner=None):
    tree = ast.parse((ROOT / relative_path).read_text(encoding="utf-8"))
    body = tree.body
    if owner:
        body = next(n for n in body if isinstance(n, ast.ClassDef) and n.name == owner).body
    functions = [n for n in body if isinstance(n, ast.AsyncFunctionDef) and n.name in names]
    assert len(functions) == len(names)
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    module = ast.fix_missing_locations(ast.Module(body=[future, *functions], type_ignores=[]))
    exec(compile(module, str(ROOT / relative_path), "exec"), scope)


def test_legacy_playlist_releases_songs_before_prose_and_preserves_metadata():
    async def run():
        prose_ready = asyncio.Event()
        closed = asyncio.Event()
        async def recommend(**kwargs):
            try:
                yield {"type": "recommendations_start", "count": 1, "exposure_id": "e"}
                yield {"type": "song", "song": {"music_id": "a", "title": "A"}, "index": 0, "total": 1, "exposure_id": "e"}
                yield {"type": "recommendations_complete"}
                await prose_ready.wait()
                yield {"type": "response", "content": "For you"}
                yield {"type": "complete", "success": True, "dialog_state": {"scene": "rain"}}
            finally:
                closed.set()
        scope = {"asyncio": asyncio, "json": json, "logger": logging.getLogger("test"),
                 "get_agent": lambda: SimpleNamespace(stream_recommendations=recommend)}
        load_bodies("api/server.py", {"_stream_playlist"}, scope)
        stream = scope["_stream_playlist"]("rain")
        received = []
        while True:
            data = json.loads((await anext(stream)).removeprefix("data: "))
            received.append(data)
            if data["type"] == "song":
                break
        assert not prose_ready.is_set()
        assert received[-1]["exposure_id"] == "e"
        assert received[-2]["type"] == "songs_start"
        prose_ready.set()
        tail = [json.loads(row.removeprefix("data: ")) async for row in stream]
        assert [row["type"] for row in tail] == ["songs_complete", "response", "complete"]
        assert tail[-1]["dialog_state"] == {"scene": "rain"}
        assert closed.is_set()
    asyncio.run(asyncio.wait_for(run(), 3))


@pytest.mark.parametrize("producer_error", [True, False])
def test_legacy_playlist_never_invents_success(producer_error):
    async def recommend(**kwargs):
        if producer_error:
            yield {"type": "error", "error": "private provider detail"}
        else:
            yield {"type": "thinking", "message": "working"}
    scope = {"asyncio": asyncio, "json": json, "logger": logging.getLogger("test"),
             "get_agent": lambda: SimpleNamespace(stream_recommendations=recommend)}
    load_bodies("api/server.py", {"_stream_playlist"}, scope)
    async def run():
        return [json.loads(row.removeprefix("data: ")) async for row in scope["_stream_playlist"]("rain")]
    result = asyncio.run(run())
    assert result[-1]["type"] == "error"
    assert not any(row["type"] == "complete" for row in result)
    assert "private provider detail" not in str(result)


def make_agent(app):
    scope = {
        "asyncio": asyncio, "suppress": suppress,
        "logger": logging.getLogger("test-stream"),
        "settings": SimpleNamespace(eval_disable_side_effects=True),
        "resolve_conversation_id": resolve_conversation_id,
        "checkpoint_thread_id": checkpoint_thread_id,
        "_listening_context": lambda _: {},
        "safe_query": lambda query: "[test]",
    }
    load_bodies("agent/music_agent.py", {"stream_recommendations"}, scope, "MusicRecommendationAgent")
    agent = SimpleNamespace(graph=SimpleNamespace(_explanation_queues={}, checkpointer=None), app=app)
    agent.stream_recommendations = MethodType(scope["stream_recommendations"], agent)
    return agent


def test_shared_agent_early_close_cancels_own_graph_and_keeps_other_request():
    async def run():
        entered = {name: asyncio.Event() for name in ("a", "b")}
        release = {name: asyncio.Event() for name in ("a", "b")}
        stopped = set()

        async def invoke(state, config):
            name = state["input"]
            entered[name].set()
            try:
                await release[name].wait()
                return {"recommendations": []}
            finally:
                stopped.add(name)

        agent = make_agent(SimpleNamespace(ainvoke=invoke))
        first = agent.stream_recommendations("a")
        second = agent.stream_recommendations("b")
        assert (await anext(first))["type"] == "thinking"
        assert (await anext(second))["type"] == "thinking"
        await asyncio.gather(*(event.wait() for event in entered.values()))
        await first.aclose()
        assert stopped == {"a"}
        assert len(agent.graph._explanation_queues) == 1
        release["b"].set()
        events = [event async for event in second]
        assert events[-1]["type"] == "complete"
        assert stopped == {"a", "b"}
        assert agent.graph._explanation_queues == {}
        assert not hasattr(agent, "_current_graph_task")

    asyncio.run(asyncio.wait_for(run(), 2))


def test_graph_exception_and_no_prose_completion_both_terminate():
    async def run():
        for fail in (False, True):
            async def invoke(state, config):
                if fail:
                    raise RuntimeError("graph failure")
                return {}

            agent = make_agent(SimpleNamespace(ainvoke=invoke))
            events = [event async for event in agent.stream_recommendations("query")]
            assert events[-1]["type"] == ("error" if fail else "complete")
            assert not agent.graph._explanation_queues

    asyncio.run(asyncio.wait_for(run(), 2))


def test_agent_cancel_propagates_instead_of_yielding_an_error():
    async def run():
        entered, stopped = asyncio.Event(), asyncio.Event()

        async def invoke(state, config):
            try:
                entered.set()
                await asyncio.Event().wait()
            finally:
                stopped.set()

        agent = make_agent(SimpleNamespace(ainvoke=invoke))

        async def consume():
            return [event async for event in agent.stream_recommendations("query")]

        task = asyncio.create_task(consume())
        await entered.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert stopped.is_set()
        assert not agent.graph._explanation_queues

    asyncio.run(asyncio.wait_for(run(), 2))


def test_api_disconnect_closes_real_agent_without_harming_other_turn(monkeypatch):
    import api.recommendation_execution as factory

    def build(*, web_search_enabled):
        name = "bob" if web_search_enabled else "alice"
        return RecommendationExecution(web_search_enabled, RecommendationModels(name, name, name, name))

    monkeypatch.setattr(factory, "build_recommendation_execution", build)

    async def run():
        entered = {name: asyncio.Event() for name in ("alice", "bob")}
        release_bob, disconnect = asyncio.Event(), asyncio.Event()
        stopped, snapshots = set(), {}

        async def invoke(state, config):
            name = state["user_id"]
            snapshots[name] = (
                web_search_allowed(), current_execution().models.intent,
                current_runtime_context().effective_user_id,
            )
            entered[name].set()
            try:
                if name == "bob":
                    await release_bob.wait()
                else:
                    await asyncio.Event().wait()
                return {}
            finally:
                stopped.add(name)

        agent = make_agent(SimpleNamespace(ainvoke=invoke))
        scope = {"asyncio": asyncio, "json": json, "get_agent": lambda: agent,
                 "logger": logging.getLogger("test-api"), "safe_query": lambda q: "[test]"}
        load_bodies("api/server.py", {"stream_recommendations", "_stream_recommendations"}, scope)

        async def probe():
            return disconnect.is_set()

        async def consume(name, web, callback):
            return [event async for event in scope["stream_recommendations"](
                query="same query", web_search_enabled=web, is_disconnected=callback,
                runtime_context=build_runtime_context(profile_id=name, session_id="same-session"),
            )]

        alice = asyncio.create_task(consume("alice", False, probe))
        bob = asyncio.create_task(consume("bob", True, None))
        await asyncio.gather(*(event.wait() for event in entered.values()))
        disconnect.set()
        await alice
        assert stopped == {"alice"}
        assert not bob.done()
        assert snapshots == {"alice": (False, "alice", "alice"), "bob": (True, "bob", "bob")}
        release_bob.set()
        events = await bob
        assert any('"type": "complete"' in event for event in events)
        assert not agent.graph._explanation_queues
        assert current_execution() is None
        assert current_runtime_context().interaction_mode == "legacy"

    asyncio.run(asyncio.wait_for(run(), 3))
