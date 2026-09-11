"""P0 request isolation without LLMs, databases, or network calls."""

import asyncio
import ast
from dataclasses import FrozenInstanceError
from pathlib import Path
from types import SimpleNamespace

import pytest

from services.recommendation_execution import (
    RecommendationExecution,
    RecommendationModels,
    current_execution,
    execution_scope,
    request_model,
    web_search_allowed,
)


def execution(name, web):
    return RecommendationExecution(web, RecommendationModels(
        main=f"{name}-main", intent=f"{name}-intent",
        conversation=f"{name}-chat", explain=f"{name}-explain",
    ))


def test_nested_scope_resets_even_on_failure(monkeypatch):
    monkeypatch.setenv("MUSIC_WEB_SEARCH_ENABLED", "false")
    outer = execution("outer", True)
    with execution_scope(outer):
        with pytest.raises(RuntimeError):
            with execution_scope(execution("inner", False)):
                assert not web_search_allowed()
                raise RuntimeError("exit")
        assert current_execution() is outer
        assert web_search_allowed()
    assert current_execution() is None
    assert not web_search_allowed()
    assert request_model("main") is None


def test_execution_choices_are_frozen():
    value = execution("a", False)
    with pytest.raises(FrozenInstanceError):
        value.web_search_enabled = True
    with pytest.raises(FrozenInstanceError):
        value.models.intent = "other"


def test_interleaved_requests_and_worker_threads_are_isolated(monkeypatch):
    monkeypatch.setenv("MUSIC_WEB_SEARCH_ENABLED", "1")

    async def run():
        started = [asyncio.Event(), asyncio.Event()]

        async def request(index, name, allowed):
            with execution_scope(execution(name, allowed)):
                started[index].set()
                await started[1 - index].wait()
                await asyncio.sleep(0)
                assert web_search_allowed() is allowed
                assert request_model("intent") == f"{name}-intent"
                assert await asyncio.to_thread(web_search_allowed) is allowed
                assert await asyncio.to_thread(request_model, "conversation") == f"{name}-chat"
            assert current_execution() is None

        await asyncio.gather(request(0, "alice", False), request(1, "bob", True))

    asyncio.run(run())
    assert web_search_allowed() is True


@pytest.mark.parametrize("value", ["0", "false", "no", "off", " FALSE "])
def test_environment_is_only_an_unscoped_default(monkeypatch, value):
    monkeypatch.setenv("MUSIC_WEB_SEARCH_ENABLED", value)
    assert not web_search_allowed()
    with execution_scope(execution("request", True)):
        assert web_search_allowed()


def test_existing_graph_getters_use_captured_roles_not_mutated_defaults():
    # Execute the real getter bodies without importing the heavy graph/tool stack.
    path = Path(__file__).resolve().parents[2] / "agent/music_graph.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names = {"get_llm", "get_intent_llm", "get_conversation_llm", "get_explain_llm"}
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    scope = {"_llm": "default-main", "_intent_llm": "default-intent",
             "_conversation_llm": "default-chat", "_explain_llm": "default-explain"}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), scope)
    with execution_scope(execution("alice", False)):
        scope.update(_llm="bob-main", _intent_llm="bob-intent",
                     _conversation_llm="bob-chat", _explain_llm="bob-explain")
        assert scope["get_llm"]() == "alice-main"
        assert scope["get_intent_llm"]() == "alice-intent"
        assert scope["get_conversation_llm"]() == "alice-chat"
        assert scope["get_explain_llm"]() == "alice-explain"
    assert scope["get_llm"]() == "bob-main"


def test_model_factory_captures_all_roles_without_global_setters(monkeypatch):
    import llms.multi_llm as factories
    from api.recommendation_execution import build_recommendation_execution

    calls = []
    monkeypatch.setattr(factories, "get_chat_model", lambda **kw: calls.append(kw) or "main")
    monkeypatch.setattr(factories, "get_intent_chat_model", lambda: "planner")
    monkeypatch.setattr(factories, "get_conversation_chat_model", lambda: "chat")
    monkeypatch.setattr(factories, "get_explain_chat_model", lambda: "prose")
    value = build_recommendation_execution(web_search_enabled=False, provider="vllm")
    assert value.models == RecommendationModels("main", "planner", "chat", "prose")
    assert calls == [{"provider": "vllm", "model_name": None}]
    assert current_execution() is None


def test_role_configuration_failure_does_not_borrow_another_requests_model(monkeypatch):
    import llms.multi_llm as factories
    from api.recommendation_execution import build_recommendation_execution

    monkeypatch.setattr(factories, "get_chat_model", lambda **kw: "main")
    monkeypatch.setattr(factories, "get_intent_chat_model", lambda: "planner")

    def bad_role():
        raise ValueError("planner-only prose role")

    monkeypatch.setattr(factories, "get_conversation_chat_model", bad_role)
    with pytest.raises(ValueError, match="planner-only"):
        build_recommendation_execution(web_search_enabled=False)
    assert current_execution() is None


def test_web_supplement_honors_request_scope_over_environment(monkeypatch):
    from retrieval.web_supplement import supplement_enabled
    import services.runtime_mode as mode

    monkeypatch.setattr(mode, "side_effects_disabled", lambda: False)
    monkeypatch.setenv("MUSIC_WEB_SEARCH_ENABLED", "1")
    monkeypatch.setenv("MUSIC_WEB_SUPPLEMENT_ENABLED", "1")
    with execution_scope(execution("alice", False)):
        assert not supplement_enabled()
    with execution_scope(execution("bob", True)):
        assert supplement_enabled()


def test_http_recommendation_paths_never_write_global_model_or_web_state():
    path = Path(__file__).resolve().parents[2] / "api/server.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names = {"stream_recommendations", "_stream_recommendations", "get_recommendations"}
    for node in tree.body:
        if isinstance(node, ast.AsyncFunctionDef) and node.name in names:
            source = ast.unparse(node)
            assert 'os.environ[' not in source
            for setter in ("set_llm(", "set_intent_llm(", "set_conversation_llm(", "set_explain_llm("):
                assert setter not in source
