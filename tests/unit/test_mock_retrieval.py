import sys
import types

if "langgraph.graph.message" not in sys.modules:
    langgraph = types.ModuleType("langgraph")
    graph = types.ModuleType("langgraph.graph")
    message = types.ModuleType("langgraph.graph.message")
    message.add_messages = lambda left, right: (left or []) + (right or [])
    sys.modules.setdefault("langgraph", langgraph)
    sys.modules.setdefault("langgraph.graph", graph)
    sys.modules.setdefault("langgraph.graph.message", message)

from retrieval.mock_retrieval import mock_retrieve


def test_mock_retrieval_is_deterministic_and_playable():
    result = mock_retrieve("quiet Japanese songs", 2)
    assert result.success is True
    assert len(result.data) == 2
    assert all(item["song"]["audio_url"] for item in result.data)
