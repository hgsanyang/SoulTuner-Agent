import asyncio

from agent.explanation import build_fast_explanation
from agent.music_graph import MusicRecommendationGraph
from config.settings import settings
from schemas.music_state import ToolOutput


def test_fast_explanation_lists_wrapped_and_direct_songs():
    response = build_fast_explanation([
        {"song": {"title": "晴天", "artist": "翻唱者"}},
        {"title": "City Lights", "artist": "Neon Band"},
    ])
    assert "为你找到了 2 首歌" in response
    assert "《晴天》 - 翻唱者" in response
    assert "《City Lights》 - Neon Band" in response


def test_fast_explanation_handles_empty_results():
    assert "没有找到" in build_fast_explanation([])


def test_graph_fast_mode_skips_explain_llm_and_closes_stream(monkeypatch):
    async def _run():
        graph = object.__new__(MusicRecommendationGraph)
        queue = asyncio.Queue()
        graph._explanation_queues = {"request-1": queue}
        monkeypatch.setattr(settings, "explanation_fast_mode", True)

        def _unexpected_llm_call():
            raise AssertionError("fast mode must not initialize the explanation LLM")

        monkeypatch.setattr("agent.music_graph.get_explain_llm", _unexpected_llm_call)
        recommendations = [{"song": {"title": "晴天", "artist": "翻唱者"}}]
        result = await graph.generate_explanation({
            "input": "想听晴天",
            "recommendations": ToolOutput(success=True, data=recommendations, raw_markdown=""),
            "metadata": {"request_id": "request-1"},
        })

        assert "《晴天》" in result["final_response"]
        assert (await queue.get())["__songs__"][0]["song"]["title"] == "晴天"
        assert "《晴天》" in await queue.get()
        assert await queue.get() is None

    asyncio.run(_run())
