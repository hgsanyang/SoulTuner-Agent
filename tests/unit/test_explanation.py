import asyncio

from agent.explanation import build_fast_explanation, emit_fast_explanation


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


def test_fast_mode_emits_song_cards_response_and_stream_end():
    async def _run():
        queue = asyncio.Queue()
        recommendations = [{"song": {"title": "晴天", "artist": "翻唱者"}}]
        response = await emit_fast_explanation(recommendations, queue)

        assert "《晴天》" in response
        assert (await queue.get())["__songs__"][0]["song"]["title"] == "晴天"
        assert "《晴天》" in await queue.get()
        assert await queue.get() is None

    asyncio.run(_run())
