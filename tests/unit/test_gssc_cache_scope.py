"""GSSC 摘要缓存的会话隔离、增量拼接与失效策略。"""

import asyncio

import retrieval.gssc_context_builder as gssc


def _long_history(prefix: str = "history") -> str:
    return "\n".join(
        f"{prefix}-{index}: 用户说明本轮音乐偏好和不喜欢的特征。" * 4
        for index in range(140)
    )


def test_cache_is_scoped_by_user_and_conversation(monkeypatch):
    gssc.clear_compression_cache()

    async def fake_compress(_: str) -> str:
        return "已验证摘要"

    monkeypatch.setattr(gssc, "_llm_compress_chat_history", fake_compress)
    history = _long_history()
    asyncio.run(gssc.pre_compress_and_cache("user-a", history, "conversation-a"))

    assert gssc.get_cached_compression("user-a", "conversation-a", history)
    assert not gssc.get_cached_compression("user-a", "conversation-b", history)
    assert not gssc.get_cached_compression("user-b", "conversation-a", history)


def test_cache_hit_returns_only_appended_turns(monkeypatch):
    gssc.clear_compression_cache()

    async def fake_compress(_: str) -> str:
        return "历史摘要"

    monkeypatch.setattr(gssc, "_llm_compress_chat_history", fake_compress)
    history = _long_history()
    asyncio.run(gssc.pre_compress_and_cache("user-a", history, "conversation-a"))

    hit = gssc.get_cached_compression(
        "user-a",
        "conversation-a",
        history + "\nuser: 这一轮请不要再推荐重金属",
    )

    assert hit is not None
    assert hit.entry.summary == "历史摘要"
    assert hit.uncovered_text == "user: 这一轮请不要再推荐重金属"


def test_rewritten_history_invalidates_cache(monkeypatch):
    gssc.clear_compression_cache()

    async def fake_compress(_: str) -> str:
        return "历史摘要"

    monkeypatch.setattr(gssc, "_llm_compress_chat_history", fake_compress)
    history = _long_history()
    asyncio.run(gssc.pre_compress_and_cache("user-a", history, "conversation-a"))

    rewritten = history.replace("history-0", "rewritten-0", 1)
    assert not gssc.get_cached_compression(
        "user-a", "conversation-a", rewritten
    )


def test_expired_cache_is_rejected(monkeypatch):
    gssc.clear_compression_cache()

    async def fake_compress(_: str) -> str:
        return "历史摘要"

    monkeypatch.setattr(gssc, "_llm_compress_chat_history", fake_compress)
    history = _long_history()
    asyncio.run(gssc.pre_compress_and_cache("user-a", history, "conversation-a"))
    entry = next(iter(gssc._compress_cache.values()))

    assert not gssc.get_cached_compression(
        "user-a",
        "conversation-a",
        history,
        now=entry.generated_at + gssc.COMPRESSION_CACHE_TTL_SECONDS + 1,
    )
