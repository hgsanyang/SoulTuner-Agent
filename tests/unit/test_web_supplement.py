"""联网补充路线的确定性护栏测试。

LLM 用注入的 fake generator 模拟，可播放解析用注入的 fake resolver 模拟。
确定性代码职责：歌名/歌手归一化模糊去重、候选与命中一致性校验、
有界分数、fail-soft、开关组合。
"""

import asyncio

from retrieval.web_supplement import (
    SUPPLEMENT_BASE_SCORE,
    WebSongCandidate,
    WebSongDiscovery,
    WebSongSupplement,
    is_duplicate_song,
    is_similar_text,
    normalize_song_text,
    supplement_enabled,
)


def _candidate(title: str, artist: str, evidence: str = "来自某音乐榜单") -> WebSongCandidate:
    return WebSongCandidate(title=title, artist=artist, evidence=evidence, source_kind="chart")


async def _hit_resolver(candidate: WebSongCandidate):
    return {
        "title": candidate.title,
        "artist": candidate.artist,
        "play_url": f"http://audio.test/{candidate.title}",
        "cover_url": "",
        "album": "测试专辑",
    }


# ---------- 归一化与相似度 ----------

def test_normalize_strips_parentheticals_noise_and_punctuation():
    assert normalize_song_text("夜曲 (Live) ") == normalize_song_text("夜曲")
    assert normalize_song_text("Hello, World! (feat. A)") == normalize_song_text("hello world")
    assert normalize_song_text("Song - Remastered 2019") == normalize_song_text("song")


def test_is_similar_text_handles_containment_and_fuzzy():
    assert is_similar_text("晴天", "晴天 (周杰伦)")
    assert is_similar_text("Viva La Vida", "viva la vida")
    assert not is_similar_text("晴天", "阴天")


def test_is_duplicate_song_requires_title_and_artist_agreement():
    existing = [("夜曲", "周杰伦"), ("Fix You", "Coldplay")]
    assert is_duplicate_song("夜曲 (Live)", "周杰伦", existing)
    assert is_duplicate_song("fix you", "COLDPLAY", existing)
    assert not is_duplicate_song("夜曲", "王力宏", existing)  # 同名不同歌手不算重复


# ---------- discover 主流程 ----------

def test_discover_resolves_candidates_with_bounded_scores():
    def fake(payload):
        assert payload["user_request"] == "来点最近很火的歌"
        return WebSongDiscovery(songs=[_candidate("歌A", "歌手甲"), _candidate("歌B", "歌手乙")])

    supplement = WebSongSupplement(generator=fake, resolver=_hit_resolver)
    items = asyncio.run(supplement.discover(query="来点最近很火的歌"))

    assert len(items) == 2
    assert items[0]["similarity_score"] == SUPPLEMENT_BASE_SCORE
    assert items[1]["similarity_score"] < items[0]["similarity_score"]
    assert all(item["similarity_score"] < 9.0 for item in items)  # 不再无条件 9.5
    assert items[0]["reason"].startswith("🌐 联网补充")
    assert items[0]["song"]["source"] == "web_supplement"
    assert items[0]["song"]["web_evidence"]


def test_discover_dedupes_candidates_before_resolving():
    def fake(payload):
        return WebSongDiscovery(songs=[
            _candidate("夜曲", "周杰伦"),
            _candidate("夜曲 (Live)", "周杰伦"),
            _candidate("稻香", "周杰伦"),
        ])

    supplement = WebSongSupplement(generator=fake, resolver=_hit_resolver)
    items = asyncio.run(supplement.discover(query="周杰伦"))
    titles = [item["song"]["title"] for item in items]
    assert titles == ["夜曲", "稻香"]


def test_discover_drops_candidates_whose_platform_hit_mismatches():
    async def mismatch_resolver(candidate):
        if candidate.title == "真歌":
            return {"title": "真歌", "artist": candidate.artist, "play_url": "http://a"}
        return None  # 平台首个命中与候选不一致时 resolver 返回 None

    def fake(payload):
        return WebSongDiscovery(songs=[_candidate("真歌", "甲"), _candidate("被顶替的歌", "乙")])

    supplement = WebSongSupplement(generator=fake, resolver=mismatch_resolver)
    items = asyncio.run(supplement.discover(query="q"))
    assert [item["song"]["title"] for item in items] == ["真歌"]


def test_discover_fails_soft_on_model_error_and_timeout():
    def broken(payload):
        raise RuntimeError("web search down")

    assert asyncio.run(WebSongSupplement(generator=broken).discover(query="q")) == []

    async def slow(payload):
        await asyncio.sleep(5)
        return WebSongDiscovery(songs=[])

    supplement = WebSongSupplement(generator=slow, timeout_seconds=0.05)
    assert asyncio.run(supplement.discover(query="q")) == []


def test_discover_passes_avoid_directions_to_model():
    captured: dict = {}

    def fake(payload):
        captured.update(payload)
        return WebSongDiscovery(songs=[])

    supplement = WebSongSupplement(generator=fake, resolver=_hit_resolver)
    asyncio.run(supplement.discover(query="安静的歌", avoid=["太吵", "悲伤"]))
    assert captured["explicitly_avoided"] == ["太吵", "悲伤"]


# ---------- 开关 ----------

def test_supplement_respects_both_switches(monkeypatch):
    monkeypatch.setenv("MUSIC_WEB_SEARCH_ENABLED", "1")
    monkeypatch.setenv("MUSIC_WEB_SUPPLEMENT_ENABLED", "1")
    assert supplement_enabled() is True

    monkeypatch.setenv("MUSIC_WEB_SEARCH_ENABLED", "0")
    assert supplement_enabled() is False  # 用户关联网 → 补充路线必须关

    monkeypatch.setenv("MUSIC_WEB_SEARCH_ENABLED", "1")
    monkeypatch.setenv("MUSIC_WEB_SUPPLEMENT_ENABLED", "0")
    assert supplement_enabled() is False  # 单独关补充路线，回退旧路径
