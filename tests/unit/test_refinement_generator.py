"""LLM-first refinement chip generator 的确定性护栏测试。

LLM 输出用注入的 fake generator 模拟；这里只验证确定性代码的职责：
schema/长度校验、数量上限、去重、显式否定冲突过滤、整组重复打破、
fail-soft 中性 fallback，以及上下文输入的组装。
"""

import asyncio

import pytest

from schemas.refinement import RefinementOption
from services.refinement_generator import (
    MAX_OPTIONS,
    NEUTRAL_FALLBACK_OPTIONS,
    RefinementChipGenerator,
    RefinementProposal,
    extract_avoid_texts,
    remember_emitted_options,
    summarize_slate,
    validate_options,
)


def _opt(label: str, prompt: str | None = None, reason: str = "r", source: str = "current_slate") -> RefinementOption:
    return RefinementOption(label=label, prompt=prompt or f"{label}一点", reason=reason, source=source)


# ---------- validate_options：纯确定性护栏 ----------

def test_validate_caps_at_six_and_dedupes():
    raw = [_opt(f"方向{i}") for i in range(10)] + [_opt("方向0")]
    valid = validate_options(raw)
    assert len(valid) == MAX_OPTIONS
    assert len({o.label for o in valid}) == MAX_OPTIONS


def test_validate_drops_empty_and_overlong():
    raw = [
        _opt("正常方向"),
        RefinementOption(label="", prompt="没有标签"),
        RefinementOption(label="标签", prompt=""),
        _opt("超长标签" * 10),
        RefinementOption(label="超长提示", prompt="很长" * 100),
    ]
    valid = validate_options(raw)
    assert [o.label for o in valid] == ["正常方向"]


def test_validate_filters_explicitly_avoided_directions():
    # 用户明确说过"不要悲伤"，不得再出现"更悲伤"方向
    raw = [_opt("更悲伤", "更悲伤一点，往下沉"), _opt("节奏慢一点")]
    valid = validate_options(raw, avoid_texts=["悲伤"])
    assert [o.label for o in valid] == ["节奏慢一点"]


def test_validate_breaks_verbatim_full_set_repetition():
    options = [_opt("方向A"), _opt("方向B"), _opt("方向C")]
    previous = [{"label": o.label, "prompt": o.prompt} for o in options]
    valid = validate_options(list(options), previous=previous)
    # 与上一轮完全相同的整组必须被打破（数量减一即集合不同）
    assert len(valid) < len(options)


def test_validate_allows_partial_overlap_with_previous_turn():
    previous = [{"label": "方向A", "prompt": "方向A一点"}]
    options = [_opt("方向A"), _opt("方向B")]
    valid = validate_options(options, previous=previous)
    assert [o.label for o in valid] == ["方向A", "方向B"]


# ---------- generator：注入 fake，LLM 失败 fail-soft ----------

def test_generator_uses_injected_typed_output_and_validates():
    def fake(payload):
        assert payload["current_user_turn"] == "雨夜想听歌"
        assert payload["current_slate_summary"], "slate 摘要必须传给模型"
        return RefinementProposal(options=[_opt(f"方向{i}") for i in range(8)])

    generator = RefinementChipGenerator(generator=fake)
    options = asyncio.run(generator.generate(
        user_id="u-typed",
        user_input="雨夜想听歌",
        recommendations=[{"song": {"title": "夜曲", "artist": "周杰伦", "genre": "流行"}}],
    ))
    assert len(options) == MAX_OPTIONS


def test_generator_fails_soft_to_neutral_fallback():
    def broken(payload):
        raise RuntimeError("model down")

    generator = RefinementChipGenerator(generator=broken)
    options = asyncio.run(generator.generate(user_id="u-fail", user_input="随便"))
    assert [o.label for o in options] == [o.label for o in NEUTRAL_FALLBACK_OPTIONS]
    assert len(options) <= 2


def test_generator_passes_previous_options_and_avoid_to_model():
    remember_emitted_options("u-prev", [_opt("上轮方向")])
    captured: dict = {}

    def fake(payload):
        captured.update(payload)
        return RefinementProposal(options=[_opt("新方向甲"), _opt("新方向乙")])

    generator = RefinementChipGenerator(generator=fake)
    options = asyncio.run(generator.generate(
        user_id="u-prev",
        user_input="继续",
        plan={"soft_intent": {"avoid": ["太吵"]}},
    ))
    assert captured["previous_refinement_options"] == [{"label": "上轮方向", "prompt": "上轮方向一点"}]
    assert captured["explicitly_avoided"] == ["太吵"]
    assert [o.label for o in options] == ["新方向甲", "新方向乙"]


def test_generator_respects_timeout():
    async def slow(payload):
        await asyncio.sleep(5)
        return RefinementProposal(options=[_opt("迟到方向")])

    generator = RefinementChipGenerator(generator=slow, timeout_seconds=0.05)
    options = asyncio.run(generator.generate(user_id="u-slow", user_input="快点"))
    assert [o.label for o in options] == [o.label for o in NEUTRAL_FALLBACK_OPTIONS]


# ---------- 输入组装 helpers ----------

def test_summarize_slate_is_compact_and_factual():
    recs = [
        {"song": {"title": "T1", "artist": "A1", "genre": "Rock", "moods": ["热血", "兴奋", "紧张", "多余"]}},
        {"not_a_song": True},
        {"song": {"title": "T2", "artist": "A2"}},
    ]
    summary = summarize_slate(recs)
    assert summary[0]["title"] == "T1"
    assert summary[0]["moods"] == ["热血", "兴奋", "紧张"]
    assert {"title": "T2", "artist": "A2"} == summary[1]


def test_extract_avoid_texts_merges_plan_and_dialog_state():
    plan = {"soft_intent": {"avoid": ["悲伤", "太吵"]}}
    dialog_state = {"soft_intent": {"avoid": ["太吵", "苦情"]}}
    assert extract_avoid_texts(plan, dialog_state) == ["悲伤", "太吵", "苦情"]


if __name__ == "__main__":
    pytest.main([__file__, "-q"])
