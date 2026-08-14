from __future__ import annotations

import asyncio

from services.assembled_context_runtime import assemble_prompt_context


def test_assembler_keeps_profile_before_relevant_memory_and_audits_sources():
    bundle = asyncio.run(assemble_prompt_context(
        user_input="今晚开车，别太吵",
        user_id="u-1",
        session_id="session-1",
        chat_history="用户: 先来点摇滚\n助手: 好的",
        explicit_profile="用户明确设置：不喜欢过强鼓点",
        long_memory="[episodic] 上周夜间驾驶时跳过了高动态歌曲",
        memory_context={
            "retrieved_records": [
                {
                    "memory_id": "m-1",
                    "field": "night_drive",
                    "value": "偏好低动态",
                    "layer": "episodic",
                    "relevance": 0.91,
                    "created_at_ms": 123,
                }
            ]
        },
        dialog_state={"soft_intent": {"avoid": ["太吵"]}},
        previous_plan="上轮使用 hybrid_search",
        listening_context={
            "timezone": "Asia/Shanghai",
            "local_hour": 23,
            "day_of_week": 4,
            "day_type": "weekday",
            "day_part": "night",
            "scene": "开车",
            "ts_ms": 456,
        },
        total_budget=600,
    ))

    assert bundle.planner_preferences.index("用户明确画像") < bundle.planner_preferences.index("长期记忆")
    assert bundle.assembled.current.session_id == "session-1"
    assert bundle.assembled.current.explicit_scene == "开车"
    assert bundle.assembled.session.rejected == ["太吵"]
    assert bundle.assembled.retrieved_memories[0].memory_id == "m-1"
    assert bundle.assembled.retrieved_memories[0].statement == "night_drive=偏好低动态"


def test_exact_counter_never_exceeds_tiny_budget_and_preserves_recent_history():
    def exact_counter(text: str) -> int:
        return len(text)

    bundle = asyncio.run(assemble_prompt_context(
        user_input="现在改成安静一点",
        user_id="u-1",
        session_id="session-1",
        chat_history="\n".join([f"旧消息{i}: 很长的内容" for i in range(30)] + ["最新纠正: 不要摇滚"]),
        explicit_profile="明确画像" * 80,
        long_memory="长期记忆" * 80,
        total_budget=180,
        token_counter=exact_counter,
    ))

    combined = bundle.chat_history + bundle.assembled.explicit_profile
    combined += "".join(memory.statement for memory in bundle.assembled.retrieved_memories)
    # The exact budget applies to the four raw context fields before prompt labels.
    raw_memory = bundle.planner_preferences.replace("暂无可用用户记忆", "").replace("【用户明确画像】\n", "").replace(
        "【与当前请求相关的长期记忆】\n", ""
    )
    combined = bundle.chat_history + raw_memory
    assert len(combined) <= 180
    assert "最新纠正" in bundle.chat_history
