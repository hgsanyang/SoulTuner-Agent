"""
测试 GSSC Token 预算分配逻辑

验证：
1. 总量在预算内时不截断
2. 超出预算时按优先级分配
3. 各源保证 min_tokens 保底
4. estimate_tokens 对中英混合文本的估算合理性
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from retrieval.gssc_context_builder import (
    ContextSource,
    PRIORITY_CHAT_HISTORY,
    PRIORITY_GRAPHZEP_FACTS,
    PRIORITY_RETRIEVAL,
    estimate_tokens,
)


class TestEstimateTokens:
    """测试 Token 估算函数"""

    def test_empty_string(self):
        """空字符串应返回 0"""
        assert estimate_tokens("") == 0

    def test_pure_chinese(self):
        """纯中文：每字约 1.5 token"""
        text = "你好世界"  # 4 个中文字
        tokens = estimate_tokens(text)
        assert 4 <= tokens <= 8  # 4*1.5=6, 允许合理范围

    def test_pure_english(self):
        """纯英文：每字符约 0.4 token"""
        text = "hello world test"  # 16 字符
        tokens = estimate_tokens(text)
        assert 4 <= tokens <= 10  # 16*0.4=6.4

    def test_mixed_chinese_english(self):
        """中英混合文本"""
        text = "周杰伦 Jay Chou 的新专辑"  # 6 中文 + 若干英文
        tokens = estimate_tokens(text)
        assert tokens > 0
        # 中文 6*1.5=9, 英文 ~12*0.4=4.8, 总 ~14
        assert 10 <= tokens <= 20

    def test_none_input(self):
        """None 输入应返回 0"""
        assert estimate_tokens(None) == 0


class TestContextSource:
    """测试 ContextSource 截断逻辑"""

    def test_no_truncation_needed(self):
        """内容短于预算时不截断"""
        src = ContextSource("test", "短文本", PRIORITY_CHAT_HISTORY, min_tokens=0)
        result = src.truncate_to(1000)
        assert result == "短文本"

    def test_truncation_appends_marker(self):
        """截断时应添加"已截断"标记"""
        long_text = "\n".join([f"第{i}行内容很长很长很长" for i in range(50)])
        src = ContextSource("test", long_text, PRIORITY_CHAT_HISTORY, min_tokens=0)
        result = src.truncate_to(50)  # 非常小的预算
        assert "已截断" in result
        assert estimate_tokens(result) <= 60  # 允许少量溢出

    def test_priority_ordering(self):
        """优先级排序：当前会话 > 长期记忆 > 检索结果。"""
        assert PRIORITY_CHAT_HISTORY < PRIORITY_GRAPHZEP_FACTS
        assert PRIORITY_GRAPHZEP_FACTS < PRIORITY_RETRIEVAL

    def test_chat_history_truncation_keeps_latest_turns(self):
        """会话超预算时应保留最近轮次，而不是最早轮次。"""
        long_text = "\n".join(
            f"turn-{index}: 这是第 {index} 轮用户纠正"
            for index in range(20)
        )
        src = ContextSource(
            "chat_history",
            long_text,
            PRIORITY_CHAT_HISTORY,
            min_tokens=0,
            preserve="tail",
        )
        result = src.truncate_to(55)

        assert result.startswith("... (较早内容已截断)")
        assert "turn-19" in result
        assert "turn-0" not in result
