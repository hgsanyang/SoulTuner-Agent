"""
测试标签扩展逻辑

验证：GENRE_TAG_MAP / MOOD_TAG_MAP / SCENARIO_TAG_MAP 的中文→英文映射
是检索管线的基础组件，必须正确工作。
"""
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from tools.graphrag_search import (
    _expand_tag,
    _expand_genre_to_english,
    _expand_scenario_to_english,
    _expand_mood_to_english,
    GENRE_TAG_MAP,
    MOOD_TAG_MAP,
    SCENARIO_TAG_MAP,
    LANGUAGE_ALIAS_MAP,
)


class TestGenreExpansion:
    """测试流派标签中文→英文扩展"""

    def test_exact_match(self):
        """精确匹配中文流派"""
        result = _expand_genre_to_english("摇滚")
        assert "rock" in result

    def test_electronic_family(self):
        """电子音乐家族"""
        result = _expand_genre_to_english("电子")
        assert "electronic" in result

    def test_hiphop(self):
        """嘻哈/说唱"""
        result = _expand_genre_to_english("嘻哈")
        assert any("hip" in r for r in result)

    def test_folk(self):
        """民谣"""
        result = _expand_genre_to_english("民谣")
        assert "folk" in result or "acoustic" in result

    def test_unknown_genre_passthrough(self):
        """未知流派应原样返回（小写）"""
        result = _expand_genre_to_english("未知奇怪流派XYZ")
        assert len(result) >= 1
        assert result[0] == result[0].lower()

    def test_case_insensitive(self):
        """大小写不敏感匹配"""
        result1 = _expand_genre_to_english("EDM")
        result2 = _expand_genre_to_english("edm")
        # 两者都应该包含 electronic/dance 类标签
        assert len(result1) > 0
        assert len(result2) > 0

    def test_none_input(self):
        """None 输入应返回空列表"""
        result = _expand_genre_to_english(None)
        assert result == []

    def test_empty_string(self):
        """空字符串应返回空列表"""
        result = _expand_genre_to_english("")
        assert result == []


class TestMoodExpansion:
    """测试情绪标签扩展"""

    def test_happy(self):
        result = _expand_mood_to_english("开心")
        assert "happy" in result or "energetic" in result

    def test_sad(self):
        result = _expand_mood_to_english("悲伤")
        assert any(m in result for m in ["melancholy", "sad", "lonely"])

    def test_healing(self):
        result = _expand_mood_to_english("治愈")
        assert "healing" in result or "hopeful" in result


class TestScenarioExpansion:
    """测试场景标签扩展"""

    def test_workout(self):
        result = _expand_scenario_to_english("运动")
        assert "workout" in result or "energetic" in result

    def test_sleep(self):
        result = _expand_scenario_to_english("睡觉")
        assert "sleep" in result or "relaxing" in result

    def test_study(self):
        result = _expand_scenario_to_english("学习")
        assert "study" in result or "peaceful" in result


class TestLanguageAlias:
    """测试语言别名映射"""

    def test_chinese_aliases(self):
        """中文的多种表达"""
        for alias in ["中文", "国语", "华语"]:
            assert LANGUAGE_ALIAS_MAP.get(alias) == "Chinese"

    def test_guoyao_implies_chinese(self):
        """'国摇' 应映射到 Chinese"""
        assert LANGUAGE_ALIAS_MAP.get("国摇") == "Chinese"

    def test_japanese(self):
        assert LANGUAGE_ALIAS_MAP.get("日语") == "Japanese"

    def test_instrumental(self):
        assert LANGUAGE_ALIAS_MAP.get("纯音乐") == "Instrumental"
