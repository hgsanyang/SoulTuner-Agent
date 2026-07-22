"""
测试 Pydantic Schema 验证逻辑

验证 MusicQueryPlan / RetrievalPlan 的数据校验和默认值行为。
"""
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from schemas.query_plan import HardConstraints, IntentHints, MusicQueryPlan, RetrievalPlan, SoftIntent


class TestRetrievalPlan:
    """测试检索执行计划 Schema"""

    def test_defaults(self):
        """默认值应全部关闭"""
        plan = RetrievalPlan()
        assert plan.use_graph is False
        assert plan.use_vector is False
        assert plan.use_web_search is False
        assert plan.graph_entities == []
        assert plan.graph_genre_filter is None

    def test_from_dict(self):
        """从字典创建"""
        plan = RetrievalPlan(
            use_graph=True,
            graph_entities=["周杰伦", "青花瓷"],
            graph_genre_filter="流行",
        )
        assert plan.use_graph is True
        assert "周杰伦" in plan.graph_entities
        assert plan.graph_genre_filter == "流行"

    def test_layered_fields_populate_legacy_fields(self):
        """新分层表示应自动补齐旧检索字段"""
        plan = RetrievalPlan(
            use_graph=True,
            hard_constraints=HardConstraints(
                artist_entities=["mol-74"],
                song_entities=["赤い頬"],
                language="Japanese",
                region="Japan",
            ),
            hints=IntentHints(genres=["indie"], mood="平静", scenario="学习"),
        )
        assert plan.graph_artist_entities == ["mol-74"]
        assert plan.graph_song_entities == ["赤い頬"]
        assert plan.graph_entities == ["mol-74", "赤い頬"]
        assert plan.graph_language_filter == "Japanese"
        assert plan.graph_region_filter == "Japan"
        assert plan.graph_genre_filter == "indie"
        assert plan.graph_mood_filter == "平静"
        assert plan.graph_scenario_filter == "学习"

    def test_legacy_fields_populate_layered_fields(self):
        """旧字段应自动补齐新分层对象，保证历史 prompt 兼容"""
        plan = RetrievalPlan(
            use_graph=True,
            graph_artist_entities=["周杰伦"],
            graph_song_entities=["晴天"],
            graph_language_filter="Chinese",
            graph_region_filter="Taiwan",
            graph_genre_filter="pop",
            graph_mood_filter="浪漫",
            graph_scenario_filter="开车",
        )
        assert plan.hard_constraints.artist_entities == ["周杰伦"]
        assert plan.hard_constraints.song_entities == ["晴天"]
        assert plan.hard_constraints.language == "Chinese"
        assert plan.hard_constraints.region == "Taiwan"
        assert plan.hints.genres == ["pop"]
        assert plan.hints.mood == "浪漫"
        assert plan.hints.scenario == "开车"

    def test_soft_intent_roundtrip(self):
        """软意图自由文本不应被枚举槽位吞掉"""
        plan = RetrievalPlan(
            use_vector=True,
            soft_intent=SoftIntent(
                goal="从 emo 里走出来",
                trajectory="低落到释然",
                avoid=["太吵", "过度鸡血"],
                vibe="安静、低动态、但逐渐变亮",
            ),
        )
        assert plan.soft_intent.goal == "从 emo 里走出来"
        assert plan.soft_intent.trajectory == "低落到释然"
        assert "太吵" in plan.soft_intent.avoid

    def test_instrumental_legacy_language_becomes_acoustic_constraint(self):
        """Instrumental is an acoustic intent, not a sparse language hard filter."""
        plan = RetrievalPlan(use_graph=True, graph_language_filter="Instrumental")

        assert plan.hard_constraints.instrumental is True
        assert plan.hard_constraints.language is None
        assert plan.graph_language_filter is None


class TestMusicQueryPlan:
    """测试统一查询计划 Schema"""

    def test_valid_intent_types(self):
        """所有有效意图类型"""
        valid_intents = [
            "graph_search", "hybrid_search", "vector_search",
            "clarification", "general_chat", "web_search",
            "acquire_music", "recommend_by_favorites",
        ]
        for intent in valid_intents:
            plan = MusicQueryPlan(intent_type=intent)
            assert plan.intent_type == intent

    def test_invalid_intent_type_raises(self):
        """无效意图类型应抛出 ValidationError"""
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            MusicQueryPlan(intent_type="invalid_intent_xyz")

    def test_default_retrieval_plan(self):
        """默认 retrieval_plan 不应为 None"""
        plan = MusicQueryPlan(intent_type="graph_search")
        assert plan.retrieval_plan is not None
        assert isinstance(plan.retrieval_plan, RetrievalPlan)

    def test_json_roundtrip(self):
        """JSON 序列化/反序列化往返一致"""
        plan = MusicQueryPlan(
            intent_type="hybrid_search",
            parameters={"query": "安静的民谣"},
            context="用户想听民谣",
            retrieval_plan=RetrievalPlan(
                use_graph=True,
                use_vector=True,
                graph_genre_filter="民谣",
            ),
            reasoning="有流派标签+主观描述",
        )
        json_str = plan.model_dump_json()
        restored = MusicQueryPlan.model_validate_json(json_str)
        assert restored.intent_type == "hybrid_search"
        assert restored.retrieval_plan.use_graph is True
        assert restored.retrieval_plan.graph_genre_filter == "民谣"

    def test_empty_parameters(self):
        """parameters 为空 dict 时不报错"""
        plan = MusicQueryPlan(intent_type="general_chat", parameters={})
        assert plan.parameters == {}
