"""
测试 Pydantic Schema 验证逻辑

验证 MusicQueryPlan / RetrievalPlan 的数据校验和默认值行为。
"""
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from schemas.query_plan import MusicQueryPlan, RetrievalPlan


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


class TestMusicQueryPlan:
    """测试统一查询计划 Schema"""

    def test_valid_intent_types(self):
        """所有有效意图类型"""
        valid_intents = [
            "graph_search", "hybrid_search", "vector_search",
            "general_chat", "web_search",
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
