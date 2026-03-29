"""
统一查询计划模型（V3 简化版）
=============================
5 类检索策略型意图 + 确定性标签提取。
LLM 只负责判断意图类型，标签提取由确定性规则完成。
"""

from typing import List, Optional, Literal
from pydantic import BaseModel, Field


class RetrievalPlan(BaseModel):
    """检索执行计划：由确定性规则自动填充，LLM 不再填写"""

    use_graph: bool = Field(default=False, description="是否启用知识图谱检索")
    graph_entities: List[str] = Field(default_factory=list, description="提取出的实体词列表，含别名/外文名")
    graph_genre_filter: Optional[str] = Field(default=None, description="流派过滤")
    graph_scenario_filter: Optional[str] = Field(default=None, description="场景过滤")
    graph_mood_filter: Optional[str] = Field(default=None, description="情绪过滤")
    graph_language_filter: Optional[str] = Field(default=None, description="语言过滤")
    graph_region_filter: Optional[str] = Field(default=None, description="地区过滤")

    use_vector: bool = Field(default=False, description="是否启用声学向量检索")
    vector_acoustic_query: Optional[str] = Field(default=None, description="用于向量检索的声学描述查询（HyDE）")

    use_web_search: bool = Field(default=False, description="是否启用联网搜索")
    web_search_keywords: str = Field(default="", description="搜索关键词")


class MusicQueryPlan(BaseModel):
    """
    统一查询计划（V3 简化版）。

    意图分类从 9 类简化为 5 类检索策略：
      - graph_search: 图谱精确检索（有具体实体：歌手/歌名/流派/语言/地区）
      - hybrid_search: 混合检索（实体 + 无法用标签穷举的主观声学描述）
      - vector_search: 纯向量检索（纯氛围/情绪，无具体实体）
      - general_chat: 闲聊
      - web_search: 联网检索（时效性内容）

    功能性意图（单独处理，不走检索管线）：
      - acquire_music: 确认下载
      - recommend_by_favorites: 查用户收藏

    LLM 只负责选择意图类型 + 提取实体名。
    标签（genre/mood/scenario/language/region）由确定性规则自动提取。
    """

    intent_type: Literal[
        "graph_search",
        "hybrid_search",
        "vector_search",
        "general_chat",
        "web_search",
        "acquire_music",
        "recommend_by_favorites",
    ] = Field(
        description="意图类型（5 类检索策略 + 2 类功能性意图）"
    )
    parameters: dict = Field(
        default_factory=dict,
        description="意图参数（如 query, entities 等）"
    )
    context: str = Field(default="", description="对用户意图的简短描述")

    retrieval_plan: RetrievalPlan = Field(
        default_factory=RetrievalPlan,
        description="检索执行计划（V3 中主要由确定性规则填充）"
    )

    reasoning: str = Field(default="", description="LLM 的简短决策推理")
