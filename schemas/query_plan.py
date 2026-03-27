"""
统一查询计划模型
将意图分析 + 检索路由 + 联网判断合并为单一数据结构
"""

from typing import List, Optional, Literal
from pydantic import BaseModel, Field


class RetrievalPlan(BaseModel):
    """检索执行计划：决定启用哪些引擎及其具体参数"""

    use_graph: bool = Field(default=False, description="是否启用知识图谱检索（歌手/流派/关系查询）")
    graph_entities: List[str] = Field(default_factory=list, description="提取出的实体词列表，含别名/外文名")
    graph_genre_filter: Optional[str] = Field(default=None, description="音乐流派过滤，如 摇滚/电子/爵士/古典/民谣/金属/嘻哈")
    graph_scenario_filter: Optional[str] = Field(default=None, description="活动场景过滤，如 开车/运动/学习/睡觉/派对/旅行")
    graph_mood_filter: Optional[str] = Field(default=None, description="情绪心情过滤，如 开心/悲伤/放松/愤怒/浪漫/治愈")
    graph_language_filter: Optional[str] = Field(default=None, description="语言过滤条件，如 Chinese/English/Japanese/Korean/Cantonese")
    graph_region_filter: Optional[str] = Field(default=None, description="地区过滤条件，如 Mainland China/Taiwan/Japan/Western")

    use_vector: bool = Field(default=False, description="是否启用声学向量检索（氛围/听感/情绪匹配）")
    # vector_acoustic_query 已移除：声学描述由下游 HyDE 专用模块自动生成，Planner 无需关心

    use_web_search: bool = Field(default=False, description="是否启用联网搜索（最新资讯/新闻/动态）")
    web_search_keywords: str = Field(default="", description="精炼的搜索引擎查询词")


class MusicQueryPlan(BaseModel):
    """
    统一查询计划：一次 LLM 调用输出的完整决策结构。
    合并了原来的 intent_type（MUSIC_INTENT_ANALYZER_PROMPT）
    和 RetrievalStrategy（MusicQueryRouter._llm_routing）的全部功能。
    """

    intent_type: Literal[
        "play_specific_song_online",
        "search",
        "acquire_music",
        "recommend_by_mood",
        "recommend_by_genre",
        "recommend_by_artist",
        "recommend_by_favorites",
        "recommend_by_activity",
        "general_chat"
    ] = Field(
        description="意图类型"
    )
    parameters: dict = Field(
        default_factory=dict,
        description="意图参数（如 query, mood, genre, artist, activity 等）"
    )
    context: str = Field(default="", description="用户提供的额外上下文信息")

    retrieval_plan: RetrievalPlan = Field(
        default_factory=RetrievalPlan,
        description="检索执行计划"
    )

    reasoning: str = Field(default="", description="LLM 的简短决策推理")
