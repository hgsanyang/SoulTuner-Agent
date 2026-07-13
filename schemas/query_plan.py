"""
统一查询计划模型（V3）
=============================
5 类检索策略型意图 + 2 类功能性意图。
意图类型、实体、五维标签、声学描述(HyDE) 均由 LLM 在一次 structured_output 调用中产出
（见 agent/music_graph.py:analyze_intent）。下游检索按本计划执行。
"""

from typing import List, Optional, Literal
from pydantic import BaseModel, Field, model_validator

from schemas.tool_plan import ToolPlan, compile_legacy_tool_plan


class HardConstraints(BaseModel):
    """必须精确满足的硬约束，适合进入 Cypher 过滤。"""

    artist_entities: List[str] = Field(default_factory=list, description="歌手/乐队实体及别名")
    song_entities: List[str] = Field(default_factory=list, description="歌曲/专辑实体及别名")
    language: Optional[str] = Field(default=None, description="语言硬约束，如 Chinese/English/Japanese；Instrumental 属于声学需求")
    region: Optional[str] = Field(default=None, description="地区硬约束，如 Mainland China/Japan/Western")
    instrumental: bool = Field(default=False, description="是否明确要求纯音乐/器乐/无人声")


class SoftIntent(BaseModel):
    """无法被固定词表完整表达的软意图，优先映射到连续声学空间。"""

    goal: str = Field(default="", description="用户想通过音乐达成的目标，如从 emo 里走出来")
    trajectory: str = Field(default="", description="期望的情绪/能量变化路径")
    avoid: List[str] = Field(default_factory=list, description="需要避开的主观体验或声学特征")
    vibe: str = Field(default="", description="自由文本氛围/画面感/声学质感")


class IntentHints(BaseModel):
    """可选提示，不再作为必须命中的硬槽位。"""

    genres: List[str] = Field(default_factory=list, description="流派提示，可用于图谱粗筛")
    mood: Optional[str] = Field(default=None, description="情绪提示")
    scenario: Optional[str] = Field(default=None, description="场景提示")


class MetadataConstraints(BaseModel):
    """由 LLM 显式产出的目录/外部知识约束，供 Catalog Gap 验证本地库存。"""

    release_year_from: Optional[int] = Field(default=None, description="发行年份下界，如 1980")
    release_year_to: Optional[int] = Field(default=None, description="发行年份上界，如 1989")
    era: Optional[str] = Field(default=None, description="用户描述的年代/时期，如 80s、oldies")
    recency_required: bool = Field(default=False, description="是否明确要求最新/近期歌曲")
    external_knowledge_required: bool = Field(default=False, description="是否需要榜单、背景、新闻等外部知识")


class RetrievalPlan(BaseModel):
    """检索执行计划：分层表达硬约束、软意图与可选提示，并保留旧字段兼容。"""

    use_graph: bool = Field(
        default=False,
        description="旧版兼容提示；R1 后不再作为图谱召回开关",
    )
    hard_constraints: HardConstraints = Field(default_factory=HardConstraints, description="实体/语言/地区/纯音乐等硬约束")
    soft_intent: SoftIntent = Field(default_factory=SoftIntent, description="自由文本软意图，供向量检索和 HyDE 使用")
    hints: IntentHints = Field(default_factory=IntentHints, description="可选标签提示，不能替代软意图")
    metadata_constraints: MetadataConstraints = Field(
        default_factory=MetadataConstraints,
        description="年份、时效性、外部知识等本地曲库覆盖能力约束",
    )

    # 旧字段保留给现有检索链路、评测和本地小模型使用。
    graph_entities: List[str] = Field(default_factory=list, description="提取出的实体词列表，含别名/外文名（向后兼容）")
    graph_artist_entities: List[str] = Field(default_factory=list, description="歌手名实体（含中英文别名），匹配 Artist.name")
    graph_song_entities: List[str] = Field(default_factory=list, description="歌曲名实体（含中英文别名），匹配 Song.title")
    graph_genre_filter: Optional[str] = Field(default=None, description="流派过滤")
    graph_scenario_filter: Optional[str] = Field(default=None, description="场景过滤")
    graph_mood_filter: Optional[str] = Field(default=None, description="情绪过滤")
    graph_language_filter: Optional[str] = Field(default=None, description="语言过滤")
    graph_region_filter: Optional[str] = Field(default=None, description="地区过滤")

    use_vector: bool = Field(
        default=False,
        description="旧版兼容提示；R1 后不再作为稠密召回开关",
    )
    vector_acoustic_query: Optional[str] = Field(default=None, description="用于向量检索的主声学描述查询（HyDE）")
    vector_acoustic_queries: List[str] = Field(
        default_factory=list,
        description=(
            "用于文搜音检索的多视角声学描述。由 LLM 生成 2-4 条互补 MusicCaps 风格英文描述，"
            "覆盖乐器/节奏/音色、能量动态、场景氛围等不同声学视角。"
        ),
    )

    use_web_search: bool = Field(default=False, description="是否启用联网搜索")
    web_search_keywords: str = Field(default="", description="搜索关键词")

    @model_validator(mode="after")
    def sync_layered_and_legacy_fields(self) -> "RetrievalPlan":
        """让新旧表示互相补齐，避免一次性重写整个检索栈。"""
        hard = self.hard_constraints
        hints = self.hints

        if hard.artist_entities and not self.graph_artist_entities:
            self.graph_artist_entities = list(hard.artist_entities)
        if hard.song_entities and not self.graph_song_entities:
            self.graph_song_entities = list(hard.song_entities)
        if not hard.artist_entities and self.graph_artist_entities:
            hard.artist_entities = list(self.graph_artist_entities)
        if not hard.song_entities and self.graph_song_entities:
            hard.song_entities = list(self.graph_song_entities)

        if not self.graph_entities:
            self.graph_entities = list(dict.fromkeys(self.graph_artist_entities + self.graph_song_entities))

        language_is_instrumental = str(hard.language or self.graph_language_filter or "").strip().casefold() in {
            "instrumental",
            "纯音乐",
            "器乐",
        }
        if language_is_instrumental:
            hard.instrumental = True
            hard.language = None
            self.graph_language_filter = None

        if hard.language and not self.graph_language_filter:
            self.graph_language_filter = hard.language
        if hard.region and not self.graph_region_filter:
            self.graph_region_filter = hard.region
        if not hard.language and self.graph_language_filter:
            hard.language = self.graph_language_filter
        if not hard.region and self.graph_region_filter:
            hard.region = self.graph_region_filter
        if str(self.graph_language_filter or "").strip().casefold() in {"instrumental", "纯音乐", "器乐"}:
            hard.instrumental = True
            hard.language = None
            self.graph_language_filter = None

        if hints.genres and not self.graph_genre_filter:
            self.graph_genre_filter = hints.genres[0]
        if hints.mood and not self.graph_mood_filter:
            self.graph_mood_filter = hints.mood
        if hints.scenario and not self.graph_scenario_filter:
            self.graph_scenario_filter = hints.scenario
        if not hints.genres and self.graph_genre_filter:
            hints.genres = [self.graph_genre_filter]
        if not hints.mood and self.graph_mood_filter:
            hints.mood = self.graph_mood_filter
        if not hints.scenario and self.graph_scenario_filter:
            hints.scenario = self.graph_scenario_filter

        acoustic_queries: list[str] = []
        for item in self.vector_acoustic_queries:
            text = str(item or "").strip()
            if text and text not in acoustic_queries:
                acoustic_queries.append(text)
        if self.vector_acoustic_query:
            primary = str(self.vector_acoustic_query).strip()
            if primary and primary not in acoustic_queries:
                acoustic_queries.insert(0, primary)
        if acoustic_queries and not self.vector_acoustic_query:
            self.vector_acoustic_query = acoustic_queries[0]
        self.vector_acoustic_queries = acoustic_queries[:4]

        return self


class MusicQueryPlan(BaseModel):
    """
    统一查询计划（V3 简化版）。

    意图分类从 9 类简化为 5 类检索策略：
      - graph_search: 图谱精确检索（有具体实体：歌手/歌名/流派/语言/地区）
      - hybrid_search: 混合检索（实体 + 无法用标签穷举的主观声学描述）
      - vector_search: 纯向量检索（纯氛围/情绪，无具体实体）
      - clarification: 澄清反问（引用不明、严重矛盾、关键槽缺失时不硬猜）
      - general_chat: 闲聊
      - web_search: 联网检索（时效性内容）

    功能性意图（单独处理，不走检索管线）：
      - acquire_music: 确认下载
      - recommend_by_favorites: 查用户收藏

    LLM 一次性输出：意图类型 + 实体名 + 五维标签(genre/mood/scenario/language/region)
    + 声学描述(HyDE)。本模型即该结构化输出的 schema。
    """

    intent_type: Literal[
        "graph_search",
        "hybrid_search",
        "vector_search",
        "clarification",
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
    tool_plan: Optional[ToolPlan] = Field(
        default=None,
        description="ToolPlan v1；由 LLM 直接选择白名单工具，缺失时由旧 RetrievalPlan 兼容编译",
    )

    @model_validator(mode="after")
    def ensure_tool_plan(self) -> "MusicQueryPlan":
        if self.tool_plan is None:
            self.tool_plan = compile_legacy_tool_plan(self)
        return self
