"""
音乐旅程计划模型
LLM Structured Output 用于将用户故事/情绪曲线解析为分段旅程规划
"""

from typing import List, Optional
from pydantic import BaseModel, Field


class JourneySegmentPlan(BaseModel):
    """旅程的单个情绪片段"""

    segment_id: int = Field(description="片段序号，从 0 开始")
    mood: str = Field(description="该片段的核心情绪关键词（中文），如 平静/活力/悲伤/浪漫")
    description: str = Field(description="该片段的场景描述（1-2 句话），解释为什么选择这种情绪")
    duration_ratio: float = Field(
        description="该片段占总旅程时长的比例，所有片段的 duration_ratio 之和应为 1.0",
        ge=0.0,
        le=1.0,
    )
    acoustic_hint: str = Field(
        default="",
        description=(
            "可选的英文声学提示（30-60 词），用于向量检索。"
            "描述该阶段音乐的乐器、节奏、氛围等声学特征。"
            "如果留空，系统会根据 mood 自动生成。"
        ),
    )
    graph_genre_filter: Optional[str] = Field(
        default=None,
        description="可选的流派过滤（英文），如 pop/rock/jazz/classical/folk/electronic",
    )
    graph_mood_filter: Optional[str] = Field(
        default=None,
        description="可选的情绪过滤（中文），如 开心/悲伤/放松/治愈/浪漫",
    )
    graph_scenario_filter: Optional[str] = Field(
        default=None,
        description="可选的场景过滤（中文），如 运动/学习/开车/睡觉",
    )
    songs_count: int = Field(
        default=3,
        description="该片段需要推荐的歌曲数量（每首歌约 4 分钟），通常 2-4 首",
        ge=1,
        le=6,
    )


class MusicJourneyPlan(BaseModel):
    """完整的音乐旅程规划"""

    title: str = Field(description="旅程标题（10 字以内），如「清晨到黄昏」「从悲伤到治愈」")
    total_segments: int = Field(description="总片段数")
    segments: List[JourneySegmentPlan] = Field(description="按时间顺序排列的情绪片段列表")
    reasoning: str = Field(
        default="",
        description="LLM 对旅程规划的简短说明（为什么这样安排情绪流转）",
    )
