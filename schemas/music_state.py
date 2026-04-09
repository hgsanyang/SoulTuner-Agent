"""
音乐推荐Agent的状态定义
"""

from typing import TypedDict, List, Dict, Any, Optional, Annotated
from pydantic import BaseModel
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

class ToolOutput(BaseModel):
    """标准的工具返回值格式"""
    success: bool
    data: Any             # 结构化的数据（如 Song 列表）
    raw_markdown: str     # 给 LLM 读的格式化报告文本
    error_message: Optional[str] = None


class MusicAgentState(TypedDict, total=False):
    """音乐推荐Agent的状态"""

    # 用户输入
    input: str  # 用户查询/请求
    chat_history: Annotated[list[BaseMessage], add_messages]  # 对话历史

    # 意图分析结果
    intent_type: str  # 意图类型
    intent_parameters: Dict[str, Any]  # 意图参数
    intent_context: str  # 意图上下文

    # 搜索和推荐结果
    search_results: List[Dict[str, Any]]  # 搜索到的歌曲
    recommendations: List[Dict[str, Any]]  # 推荐结果

    # 用户偏好数据
    user_preferences: Dict[str, Any]  # 用户偏好
    favorite_songs: List[Dict[str, str]]  # 用户喜欢的歌曲

    # 生成的内容
    explanation: str  # 推荐解释
    final_response: str  # 最终回复
    playlist: Optional[Dict[str, Any]]  # 生成的播放列表

    # 执行状态
    step_count: int  # 执行步数
    error_log: List[Dict[str, Any]]  # 错误日志

    # 额外信息
    metadata: Dict[str, Any]  # 元数据
    retrieval_plan: Optional[Dict[str, Any]]  # 统一检索计划（来自 MusicQueryPlan）

    # GraphZep 记忆上下文（新增）
    graphzep_facts: str           # 从 GraphZep 召回的事实文本
    graphzep_group_id: str        # 当前会话的 group ID

    # 歌单相关（新增）
    playlist_candidates: List[Dict[str, Any]]  # 歌单候选歌曲
    playlist_balance_config: Dict[str, Any]  # 平衡配置
    created_playlist: Optional[Dict[str, Any]]  # 创建的播放列表信息

    # 流式输出队列（内部使用）
    _explanation_queue: Any  # asyncio.Queue，用于流式推送推荐解释 chunk

    # 本地未命中 → 联网降级标志（内部使用，节点间通信）
    _need_web_fallback: bool   # True 时 route_after_search 会跳转到 web_fallback 节点
    _web_fallback_query: str   # 传给网易云 API 的干净搜索词（歌名 歌手）


class UserPreferences(TypedDict, total=False):
    """用户音乐偏好"""

    favorite_genres: List[str]  # 喜欢的流派
    favorite_artists: List[str]  # 喜欢的艺术家
    favorite_decades: List[str]  # 喜欢的年代
    avoid_genres: List[str]  # 不喜欢的流派
    mood_preferences: List[str]  # 心情偏好
    activity_contexts: List[str]  # 活动场景偏好
    language_preference: str  # 语言偏好（中文/英文等）


class PlaylistInfo(TypedDict, total=False):
    """播放列表信息"""

    playlist_name: str  # 播放列表名称
    description: str  # 描述
    songs: List[Dict[str, Any]]  # 歌曲列表
    total_duration: int  # 总时长（秒）
    mood_progression: str  # 情绪变化描述
    created_at: str  # 创建时间
    theme: str  # 主题
    # Spotify 相关字段（新增）
    id: Optional[str]  # Spotify 播放列表 ID
    url: Optional[str]  # Spotify 播放列表 URL
    track_count: Optional[int]  # 歌曲数量

