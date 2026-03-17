import asyncio
from typing import List, Dict, Any, Optional
from config.logging_config import get_logger
from tools.music_tools import get_music_search_tool, get_music_recommender, Song, MusicRecommendation

logger = get_logger(__name__)

class MusicApiService:
    """
    纯粹的音乐业务 API 服务层。
    提供基础的歌曲检索、场景推荐等能力，供前端或非 Agent 编排场景直接调用。
    将原本在 MusicRecommendationAgent 里混杂的工具调用转移到这里。
    """
    def __init__(self):
        self.search_tool = get_music_search_tool()
        self.recommender = get_music_recommender()
        self._is_ready = True
        
    async def search_music(
        self, 
        query: str, 
        genre: Optional[str] = None, 
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """基础搜索功能（不经过LangGraph）"""
        try:
            songs = await self.search_tool.search_songs(query, genre, limit)
            return [song.to_dict() for song in songs]
        except Exception as e:
            logger.error(f"基础搜索失败: {str(e)}")
            return []
            
    async def get_recommendations_by_mood(
        self, 
        mood: str, 
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """根据心情推荐音乐（不经过LangGraph）"""
        try:
            recs = await self.recommender.recommend_by_mood(mood, limit)
            return [rec.to_dict() for rec in recs]
        except Exception as e:
            logger.error(f"根据心情推荐失败: {str(e)}")
            return []
            
    async def get_recommendations_by_activity(
        self, 
        activity: str, 
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """根据场景推荐音乐（不经过LangGraph）"""
        try:
            recs = await self.recommender.recommend_by_activity(activity, limit)
            return [rec.to_dict() for rec in recs]
        except Exception as e:
            logger.error(f"根据场景推荐失败: {str(e)}")
            return []
            
    async def get_similar_songs(
        self, 
        song_title: str, 
        artist: str, 
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """获取相似歌曲（不经过LangGraph）"""
        try:
            songs = await self.search_tool.get_similar_songs(song_title, artist, limit)
            return [song.to_dict() for song in songs]
        except Exception as e:
            logger.error(f"获取相似歌曲失败: {str(e)}")
            return []
