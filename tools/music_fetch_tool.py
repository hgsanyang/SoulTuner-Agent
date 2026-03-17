import aiohttp
from typing import List, Optional
from pydantic import BaseModel
from langchain_core.tools import tool
from config.logging_config import get_logger
from config.settings import settings
from schemas.music_state import ToolOutput
import json
import os

logger = get_logger(__name__)

class OnlineSong(BaseModel):
    title: str
    artist: str
    platform: str
    song_id: str
    play_url: Optional[str] = None
    cover_url: Optional[str] = None
    
    def dict(self, *args, **kwargs):
        return super().model_dump(*args, **kwargs)

class MusicFetcher:
    """音乐抓取服务，支持多种数据源降级策略"""
    def __init__(self):
        # 优先使用环境变量，如果没有则回退到本地默认端口 (需用户自行部署 API 服务)
        self.netease_api_base = settings.netease_api_base
        
    async def search_netease(self, query: str, limit: int = 3) -> List[OnlineSong]:
        """使用公开网易云 API 接口检索音乐和真实播放链接"""
        async with aiohttp.ClientSession() as session:
            try:
                # 1. 搜索获取 ID
                search_url = f"{self.netease_api_base}/search?keywords={query}&limit={limit}"
                async with session.get(search_url, timeout=settings.netease_api_timeout) as resp:
                    if resp.status != 200:
                        return []
                    data = await resp.json()
                    songs = data.get("result", {}).get("songs", [])
                    if not songs:
                        return []
                        
                results = []
                # 2. 批量获取播放链接
                song_ids = [str(s["id"]) for s in songs]
                ids_str = ",".join(song_ids)
                url_api = f"{self.netease_api_base}/song/url?id={ids_str}&level=exhigh"
                
                async with session.get(url_api, timeout=settings.netease_api_timeout) as resp:
                    urls_data = {}
                    if resp.status == 200:
                        ud = await resp.json()
                        for item in ud.get("data", []):
                            if item.get("url"):
                                urls_data[str(item["id"])] = item["url"]
                                
                for s in songs:
                    sid = str(s["id"])
                    play_url = urls_data.get(sid)
                    
                    # 取出作者信息
                    artists = ",".join([a["name"] for a in s.get("artists", [])])
                    
                    if play_url:
                        # 3. 过滤掉无法获取真实播放链接的（灰色版权 / VIP 限制）
                        # TODO: 封面可以在 /song/detail 接口拿，这里为了提速省略
                        results.append(OnlineSong(
                            title=s.get("name", "Unknown"),
                            artist=artists,
                            platform="netease",
                            song_id=sid,
                            play_url=play_url,
                            cover_url=None
                        ))
                return results
            except Exception as e:
                logger.error(f"网易云检索失败: {e}")
                return []

    async def fetch(self, query: str) -> List[OnlineSong]:
        """执行降级检索策略"""
        logger.info(f"开启互联网寻歌: {query}")
        # 1. 尝试网易云
        songs = await self.search_netease(query)
        if songs:
            logger.info(f"网易云成功获取 {len(songs)} 首播放链接")
            return songs
            
        # 2. 如果网易云失败，日志提醒回退
        logger.warning(f"网易云未找到《{query}》的可用播放源，或者版权受限。")
        return []

# 全局单例
music_fetcher_instance = MusicFetcher()

async def execute_search_online_music(query: str) -> ToolOutput:
    """内部直接调用的异步原始函数"""
    songs = await music_fetcher_instance.fetch(query)
    if not songs:
        return ToolOutput(
            success=False,
            data=[],
            raw_markdown="❌ 未能在互联网音乐平台找到该歌曲的可用播放源（可能因版权受限）。",
            error_message="Not found"
        )
        
    markdown_str = "🎵 **已从互联网平台为您找到以下可播放音源：**\n\n"
    for i, s in enumerate(songs, 1):
        markdown_str += f"{i}. **{s.title}** - {s.artist} [{s.platform}]\n"
        markdown_str += f"   🔈 [点击播放音频]({s.play_url})\n"
        # 采用 HTML5 原生播放器
        markdown_str += f"   <audio src=\"{s.play_url}\" controls preload=\"none\"></audio>\n\n"
        
    return ToolOutput(
        success=True,
        data=[{**s.dict(), "preview_url": s.play_url} for s in songs],
        raw_markdown=markdown_str
    )

@tool
async def search_online_music(query: str) -> ToolOutput:
    """
    当本地曲库没有找到用户想听的歌曲，或者用户明确要求“播放网页/最新/某首特定的歌”时调用。
    该工具会从各大音乐平台（网易云等）抓取真实的音频试听链接。
    
    Args:
        query: 歌曲名称和歌手，例如 "周杰伦 稻香"
    """
    return await execute_search_online_music(query)
