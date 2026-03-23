"""
FastAPI后端服务器
支持SSE流式输出音乐推荐
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import AsyncGenerator, Dict, Any, Optional, List

# 添加项目根目录到Python路径(如果还没有)
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel


from config.logging_config import get_logger
from agent.music_agent import MusicRecommendationAgent
from tools.music_tools import get_music_search_tool

logger = get_logger(__name__)

app = FastAPI(title="Music Recommendation API", version="1.0.0")

# 注册用户画像路由
from api.user_profile import router as user_profile_router
app.include_router(user_profile_router)

@app.on_event("startup")
async def startup_event():
    """在服务器启动时预加载 M2D-CLAP 跨模态模型"""
    logger.info("🚀 正在预加载 M2D-CLAP 跨模态检索模型到内存...")
    try:
        from retrieval.audio_embedder import get_m2d2_model
        import asyncio
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, get_m2d2_model)
        logger.info("✅ M2D-CLAP 模型预加载完成!")
    except Exception as e:
        logger.error(f"❌ M2D-CLAP 模型预加载失败: {e}")

# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000", 
        "http://127.0.0.1:3000",
        "http://localhost:3003",   # Frontend (Next.js)
        "http://127.0.0.1:3003",
        "http://localhost:3100",   # GraphZep Server
        "http://127.0.0.1:3100",
        "http://localhost:31000",
        "http://127.0.0.1:31000"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 挂载静态音频/封面/歌词文件目录
# 音频实际存放在 data/processed_audio/audio/(非 data/raw_audio)
from fastapi.staticfiles import StaticFiles
PROCESSED_AUDIO_ROOT = Path(r"C:\Users\sanyang\sanyangworkspace\music_recommendation\data\processed_audio")
audio_dir = PROCESSED_AUDIO_ROOT / "audio"
if audio_dir.exists():
    app.mount("/static/audio", StaticFiles(directory=str(audio_dir)), name="audio")
    logger.info(f"✅ 音频静态文件挂载成功: {audio_dir}")
else:
    logger.warning(f"音频目录不存在,无法提供静态音频挂载: {audio_dir}")
cover_dir = PROCESSED_AUDIO_ROOT / "covers"
if cover_dir.exists():
    app.mount("/static/covers", StaticFiles(directory=str(cover_dir)), name="covers")
    logger.info(f"✅ 封面静态文件挂载成功: {cover_dir}")
else:
    logger.warning(f"封面目录不存在,无法提供静态封面挂载: {cover_dir}")
lyrics_dir = PROCESSED_AUDIO_ROOT / "lyrics"
if lyrics_dir.exists():
    app.mount("/static/lyrics", StaticFiles(directory=str(lyrics_dir)), name="lyrics")
    logger.info(f"✅ 歌词静态文件挂载成功: {lyrics_dir}")
else:
    logger.warning(f"歌词目录不存在,无法提供静态歌词挂载: {lyrics_dir}")

# MTG 数据集音频：使用显式路由（避免 StaticFiles 挂载顺序问题）
MTG_AUDIO_DIR = Path(r"C:\Users\sanyang\sanyangworkspace\music_recommendation\data\mtg_sample\audio")

from fastapi.responses import FileResponse

@app.get("/static/mtg_audio/{filename:path}")
async def serve_mtg_audio(filename: str):
    """提供 MTG 数据集音频文件"""
    file_path = MTG_AUDIO_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"MTG audio not found: {filename}")
    return FileResponse(str(file_path), media_type="audio/mpeg")


# 联网获取的音频/封面/歌词(独立目录 data/online_acquired/)
ONLINE_AUDIO_ROOT = Path(r"C:\Users\sanyang\sanyangworkspace\music_recommendation\data\online_acquired")
online_audio_dir = ONLINE_AUDIO_ROOT / "audio"
online_audio_dir.mkdir(parents=True, exist_ok=True)
app.mount("/static/online_audio", StaticFiles(directory=str(online_audio_dir)), name="online_audio")
logger.info(f"✅ 联网音频静态文件挂载: {online_audio_dir}")
online_cover_dir = ONLINE_AUDIO_ROOT / "covers"
online_cover_dir.mkdir(parents=True, exist_ok=True)
app.mount("/static/online_covers", StaticFiles(directory=str(online_cover_dir)), name="online_covers")
logger.info(f"✅ 联网封面静态文件挂载: {online_cover_dir}")
online_lyrics_dir = ONLINE_AUDIO_ROOT / "lyrics"
online_lyrics_dir.mkdir(parents=True, exist_ok=True)
app.mount("/static/online_lyrics", StaticFiles(directory=str(online_lyrics_dir)), name="online_lyrics")
logger.info(f"✅ 联网歌词静态文件挂载: {online_lyrics_dir}")

# 全局Agent实例
_agent: Optional[MusicRecommendationAgent] = None


def get_agent() -> MusicRecommendationAgent:
    """获取Agent实例(单例模式)"""
    global _agent
    if _agent is None:
        _agent = MusicRecommendationAgent()
    return _agent


# 请求模型
class RecommendationRequest(BaseModel):
    query: str
    genre: Optional[str] = None
    mood: Optional[str] = None
    user_preferences: Optional[Dict[str, Any]] = None
    chat_history: Optional[List[Dict[str, str]]] = None
    llm_provider: str = "siliconflow"          # 模型供应商: siliconflow / dashscope / google / ...
    web_search_enabled: bool = True           # 是否开启联网搜索


class PlaylistRequest(BaseModel):
    query: str
    target_size: int = 30
    public: bool = False
    user_preferences: Optional[Dict[str, Any]] = None


class JourneyRequest(BaseModel):
    story: Optional[str] = None
    mood_transitions: Optional[List[Dict[str, Any]]] = None  # [{time, mood, intensity}]
    duration: int = 60  # 总时长(分钟)
    user_preferences: Optional[Dict[str, Any]] = None
    context: Optional[Dict[str, Any]] = None  # 天气、地点、时间等


class SearchRequest(BaseModel):
    """歌曲搜索请求"""
    query: str
    genre: Optional[str] = None
    limit: int = 20


class AcquireSongRequest(BaseModel):
    """单曲加入本地请求"""
    title: str
    artist: str
    song_id: Optional[str] = None
    platform: str = "netease"


@app.post("/api/acquire-song")
async def acquire_song_endpoint(request: AcquireSongRequest):
    """
    按需触发数据飞轮:下载单首歌曲的音频/歌词/封面并入库 Neo4j.
    前端点击"加入本地"按钮时调用.
    """
    import aiohttp
    from tools.acquire_music import OnlineMusicAcquirer, _quick_ingest_to_neo4j, _background_flywheel

    query = f"{request.title} {request.artist}"
    logger.info(f"🎯 [acquire-song] 用户请求加入本地: {query}")

    acquirer = OnlineMusicAcquirer()
    async with aiohttp.ClientSession() as session:
        acquired = await acquirer.search_and_acquire([query], session)

    if not acquired:
        raise HTTPException(status_code=404, detail="未能获取该歌曲的音频资源(可能因版权限制)")

    # 秒级写入 Neo4j
    await _quick_ingest_to_neo4j(acquired)

    # 启动后台飞轮(歌词标注 + 向量提取)
    asyncio.create_task(_background_flywheel(acquired))

    song = acquired[0]
    logger.info(f"✅ [acquire-song] 成功入库: {song['title']} - {song['artist']}")
    return {
        "success": True,
        "message": f"已成功将《{song['title']}》加入本地曲库",
        "song": {
            "title": song["title"],
            "artist": song["artist"],
            "album": song.get("album", ""),
            "audio_url": song["audio_url"],
            "cover_url": song.get("cover_url", ""),
        }
    }

async def stream_recommendations(
    query: str,
    genre: Optional[str] = None,
    mood: Optional[str] = None,
    user_preferences: Optional[Dict[str, Any]] = None,
    chat_history: Optional[List[Dict[str, str]]] = None,
    llm_provider: str = "siliconflow",
    web_search_enabled: bool = True,
) -> AsyncGenerator[str, None]:
    """
    流式生成推荐结果 (真流式：推荐解释逐 chunk 推送)
    
    Yields:
        SSE格式的数据块
    """
    try:
        agent = get_agent()
        
        # 根据前端传入的 provider 动态切换 LLM
        try:
            from llms.multi_llm import get_chat_model
            from agent.music_graph import set_llm
            new_llm = get_chat_model(provider=llm_provider)
            set_llm(new_llm)
            logger.info(f"切换 LLM provider 到 {llm_provider}")
        except Exception as e:
            logger.warning(f"切换 LLM 失败,使用默认配置: {e}")

        # 通过环境变量传递联网搜索开关
        os.environ["MUSIC_WEB_SEARCH_ENABLED"] = "1" if web_search_enabled else "0"
        logger.info(f"联网搜索: {'ON' if web_search_enabled else 'OFF'}")
        
        logger.info("\n" + "🚀" * 30)
        logger.info(f"🆕 [NEW RECOMMENDATION] User Query: {query}")
        logger.info("-" * 40)
        
        # 发送开始事件
        yield f"data: {json.dumps({'type': 'start', 'message': '开始分析你的需求...'}, ensure_ascii=False)}\n\n"
        await asyncio.sleep(0.1)
        
        # 使用流式推荐方法：推荐解释会逐 chunk 实时推送
        async for event in agent.stream_recommendations(
            query=query,
            chat_history=chat_history,
            user_preferences=user_preferences
        ):
            event_type = event.get("type")
            
            if event_type == "thinking":
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
                
            elif event_type == "response":
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
                # 流式 chunk 不需要 sleep，尽快推送
                
            elif event_type == "recommendations_start":
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
                
            elif event_type == "song":
                song = event.get("song", {})
                # 跳过无法播放的条目
                is_playable = isinstance(song, dict) and (song.get("audio_url") or song.get("preview_url"))
                if isinstance(song, dict) and song.get("title"):
                    yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
                    await asyncio.sleep(0.1)
                    
            elif event_type == "recommendations_complete":
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
                
            elif event_type == "complete":
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
                
            elif event_type == "error":
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
        
    except Exception as e:
        logger.error(f"流式推荐失败: {str(e)}", exc_info=True)
        yield f"data: {json.dumps({'type': 'error', 'message': str(e)}, ensure_ascii=False)}\n\n"


async def stream_playlist(
    query: str,
    target_size: int = 30,
    public: bool = False,
    user_preferences: Optional[Dict[str, Any]] = None
) -> AsyncGenerator[str, None]:
    """
    流式生成歌单(已降级为基于推荐引擎的本地歌单)
    """
    try:
        agent = get_agent()
        
        yield f"data: {json.dumps({'type': 'start', 'message': '开始生成你的专属歌单...'}, ensure_ascii=False)}\n\n"
        await asyncio.sleep(0.1)
        
        yield f"data: {json.dumps({'type': 'thinking', 'message': '正在通过推荐引擎分析...'}, ensure_ascii=False)}\n\n"
        
        # 使用推荐引擎生成歌单(替代已废弃的 Spotify 服务)
        result = await agent.get_recommendations(
            query=query,
            user_preferences=user_preferences or {}
        )
        
        if result.get("success") and result.get("recommendations"):
            raw_songs = result["recommendations"]
            songs = getattr(raw_songs, "data", raw_songs)
            if not isinstance(songs, list):
                songs = []
            yield f"data: {json.dumps({'type': 'songs_start', 'count': len(songs)}, ensure_ascii=False)}\n\n"
            for i, song in enumerate(songs):
                song_data = song.get("song", song) if isinstance(song, dict) else song
                yield f"data: {json.dumps({'type': 'song', 'song': song_data, 'index': i, 'total': len(songs)}, ensure_ascii=False)}\n\n"
                await asyncio.sleep(0.05)
            yield f"data: {json.dumps({'type': 'songs_complete'}, ensure_ascii=False)}\n\n"
        
        yield f"data: {json.dumps({'type': 'complete', 'success': True}, ensure_ascii=False)}\n\n"
        
    except Exception as e:
        logger.error(f"流式歌单生成失败: {str(e)}", exc_info=True)
        yield f"data: {json.dumps({'type': 'error', 'message': str(e)}, ensure_ascii=False)}\n\n"


async def stream_journey(
    story: Optional[str] = None,
    mood_transitions: Optional[List[Dict[str, Any]]] = None,
    duration: int = 60,
    user_preferences: Optional[Dict[str, Any]] = None,
    context: Optional[Dict[str, Any]] = None
) -> AsyncGenerator[str, None]:
    """
    流式生成音乐旅程 - 委托给 music_journey.stream_journey_events()
    """
    try:
        logger.info(f"[Journey SSE] ✅ 开始处理旅程: story={story!r}, "
                    f"mood_transitions={mood_transitions}, duration={duration}")
        yield f"data: {json.dumps({'type': 'journey_start', 'message': '正在连接旅程引擎...'}, ensure_ascii=False)}\n\n"

        from retrieval.music_journey import stream_journey_events
        logger.info("[Journey SSE] ✅ music_journey 模块导入成功")

        event_count = 0
        async for event in stream_journey_events(
            story=story,
            mood_transitions=mood_transitions,
            duration=duration,
            context=context,
        ):
            event_count += 1
            logger.info(f"[Journey SSE] 事件 #{event_count}: type={event.get('type')}")
            yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
            await asyncio.sleep(0.05)

        logger.info(f"[Journey SSE] ✅ 旅程生成完成,共 {event_count} 个事件")

    except Exception as e:
        logger.error(f"[Journey SSE] ❌ 流式旅程生成失败: {str(e)}", exc_info=True)
        yield f"data: {json.dumps({'type': 'error', 'error': str(e)}, ensure_ascii=False)}\n\n"


@app.get("/")
async def root():
    """健康检查"""
    return {"status": "ok", "service": "Music Recommendation API"}


@app.get("/health")
async def health():
    """健康检查"""
    return {"status": "healthy"}


@app.post("/api/recommendations/stream")
async def get_stream_recommendations(request: RecommendationRequest):
    """
    流式获取音乐推荐
    
    SSE流式接口,会逐步发送分析进度和结果
    """
    return StreamingResponse(
        stream_recommendations(
            query=request.query,
            genre=request.genre,
            mood=request.mood,
            user_preferences=request.user_preferences,
            chat_history=request.chat_history,
            llm_provider=request.llm_provider,
            web_search_enabled=request.web_search_enabled
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


@app.post("/api/playlist/stream")
async def stream_playlist_endpoint(request: PlaylistRequest):
    """
    流式生成歌单(SSE)
    """
    return StreamingResponse(
        stream_playlist(
            query=request.query,
            target_size=request.target_size,
            public=request.public,
            user_preferences=request.user_preferences
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


@app.post("/api/recommendations")
async def get_recommendations(request: RecommendationRequest):
    """
    获取音乐推荐(非流式,兼容旧接口)
    """
    try:
        # 根据前端传入的 provider 动态切换 LLM
        try:
            from llms.multi_llm import get_chat_model
            from agent.music_graph import set_llm
            new_llm = get_chat_model(provider=request.llm_provider)
            set_llm(new_llm)
            logger.info(f"切换 LLM provider 到 {request.llm_provider}")
        except Exception as e:
            logger.warning(f"切换 LLM 失败,使用默认配置: {e}")

        # 通过环境变量传递联网搜索开关
        os.environ["MUSIC_WEB_SEARCH_ENABLED"] = "1" if request.web_search_enabled else "0"

        agent = get_agent()
        result = await agent.get_recommendations(
            query=request.query,
            user_preferences=request.user_preferences,
            chat_history=request.chat_history
        )
        return result
    except Exception as e:
        logger.error(f"获取推荐失败: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/playlist")
async def generate_playlist(request: PlaylistRequest):
    """
    生成歌单(使用推荐引擎,替代废弃的 Spotify 服务)
    """
    try:
        agent = get_agent()
        result = await agent.get_recommendations(
            query=request.query,
            user_preferences=request.user_preferences or {}
        )
        return {"success": result.get("success", False), **result}
    except Exception as e:
        logger.error(f"生成歌单失败: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/journey/stream")
async def stream_journey_endpoint(request: JourneyRequest):
    """
    流式生成音乐旅程(SSE)
    """
    import datetime
    print(f"\n🔥🔥🔥 [Journey Endpoint] CALLED at {datetime.datetime.now()} "
          f"story={request.story!r} duration={request.duration}\n", flush=True)
    return StreamingResponse(
        stream_journey(
            story=request.story,
            mood_transitions=request.mood_transitions,
            duration=request.duration,
            user_preferences=request.user_preferences,
            context=request.context
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


@app.post("/api/journey")
async def generate_journey(request: JourneyRequest):
    """
    生成音乐旅程(简化版,使用推荐引擎)
    """
    try:
        agent = get_agent()
        query = request.story or "生成一段音乐旅程"
        result = await agent.get_recommendations(
            query=query,
            user_preferences=request.user_preferences or {}
        )
        return result
    except Exception as e:
        logger.error(f"生成旅程失败: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/search")
async def search_music(request: SearchRequest):
    """
    搜索歌曲
    - 优先 TavilyAPI 在线搜索
    - 无结果时使用本地 JSON 数据库模糊匹配
    """
    try:
        search_tool = get_music_search_tool()
        songs = await search_tool.search_songs(
            query=request.query,
            genre=request.genre,
            limit=request.limit,
        )
        return {
            "success": True,
            "count": len(songs),
            "songs": [s.to_dict() for s in songs],
        }
    except Exception as e:
        logger.error(f"搜索歌曲失败: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ---- 设置管理 API ----
@app.get("/api/settings")
async def get_settings_endpoint():
    """
    返回当前所有可配置设置（供前端设置面板加载）。
    """
    from config.settings import settings
    import retrieval.hybrid_retrieval as hr

    return {
        # 模型配置
        "llm_default_provider": settings.llm_default_provider,
        "llm_default_model": settings.llm_default_model,
        "intent_llm_provider": settings.intent_llm_provider,
        "intent_llm_model": settings.intent_llm_model,
        "hyde_llm_provider": settings.hyde_llm_provider,
        "hyde_llm_model": settings.hyde_llm_model,
        "finetuned_model_path": settings.finetuned_model_path,
        "llm_timeout": settings.llm_timeout,
        # 路径配置
        "audio_data_dir": settings.audio_data_dir,
        "mtg_audio_dir": settings.mtg_audio_dir,
        "online_acquired_dir": settings.online_acquired_dir,
        # 检索参数
        "graph_search_limit": settings.graph_search_limit,
        "semantic_search_limit": settings.semantic_search_limit,
        "hybrid_retrieval_limit": settings.hybrid_retrieval_limit,
        "web_search_max_results": settings.web_search_max_results,
        # RRF 融合参数
        "rrf_weight_vector": hr.RRF_WEIGHT_VECTOR,
        "rrf_weight_graph": hr.RRF_WEIGHT_GRAPH,
        # 图距离参数
        "graph_affinity_enabled": hr.GRAPH_AFFINITY_ENABLED,
        "graph_affinity_weight": hr.GRAPH_AFFINITY_WEIGHT,
        "graph_affinity_max_hops": hr.GRAPH_AFFINITY_MAX_HOPS,
        # 记忆系统
        "memory_retain_rounds": settings.memory_retain_rounds,
        "default_user_id": settings.default_user_id,
    }


class SettingsUpdateRequest(BaseModel):
    """前端设置面板提交的配置更新"""
    # 所有字段都可选 —— 前端只发送修改了的字段
    llm_default_provider: str | None = None
    llm_default_model: str | None = None
    intent_llm_provider: str | None = None
    intent_llm_model: str | None = None
    hyde_llm_provider: str | None = None
    hyde_llm_model: str | None = None
    finetuned_model_path: str | None = None
    llm_timeout: int | None = None
    audio_data_dir: str | None = None
    mtg_audio_dir: str | None = None
    online_acquired_dir: str | None = None
    graph_search_limit: int | None = None
    semantic_search_limit: int | None = None
    hybrid_retrieval_limit: int | None = None
    web_search_max_results: int | None = None
    rrf_weight_vector: float | None = None
    graph_affinity_enabled: bool | None = None
    graph_affinity_weight: float | None = None
    graph_affinity_max_hops: int | None = None
    memory_retain_rounds: int | None = None
    default_user_id: str | None = None


@app.post("/api/settings")
async def update_settings_endpoint(request: SettingsUpdateRequest):
    """
    动态更新运行时设置（不重启服务）。
    前端只发送修改了的字段，未发送的字段保持不变。
    """
    from config.settings import settings
    import retrieval.hybrid_retrieval as hr

    updated_fields = []

    # 遍历请求中所有非 None 字段
    update_data = request.model_dump(exclude_none=True)
    for key, val in update_data.items():
        # RRF / GraphAffinity 参数 → 更新模块级常量
        if key == "rrf_weight_vector":
            hr.RRF_WEIGHT_VECTOR = val
            hr.RRF_WEIGHT_GRAPH = round(1.0 - val, 2)
            updated_fields.extend(["rrf_weight_vector", "rrf_weight_graph"])
        elif key == "graph_affinity_enabled":
            hr.GRAPH_AFFINITY_ENABLED = val
            updated_fields.append(key)
        elif key == "graph_affinity_weight":
            hr.GRAPH_AFFINITY_WEIGHT = val
            updated_fields.append(key)
        elif key == "graph_affinity_max_hops":
            hr.GRAPH_AFFINITY_MAX_HOPS = val
            updated_fields.append(key)
        # 其他字段 → 更新 pydantic settings 对象
        elif hasattr(settings, key):
            setattr(settings, key, val)
            updated_fields.append(key)

    # 如果切换了主 LLM provider，热切换模型
    if "llm_default_provider" in update_data or "llm_default_model" in update_data:
        try:
            from llms.multi_llm import get_chat_model
            from agent.music_graph import set_llm
            provider = update_data.get("llm_default_provider", settings.llm_default_provider)
            new_llm = get_chat_model(provider=provider)
            set_llm(new_llm)
            logger.info(f"[Settings] LLM 热切换至 {provider}")
        except Exception as e:
            logger.warning(f"[Settings] LLM 切换失败: {e}")

    logger.info(f"[Settings] 已更新配置: {updated_fields}")
    return {"success": True, "updated": updated_fields}


@app.post("/api/settings/reset")
async def reset_settings_endpoint():
    """
    还原所有配置为默认值（从环境变量 + 代码默认值重新加载）。
    """
    import config.settings as settings_module
    import retrieval.hybrid_retrieval as hr
    from config.settings import GlobalSettings

    # 重新实例化 settings（拾取 .env / 环境变量中的默认值）
    fresh = GlobalSettings()
    settings_module.settings = fresh

    # 重置 RRF / GraphAffinity 模块级常量
    hr.RRF_WEIGHT_VECTOR = fresh.rrf_weight_vector
    hr.RRF_WEIGHT_GRAPH = fresh.rrf_weight_graph
    hr.GRAPH_AFFINITY_ENABLED = fresh.graph_affinity_enabled
    hr.GRAPH_AFFINITY_WEIGHT = fresh.graph_affinity_weight
    hr.GRAPH_AFFINITY_MAX_HOPS = fresh.graph_affinity_max_hops

    # 返回新的完整设置给前端
    return {
        "success": True,
        "settings": {
            "llm_default_provider": fresh.llm_default_provider,
            "llm_default_model": fresh.llm_default_model,
            "intent_llm_provider": fresh.intent_llm_provider,
            "intent_llm_model": fresh.intent_llm_model,
            "hyde_llm_provider": fresh.hyde_llm_provider,
            "hyde_llm_model": fresh.hyde_llm_model,
            "finetuned_model_path": fresh.finetuned_model_path,
            "llm_timeout": fresh.llm_timeout,
            "audio_data_dir": fresh.audio_data_dir,
            "mtg_audio_dir": fresh.mtg_audio_dir,
            "online_acquired_dir": fresh.online_acquired_dir,
            "graph_search_limit": fresh.graph_search_limit,
            "semantic_search_limit": fresh.semantic_search_limit,
            "hybrid_retrieval_limit": fresh.hybrid_retrieval_limit,
            "web_search_max_results": fresh.web_search_max_results,
            "rrf_weight_vector": fresh.rrf_weight_vector,
            "rrf_weight_graph": fresh.rrf_weight_graph,
            "graph_affinity_enabled": fresh.graph_affinity_enabled,
            "graph_affinity_weight": fresh.graph_affinity_weight,
            "graph_affinity_max_hops": fresh.graph_affinity_max_hops,
            "memory_retain_rounds": fresh.memory_retain_rounds,
            "default_user_id": fresh.default_user_id,
        },
    }


# ---- 新增:行为事件请求模型 ----
class UserEventRequest(BaseModel):
    """用户行为事件"""
    event_type: str           # like / unlike / save / unsave / skip / full_play / repeat
    song_title: str           # 歌曲名
    artist: str = "未知"      # 歌手
    user_id: str = "local_admin"
    extra: Optional[str] = None  # 额外信息(如播放时长)


# ---- 新增:行为事件转自然语言 ----
EVENT_TEMPLATES = {
    "like":      "用户对《{title}》{artist} 点了赞,表示喜欢这首歌",
    "unlike":    "用户取消了对《{title}》{artist} 的点赞,可能不再感兴趣",
    "save":      "用户收藏了《{title}》{artist},非常喜欢这首歌",
    "unsave":    "用户取消了《{title}》{artist} 的收藏",
    "skip":      "用户在播放《{title}》{artist} 时迅速跳过了,可能不喜欢",
    "full_play": "用户完整听完了《{title}》{artist},表示认可",
    "repeat":    "用户反复播放了《{title}》{artist},非常喜欢这首歌",
    "dislike":   "用户明确表示不喜欢《{title}》{artist}",
}


@app.post("/api/user-event")
async def capture_user_event(request: UserEventRequest):
    """
    接收前端行为事件，直写 Neo4j 用户关系 + 异步送 GraphZep 补充上下文。

    关系类型与权重:
      like     → LIKES (weight=1.0)    点赞 = 显式正向信号
      save     → SAVES (weight=0.8)    收藏 = 组织性信号（略低于点赞）
      repeat   → LIKES (weight+0.5)    循环播放 = 最强隐式信号
      unlike   → 删除 LIKES
      unsave   → 删除 SAVES
      skip     → SKIPPED (count++)     跳过 = 弱负向（>=3次才降权）
      dislike  → DISLIKES              明确不喜欢 = 推荐时排除
      full_play→ LISTENED_TO (count++) 完整播放 = 隐式正向
    """
    try:
        from retrieval.user_memory import UserMemoryManager
        memory = UserMemoryManager()
        memory.ensure_user_exists(request.user_id)

        # ① 直写 Neo4j 关系（精确、快速、0.1s 内完成）
        event = request.event_type
        title = request.song_title
        artist = request.artist

        if event == "like":
            memory.record_liked_song(request.user_id, title, artist)
        elif event == "save":
            memory.record_saved_song(request.user_id, title, artist)
        elif event == "repeat":
            # 循环播放：先确保 LIKES 存在，再额外加权
            memory.record_liked_song(request.user_id, title, artist)
        elif event == "unlike":
            memory.remove_like(request.user_id, title, artist)
        elif event == "unsave":
            memory.remove_save(request.user_id, title, artist)
        elif event == "skip":
            memory.record_skipped(request.user_id, title, artist)
        elif event == "dislike":
            memory.record_dislike(request.user_id, title, artist)
        elif event == "full_play":
            memory.record_listened_song(request.user_id, title, artist)

        # ② GraphZep 异步写入（仅作为补充上下文，不作为主记忆源）
        template = EVENT_TEMPLATES.get(
            event,
            "用户对《{title}》{artist} 执行了" + event + " 操作"
        )
        description = template.format(title=title, artist=artist)

        from services.graphzep_client import get_graphzep_client
        client = get_graphzep_client()
        asyncio.create_task(
            client.add_user_event(event_description=description)
        )

        return {"success": True, "event_recorded": description}

    except Exception as e:
        logger.error(f"行为事件记录失败: {e}")
        return {"success": False, "error": str(e)}


if __name__ == "__main__":
    import uvicorn
    
    port = int(os.getenv("API_PORT", "8501"))
    uvicorn.run(
        "api.server:app",
        host="0.0.0.0",
        port=port,
        reload=True,
        log_level="info"
    )

