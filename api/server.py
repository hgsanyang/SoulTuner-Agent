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

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel


from config.logging_config import get_logger
from agent.music_agent import MusicRecommendationAgent
from api.security import (
    reject_shared_safe_action,
    require_admin_api_key,
    safe_resolve_child,
    safe_static_url_to_path,
    shared_safe_mode_enabled,
)

logger = get_logger(__name__)

app = FastAPI(title="Music Recommendation API", version="1.0.0")

# 注册用户画像路由
from api.user_profile import router as user_profile_router
app.include_router(user_profile_router)

# 注册动态用户画像路由（Profile Synthesizer）
from api.user_portrait import router as user_portrait_router
app.include_router(user_portrait_router)

@app.on_event("startup")
async def startup_event():
    """在服务器启动时预加载关键组件，避免首次请求冷启动延迟"""
    import time as _t
    _t0 = _t.time()
    logger.info("🚀 开始预加载关键组件...")
    if os.getenv("MUSIC_MOCK_MODE", "0").lower() in {"1", "true", "yes"}:
        get_agent()
        logger.info("🧪 Mock 模式：跳过模型、Neo4j、GraphZep 与 KV Cache 预热")
        return
    
    # 1. CPU profile预加载M2D；GPU profile先让MuQ按需进入显存，避免两个大模型的加载峰值叠加。
    loop = asyncio.get_running_loop()
    preload_m2d = os.getenv("MUSIC_M2D_MODEL_WARMUP", "true").lower() not in {"0", "false", "off"}
    if preload_m2d:
        try:
            from retrieval.audio_embedder import get_m2d2_model, encode_text_to_embedding

            await loop.run_in_executor(None, get_m2d2_model)
            logger.info(f"  ✅ M2D-CLAP 音频骨干预加载完成 ({_t.time()-_t0:.1f}s)")

            async def _warmup_text_encoder_background():
                try:
                    await loop.run_in_executor(None, lambda: encode_text_to_embedding("warmup"))
                    logger.info(f"  ✅ M2D-CLAP 文本编码器后台预热完成 ({_t.time()-_t0:.1f}s)")
                except Exception as warmup_error:
                    logger.warning(f"  ⚠️ M2D-CLAP 文本编码器后台预热失败，向量召回将按路超时降级: {warmup_error}")

            if os.getenv("MUSIC_M2D_TEXT_WARMUP", "background").lower() not in {"0", "false", "off"}:
                asyncio.create_task(_warmup_text_encoder_background())
                logger.info("  🔄 M2D-CLAP 文本编码器后台预热已启动")
        except Exception as e:
            logger.error(f"  ❌ M2D-CLAP 模型预加载失败: {e}")
    else:
        logger.info("  ⏭️ M2D-CLAP 启动预热已关闭，将在语义精排首次使用时按需加载")
    
    # 2. 预初始化 Agent 实例（编译 LangGraph 工作流 + MemorySaver）
    try:
        _agent_inst = get_agent()
        logger.info(f"  ✅ Agent 实例预初始化完成 ({_t.time()-_t0:.1f}s)")
    except Exception as e:
        logger.error(f"  ❌ Agent 预初始化失败: {e}")
    
    # 3. 预热 Neo4j 连接池（首次查询建立 TCP 连接 + Bolt 握手）
    try:
        from retrieval.neo4j_client import get_neo4j_client
        neo4j = get_neo4j_client()
        if neo4j and neo4j.driver:
            await loop.run_in_executor(None, lambda: neo4j.execute_query("RETURN 1 AS warmup", {}))
            logger.info(f"  ✅ Neo4j 连接预热完成 ({_t.time()-_t0:.1f}s)")
            
            # ★ Thompson Sampling 时间衰减：每次重启服务，ts_beta 衰减 20%
            # 这样长时间没被推荐的歌自然"恢复"，避免被永久封杀
            # 同时确保所有 Song 节点都有 ts_alpha/ts_beta 属性
            ts_decay_query = """
            MATCH (s:Song)
            SET s.ts_alpha = coalesce(s.ts_alpha, 1),
                s.ts_beta  = CASE 
                    WHEN s.ts_beta IS NOT NULL AND s.ts_beta > 1 
                    THEN s.ts_beta * 0.8 
                    ELSE 1 
                END
            RETURN count(s) AS total,
                   avg(s.ts_beta) AS avg_beta
            """
            ts_result = await loop.run_in_executor(
                None, lambda: neo4j.execute_query(ts_decay_query, {})
            )
            if ts_result:
                r = ts_result[0]
                logger.info(
                    f"  ✅ TS 时间衰减完成: {r['total']} 首歌, "
                    f"avg(beta)={r['avg_beta']:.2f} ({_t.time()-_t0:.1f}s)"
                )
    except Exception as e:
        logger.warning(f"  ⚠️ Neo4j 预热失败（不影响启动）: {e}")
    
    # 4. 加载用户画像缓存（从 Neo4j 读取上次保存的画像）
    try:
        from services.profile_synthesizer import get_profile_synthesizer
        synth = get_profile_synthesizer()
        portrait = await synth.load_portrait()
        if portrait:
            logger.info(f"  ✅ 用户画像已加载 | confidence={portrait.confidence} ({_t.time()-_t0:.1f}s)")
        else:
            logger.info(f"  ℹ️ 未找到历史画像，将在对话后自动生成")
    except Exception as e:
        logger.warning(f"  ⚠️ 用户画像加载失败（不影响启动）: {e}")
    
    # 5. 异步预热 KV Prefix Cache（后台执行，不阻塞服务就绪）
    try:
        _agent_inst = get_agent()
        asyncio.create_task(_agent_inst.graph.warmup_kv_cache())
        logger.info(f"  🔥 KV Cache 预热已启动（后台运行）")
    except Exception as e:
        logger.warning(f"  ⚠️ KV Cache 预热启动失败: {e}")
    
    logger.info(f"🏁 预加载完成，总耗时 {_t.time()-_t0:.1f}s")

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
    allow_credentials=not shared_safe_mode_enabled(),
    allow_methods=["GET", "POST", "OPTIONS"] if shared_safe_mode_enabled() else ["*"],
    allow_headers=["*"],
)

# 挂载静态音频/封面/歌词文件目录
# Docker 内为 /app/data，本地开发为 Windows 路径
from fastapi.staticfiles import StaticFiles
from starlette.middleware.cors import CORSMiddleware as _StarletteCORS

# StaticFiles 是独立的 ASGI sub-app，主 app 的 CORSMiddleware 不覆盖它。
# 必须单独包裹，否则 <audio> 和 fetch(lyrics) 会被浏览器 CORS 拦截。
_CORS_ORIGINS = [
    "http://localhost:3000", "http://127.0.0.1:3000",
    "http://localhost:3003", "http://127.0.0.1:3003",
    "http://localhost:31000", "http://127.0.0.1:31000",
]

def _cors_static(directory: str) -> CORSMiddleware:
    """给 StaticFiles 包一层 CORS，让 <audio> / fetch 跨域正常工作"""
    static_app = StaticFiles(directory=directory)
    return _StarletteCORS(
        static_app,
        allow_origins=_CORS_ORIGINS,
        allow_methods=["GET", "HEAD", "OPTIONS"],
        allow_headers=["*"],
    )

_DATA_ROOT = Path(os.environ.get("MUSIC_DATA_ROOT", r"C:\Users\sanyang\sanyangworkspace\music_recommendation\data"))
PROCESSED_AUDIO_ROOT = _DATA_ROOT / "processed_audio"
audio_dir = PROCESSED_AUDIO_ROOT / "audio"
if audio_dir.exists():
    app.mount("/static/audio", _cors_static(str(audio_dir)), name="audio")
    logger.info(f"✅ 音频静态文件挂载成功: {audio_dir}")
else:
    logger.warning(f"音频目录不存在,无法提供静态音频挂载: {audio_dir}")
cover_dir = PROCESSED_AUDIO_ROOT / "covers"
if cover_dir.exists():
    app.mount("/static/covers", _cors_static(str(cover_dir)), name="covers")
    logger.info(f"✅ 封面静态文件挂载成功: {cover_dir}")
else:
    logger.warning(f"封面目录不存在,无法提供静态封面挂载: {cover_dir}")
lyrics_dir = PROCESSED_AUDIO_ROOT / "lyrics"
if lyrics_dir.exists():
    app.mount("/static/lyrics", _cors_static(str(lyrics_dir)), name="lyrics")
    logger.info(f"✅ 歌词静态文件挂载成功: {lyrics_dir}")
else:
    logger.warning(f"歌词目录不存在,无法提供静态歌词挂载: {lyrics_dir}")

# MTG 数据集音频：使用显式路由（避免 StaticFiles 挂载顺序问题）
MTG_AUDIO_DIR = _DATA_ROOT / "mtg_sample" / "audio"

from fastapi.responses import FileResponse

@app.get("/static/mtg_audio/{filename:path}")
async def serve_mtg_audio(filename: str):
    """提供 MTG 数据集音频文件"""
    try:
        file_path = safe_resolve_child(MTG_AUDIO_DIR, filename)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid MTG audio path")
    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"MTG audio not found: {filename}")
    return FileResponse(str(file_path), media_type="audio/mpeg")


# 联网获取的音频/封面/歌词(独立目录 data/online_acquired/)
ONLINE_AUDIO_ROOT = _DATA_ROOT / "online_acquired"
online_audio_dir = ONLINE_AUDIO_ROOT / "audio"
online_audio_dir.mkdir(parents=True, exist_ok=True)
app.mount("/static/online_audio", _cors_static(str(online_audio_dir)), name="online_audio")
logger.info(f"✅ 联网音频静态文件挂载: {online_audio_dir}")
online_cover_dir = ONLINE_AUDIO_ROOT / "covers"
online_cover_dir.mkdir(parents=True, exist_ok=True)
app.mount("/static/online_covers", _cors_static(str(online_cover_dir)), name="online_covers")
logger.info(f"✅ 联网封面静态文件挂载: {online_cover_dir}")
online_lyrics_dir = ONLINE_AUDIO_ROOT / "lyrics"
online_lyrics_dir.mkdir(parents=True, exist_ok=True)
app.mount("/static/online_lyrics", _cors_static(str(online_lyrics_dir)), name="online_lyrics")
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
    dialog_state: Optional[Dict[str, Any]] = None
    user_id: str = "local_admin"
    llm_provider: str = "dashscope"            # 模型供应商: dashscope / siliconflow / google / ...
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
    llm_provider: str = "dashscope"  # 模型提供商，和推荐页保持一致


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
async def acquire_song_endpoint(
    request: AcquireSongRequest,
    _: None = Depends(require_admin_api_key),
):
    """
    下载单首歌曲的音频/歌词/封面到本地待入库目录。
    不再自动入库 Neo4j，需要用户在待入库页面确认后才入库。
    """
    reject_shared_safe_action("acquire song")
    import aiohttp
    from tools.acquire_music import OnlineMusicAcquirer

    query = f"{request.title} {request.artist}"
    logger.info(f"🎯 [acquire-song] 用户请求下载到待入库: {query}")

    acquirer = OnlineMusicAcquirer()
    async with aiohttp.ClientSession() as session:
        acquired = await acquirer.search_and_acquire([query], session)

    if not acquired:
        raise HTTPException(status_code=404, detail="未能获取该歌曲的音频资源(可能因版权限制)")

    song = acquired[0]
    logger.info(f"✅ [acquire-song] 已下载到待入库: {song['title']} - {song['artist']}")
    return {
        "success": True,
        "message": f"已将《{song['title']}》下载到待入库",
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
    dialog_state: Optional[Dict[str, Any]] = None,
    user_id: str = "local_admin",
    web_search_enabled: bool = True,
    is_disconnected=None,
) -> AsyncGenerator[str, None]:
    """
    流式生成推荐结果 (真流式：推荐解释逐 chunk 推送)
    
    Args:
        is_disconnected: 可选的异步回调，检测客户端是否已断开连接。
                         由 FastAPI 端点通过 request.is_disconnected 注入。
    Yields:
        SSE格式的数据块
    """
    try:
        agent = get_agent()
        
        # 根据 settings 配置初始化 LLM（provider/model 统一由设置面板管理）
        try:
            from llms.multi_llm import get_chat_model, get_intent_chat_model, get_explain_chat_model
            from agent.music_graph import set_llm, set_intent_llm, set_explain_llm
            from config.settings import settings as _req_settings
            
            _provider = _req_settings.llm_default_provider or "dashscope"
            _model = _req_settings.llm_default_model
            
            new_llm = get_chat_model(provider=_provider, model_name=_model)
            set_llm(new_llm)
            
            # 同步切换意图分析 LLM（如果没有独立配置，跟随主模型）
            if not _req_settings.intent_llm_model:
                new_intent = get_intent_chat_model()
                set_intent_llm(new_intent)
            
            # 同步切换解释生成 LLM（如果没有独立配置，跟随主模型）
            if not _req_settings.explain_llm_model:
                new_explain = get_explain_chat_model()
                set_explain_llm(new_explain)
            
            logger.info(f"LLM 初始化: {_provider} / {_model}")
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
            user_preferences=user_preferences,
            dialog_state=dialog_state,
            user_id=user_id,
        ):
            # ★ 检测客户端是否已断开（用户点击了"停止生成"）
            if is_disconnected:
                try:
                    disconnected = await is_disconnected()
                except Exception:
                    disconnected = False
                if disconnected:
                    logger.info("🛑 [SSE] 检测到客户端断开连接，停止推送并取消后台任务")
                    # 取消 agent 内部的 graph_task
                    _internal_task = getattr(agent, '_current_graph_task', None)
                    if _internal_task and not _internal_task.done():
                        _internal_task.cancel()
                        logger.info("🛑 [SSE] 后台 graph_task 已取消")
                    return

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

            elif event_type == "clarification_required":
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
                
            elif event_type == "complete":
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
                
            elif event_type == "error":
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
        
    except asyncio.CancelledError:
        logger.info("🛑 [SSE] 流式推荐被取消")
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
async def get_stream_recommendations(request: RecommendationRequest, raw_request: Request):
    """
    流式获取音乐推荐
    
    SSE流式接口,会逐步发送分析进度和结果。
    注入 raw_request.is_disconnected 以支持客户端断开时取消后台任务。
    """
    return StreamingResponse(
        stream_recommendations(
            query=request.query,
            genre=request.genre,
            mood=request.mood,
            user_preferences=request.user_preferences,
            chat_history=request.chat_history,
            dialog_state=request.dialog_state,
            user_id=request.user_id,
            web_search_enabled=request.web_search_enabled,
            is_disconnected=raw_request.is_disconnected,
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
            chat_history=request.chat_history,
            dialog_state=request.dialog_state,
            user_id=request.user_id,
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

    # 根据前端传入的 provider 切换 LLM（与推荐页保持一致）
    try:
        from llms.multi_llm import get_chat_model
        from agent.music_graph import set_llm
        new_llm = get_chat_model(provider=request.llm_provider)
        set_llm(new_llm)
        logger.info(f"[Journey] 切换 LLM provider 到 {request.llm_provider}")
    except Exception as _e:
        logger.warning(f"[Journey] 切换 LLM 失败，使用默认: {_e}")

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

    return {
        # 模型配置
        "llm_default_provider": settings.llm_default_provider,
        "llm_default_model": settings.llm_default_model,
        "intent_llm_provider": settings.intent_llm_provider,
        "intent_llm_model": settings.intent_llm_model,
        "hyde_llm_provider": settings.hyde_llm_provider,
        "hyde_llm_model": settings.hyde_llm_model,
        "explain_llm_provider": settings.explain_llm_provider,
        "explain_llm_model": settings.explain_llm_model,
        "compress_llm_provider": settings.compress_llm_provider,
        "compress_llm_model": settings.compress_llm_model,
        "explanation_mode": settings.explanation_mode,
        "explanation_fast_mode": settings.explanation_fast_mode,
        "context_total_budget": settings.context_total_budget,
        "intent_max_tokens": settings.intent_max_tokens,
        "finetuned_model_path": settings.finetuned_model_path,
        "llm_timeout": settings.llm_timeout,
        # 路径配置
        "audio_data_dir": settings.audio_data_dir,
        "mtg_audio_dir": settings.mtg_audio_dir,
        "online_acquired_dir": settings.online_acquired_dir,
        # 检索参数
        "graph_search_limit": settings.graph_search_limit,
        "semantic_search_limit": settings.semantic_search_limit,
        "mixed_retrieval_limit": settings.mixed_retrieval_limit,
        "hybrid_retrieval_limit": settings.hybrid_retrieval_limit,
        "web_search_max_results": settings.web_search_max_results,
        # 粗排 & 探索
        "graph_affinity_enabled": settings.graph_affinity_enabled,
        "graph_affinity_max_hops": settings.graph_affinity_max_hops,
        "coarse_cut_ratio": settings.coarse_cut_ratio,
        "exploration_ratio": settings.exploration_ratio,
        # 三锚精排权重
        "tri_anchor_w_semantic": settings.tri_anchor_w_semantic,
        "tri_anchor_w_acoustic": settings.tri_anchor_w_acoustic,
        "tri_anchor_w_personal": settings.tri_anchor_w_personal,
        # 多样性参数
        "max_songs_per_artist": settings.max_songs_per_artist,
        "mmr_lambda": settings.mmr_lambda,
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
    compress_llm_provider: str | None = None
    compress_llm_model: str | None = None
    explain_llm_provider: str | None = None
    explain_llm_model: str | None = None
    explanation_mode: str | None = None
    explanation_fast_mode: bool | None = None
    context_total_budget: int | None = None
    intent_max_tokens: int | None = None
    finetuned_model_path: str | None = None
    llm_timeout: int | None = None
    audio_data_dir: str | None = None
    mtg_audio_dir: str | None = None
    online_acquired_dir: str | None = None
    graph_search_limit: int | None = None
    semantic_search_limit: int | None = None
    mixed_retrieval_limit: int | None = None
    hybrid_retrieval_limit: int | None = None
    web_search_max_results: int | None = None
    graph_affinity_enabled: bool | None = None
    graph_affinity_max_hops: int | None = None
    coarse_cut_ratio: float | None = None
    exploration_ratio: float | None = None
    tri_anchor_w_semantic: float | None = None
    tri_anchor_w_acoustic: float | None = None
    tri_anchor_w_personal: float | None = None
    max_songs_per_artist: int | None = None
    mmr_lambda: float | None = None
    memory_retain_rounds: int | None = None
    default_user_id: str | None = None


@app.post("/api/settings")
async def update_settings_endpoint(request: SettingsUpdateRequest):
    """
    动态更新运行时设置（不重启服务）。
    前端只发送修改了的字段，未发送的字段保持不变。
    """
    from config.settings import settings

    updated_fields = []

    # 遍历请求中所有非 None 字段，统一更新 settings 对象
    update_data = request.model_dump(exclude_none=True)
    for key, val in update_data.items():
        if hasattr(settings, key):
            setattr(settings, key, val)
            updated_fields.append(key)

    # 如果切换了主 LLM provider，热切换模型
    if "llm_default_provider" in update_data or "llm_default_model" in update_data:
        try:
            from llms.multi_llm import get_chat_model
            from agent.music_graph import set_llm
            provider = update_data.get("llm_default_provider", settings.llm_default_provider)
            model = update_data.get("llm_default_model", settings.llm_default_model)
            new_llm = get_chat_model(provider=provider, model_name=model)
            set_llm(new_llm)
            logger.info(f"[Settings] LLM 热切换至 {provider} / {model}")
        except Exception as e:
            logger.warning(f"[Settings] LLM 切换失败: {e}")

    # 如果切换了意图分析 LLM（或主模型变更且意图使用主模型，或 max_tokens 变更），热切换意图模型
    _intent_changed = ("intent_llm_provider" in update_data or "intent_llm_model" in update_data
                       or "intent_max_tokens" in update_data
                       or "llm_default_provider" in update_data or "llm_default_model" in update_data)
    if _intent_changed:
        try:
            from llms.multi_llm import get_intent_chat_model
            from agent.music_graph import set_intent_llm
            new_intent_llm = get_intent_chat_model()
            set_intent_llm(new_intent_llm)
            _intent_model = settings.intent_llm_model or settings.llm_default_model
            logger.info(f"[Settings] 意图分析 LLM 热切换至 {_intent_model}")
        except Exception as e:
            logger.warning(f"[Settings] 意图分析 LLM 切换失败: {e}")

    # 如果切换了解释生成 LLM，或主模型变更且解释复用主模型，热切换
    _explain_needs_update = ("explain_llm_provider" in update_data or "explain_llm_model" in update_data)
    if not _explain_needs_update and ("llm_default_provider" in update_data or "llm_default_model" in update_data):
        # 解释模型没有单独配置 → 复用主模型 → 主模型变了就要跟着更新
        if not settings.explain_llm_model:
            _explain_needs_update = True
    if _explain_needs_update:
        try:
            from llms.multi_llm import get_explain_chat_model
            from agent.music_graph import set_explain_llm
            new_explain_llm = get_explain_chat_model()
            set_explain_llm(new_explain_llm)
            _explain_model = settings.explain_llm_model or settings.llm_default_model
            logger.info(f"[Settings] 解释生成 LLM 热切换至 {_explain_model}")
        except Exception as e:
            logger.warning(f"[Settings] 解释生成 LLM 切换失败: {e}")

    logger.info(f"[Settings] 已更新配置: {updated_fields}")

    # 持久化到 JSON 文件，重启后自动恢复
    try:
        from config.settings import save_user_settings
        save_user_settings(settings, updated_fields)
        logger.info(f"[Settings] 已持久化到 user_settings.json")
    except Exception as e:
        logger.warning(f"[Settings] 持久化失败: {e}")

    return {"success": True, "updated": updated_fields}


@app.post("/api/settings/reset")
async def reset_settings_endpoint():
    """
    还原所有配置为默认值（从环境变量 + 代码默认值重新加载）。
    """
    import config.settings as settings_module
    from config.settings import GlobalSettings

    # 重新实例化 settings（拾取 .env / 环境变量中的默认值）
    fresh = GlobalSettings()
    settings_module.settings = fresh

    # 清除持久化文件
    try:
        from config.settings import clear_user_settings
        clear_user_settings()
        logger.info("[Settings] 已清除 user_settings.json")
    except Exception as e:
        logger.warning(f"[Settings] 清除持久化文件失败: {e}")

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
            "compress_llm_provider": fresh.compress_llm_provider,
            "compress_llm_model": fresh.compress_llm_model,
            "explain_llm_provider": fresh.explain_llm_provider,
            "explain_llm_model": fresh.explain_llm_model,
            "explanation_mode": fresh.explanation_mode,
            "explanation_fast_mode": fresh.explanation_fast_mode,
            "context_total_budget": fresh.context_total_budget,
            "intent_max_tokens": fresh.intent_max_tokens,
            "finetuned_model_path": fresh.finetuned_model_path,
            "llm_timeout": fresh.llm_timeout,
            "audio_data_dir": fresh.audio_data_dir,
            "mtg_audio_dir": fresh.mtg_audio_dir,
            "online_acquired_dir": fresh.online_acquired_dir,
            "graph_search_limit": fresh.graph_search_limit,
            "semantic_search_limit": fresh.semantic_search_limit,
            "mixed_retrieval_limit": fresh.mixed_retrieval_limit,
            "hybrid_retrieval_limit": fresh.hybrid_retrieval_limit,
            "web_search_max_results": fresh.web_search_max_results,
            "graph_affinity_enabled": fresh.graph_affinity_enabled,
            "graph_affinity_max_hops": fresh.graph_affinity_max_hops,
            "coarse_cut_ratio": fresh.coarse_cut_ratio,
            "exploration_ratio": fresh.exploration_ratio,
            "tri_anchor_w_semantic": fresh.tri_anchor_w_semantic,
            "tri_anchor_w_acoustic": fresh.tri_anchor_w_acoustic,
            "tri_anchor_w_personal": fresh.tri_anchor_w_personal,
            "max_songs_per_artist": fresh.max_songs_per_artist,
            "mmr_lambda": fresh.mmr_lambda,
            "memory_retain_rounds": fresh.memory_retain_rounds,
            "default_user_id": fresh.default_user_id,
        },
    }


# ---- 新增:行为事件请求模型 ----
class UserEventRequest(BaseModel):
    """用户行为事件"""
    event_type: str           # like/save/skip/full_play/repeat/dislike/play_start...
    song_title: str           # 歌曲名
    artist: str = "未知"      # 歌手
    user_id: str = "local_admin"
    exposure_id: Optional[str] = None
    extra: Optional[Any] = None  # position/play_duration_ms/progress_ratio/session_id


class SlateFeedbackRequest(BaseModel):
    """一整组推荐结果的反馈。"""
    exposure_id: str
    rating: str
    reasons: List[str] = []
    note: str = ""
    user_id: str = "local_admin"
    extra: Optional[Dict[str, Any]] = None


class RankingPolicyReplayRequest(BaseModel):
    """Generate a validation-gated ranking-policy candidate from feedback logs."""
    min_events: int = 20
    per_user_min_events: int = 30
    validation_ratio: float = 0.2
    slate_top_k: int = 5
    include_slate_feedback: bool = True
    write_candidate: bool = True


class MemoryPreferenceUpdateRequest(BaseModel):
    """Editable memory update from the user profile panel."""
    user_id: str = "local_admin"
    preferences: Dict[str, Any]


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
    "play_start": "用户开始播放《{title}》{artist}",
}

SUPPORTED_USER_EVENTS = set(EVENT_TEMPLATES) | {"unsave"}

SUPPORTED_SLATE_RATINGS = {
    "great",
    "partial",
    "off",
    "too_familiar",
    "more_discovery",
    "too_noisy",
    "too_quiet",
    "too_sad",
    "too_generic",
    "more_niche",
    "closer_to_seed",
    "wrong_context",
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
        if request.event_type not in SUPPORTED_USER_EVENTS:
            raise HTTPException(status_code=422, detail="Unsupported user event type")
        event = request.event_type
        title = request.song_title
        artist = request.artist

        # MemoryGateway 统一写入 Neo4j 热路径、JSONL 反馈日志和可选 GraphZep 旁路。
        from services.memory_gateway import get_memory_gateway

        result = await get_memory_gateway().remember_event(
            event_type=event,
            title=title,
            artist=artist,
            user_id=request.user_id,
            exposure_id=request.exposure_id,
            extra=request.extra if isinstance(request.extra, dict) else {"value": request.extra} if request.extra is not None else {},
        )
        if not result.success:
            raise HTTPException(status_code=422, detail=result.error or "user event failed")

        return {
            "success": True,
            "event_recorded": result.description,
            "feedback_event_id": result.feedback_event_id,
            "graphzep_scheduled": result.graphzep_scheduled,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"行为事件记录失败: {e}")
        return {"success": False, "error": str(e)}


@app.post("/api/slate-feedback")
async def capture_slate_feedback(request: SlateFeedbackRequest):
    """Record feedback for an entire recommendation slate.

    This is the A3-lite bridge between user-facing evaluation buttons and
    offline ranking replay. MemoryGateway also turns high-confidence slate
    feedback into conservative hot-path preference updates.
    """
    try:
        if not request.exposure_id.strip():
            raise HTTPException(status_code=422, detail="exposure_id is required")
        if request.rating not in SUPPORTED_SLATE_RATINGS:
            raise HTTPException(status_code=422, detail="Unsupported slate rating")

        from services.memory_gateway import get_memory_gateway

        result = await get_memory_gateway().remember_slate_feedback(
            exposure_id=request.exposure_id,
            rating=request.rating,
            reasons=request.reasons,
            note=request.note,
            user_id=request.user_id,
            extra=request.extra or {},
        )
        return {
            "success": True,
            "feedback_id": result.slate_feedback_id,
            "preference_update": result.preference_update,
            "graphzep_scheduled": result.graphzep_scheduled,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"歌单级反馈记录失败: {e}")
        return {"success": False, "error": str(e)}


@app.get("/api/ranking-policy/status")
async def ranking_policy_status():
    """Return non-sensitive A3 policy state for diagnostics and UI tooling."""
    from services.feedback_logger import load_jsonl
    from services.feedback_diagnostics import summarize_feedback_quality
    from services.ranking_policy import (
        ACTIVE_FILE,
        CANDIDATE_FILE,
        feedback_dir,
        policy_path,
        summarize_policy_readiness,
    )

    root = feedback_dir()
    exposures = load_jsonl(root / "exposures.jsonl")
    events = load_jsonl(root / "events.jsonl")
    slate_feedback = load_jsonl(root / "slate_feedback.jsonl")

    def _summary(path: Path) -> Dict[str, Any] | None:
        if not path.exists():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            global_model = payload.get("global") or {}
            return {
                "schema_version": payload.get("schema_version"),
                "status": payload.get("status"),
                "gate_passed": payload.get("gate_passed"),
                "created_at": payload.get("created_at"),
                "global_status": global_model.get("status"),
                "events": global_model.get("events"),
                "validation_events": global_model.get("validation_events"),
            }
        except Exception:
            return {"status": "invalid"}

    active = _summary(policy_path(ACTIVE_FILE))
    candidate = _summary(policy_path(CANDIDATE_FILE))
    readiness = summarize_policy_readiness(
        num_exposures=len(exposures),
        num_events=len(events),
        num_slate_feedback=len(slate_feedback),
        active=active,
        candidate=candidate,
    )
    feedback_quality = summarize_feedback_quality(
        exposures,
        events,
        slate_feedback,
    )

    return {
        "feedback_dir": str(root),
        "num_exposures": len(exposures),
        "num_events": len(events),
        "num_slate_feedback": len(slate_feedback),
        "active": active,
        "candidate": candidate,
        "readiness": readiness,
        "feedback_quality": feedback_quality,
    }


@app.get("/api/memory/profile")
async def memory_profile(user_id: str = "local_admin"):
    """Return editable hot-path memory and configured episodic sidecars."""
    try:
        from services.memory_gateway import get_memory_gateway

        return {
            "success": True,
            "memory": get_memory_gateway().explain_memory(user_id=user_id),
        }
    except Exception as e:
        logger.error(f"[MemoryAPI] 读取记忆画像失败: {e}")
        return {"success": False, "error": str(e)}


@app.post("/api/memory/preference")
async def add_memory_preference(request: MemoryPreferenceUpdateRequest):
    """Add structured preferences to the Neo4j hot path."""
    reject_shared_safe_action("update memory preference")
    try:
        from services.memory_gateway import get_memory_gateway

        result = get_memory_gateway().remember_preference(
            user_id=request.user_id,
            preferences=request.preferences,
        )
        return {
            "success": result.success,
            "preference_update": result.preference_update,
        }
    except Exception as e:
        logger.error(f"[MemoryAPI] 写入记忆偏好失败: {e}")
        return {"success": False, "error": str(e)}


@app.delete("/api/memory/preference")
async def delete_memory_preference(
    field: str,
    value: str,
    user_id: str = "local_admin",
):
    """Delete one learned preference item from the Neo4j hot path."""
    reject_shared_safe_action("delete memory preference")
    try:
        from services.memory_gateway import get_memory_gateway

        ok = get_memory_gateway().forget_preference_item(
            user_id=user_id,
            field=field,
            value=value,
        )
        if not ok:
            raise HTTPException(status_code=422, detail="Unsupported memory field or value")
        return {"success": True}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[MemoryAPI] 删除记忆偏好失败: {e}")
        return {"success": False, "error": str(e)}


@app.delete("/api/memory/profile")
async def clear_learned_memory_profile(
    user_id: str = "local_admin",
):
    """Clear learned hot-path preference fields, keeping manual profile and song relations."""
    reject_shared_safe_action("clear learned memory profile")
    try:
        from services.memory_gateway import get_memory_gateway

        ok = get_memory_gateway().clear_learned_preferences(user_id=user_id)
        if not ok:
            raise HTTPException(status_code=422, detail="Unable to clear learned preferences")
        return {"success": True}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[MemoryAPI] 清空学习记忆失败: {e}")
        return {"success": False, "error": str(e)}


@app.post("/api/ranking-policy/replay")
async def ranking_policy_replay(
    request: RankingPolicyReplayRequest,
    _: None = Depends(require_admin_api_key),
):
    """Fit a candidate ranking policy from logged exposures and feedback."""
    reject_shared_safe_action("learn ranking policy")
    from services.feedback_logger import load_jsonl
    from services.ranking_learning import learn_ranking_policy
    from services.ranking_policy import feedback_dir, write_candidate

    root = feedback_dir()
    exposures = load_jsonl(root / "exposures.jsonl")
    events = load_jsonl(root / "events.jsonl")
    slate_feedback = (
        load_jsonl(root / "slate_feedback.jsonl")
        if request.include_slate_feedback
        else []
    )
    report = learn_ranking_policy(
        exposures,
        events,
        slate_feedback=slate_feedback,
        min_events=max(1, int(request.min_events)),
        per_user_min_events=max(1, int(request.per_user_min_events)),
        validation_ratio=max(0.05, min(float(request.validation_ratio), 0.5)),
        slate_top_k=max(1, min(int(request.slate_top_k), 10)),
    )
    report["num_exposures"] = len(exposures)
    report["num_events"] = len(events)
    report["num_slate_feedback"] = len(slate_feedback)
    candidate_path = None
    if request.write_candidate:
        candidate_path = str(write_candidate(report, root))
    return {
        "success": True,
        "candidate_path": candidate_path,
        "report": report,
    }


@app.post("/api/ranking-policy/promote")
async def ranking_policy_promote(_: None = Depends(require_admin_api_key)):
    """Promote a gate-passed candidate policy to the active runtime policy."""
    reject_shared_safe_action("promote ranking policy")
    from services.ranking_policy import feedback_dir, promote_candidate

    path = promote_candidate(feedback_dir())
    return {"success": True, "active_path": str(path)}


@app.post("/api/ranking-policy/rollback")
async def ranking_policy_rollback(_: None = Depends(require_admin_api_key)):
    """Restore the previous active ranking policy."""
    reject_shared_safe_action("rollback ranking policy")
    from services.ranking_policy import feedback_dir, rollback_policy

    path = rollback_policy(feedback_dir())
    return {"success": True, "active_path": str(path)}


@app.get("/api/catalog-diagnostics")
async def catalog_diagnostics(limit: int = 50):
    """Return catalog-bias and recent-exposure diagnostics for the UI."""
    try:
        from retrieval.neo4j_client import get_neo4j_client
        from services.catalog_diagnostics import load_feedback_diagnostics, summarize_catalog_bias
        from services.ranking_policy import feedback_dir

        query = """
        MATCH (s:Song)
        OPTIONAL MATCH (s)-[:BELONGS_TO_GENRE]->(g:Genre)
        OPTIONAL MATCH (s)-[:HAS_MOOD]->(m:Mood)
        OPTIONAL MATCH (s)-[:HAS_THEME]->(t:Theme)
        OPTIONAL MATCH (s)-[:FITS_SCENARIO]->(sc:Scenario)
        RETURN s.music_id AS music_id,
               s.title AS title,
               s.audio_url AS audio_url,
               coalesce(s.language, 'Unknown') AS language,
               coalesce(s.source, 'local') AS source,
               s.muq_embedding IS NOT NULL AS has_muq_embedding,
               s.m2d2_embedding IS NOT NULL AS has_m2d2_embedding,
               collect(DISTINCT g.name) AS genres,
               collect(DISTINCT m.name) AS moods,
               collect(DISTINCT t.name) AS themes,
               collect(DISTINCT sc.name) AS scenarios
        """
        catalog_rows = [record for record in get_neo4j_client().execute_query(query)]
        exposures, slate_feedback = load_feedback_diagnostics(feedback_dir())
        report = summarize_catalog_bias(
            catalog_rows,
            exposures,
            slate_feedback,
            recent_exposure_limit=max(1, min(int(limit), 200)),
        )
        return {"success": True, **report}
    except Exception as e:
        logger.error(f"曲库诊断失败: {e}")
        return {"success": False, "error": str(e)}


# ================================================================
# 用户收藏 / 不喜欢 查询 API（供前端同步使用）
# ================================================================

@app.get("/api/liked-songs")
async def get_liked_songs(user_id: str = "local_admin", limit: int = 50):
    """
    查询用户点赞+收藏的歌曲列表（从 Neo4j 读取）。
    前端启动时调用此接口同步 liked songs 状态。
    """
    try:
        from retrieval.user_memory import UserMemoryManager
        memory = UserMemoryManager()
        songs = memory.get_liked_songs(user_id=user_id, limit=limit)
        return {"success": True, "songs": songs, "total": len(songs)}
    except Exception as e:
        logger.error(f"查询 liked songs 失败: {e}")
        return {"success": False, "songs": [], "error": str(e)}


@app.get("/api/disliked-songs")
async def get_disliked_songs(user_id: str = "local_admin", limit: int = 50):
    """
    查询用户标记为「不喜欢」的歌曲列表（从 Neo4j 读取）。
    供前端展示「不喜欢」管理页面。
    """
    try:
        from retrieval.neo4j_client import get_neo4j_client
        client = get_neo4j_client()
        query = """
        MATCH (u:User {id: $user_id})-[r:DISLIKES]->(s:Song)
        OPTIONAL MATCH (s)-[:PERFORMED_BY]->(a:Artist)
        RETURN s.title AS title, coalesce(a.name, s.artist, '未知') AS artist,
               s.audio_url AS audio_url, s.cover_url AS cover_url,
               s.album AS album,
               r.created_at AS disliked_at
        ORDER BY r.created_at DESC
        LIMIT $limit
        """
        results = client.execute_query(query, {"user_id": user_id, "limit": limit})
        songs = [
            {
                "title": r.get("title", ""),
                "artist": r.get("artist", ""),
                "audio_url": r.get("audio_url", ""),
                "cover_url": r.get("cover_url", ""),
                "album": r.get("album", ""),
                "disliked_at": r.get("disliked_at"),
            }
            for r in results
        ]
        return {"success": True, "songs": songs, "total": len(songs)}
    except Exception as e:
        logger.error(f"查询 disliked songs 失败: {e}")
        return {"success": False, "songs": [], "error": str(e)}


@app.delete("/api/disliked-songs")
async def remove_dislike(
    song_title: str,
    artist: str,
    user_id: str = "local_admin",
    _: None = Depends(require_admin_api_key),
):
    """从「不喜欢」列表中移除一首歌"""
    reject_shared_safe_action("remove disliked song")
    try:
        from retrieval.neo4j_client import get_neo4j_client
        client = get_neo4j_client()
        query = """
        MATCH (u:User {id: $user_id})-[r:DISLIKES]->(s:Song {title: $title})
        WHERE s.artist = $artist
              OR EXISTS((s)-[:PERFORMED_BY]->(:Artist {name: $artist}))
        DELETE r
        """
        client.execute_query(query, {"user_id": user_id, "title": song_title, "artist": artist})
        logger.info(f"用户 {user_id} 撤销不喜欢: {song_title}")
        return {"success": True}
    except Exception as e:
        logger.error(f"撤销不喜欢失败: {e}")
        return {"success": False, "error": str(e)}


# ================================================================
# 歌曲完整删除 API（图谱 + 文件系统）
# ================================================================

@app.delete("/api/songs")
async def delete_song_completely(
    song_title: str,
    artist: str,
    _: None = Depends(require_admin_api_key),
):
    """
    从系统中彻底删除一首歌，包括：
    1. Neo4j 图谱中的 Song 节点及所有关联边（保留 Artist 节点）
    2. 文件系统中的音频、封面、歌词、元数据文件

    ⚠️ 此操作不可逆！
    """
    deleted_files = []
    errors = []
    reject_shared_safe_action("delete song")

    try:
        from retrieval.neo4j_client import get_neo4j_client
        client = get_neo4j_client()

        # ── Step 1: 从 Neo4j 查询 Song 节点的文件路径信息 ──
        query_info = """
        MATCH (s:Song {title: $title})
        WHERE s.artist = $artist
              OR EXISTS((s)-[:PERFORMED_BY]->(:Artist {name: $artist}))
        RETURN s.music_id AS music_id,
               s.audio_url AS audio_url,
               s.cover_url AS cover_url,
               s.lrc_url AS lrc_url,
               s.dataset AS dataset
        LIMIT 1
        """
        info_result = client.execute_query(query_info, {"title": song_title, "artist": artist})

        if not info_result:
            raise HTTPException(status_code=404, detail=f"歌曲未找到: 《{song_title}》 - {artist}")

        song_info = info_result[0]
        music_id = song_info.get("music_id", "")
        audio_url = song_info.get("audio_url", "")
        cover_url = song_info.get("cover_url", "")
        lrc_url = song_info.get("lrc_url", "")
        dataset = song_info.get("dataset", "personal")

        logger.info(f"🗑️ 开始删除歌曲: 《{song_title}》 - {artist} (music_id={music_id}, dataset={dataset})")

        # ── Step 2: 从 Neo4j 删除 Song 节点及所有关联边 ──
        # DETACH DELETE 会自动删除所有关联的边（PERFORMED_BY, HAS_MOOD 等）
        # Artist 节点本身不会被删除（只删除边）
        delete_query = """
        MATCH (s:Song {title: $title})
        WHERE s.artist = $artist
              OR EXISTS((s)-[:PERFORMED_BY]->(:Artist {name: $artist}))
        DETACH DELETE s
        RETURN count(s) AS deleted_count
        """
        del_result = client.execute_query(delete_query, {"title": song_title, "artist": artist})
        deleted_count = del_result[0].get("deleted_count", 0) if del_result else 0

        if deleted_count == 0:
            raise HTTPException(status_code=404, detail=f"歌曲删除失败: 图谱中未找到匹配节点")

        logger.info(f"  ✅ Neo4j: 已删除 {deleted_count} 个 Song 节点及其关联边")

        # ── Step 3: 删除文件系统中的关联文件 ──
        def try_delete_file(file_path: Path, desc: str):
            """尝试删除文件，记录结果"""
            if file_path.exists():
                try:
                    file_path.unlink()
                    deleted_files.append(str(file_path))
                    logger.info(f"  🗑️ 已删除{desc}: {file_path}")
                except Exception as e:
                    errors.append(f"删除{desc}失败: {e}")
                    logger.warning(f"  ⚠️ 删除{desc}失败: {file_path} - {e}")

        # 根据 URL 路径推断文件系统位置
        def resolve_static_path(url: str) -> Path | None:
            """将 /static/xxx/yyy 格式的 URL 映射回文件系统路径"""
            try:
                return (
                    safe_static_url_to_path(url, "/static/audio/", PROCESSED_AUDIO_ROOT / "audio")
                    or safe_static_url_to_path(url, "/static/covers/", PROCESSED_AUDIO_ROOT / "covers")
                    or safe_static_url_to_path(url, "/static/lyrics/", PROCESSED_AUDIO_ROOT / "lyrics")
                    or safe_static_url_to_path(url, "/static/online_audio/", ONLINE_AUDIO_ROOT / "audio")
                    or safe_static_url_to_path(url, "/static/online_covers/", ONLINE_AUDIO_ROOT / "covers")
                    or safe_static_url_to_path(url, "/static/online_lyrics/", ONLINE_AUDIO_ROOT / "lyrics")
                )
            except ValueError as e:
                errors.append(f"忽略越界静态路径: {e}")
                logger.warning("  ⚠️ 忽略越界静态路径: %s", url)
                return None

        # 删除音频文件
        audio_path = resolve_static_path(audio_url)
        if audio_path:
            try_delete_file(audio_path, "音频文件")
            # 同时尝试删除对应的元数据文件
            basename = audio_path.stem  # 不含扩展名
            try:
                meta_path = safe_resolve_child(PROCESSED_AUDIO_ROOT / "metadata", f"{basename}_meta.json")
                try_delete_file(meta_path, "元数据文件")
            except ValueError:
                errors.append("忽略越界元数据路径")

        # 删除封面文件
        cover_path = resolve_static_path(cover_url)
        if cover_path:
            try_delete_file(cover_path, "封面文件")

        # 删除歌词文件
        lrc_path = resolve_static_path(lrc_url)
        if lrc_path:
            try_delete_file(lrc_path, "歌词文件")

        # 联网获取的歌曲额外检查（可能有歌词在 online_acquired/lyrics/）
        if dataset == "online":
            try:
                online_lrc = safe_resolve_child(ONLINE_AUDIO_ROOT / "lyrics", f"{music_id}.lrc")
                try_delete_file(online_lrc, "联网歌词文件")
            except ValueError:
                errors.append("忽略越界联网歌词路径")

        logger.info(f"🗑️ 歌曲删除完成: 《{song_title}》 - {artist}, 删除了 {len(deleted_files)} 个文件")

        return {
            "success": True,
            "message": f"《{song_title}》已从系统中彻底删除",
            "deleted_graph_nodes": deleted_count,
            "deleted_files": deleted_files,
            "errors": errors if errors else None,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"歌曲删除失败: {e}", exc_info=True)
        return {"success": False, "error": str(e)}


# ================================================================
# 待入库 (Pending) 管理 API
# ================================================================

@app.get("/api/pending-songs")
async def get_pending_songs():
    """
    扫描 data/online_acquired/metadata/ 目录中的 _meta.json，
    排除已在 Neo4j 中入库的歌曲，返回待入库列表。
    """
    import json as _json
    from services.library_quality import pending_asset_status

    pending = []
    meta_dir = ONLINE_AUDIO_ROOT / "metadata"
    if not meta_dir.exists():
        return {"success": True, "songs": [], "total": 0}

    # 查询 Neo4j 中已有的 online 歌曲 music_id 集合
    ingested_ids = set()
    try:
        from retrieval.neo4j_client import get_neo4j_client
        client = get_neo4j_client()
        rows = client.execute_query(
            "MATCH (s:Song) WHERE s.source = 'online' RETURN s.music_id AS mid", {}
        )
        ingested_ids = {str(r["mid"]) for r in rows if r.get("mid")}
    except Exception as e:
        logger.warning(f"[pending] Neo4j 查询已入库歌曲失败: {e}")

    for meta_file in sorted(meta_dir.glob("*_meta.json"), key=lambda f: f.stat().st_mtime, reverse=True):
        try:
            with open(meta_file, "r", encoding="utf-8") as f:
                meta = _json.load(f)
            music_id = str(meta.get("musicId", ""))
            if music_id in ingested_ids:
                continue  # 已入库，跳过

            # 推导文件名
            title = meta.get("musicName", "Unknown")
            artists = meta.get("artist", [])
            artist_str = "、".join([a[0] if isinstance(a, list) else str(a) for a in artists]) if artists else "Unknown"
            fmt = meta.get("format", "mp3")
            # 安全文件名（与 acquire_music.py 一致）
            safe_title = "".join(c for c in title if c not in r'\\/:*?"<>|').strip()
            safe_artist = "".join(c for c in artist_str if c not in r'\\/:*?"<>|').strip()
            file_basename = f"{safe_title} - {safe_artist}"

            # 检查音频文件是否存在
            audio_path = ONLINE_AUDIO_ROOT / "audio" / f"{file_basename}.{fmt}"
            has_audio = audio_path.exists()
            if not has_audio:
                # 尝试其他格式
                for ext in ["mp3", "flac", "m4a"]:
                    alt = ONLINE_AUDIO_ROOT / "audio" / f"{file_basename}.{ext}"
                    if alt.exists():
                        fmt = ext
                        has_audio = True
                        break
            has_cover = (ONLINE_AUDIO_ROOT / "covers" / f"{file_basename}_cover.jpg").exists()
            has_lyrics = (ONLINE_AUDIO_ROOT / "lyrics" / f"{file_basename}.lrc").exists()
            asset_status = pending_asset_status(
                has_audio=has_audio,
                has_cover=has_cover,
                has_lyrics=has_lyrics,
            )

            pending.append({
                "music_id": music_id,
                "title": title,
                "artist": artist_str,
                "album": meta.get("album", "Unknown"),
                "duration": meta.get("duration", 0),
                "format": fmt,
                "file_basename": file_basename,
                "audio_url": f"/static/online_audio/{file_basename}.{fmt}",
                "cover_url": f"/static/online_covers/{file_basename}_cover.jpg",
                "lrc_url": f"/static/online_lyrics/{file_basename}.lrc",
                "acquired_at": meta.get("acquired_at", ""),
                "release_year": meta.get("release_year"),
                "source_platform": meta.get("source_platform") or meta.get("source", ""),
                "source_id": str(meta.get("source_id") or meta.get("musicId", "")),
                "metadata_source": meta.get("metadata_source", ""),
                **asset_status,
            })
        except Exception as e:
            logger.warning(f"[pending] 解析元数据失败 {meta_file.name}: {e}")

    return {"success": True, "songs": pending, "total": len(pending)}


class PendingIngestItem(BaseModel):
    file_basename: str
    ext: str = "mp3"
    music_id: str = ""
    title: str = ""
    artist: str = ""
    album: str = "Unknown"
    duration: int = 0
    release_year: Optional[int] = None
    source_platform: str = ""
    source_id: str = ""
    metadata_source: str = ""


class PendingIngestRequest(BaseModel):
    songs: List[PendingIngestItem]


class LibraryTagUpdateRequest(BaseModel):
    music_id: str = ""
    title: str = ""
    artist: str = ""
    genres: List[str] = []
    moods: List[str] = []
    themes: List[str] = []
    scenarios: List[str] = []
    language: str = ""


@app.post("/api/pending-songs/ingest")
async def ingest_pending_songs(
    request: PendingIngestRequest,
    _: None = Depends(require_admin_api_key),
):
    """
    秒级写入歌曲元数据，并将耗时的歌词/向量增强交给独立 Worker。
    """
    reject_shared_safe_action("ingest pending songs")
    from tools.acquire_music import _quick_ingest_to_neo4j, _background_flywheel
    from services.ingest_queue import IngestQueueValidationError, enqueue_songs

    songs_to_ingest = []
    for item in request.songs:
        song_data = {
            "song_id": item.music_id,
            "title": item.title,
            "artist": item.artist,
            "album": item.album,
            "duration": item.duration,
            "audio_url": f"/static/online_audio/{item.file_basename}.{item.ext}",
            "cover_url": f"/static/online_covers/{item.file_basename}_cover.jpg",
            "lrc_url": f"/static/online_lyrics/{item.file_basename}.lrc",
            "file_basename": item.file_basename,
            "ext": item.ext,
            "release_year": item.release_year,
            "source_platform": item.source_platform or "netease",
            "source_id": item.source_id or item.music_id,
            "metadata_source": item.metadata_source or "netease",
        }
        songs_to_ingest.append(song_data)

    if not songs_to_ingest:
        return {"success": False, "message": "没有要入库的歌曲", "ingested": 0}

    # 秒级写入 Neo4j
    await _quick_ingest_to_neo4j(songs_to_ingest)

    inline_ingest = os.getenv("MUSIC_INLINE_INGEST_ENABLED", "0").lower() in {"1", "true", "yes"}
    job_id = None
    if inline_ingest:
        asyncio.create_task(_background_flywheel(songs_to_ingest))
    else:
        try:
            job_id = enqueue_songs(songs_to_ingest)
        except IngestQueueValidationError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    logger.info(
        "✅ [pending-ingest] 元数据入库 %s 首，增强模式=%s job=%s",
        len(songs_to_ingest),
        "inline" if inline_ingest else "worker",
        job_id or "-",
    )
    return {
        "success": True,
        "message": f"已写入 {len(songs_to_ingest)} 首歌曲，音频特征将在后台补齐",
        "ingested": len(songs_to_ingest),
        "enrichment": "inline" if inline_ingest else "queued",
        "job_id": job_id,
    }


class PendingDeleteRequest(BaseModel):
    file_basename: str
    ext: str = "mp3"


@app.delete("/api/pending-songs")
async def delete_pending_song(
    file_basename: str,
    ext: str = "mp3",
    _: None = Depends(require_admin_api_key),
):
    """
    删除待入库歌曲的所有本地文件（音频、封面、歌词、元数据）。
    """
    reject_shared_safe_action("delete pending song")
    deleted = []
    errors = []

    ext = ext.lower().lstrip(".")
    if ext not in {"mp3", "flac", "wav", "m4a", "aac", "ogg"}:
        raise HTTPException(status_code=400, detail="Unsupported audio extension")
    try:
        file_paths = [
            (safe_resolve_child(ONLINE_AUDIO_ROOT / "audio", f"{file_basename}.{ext}"), "音频"),
            (safe_resolve_child(ONLINE_AUDIO_ROOT / "covers", f"{file_basename}_cover.jpg"), "封面"),
            (safe_resolve_child(ONLINE_AUDIO_ROOT / "lyrics", f"{file_basename}.lrc"), "歌词"),
            (safe_resolve_child(ONLINE_AUDIO_ROOT / "metadata", f"{file_basename}_meta.json"), "元数据"),
        ]
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid pending song path")

    for fpath, desc in file_paths:
        if fpath.exists():
            try:
                fpath.unlink()
                deleted.append(str(fpath))
                logger.info(f"🗑️ [pending-delete] 已删除{desc}: {fpath.name}")
            except Exception as e:
                errors.append(f"删除{desc}失败: {e}")

    return {
        "success": len(errors) == 0,
        "deleted_files": deleted,
        "errors": errors if errors else None,
    }


@app.get("/api/ingest-jobs")
async def get_ingest_jobs(limit: int = 30):
    """Return recent offline ingestion worker jobs for the pending-import UI."""
    try:
        from services.ingest_queue import list_jobs

        jobs = list_jobs(limit=limit)
        counts: dict[str, int] = {}
        for job in jobs:
            status = str(job.get("status") or "unknown")
            counts[status] = counts.get(status, 0) + 1
        return {"success": True, "jobs": jobs, "counts": counts}
    except Exception as e:
        logger.error(f"[ingest-jobs] 查询队列失败: {e}")
        return {"success": False, "jobs": [], "counts": {}, "error": str(e)}


@app.post("/api/ingest-jobs/{job_id}/retry")
async def retry_ingest_job(
    job_id: str,
    _: None = Depends(require_admin_api_key),
):
    """Retry a failed enrichment job by moving it back to the pending queue."""
    reject_shared_safe_action("retry ingest job")
    try:
        from services.ingest_queue import retry_failed_job

        ok = retry_failed_job(job_id)
        if not ok:
            raise HTTPException(status_code=404, detail="Failed job not found")
        return {"success": True}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[ingest-jobs] 重试任务失败: {e}")
        return {"success": False, "error": str(e)}


# ================================================================
# 我的曲库 (Library) API — 查询 Neo4j 图谱中的全部歌曲
# ================================================================

@app.get("/api/library-songs")
async def get_library_songs(offset: int = 0, limit: int = 200):
    """
    查询 Neo4j 中所有 Song 节点（含 Artist 关联），返回曲库列表。
    """
    try:
        from retrieval.neo4j_client import get_neo4j_client
        from services.library_quality import (
            duplicate_key,
            missing_fields_for_song,
            quality_score,
            vector_coverage_from_dims,
        )

        client = get_neo4j_client()
        query = """
        MATCH (s:Song)
        OPTIONAL MATCH (s)-[:PERFORMED_BY]->(a:Artist)
        OPTIONAL MATCH (s)-[:HAS_MOOD]->(m:Mood)
        OPTIONAL MATCH (s)-[:HAS_THEME]->(t:Theme)
        OPTIONAL MATCH (s)-[:BELONGS_TO_GENRE]->(g:Genre)
        OPTIONAL MATCH (s)-[:FITS_SCENARIO]->(sc:Scenario)
        OPTIONAL MATCH (s)-[:IN_LANGUAGE]->(lang:Language)
        OPTIONAL MATCH (s)-[:HAS_LANGUAGE]->(lang2:Language)
        OPTIONAL MATCH (s)-[:HAS_KNOWLEDGE]->(kc:KnowledgeCard)
        WITH s, a,
             collect(DISTINCT m.name) AS moods,
             collect(DISTINCT t.name) AS themes,
             collect(DISTINCT g.name) AS genres,
             collect(DISTINCT sc.name) AS scenarios,
             collect(DISTINCT lang.name) + collect(DISTINCT lang2.name) AS languages,
             collect(DISTINCT {
                key: kc.key,
                kind: kc.kind,
                summary: kc.summary,
                source: kc.source,
                source_url: kc.source_url,
                confidence: kc.confidence,
                release_year: kc.release_year,
                style_tags_json: kc.style_tags_json
             }) AS knowledge_cards
        RETURN s.title AS title,
               coalesce(a.name, s.artist, 'Unknown') AS artist,
               s.album AS album,
               s.audio_url AS audio_url,
               s.cover_url AS cover_url,
               s.lrc_url AS lrc_url,
               s.source AS source,
               s.music_id AS music_id,
               s.duration AS duration,
               s.format AS format,
               s.vibe AS vibe,
               s.source_platform AS source_platform,
               s.source_id AS source_id,
               s.metadata_source AS metadata_source,
               s.release_year AS release_year,
               s.tag_source AS tag_source,
               s.tag_confidence_json AS tag_confidence_json,
               size(coalesce(s.muq_embedding, [])) AS muq_dim,
               size(coalesce(s.m2d2_embedding, [])) AS m2d_dim,
               size(coalesce(s.omar_embedding, [])) AS omar_dim,
               moods, themes, genres, scenarios,
               coalesce(s.language, languages[0], '') AS language,
               knowledge_cards
        ORDER BY s.updated_at DESC
        SKIP $offset LIMIT $limit
        """
        results = client.execute_query(query, {"offset": offset, "limit": limit})

        # 获取总数
        count_result = client.execute_query("MATCH (s:Song) RETURN count(s) AS total", {})
        total = count_result[0]["total"] if count_result else 0

        songs = []
        for r in results:
            title = r.get("title", "")
            artist = r.get("artist", "Unknown")
            vector_coverage = vector_coverage_from_dims(
                muq_dim=r.get("muq_dim"),
                m2d_dim=r.get("m2d_dim"),
                omar_dim=r.get("omar_dim"),
            )
            knowledge_cards = [
                card
                for card in (r.get("knowledge_cards") or [])
                if isinstance(card, dict) and card.get("summary")
            ][:3]
            missing_fields = missing_fields_for_song(
                {
                    "audio_url": r.get("audio_url"),
                    "cover_url": r.get("cover_url"),
                    "lrc_url": r.get("lrc_url"),
                    "artist": artist,
                    "language": r.get("language"),
                    "release_year": r.get("release_year"),
                },
                vector_coverage,
            )
            songs.append({
                "title": title,
                "artist": artist,
                "album": r.get("album", ""),
                "audio_url": r.get("audio_url", ""),
                "cover_url": r.get("cover_url", ""),
                "lrc_url": r.get("lrc_url", ""),
                "source": r.get("source", "local"),
                "music_id": r.get("music_id", ""),
                "duration": r.get("duration", 0),
                "format": r.get("format", ""),
                "vibe": r.get("vibe", ""),
                "moods": r.get("moods", []),
                "themes": r.get("themes", []),
                "genres": r.get("genres", []),
                "scenarios": r.get("scenarios", []),
                "language": r.get("language", ""),
                "release_year": r.get("release_year"),
                "source_platform": r.get("source_platform", ""),
                "source_id": r.get("source_id", ""),
                "metadata_source": r.get("metadata_source", ""),
                "tag_source": r.get("tag_source", ""),
                "tag_confidence_json": r.get("tag_confidence_json", ""),
                "vector_coverage": vector_coverage,
                "missing_fields": missing_fields,
                "quality_score": quality_score(missing_fields),
                "duplicate_key": duplicate_key(title, artist),
                "knowledge_cards": knowledge_cards,
            })

        return {"success": True, "songs": songs, "total": total}
    except Exception as e:
        logger.error(f"查询曲库失败: {e}")
        return {"success": False, "songs": [], "total": 0, "error": str(e)}


@app.patch("/api/library-songs/tags")
async def update_library_song_tags(
    request: LibraryTagUpdateRequest,
    _: None = Depends(require_admin_api_key),
):
    """Update curated tags on one Song node and its tag relationships."""
    reject_shared_safe_action("update library song tags")
    if not request.music_id and not (request.title and request.artist):
        raise HTTPException(status_code=400, detail="music_id or title+artist is required")

    from services.catalog_enrichment import prepare_tag_enrichment

    relationship_specs = {
        "genres": ("BELONGS_TO_GENRE", "Genre"),
        "moods": ("HAS_MOOD", "Mood"),
        "themes": ("HAS_THEME", "Theme"),
        "scenarios": ("FITS_SCENARIO", "Scenario"),
    }
    enriched_tags = prepare_tag_enrichment(request.model_dump(), source="manual", confidence=1.0)
    values_by_field = {field: enriched_tags[field] for field in relationship_specs}
    language = str(request.language or "").strip()[:40]

    try:
        from retrieval.neo4j_client import get_neo4j_client

        client = get_neo4j_client()
        match = """
        MATCH (s:Song)
        WHERE ($music_id <> '' AND s.music_id = $music_id)
           OR ($music_id = '' AND s.title = $title AND coalesce(s.artist, '') = $artist)
        WITH s LIMIT 1
        """
        exists = client.execute_query(
            match + " RETURN elementId(s) AS sid",
            {
                "music_id": request.music_id,
                "title": request.title,
                "artist": request.artist,
            },
        )
        if not exists:
            raise HTTPException(status_code=404, detail="Song not found")

        for field, values in values_by_field.items():
            rel, label = relationship_specs[field]
            client.execute_query(
                match
                + f"""
                OPTIONAL MATCH (s)-[old:{rel}]->(:{label})
                DELETE old
                WITH s
                SET s.{field} = $values,
                    s.tag_source = $tag_source,
                    s.tag_confidence_json = $tag_confidence_json,
                    s.tag_sources_json = $tag_sources_json,
                    s.updated_at = timestamp()
                WITH s
                UNWIND $values AS tag_name
                MERGE (tag:{label} {{name: tag_name}})
                MERGE (s)-[:{rel}]->(tag)
                RETURN count(tag) AS updated
                """,
                {
                    "music_id": request.music_id,
                    "title": request.title,
                    "artist": request.artist,
                    "values": values,
                    "tag_source": enriched_tags["tag_source"],
                    "tag_confidence_json": enriched_tags["tag_confidence_json"],
                    "tag_sources_json": enriched_tags["tag_sources_json"],
                },
            )

        if language:
            client.execute_query(
                match
                + """
                OPTIONAL MATCH (s)-[old:IN_LANGUAGE]->(:Language)
                DELETE old
                WITH s
                MERGE (lang:Language {name: $language})
                MERGE (s)-[:IN_LANGUAGE]->(lang)
                SET s.language = $language,
                    s.updated_at = timestamp()
                RETURN lang.name AS language
                """,
                {
                    "music_id": request.music_id,
                    "title": request.title,
                    "artist": request.artist,
                    "language": language,
                },
            )

        return {
            "success": True,
            "tags": {**values_by_field, "language": language},
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[library-tags] 更新标签失败: {e}", exc_info=True)
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
