# ============================================================
# 【联网音乐自动获取工具】── LangGraph Tool
#
# 功能：当本地曲库未命中时，自动从网易云 API 下载音频/歌词/封面,
#      秒级写入 Neo4j（立即可播），后台异步触发歌词标签/向量提取。
#
# 设计：
#   - 文件存入 data/online_acquired/（与本地 processed_audio 隔离）
#   - Neo4j 节点标记 source='online'，后续本地入库自动覆盖
#   - 下载失败或 API 不可用时静默降级，不影响主流程
# ============================================================

import asyncio
import aiohttp
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

from langchain_core.tools import tool

# 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.logging_config import get_logger
from config.settings import settings
from schemas.music_state import ToolOutput

logger = get_logger(__name__)

# ---- 存储目录（与 processed_audio 隔离）---
ONLINE_DATA_ROOT = os.path.join(
    str(PROJECT_ROOT.parent),   # music_recommendation/
    "data", "online_acquired"
)
ONLINE_AUDIO_DIR = os.path.join(ONLINE_DATA_ROOT, "audio")
ONLINE_COVER_DIR = os.path.join(ONLINE_DATA_ROOT, "covers")
ONLINE_LYRICS_DIR = os.path.join(ONLINE_DATA_ROOT, "lyrics")
ONLINE_META_DIR = os.path.join(ONLINE_DATA_ROOT, "metadata")

# 静态 URL 前缀（需要 server.py 挂载）
STATIC_PREFIX_ONLINE = "/static/online_audio"
STATIC_PREFIX_ONLINE_COVERS = "/static/online_covers"
STATIC_PREFIX_ONLINE_LYRICS = "/static/online_lyrics"

# NeteaseAPI 基础地址
NETEASE_API_BASE = settings.netease_api_base


def _ensure_dirs():
    """确保输出目录存在"""
    for d in [ONLINE_AUDIO_DIR, ONLINE_COVER_DIR, ONLINE_LYRICS_DIR, ONLINE_META_DIR]:
        os.makedirs(d, exist_ok=True)


def _safe_filename(text: str) -> str:
    """生成安全的文件名"""
    return "".join(c for c in text if c not in r'\/:*?"<>|').strip()


class OnlineMusicAcquirer:
    """联网音乐获取器：搜索 → 下载 → 快速入库 → 后台飞轮"""

    def __init__(self):
        self.api_base = NETEASE_API_BASE

    async def search_and_acquire(
        self, queries: List[str], session: aiohttp.ClientSession
    ) -> List[Dict[str, Any]]:
        """
        批量搜索并获取音乐。每个 query 取最佳匹配的 1 首。        返回成功获取的歌曲信息列表。        """
        _ensure_dirs()
        acquired = []

        for query in queries:
            try:
                song = await self._acquire_one(query, session)
                if song:
                    acquired.append(song)
            except Exception as e:
                logger.warning(f"获取 '{query}' 失败: {e}")

        return acquired

    async def _acquire_one(
        self, query: str, session: aiohttp.ClientSession
    ) -> Optional[Dict[str, Any]]:
        """搜索并下载单首歌曲的全部资源"""

        # 1. 搜索（清理特殊字符）
        import re
        clean_query = re.sub(r'[《》\[\]【】]', ' ', query)
        clean_query = re.sub(r'\s+[xX×]\s+', ' ', clean_query)  # "A x B" → "A B"
        clean_query = clean_query.strip()
        search_url = f"{self.api_base}/search?keywords={clean_query}&limit={settings.netease_search_limit}"
        async with session.get(search_url, timeout=settings.netease_api_timeout) as resp:
            if resp.status != 200:
                logger.warning(f"搜索失败 status={resp.status}: {query}")
                return None
            data = await resp.json()
            songs = data.get("result", {}).get("songs", [])
            if not songs:
                logger.warning(f"搜索无结果: {query}")
                return None

        # 取第一首
        song = songs[0]
        song_id = str(song["id"])
        title = song.get("name", "Unknown")
        artists = [a["name"] for a in song.get("artists", [])]
        artist_str = "、".join(artists) if artists else "Unknown"
        album = song.get("album", {}).get("name", "Unknown")
        duration = song.get("duration", 0)

        safe_title = _safe_filename(title)
        safe_artist = _safe_filename(artist_str)
        file_basename = f"{safe_title} - {safe_artist}"

        # 防重：如果音频文件已存在则跳过下载
        existing_audio = None
        for ext in ["mp3", "flac", "m4a"]:
            candidate = os.path.join(ONLINE_AUDIO_DIR, f"{file_basename}.{ext}")
            if os.path.exists(candidate):
                existing_audio = candidate
                break

        if existing_audio:
            logger.info(f"⏭️ 已存在，跳过下载: {file_basename}")
            ext = os.path.splitext(existing_audio)[1].lstrip(".")
            has_lyrics = os.path.exists(os.path.join(ONLINE_LYRICS_DIR, f"{file_basename}.lrc"))
            return self._build_result(
                song_id, title, artist_str, album, duration, file_basename, ext, has_lyrics
            )

        # 2. 并发获取播放链接 + 歌词 + 封面
        play_url, lyrics_text, cover_url, song_detail = await asyncio.gather(
            self._get_play_url(song_id, session),
            self._get_lyrics(song_id, session),
            self._get_cover_url(song_id, session),
            self._get_song_detail(song_id, session),
            return_exceptions=True,
        )

        # 处理异常
        if isinstance(play_url, Exception):
            play_url = None
        if isinstance(lyrics_text, Exception):
            lyrics_text = None
        if isinstance(cover_url, Exception):
            cover_url = None
        if isinstance(song_detail, Exception):
            song_detail = {}

        if not play_url:
            logger.warning(f"⚠️ 无法获取播放链接（版权限制）: {title} - {artist_str}")
            return None

        # 3. 下载音频
        ext = "mp3"  # 网易云 API 默认返回 mp3
        audio_path = os.path.join(ONLINE_AUDIO_DIR, f"{file_basename}.{ext}")
        downloaded = await self._download_file(play_url, audio_path, session)
        if not downloaded:
            return None

        logger.info(f"✅ 音频下载成功: {file_basename}.{ext}")

        # 4. 保存歌词
        has_lyrics = False
        if lyrics_text:
            lrc_path = os.path.join(ONLINE_LYRICS_DIR, f"{file_basename}.lrc")
            try:
                with open(lrc_path, "w", encoding="utf-8") as f:
                    f.write(lyrics_text)
                logger.info(f"✅ 歌词保存成功: {file_basename}.lrc")
                has_lyrics = True
            except Exception as e:
                logger.warning(f"歌词保存失败: {e}")

        # 5. 下载封面
        if cover_url and isinstance(cover_url, str):
            cover_path = os.path.join(ONLINE_COVER_DIR, f"{file_basename}_cover.jpg")
            await self._download_file(cover_url, cover_path, session)

        # 6. 保存元数据(兼容 ingest_to_neo4j.py 的 _meta.json 格式)
        meta = {
            "musicId": int(song_id),
            "musicName": title,
            "artist": [[a, 0] for a in artists],  # NCM 格式: [[name, id], ...]
            "album": album,
            "duration": duration,
            "format": ext,
            "source": "online",
            "acquired_at": datetime.now().isoformat(),
            "bitrate": song_detail.get("bitrate", 0) if isinstance(song_detail, dict) else 0,
        }
        meta_path = os.path.join(ONLINE_META_DIR, f"{file_basename}_meta.json")
        try:
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning(f"元数据保存失败: {e}")

        return self._build_result(
            song_id, title, artist_str, album, duration, file_basename, ext, has_lyrics
        )

    def _build_result(
        self, song_id, title, artist, album, duration, file_basename, ext, has_lyrics=False
    ) -> Dict[str, Any]:
        """构建返回结果"""
        lrc_url = f"{STATIC_PREFIX_ONLINE_LYRICS}/{file_basename}.lrc" if has_lyrics else ""
        audio = f"{STATIC_PREFIX_ONLINE}/{file_basename}.{ext}"
        return {
            "song_id": song_id,
            "title": title,
            "artist": artist,
            "album": album,
            "duration": duration,
            "audio_url": audio,
            "preview_url": audio,  # 前端播放器用 preview_url
            "cover_url": f"{STATIC_PREFIX_ONLINE_COVERS}/{file_basename}_cover.jpg",
            "lrc_url": lrc_url,
            "file_basename": file_basename,
            "ext": ext,
            "source": "online",
        }

    # ---- NeteaseAPI 辅助方法 ----

    async def _get_play_url(
        self, song_id: str, session: aiohttp.ClientSession
    ) -> Optional[str]:
        """获取音频播放/下载链接"""
        url = f"{self.api_base}/song/url?id={song_id}&level=exhigh"
        async with session.get(url, timeout=settings.netease_api_timeout) as resp:
            if resp.status != 200:
                return None
            data = await resp.json()
            for item in data.get("data", []):
                if item.get("url"):
                    return item["url"]
        return None

    async def _get_lyrics(
        self, song_id: str, session: aiohttp.ClientSession
    ) -> Optional[str]:
        """获取 LRC 歌词"""
        url = f"{self.api_base}/lyric?id={song_id}"
        async with session.get(url, timeout=settings.netease_api_timeout) as resp:
            if resp.status != 200:
                return None
            data = await resp.json()
            return data.get("lrc", {}).get("lyric")

    async def _get_cover_url(
        self, song_id: str, session: aiohttp.ClientSession
    ) -> Optional[str]:
        """获取专辑封面 URL"""
        url = f"{self.api_base}/song/detail?ids={song_id}"
        async with session.get(url, timeout=settings.netease_api_timeout) as resp:
            if resp.status != 200:
                return None
            data = await resp.json()
            songs = data.get("songs", [])
            if songs:
                return songs[0].get("al", {}).get("picUrl")
        return None

    async def _get_song_detail(
        self, song_id: str, session: aiohttp.ClientSession
    ) -> Dict:
        """获取歌曲详情"""
        url = f"{self.api_base}/song/detail?ids={song_id}"
        async with session.get(url, timeout=settings.netease_api_timeout) as resp:
            if resp.status != 200:
                return {}
            data = await resp.json()
            songs = data.get("songs", [])
            return songs[0] if songs else {}

    async def _download_file(
        self, url: str, save_path: str, session: aiohttp.ClientSession
    ) -> bool:
        """下载文件到本地"""
        try:
            async with session.get(url, timeout=settings.audio_download_timeout) as resp:
                if resp.status != 200:
                    logger.warning(f"下载失败 status={resp.status}: {url[:80]}")
                    return False
                content = await resp.read()
                with open(save_path, "wb") as f:
                    f.write(content)
                size_mb = len(content) / 1024 / 1024
                logger.info(f"📥 已下载 {size_mb:.1f}MB → {os.path.basename(save_path)}")
                return True
        except Exception as e:
            logger.warning(f"下载异常: {e}")
            return False


async def _quick_ingest_to_neo4j(songs: List[Dict[str, Any]]):
    """
    秒级快速写入 Neo4j：只写元数据 + audio_url，不提取向量。    使用 MERGE 幂等写入，后续本地入库可覆盖。    """
    try:
        from retrieval.neo4j_client import get_neo4j_client
        client = get_neo4j_client()

        for song in songs:
            query = """
            MERGE (s:Song {music_id: $music_id})
            SET s.title = $title,
                s.album = $album,
                s.duration = $duration,
                s.format = $format,
                s.audio_url = $audio_url,
                s.cover_url = $cover_url,
                s.lrc_url = $lrc_url,
                s.source = 'online',
                s.acquired_at = $acquired_at,
                s.updated_at = timestamp()

            MERGE (a:Artist {name: $artist_name})
            MERGE (s)-[:PERFORMED_BY]->(a)
            """
            params = {
                "music_id": song["song_id"],
                "title": song["title"],
                "artist_name": song["artist"],
                "album": song.get("album", "Unknown"),
                "duration": song.get("duration", 0),
                "format": song.get("ext", "mp3"),
                "audio_url": song["audio_url"],
                "cover_url": song.get("cover_url", ""),
                "lrc_url": song.get("lrc_url", ""),
                "acquired_at": datetime.now().isoformat(),
            }
            client.execute_query(query, params)
            logger.info(f"✅ Neo4j 秒级入库: {song['title']} - {song['artist']}")

    except Exception as e:
        logger.error(f"Neo4j 快速入库失败: {e}")


async def _background_flywheel(songs: List[Dict[str, Any]]):
    """
    后台异步数据飞轮：歌词标签提取 + 向量提取 + Neo4j 更新。    在后台静默运行，不阻塞用户交互。    """
    try:
        logger.info(f"🔄 [后台飞轮] 开始处理 {len(songs)} 首歌...")

        from retrieval.neo4j_client import get_neo4j_client
        client = get_neo4j_client()

        for song in songs:
            basename = song["file_basename"]
            ext = song["ext"]
            audio_path = os.path.join(ONLINE_AUDIO_DIR, f"{basename}.{ext}")
            lrc_path = os.path.join(ONLINE_LYRICS_DIR, f"{basename}.lrc")

            # ---- 步骤1: 歌词标签提取 ----
            if os.path.exists(lrc_path):
                try:
                    tags = await _extract_lyrics_tags(basename, lrc_path)
                    if tags:
                        # 将标签写入 Neo4j
                        tag_query = """
                        MATCH (s:Song {music_id: $music_id})
                        SET s.vibe = $vibe

                        WITH s
                        FOREACH (mood IN $moods |
                            MERGE (m:Mood {name: mood})
                            MERGE (s)-[:HAS_MOOD]->(m)
                        )
                        WITH s
                        FOREACH (theme IN $themes |
                            MERGE (t:Theme {name: theme})
                            MERGE (s)-[:HAS_THEME]->(t)
                        )
                        WITH s
                        FOREACH (scenario IN $scenarios |
                            MERGE (sc:Scenario {name: scenario})
                            MERGE (s)-[:FITS_SCENARIO]->(sc)
                        )
                        """
                        client.execute_query(tag_query, {
                            "music_id": song["song_id"],
                            "moods": tags.get("moods", []),
                            "themes": tags.get("themes", []),
                            "scenarios": tags.get("scenarios", []),
                            "vibe": tags.get("vibe", ""),
                        })
                        logger.info(f"🏷️ [后台飞轮] 歌词标签入库: {song['title']}")
                except Exception as e:
                    logger.warning(f"[后台飞轮] 歌词标签提取失败 {song['title']}: {e}")

            # ---- 步骤2: 向量提取 ----
            if os.path.exists(audio_path):
                try:
                    embeddings = await _extract_embeddings(audio_path)
                    if embeddings:
                        embed_query = """
                        MATCH (s:Song {music_id: $music_id})
                        SET s.m2d2_embedding = $m2d2_embedding,
                            s.omar_embedding = $omar_embedding
                        """
                        client.execute_query(embed_query, {
                            "music_id": song["song_id"],
                            "m2d2_embedding": embeddings.get("m2d2_embedding", []),
                            "omar_embedding": embeddings.get("omar_embedding", []),
                        })
                        logger.info(f"🧠 [后台飞轮] 向量入库: {song['title']}")
                except Exception as e:
                    logger.warning(f"[后台飞轮] 向量提取失败 {song['title']}: {e}")

        logger.info(f"✅ [后台飞轮] 全部完成！{len(songs)} 首歌已入库")

    except Exception as e:
        logger.error(f"[后台飞轮] 整体失败: {e}")


async def _extract_lyrics_tags(basename: str, lrc_path: str) -> Optional[Dict]:
    """调用 LLM 提取歌词标签（简化版，单首处理）"""
    import re

    try:
        with open(lrc_path, "r", encoding="utf-8") as f:
            raw_lyrics = f.read()

        # 清洗歌词
        cleaned = re.sub(r"\[\d{2}:\d{2}\.\d{2,3}\]", "", raw_lyrics)
        cleaned = re.sub(
            r"^\[(ar|ti|al|by|offset|hash|total|sign):.*\]$",
            "", cleaned, flags=re.MULTILINE | re.IGNORECASE,
        )
        cleaned = "\n".join(line.strip() for line in cleaned.split("\n") if line.strip())

        if len(cleaned) < 20:
            return {"moods": ["Instrumental"], "themes": [], "scenarios": [], "vibe": ""}

        # 调用 LLM
        from llms.multi_llm import MultiLLM
        llm = MultiLLM(provider="siliconflow", temperature=0.3)

        prompt = f"""分析以下歌词，返回纯 JSON 对象（不加 markdown 代码块）。{{
  "moods": ["1-3个情绪标签，如 Happy/Melancholy/Healing"],
  "themes": ["1-3个主题标签，如 Love/Youth/Life"],
  "scenarios": ["1-2个场景标签，如 Late Night/Driving"],
  "vibe": "1个氛围标签，如 Indie/Acoustic/Lo-fi"
}}

歌曲: {basename}
歌词:
{cleaned[:2000]}"""

        response = llm.invoke(
            system_prompt="你只返回纯 JSON，不加任何解释文字。",
            user_prompt=prompt,
            max_tokens=500,
        )

        # 解析 JSON
        text = response.strip()
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0].strip()
        elif "```" in text:
            text = text.split("```")[1].split("```")[0].strip()
        return json.loads(text)

    except Exception as e:
        logger.warning(f"歌词标签提取失败: {e}")
        return None


async def _extract_embeddings(audio_path: str) -> Optional[Dict[str, List[float]]]:
    """提取双模型音频向量（在线程池中执行，避免阻塞事件循环）"""
    try:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, _sync_extract_embeddings, audio_path)
    except Exception as e:
        logger.warning(f"向量提取失败: {e}")
        return None


def _sync_extract_embeddings(audio_path: str) -> Dict[str, List[float]]:
    """同步版本的向量提取"""
    import librosa
    from retrieval.audio_embedder import encode_audio_to_embedding, extract_audio_representation

    MAX_SECONDS = 300
    file_duration = librosa.get_duration(path=audio_path)
    load_duration = MAX_SECONDS if file_duration > MAX_SECONDS else None

    audio_np, sr = librosa.load(audio_path, sr=None, mono=True, duration=load_duration)
    audio_16k = librosa.resample(audio_np, orig_sr=sr, target_sr=16000)

    m2d2_emb = encode_audio_to_embedding(audio_16k, sample_rate=16000)
    omar_emb = extract_audio_representation(audio_16k, sample_rate=16000)

    return {"m2d2_embedding": m2d2_emb, "omar_embedding": omar_emb}


# ---- 全局单例 ----
_acquirer = OnlineMusicAcquirer()


@tool
async def acquire_online_music(song_queries: list[str]) -> ToolOutput:
    """
    当用户确认要获取联网搜索推荐的歌曲时调用此工具。    它会自动从网易云等平台下载音频、歌词、封面，存入本地并写入 Neo4j 图谱。    下载完成即可在前端播放。后台会异步进行歌词标签分析和音频向量提取。
    Args:
        song_queries: 要获取的歌曲列表，格式为 ["歌名 歌手", "歌名 歌手", ...]
                      例如 ["稻香 周杰伦", "平凡之路 朴树"]
    """
    logger.info(f"🎵 开始联网获取 {len(song_queries)} 首歌曲..")

    async with aiohttp.ClientSession() as session:
        acquired = await _acquirer.search_and_acquire(song_queries, session)

    if not acquired:
        return ToolOutput(
            success=False,
            data=[],
            raw_markdown="❌ 未能获取任何歌曲的音频资源（可能因版权限制或网络问题）。",
            error_message="No songs acquired",
        )

    # 秒级写入 Neo4j（元数据 + audio_url）
    await _quick_ingest_to_neo4j(acquired)

    # 启动后台飞轮（歌词标签 + 向量提取），不阻塞当前响应
    asyncio.create_task(_background_flywheel(acquired))

    # 构建返回给前端的 markdown
    md = f"🎵 **已成功获取 {len(acquired)} 首歌曲并入库！**\n\n"
    for i, s in enumerate(acquired, 1):
        md += f"{i}. **{s['title']}** - {s['artist']}\n"
        md += f"   📀 专辑: {s.get('album', 'Unknown')}\n"
        md += f"   🔈 已可播放（点击前端播放按钮）\n\n"

    md += "\n> 💡 歌词标签和音频特征分析正在后台处理中，完成后自动更新图谱。"

    return ToolOutput(
        success=True,
        data=[{
            "song": {
                "title": s["title"],
                "artist": s["artist"],
                "album": s.get("album", "Unknown"),
                "audio_url": s["audio_url"],
                "preview_url": s.get("preview_url", s["audio_url"]),
                "cover_url": s.get("cover_url", ""),
                "lrc_url": s.get("lrc_url", ""),
                "song_id": s.get("song_id", ""),
                "source": "online",
                "platform": "netease",
            }
        } for s in acquired],
        raw_markdown=md,
    )
