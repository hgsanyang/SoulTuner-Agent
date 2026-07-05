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

from config.logging_config import get_logger  # noqa: E402
from config.settings import settings  # noqa: E402
from services.catalog_enrichment import (  # noqa: E402
    extract_release_year,
    normalize_acquisition_metadata,
    prepare_tag_enrichment,
)
from schemas.music_state import ToolOutput  # noqa: E402

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
        artist_ids = [str(a.get("id")) for a in song.get("artists", []) if a.get("id")]
        artist_str = "、".join(artists) if artists else "Unknown"
        album_info = song.get("album", {}) or {}
        album = album_info.get("name", "Unknown")
        album_id = str(album_info.get("id") or "")
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
        release_year = extract_release_year(song_detail if isinstance(song_detail, dict) else {})
        meta = {
            "musicId": int(song_id),
            "musicName": title,
            "artist": [[a, 0] for a in artists],  # NCM 格式: [[name, id], ...]
            "album": album,
            "album_id": album_id,
            "duration": duration,
            "format": ext,
            "source": "online",
            "source_platform": "netease",
            "source_id": song_id,
            "metadata_source": "netease",
            "acquired_at": datetime.now().isoformat(),
            "bitrate": song_detail.get("bitrate", 0) if isinstance(song_detail, dict) else 0,
            "release_year": release_year,
            "publishTime": song_detail.get("publishTime") if isinstance(song_detail, dict) else None,
            "cover_url": cover_url or "",
            "lyrics_available": has_lyrics,
            "artist_ids": artist_ids,
            "aliases": song_detail.get("alia", []) if isinstance(song_detail, dict) else [],
            "popularity": song_detail.get("pop") if isinstance(song_detail, dict) else None,
        }
        meta_path = os.path.join(ONLINE_META_DIR, f"{file_basename}_meta.json")
        try:
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning(f"元数据保存失败: {e}")

        return self._build_result(
            song_id,
            title,
            artist_str,
            album,
            duration,
            file_basename,
            ext,
            has_lyrics,
            release_year=release_year,
            source_platform="netease",
            metadata_source="netease",
        )

    def _build_result(
        self,
        song_id,
        title,
        artist,
        album,
        duration,
        file_basename,
        ext,
        has_lyrics=False,
        *,
        release_year=None,
        source_platform="netease",
        metadata_source="netease",
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
            "source_platform": source_platform,
            "source_id": song_id,
            "metadata_source": metadata_source,
            "release_year": release_year,
        }

    # ---- NeteaseAPI 辅助方法 ----

    async def _get_play_url(
        self, song_id: str, session: aiohttp.ClientSession
    ) -> Optional[str]:
        """获取音频播放/下载链接（含 30s 试听检测）"""
        url = f"{self.api_base}/song/url?id={song_id}&level=exhigh"
        async with session.get(url, timeout=settings.netease_api_timeout) as resp:
            if resp.status != 200:
                return None
            data = await resp.json()
            for item in data.get("data", []):
                if item.get("url"):
                    # 检测 30s 试听片段
                    trial_info = item.get("freeTrialInfo")
                    if trial_info is not None:
                        logger.warning(
                            f"⚠️ 歌曲 {song_id} 为试听版 "
                            f"(freeTrialInfo: {trial_info.get('start',0)}-{trial_info.get('end',30)}s)"
                        )
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
    秒级快速写入 Neo4j：只写元数据 + audio_url，不提取向量。

    去重策略（修复与初始数据集的兼容性）：
    1. 先通过 title + PERFORMED_BY->Artist.name 查找已有节点（兼容无 s.artist 属性的旧数据）
    2. 如果找到则 SET 更新属性
    3. 如果没有则通过 MERGE {title, artist} 创建新节点
    """
    try:
        from retrieval.neo4j_client import get_neo4j_client
        client = get_neo4j_client()

        for song in songs:
            title = song["title"]
            artist = song["artist"]

            # ── 第一步：检查是否已存在（通过关系匹配，兼容初始数据集） ──
            existing = client.execute_query(
                """MATCH (s:Song)-[:PERFORMED_BY]->(a:Artist)
                WHERE s.title = $title AND a.name = $artist
                RETURN elementId(s) AS eid LIMIT 1""",
                {"title": title, "artist": artist}
            )

            if existing:
                # ── 已存在：更新属性（不创建新节点） ──
                query = """
                MATCH (s:Song)-[:PERFORMED_BY]->(a:Artist)
                WHERE s.title = $title AND a.name = $artist_name
                WITH s LIMIT 1
                SET s.music_id = $music_id,
                    s.artist = $artist_name,
                    s.album = $album,
                    s.duration = $duration,
                    s.format = $format,
                    s.audio_url = $audio_url,
                    s.cover_url = $cover_url,
                    s.lrc_url = $lrc_url,
                    s.source_platform = $source_platform,
                    s.source_id = $source_id,
                    s.metadata_source = $metadata_source,
                    s.release_year = $release_year,
                    s.album_id = $album_id,
                    s.source = 'online',
                    s.acquired_at = $acquired_at,
                    s.updated_at = timestamp()
                """
                logger.info(f"🔄 Neo4j 更新已有歌曲: {title} - {artist}")
            else:
                # ── 不存在：创建新节点 ──
                query = """
                MERGE (s:Song {title: $title, artist: $artist_name})
                SET s.music_id = $music_id,
                    s.album = $album,
                    s.duration = $duration,
                    s.format = $format,
                    s.audio_url = $audio_url,
                    s.cover_url = $cover_url,
                    s.lrc_url = $lrc_url,
                    s.source_platform = $source_platform,
                    s.source_id = $source_id,
                    s.metadata_source = $metadata_source,
                    s.release_year = $release_year,
                    s.album_id = $album_id,
                    s.source = 'online',
                    s.acquired_at = $acquired_at,
                    s.updated_at = timestamp()

                MERGE (a:Artist {name: $artist_name})
                MERGE (s)-[:PERFORMED_BY]->(a)
                """
                logger.info(f"✅ Neo4j 秒级入库: {title} - {artist}")

            normalized_meta = normalize_acquisition_metadata(
                {
                    "musicId": song.get("song_id"),
                    "musicName": title,
                    "artist": [[artist, 0]],
                    "album": song.get("album", "Unknown"),
                    "album_id": song.get("album_id", ""),
                    "duration": song.get("duration", 0),
                    "format": song.get("ext", "mp3"),
                    "source": "online",
                    "source_platform": song.get("platform") or "netease",
                    "metadata_source": song.get("metadata_source") or "netease",
                    "release_year": song.get("release_year"),
                    "cover_url": song.get("cover_url", ""),
                    "lrc_url": song.get("lrc_url", ""),
                }
            )
            params = {
                "music_id": song["song_id"],
                "title": title,
                "artist_name": artist,
                "album": song.get("album", "Unknown"),
                "duration": song.get("duration", 0),
                "format": song.get("ext", "mp3"),
                "audio_url": song["audio_url"],
                "cover_url": song.get("cover_url", ""),
                "lrc_url": song.get("lrc_url", ""),
                "source_platform": normalized_meta.get("source_platform", "netease"),
                "source_id": normalized_meta.get("source_id", song.get("song_id", "")),
                "metadata_source": normalized_meta.get("metadata_source", "netease"),
                "release_year": normalized_meta.get("release_year"),
                "album_id": normalized_meta.get("album_id", ""),
                "acquired_at": datetime.now().isoformat(),
            }
            client.execute_query(query, params)

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
                        enriched_tags = prepare_tag_enrichment(tags, source="llm_lyrics")
                        # 将标签写入 Neo4j（统一用 title+artist 匹配）
                        tag_query = """
                        MATCH (s:Song {title: $title, artist: $artist_name})
                        SET s.vibe = $vibe,
                            s.language = $language,
                            s.region = $region,
                            s.tag_source = $tag_source,
                            s.tag_confidence_json = $tag_confidence_json,
                            s.tag_sources_json = $tag_sources_json,
                            s.updated_at = timestamp()

                        WITH s
                        OPTIONAL MATCH (s)-[old_m:HAS_MOOD]->(:Mood)
                        DELETE old_m
                        WITH s
                        OPTIONAL MATCH (s)-[old_t:HAS_THEME]->(:Theme)
                        DELETE old_t
                        WITH s
                        OPTIONAL MATCH (s)-[old_sc:FITS_SCENARIO]->(:Scenario)
                        DELETE old_sc
                        WITH s
                        OPTIONAL MATCH (s)-[old_g:BELONGS_TO_GENRE]->(:Genre)
                        DELETE old_g
                        WITH s
                        OPTIONAL MATCH (s)-[old_l:HAS_LANGUAGE]->(:Language)
                        DELETE old_l
                        WITH s
                        OPTIONAL MATCH (s)-[old_r:IN_REGION]->(:Region)
                        DELETE old_r

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
                        WITH s
                        FOREACH (genre IN $genres |
                            MERGE (g:Genre {name: genre})
                            MERGE (s)-[:BELONGS_TO_GENRE]->(g)
                        )
                        WITH s
                        FOREACH (_ IN CASE WHEN $language <> '' THEN [1] ELSE [] END |
                            MERGE (lang:Language {name: $language})
                            MERGE (s)-[:HAS_LANGUAGE]->(lang)
                        )
                        WITH s
                        FOREACH (_ IN CASE WHEN $region <> '' THEN [1] ELSE [] END |
                            MERGE (reg:Region {name: $region})
                            MERGE (s)-[:IN_REGION]->(reg)
                        )
                        """
                        client.execute_query(tag_query, {
                            "title": song["title"],
                            "artist_name": song["artist"],
                            "moods": enriched_tags.get("moods", []),
                            "themes": enriched_tags.get("themes", []),
                            "scenarios": enriched_tags.get("scenarios", []),
                            "genres": enriched_tags.get("genres", []),
                            "vibe": tags.get("vibe", ""),
                            "language": str(tags.get("language") or "").strip()[:40],
                            "region": str(tags.get("region") or "").strip()[:60],
                            "tag_source": enriched_tags.get("tag_source", "llm_lyrics"),
                            "tag_confidence_json": enriched_tags.get("tag_confidence_json", "{}"),
                            "tag_sources_json": enriched_tags.get("tag_sources_json", "{}"),
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
                        MATCH (s:Song {title: $title, artist: $artist_name})
                        SET s.m2d2_embedding = $m2d2_embedding,
                            s.omar_embedding = $omar_embedding,
                            s.muq_embedding = $muq_embedding
                        """
                        client.execute_query(embed_query, {
                            "title": song["title"],
                            "artist_name": song["artist"],
                            "m2d2_embedding": embeddings.get("m2d2_embedding", []),
                            "omar_embedding": embeddings.get("omar_embedding", []),
                            "muq_embedding": embeddings.get("muq_embedding", []),
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
        from llms.multi_llm import get_chat_model
        llm = get_chat_model(
            provider=settings.llm_default_provider,
            model_name=settings.llm_default_model,
            temperature=0.3,
            max_tokens=800,
        )

        prompt = f"""分析以下歌词，返回纯 JSON 对象（不加 markdown 代码块）。{{
  "moods": ["1-5个情绪标签，按实际内容选择，如 Melancholy/Healing/Nostalgic/Dreamy"],
  "themes": ["0-5个主题标签，按实际内容选择，如 Love/Youth/Life/Journey"],
  "scenarios": ["1-5个场景标签，按实际适配场景选择，如 Late Night/Driving/Rainy Day"],
  "vibe": "1个氛围标签，如 Indie/Acoustic/Lo-fi",
  "genres": ["1-5个流派标签，按实际风格选择，如 Rock/Indie/Pop/Ballad"],
  "language": "English/Chinese/Japanese/Korean/Cantonese/Instrumental/Mixed/Other/Unknown",
  "region": "Western/Mainland China/Taiwan/Hong Kong/Japan/Korea/Other/Unknown"
}}

不要为了凑数量硬填标签；不确定就少填。

歌曲: {basename}
歌词:
{cleaned[:2000]}"""

        response = llm.invoke(
            [
                ("system", "你只返回纯 JSON，不加任何解释文字。"),
                ("human", prompt),
            ]
        )

        # 解析 JSON
        text = response.content if hasattr(response, "content") else str(response)
        text = text.strip()
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
    from retrieval.muq_embedder import encode_audio_to_muq

    MAX_SECONDS = 300
    file_duration = librosa.get_duration(path=audio_path)
    load_duration = MAX_SECONDS if file_duration > MAX_SECONDS else None

    audio_np, sr = librosa.load(audio_path, sr=None, mono=True, duration=load_duration)
    audio_16k = librosa.resample(audio_np, orig_sr=sr, target_sr=16000)
    audio_24k = librosa.resample(audio_np, orig_sr=sr, target_sr=24000)

    m2d2_emb = encode_audio_to_embedding(audio_16k, sample_rate=16000)
    omar_emb = extract_audio_representation(audio_16k, sample_rate=16000)
    try:
        muq_emb = encode_audio_to_muq(audio_24k, sample_rate=24000)
    except Exception as exc:
        logger.warning("MuQ embedding 提取失败，在线歌曲将暂时缺少主文搜音锚: %s", exc)
        muq_emb = []

    return {"m2d2_embedding": m2d2_emb, "omar_embedding": omar_emb, "muq_embedding": muq_emb}


# ---- 全局单例 ----
_acquirer = OnlineMusicAcquirer()


@tool
async def acquire_online_music(song_queries: list[str]) -> ToolOutput:
    """
    当用户确认要获取联网搜索推荐的歌曲时调用此工具。
    它会自动从网易云等平台下载音频、歌词、封面到本地待入库目录。
    下载完成后歌曲进入「待入库」状态，用户可在前端待入库页面试听、
    勾选并确认入库到知识图谱。
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

    # ★ 不再自动入库 Neo4j 和触发飞轮
    # 歌曲仅下载到 data/online_acquired/，用户在前端待入库页面确认后才入库

    # 构建返回给前端的 markdown
    md = f"🎵 **已成功下载 {len(acquired)} 首歌曲到待入库！**\n\n"
    for i, s in enumerate(acquired, 1):
        md += f"{i}. **{s['title']}** — {s['artist']}\n"
        md += f"   📀 专辑：{s.get('album', 'Unknown')}\n\n"

    md += "\n> 💡 请前往 **音乐库 → 待入库** 页面试听确认，勾选后即可入库到知识图谱。"

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
