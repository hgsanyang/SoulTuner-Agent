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
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

from langchain_core.tools import tool

# 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.logging_config import get_logger, safe_query  # noqa: E402
from config.settings import settings  # noqa: E402
from services.catalog_enrichment import (  # noqa: E402
    extract_release_year,
    normalize_acquisition_metadata,
    prepare_tag_enrichment,
)
from schemas.music_state import ToolOutput  # noqa: E402

logger = get_logger(__name__)


def _normalize_identity_text(value: Any) -> str:
    """Collapse Unicode whitespace so equivalent catalog identities stay stable."""
    return " ".join(str(value or "").strip().split())


@dataclass
class EmbeddingExtraction:
    vectors: Dict[str, List[float]] = field(default_factory=dict)
    errors: Dict[str, str] = field(default_factory=dict)


class EnrichmentIncompleteError(RuntimeError):
    """Raised when a queue job lacks an embedding required by live retrieval."""

# ---- 存储目录（与 processed_audio 隔离）---
ONLINE_DATA_ROOT = os.path.join(
    os.getenv("MUSIC_DATA_ROOT", str(PROJECT_ROOT.parent / "data")),
    "online_acquired",
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


def _resolve_enrichment_paths(
    song: Dict[str, Any],
    basename: str,
    ext: str,
) -> tuple[str, str]:
    """Resolve both online-candidate and durable-library assets in host or Docker."""
    explicit_audio = str(song.get("audio_path") or "").strip()
    explicit_lrc = str(song.get("lrc_path") or "").strip()
    if explicit_audio and os.path.exists(explicit_audio):
        audio_path = explicit_audio
    else:
        audio_url = str(song.get("audio_url") or "").replace("\\", "/")
        filename = Path(audio_url).name or f"{basename}.{ext}"
        if audio_url.startswith("/static/audio/"):
            data_root = Path(os.getenv("MUSIC_DATA_ROOT", str(PROJECT_ROOT.parent / "data")))
            audio_path = str(data_root / "processed_audio" / "audio" / filename)
        else:
            audio_path = os.path.join(ONLINE_AUDIO_DIR, filename)

    if explicit_lrc and os.path.exists(explicit_lrc):
        lrc_path = explicit_lrc
    else:
        lrc_url = str(song.get("lrc_url") or "").replace("\\", "/")
        lyric_name = Path(lrc_url).name or f"{basename}.lrc"
        if lrc_url.startswith("/static/lyrics/"):
            data_root = Path(os.getenv("MUSIC_DATA_ROOT", str(PROJECT_ROOT.parent / "data")))
            lrc_path = str(data_root / "processed_audio" / "lyrics" / lyric_name)
        else:
            lrc_path = os.path.join(ONLINE_LYRICS_DIR, lyric_name)
    return audio_path, lrc_path


def _artist_names_from_payload(song: Dict[str, Any]) -> List[str]:
    """Return artist names from Netease search/detail shaped payloads."""
    artists = song.get("artists") or song.get("ar") or []
    names: list[str] = []
    for artist in artists:
        if isinstance(artist, dict):
            name = str(artist.get("name") or "").strip()
        elif isinstance(artist, (list, tuple)) and artist:
            name = str(artist[0] or "").strip()
        else:
            name = str(artist or "").strip()
        if name:
            names.append(name)
    if not names and song.get("artist"):
        names = [part.strip() for part in str(song.get("artist")).replace("/", "、").split("、") if part.strip()]
    return names


def _artist_ids_from_payload(song: Dict[str, Any]) -> List[str]:
    artists = song.get("artists") or song.get("ar") or []
    ids: list[str] = []
    for artist in artists:
        if isinstance(artist, dict) and artist.get("id"):
            ids.append(str(artist.get("id")))
    return ids


def _album_from_payload(song: Dict[str, Any]) -> Dict[str, Any]:
    album = song.get("album") or song.get("al") or {}
    return album if isinstance(album, dict) else {}


def _meta_music_id(song_id: str) -> int | str:
    try:
        return int(song_id)
    except (TypeError, ValueError):
        return str(song_id or "")


def _write_meta_file(file_basename: str, meta: Dict[str, Any]) -> None:
    _ensure_dirs()
    meta_path = os.path.join(ONLINE_META_DIR, f"{file_basename}_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def mark_online_audio_retained(
    *,
    title: str,
    artist: str,
    song_id: str = "",
    retention_reason: str = "user_saved",
) -> bool:
    """Mark an already acquired online audio file as long-term retained."""
    _ensure_dirs()
    safe_title = _safe_filename(title)
    safe_artist = _safe_filename(artist)
    expected = f"{safe_title} - {safe_artist}_meta.json"
    candidates = [os.path.join(ONLINE_META_DIR, expected)]
    candidates.extend(
        os.path.join(ONLINE_META_DIR, name)
        for name in os.listdir(ONLINE_META_DIR)
        if name.endswith("_meta.json")
    )

    seen: set[str] = set()
    for meta_path in candidates:
        if meta_path in seen or not os.path.exists(meta_path):
            continue
        seen.add(meta_path)
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            meta_title = str(meta.get("musicName") or "").strip()
            artists = meta.get("artist") or []
            meta_artist = "、".join([a[0] if isinstance(a, list) else str(a) for a in artists]) if artists else ""
            meta_id = str(meta.get("musicId") or meta.get("source_id") or "")
            if song_id and meta_id and meta_id != str(song_id):
                continue
            if not song_id and (meta_title != title or meta_artist != artist):
                continue
            meta["audio_retention"] = "saved"
            meta["retention_reason"] = retention_reason
            meta["retained_at"] = datetime.now().isoformat()
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)
            return True
        except Exception as exc:
            logger.debug("标记在线音频保留失败: %s", exc)
    return False


def _promote_external_candidate_feedback(client: Any, *, title: str, artist: str) -> None:
    """Move feedback from an external candidate onto the newly ingested Song."""
    relation_pairs = (
        ("LIKES_CANDIDATE", "LIKES", "r.weight = coalesce(old.weight, 1.0), r.created_at = coalesce(old.created_at, timestamp())"),
        ("SAVES_CANDIDATE", "SAVES", "r.weight = coalesce(old.weight, 0.8), r.created_at = coalesce(old.created_at, timestamp())"),
        ("DISLIKES_CANDIDATE", "DISLIKES", "r.weight = coalesce(old.weight, 1.0), r.created_at = coalesce(old.created_at, timestamp())"),
        ("SKIPPED_CANDIDATE", "SKIPPED", "r.skip_count = coalesce(old.skip_count, 1), r.last_skipped = timestamp()"),
        ("LISTENED_TO_CANDIDATE", "LISTENED_TO", "r.play_count = coalesce(old.play_count, 1), r.total_duration = coalesce(old.duration, 0), r.last_played = timestamp()"),
    )
    for candidate_rel, song_rel, set_clause in relation_pairs:
        query = f"""
        MATCH (u:User)-[old:{candidate_rel}]->(c:ExternalTrackCandidate {{title: $title, artist: $artist}})
        MATCH (s:Song {{title: $title, artist: $artist}})
        MERGE (u)-[r:{song_rel}]->(s)
        SET {set_clause}
        DELETE old
        """
        client.execute_query(query, {"title": title, "artist": artist})
    client.execute_query(
        """
        MATCH (c:ExternalTrackCandidate {title: $title, artist: $artist})
        WHERE NOT (c)--()
        DELETE c
        """,
        {"title": title, "artist": artist},
    )


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
                logger.warning(f"获取 '{safe_query(query)}' 失败: {e}")

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
                logger.warning(f"搜索失败 status={resp.status}: {safe_query(query)}")
                return None
            data = await resp.json()
            songs = data.get("result", {}).get("songs", [])
            if not songs:
                logger.warning(f"搜索无结果: {safe_query(query)}")
                return None

        return await self.acquire_resolved_song(
            songs[0],
            session,
            retention="saved",
            requested_by="explicit_acquire",
        )

    async def acquire_resolved_song(
        self,
        song: Dict[str, Any],
        session: aiohttp.ClientSession,
        *,
        retention: str = "temporary",
        requested_by: str = "auto_recommendation",
    ) -> Optional[Dict[str, Any]]:
        """Download a specific resolved online song by source id, avoiding fuzzy re-search."""

        song_id = str(song.get("id") or song.get("song_id") or song.get("source_id") or "").strip()
        if not song_id:
            logger.warning("跳过联网获取：缺少 song_id: %s", song)
            return None

        title = str(song.get("name") or song.get("title") or "Unknown").strip() or "Unknown"
        artists = _artist_names_from_payload(song)
        artist_ids = _artist_ids_from_payload(song)
        artist_str = "、".join(artists) if artists else "Unknown"
        album_info = _album_from_payload(song)
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
            logger.info(f"已存在，跳过下载: {file_basename}")
            ext = os.path.splitext(existing_audio)[1].lstrip(".")
            has_lyrics = os.path.exists(os.path.join(ONLINE_LYRICS_DIR, f"{file_basename}.lrc"))
            is_trial = False
            existing_meta = os.path.join(ONLINE_META_DIR, f"{file_basename}_meta.json")
            if os.path.exists(existing_meta):
                try:
                    with open(existing_meta, "r", encoding="utf-8") as f:
                        is_trial = bool(json.load(f).get("is_trial"))
                except Exception:
                    pass
            if retention == "saved":
                mark_online_audio_retained(
                    title=title,
                    artist=artist_str,
                    song_id=song_id,
                    retention_reason=requested_by,
                )
            return self._build_result(
                song_id,
                title,
                artist_str,
                album,
                duration,
                file_basename,
                ext,
                has_lyrics,
                source_platform="netease",
                metadata_source="netease",
                retention=retention,
                requested_by=requested_by,
                is_trial=is_trial,
            )

        # 2. 并发获取播放链接 + 歌词 + 封面
        play_result, lyrics_text, cover_url, song_detail = await asyncio.gather(
            self._get_play_url(song_id, session),
            self._get_lyrics(song_id, session),
            self._get_cover_url(song_id, session),
            self._get_song_detail(song_id, session),
            return_exceptions=True,
        )

        # 处理异常
        if isinstance(play_result, Exception):
            play_url, is_trial = None, False
        else:
            play_url, is_trial = play_result
        if isinstance(lyrics_text, Exception):
            lyrics_text = None
        if isinstance(cover_url, Exception):
            cover_url = None
        if isinstance(song_detail, Exception):
            song_detail = {}

        if not play_url:
            logger.warning(f"无法获取播放链接（版权限制）: {title} - {artist_str}")
            self._write_failed_meta(
                title=title,
                artist=artist_str,
                song_id=song_id,
                album=album,
                duration=duration,
                file_basename=file_basename,
                error="play_url_unavailable",
                retention=retention,
                requested_by=requested_by,
            )
            return None

        # 3. 下载音频
        ext = "mp3"  # 网易云 API 默认返回 mp3
        audio_path = os.path.join(ONLINE_AUDIO_DIR, f"{file_basename}.{ext}")
        downloaded = await self._download_file(play_url, audio_path, session)
        if not downloaded:
            self._write_failed_meta(
                title=title,
                artist=artist_str,
                song_id=song_id,
                album=album,
                duration=duration,
                file_basename=file_basename,
                error="audio_download_failed",
                retention=retention,
                requested_by=requested_by,
            )
            return None

        logger.info(f"音频下载成功: {file_basename}.{ext}")

        # 4. 保存歌词
        has_lyrics = False
        if lyrics_text:
            lrc_path = os.path.join(ONLINE_LYRICS_DIR, f"{file_basename}.lrc")
            try:
                with open(lrc_path, "w", encoding="utf-8") as f:
                    f.write(lyrics_text)
                logger.info(f"歌词保存成功: {file_basename}.lrc")
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
            "musicId": _meta_music_id(song_id),
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
            "acquire_status": "ready",
            "audio_retention": retention,
            "retention_reason": requested_by if retention == "saved" else "",
            "requested_by": requested_by,
            "is_trial": is_trial,
        }
        try:
            _write_meta_file(file_basename, meta)
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
            retention=retention,
            requested_by=requested_by,
            is_trial=is_trial,
        )

    def _write_failed_meta(
        self,
        *,
        title: str,
        artist: str,
        song_id: str,
        album: str,
        duration: int,
        file_basename: str,
        error: str,
        retention: str,
        requested_by: str,
    ) -> None:
        meta = {
            "musicId": _meta_music_id(song_id),
            "musicName": title,
            "artist": [[artist, 0]] if artist else [],
            "album": album or "Unknown",
            "duration": duration or 0,
            "format": "mp3",
            "source": "online",
            "source_platform": "netease",
            "source_id": song_id,
            "metadata_source": "netease",
            "acquired_at": datetime.now().isoformat(),
            "acquire_status": "failed",
            "acquire_error": error,
            "audio_retention": retention,
            "retention_reason": requested_by if retention == "saved" else "",
            "requested_by": requested_by,
        }
        try:
            _write_meta_file(file_basename, meta)
        except Exception as exc:
            logger.warning("失败元数据保存失败: %s", exc)

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
        retention="temporary",
        requested_by="auto_recommendation",
        is_trial=False,
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
            "audio_retention": retention,
            "requested_by": requested_by,
            "is_trial": bool(is_trial),
        }

    # ---- NeteaseAPI 辅助方法 ----

    async def _get_play_url(
        self, song_id: str, session: aiohttp.ClientSession
    ) -> tuple[Optional[str], bool]:
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
                            f"歌曲 {song_id} 为试听版 "
                            f"(freeTrialInfo: {trial_info.get('start',0)}-{trial_info.get('end',30)}s)"
                        )
                    return item["url"], trial_info is not None
        return None, False

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
                logger.info(f"已下载 {size_mb:.1f}MB -> {os.path.basename(save_path)}")
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

        from services.catalog_tier import resolve_ingest_tier

        for song in songs:
            title = _normalize_identity_text(song["title"])
            artist = _normalize_identity_text(song["artist"])
            music_id = _normalize_identity_text(song.get("song_id") or song.get("source_id"))
            source = str(song.get("source") or "online")
            cache_import_run_id = str(song.get("cache_import_run_id") or "")

            # ── 第零步：定档。自动飞轮抓来的只是临时候选，不进"我的曲库" ──
            # 单独查一次而不是塞进下面的 SET：同一个 SET 子句里 s.source 会先被
            # 改成 'online'，再用它判断层级就永远是 candidate，本地曲库会被误降级。
            tier_rows = client.execute_query(
                """MATCH (s:Song)
                   WHERE ($music_id <> '' AND
                          (toString(s.music_id) = $music_id OR toString(s.source_id) = $music_id))
                      OR (s.title = $title
                          AND (coalesce(s.artist, '') = $artist
                               OR EXISTS {
                                   MATCH (s)-[:PERFORMED_BY]->(a:Artist)
                                   WHERE a.name = $artist
                               }))
                   RETURN coalesce(s.catalog_tier, '') AS tier,
                          coalesce(s.source, '') AS source LIMIT 1""",
                {"music_id": music_id, "title": title, "artist": artist},
            )
            catalog_tier = resolve_ingest_tier(
                song,
                existing_tier=(tier_rows[0].get("tier") if tier_rows else ""),
                existing_source=(tier_rows[0].get("source") if tier_rows else ""),
                node_exists=bool(tier_rows),
            )

            # ── 第一步：检查是否已存在（通过关系匹配，兼容初始数据集） ──
            existing = client.execute_query(
                """MATCH (s:Song)
                WHERE ($music_id <> '' AND
                       (toString(s.music_id) = $music_id OR toString(s.source_id) = $music_id))
                   OR (s.title = $title
                       AND (coalesce(s.artist, '') = $artist
                            OR EXISTS {
                                MATCH (s)-[:PERFORMED_BY]->(a:Artist)
                                WHERE a.name = $artist
                            }))
                RETURN elementId(s) AS eid LIMIT 1""",
                {"music_id": music_id, "title": title, "artist": artist}
            )

            if existing:
                # ── 已存在：更新属性（不创建新节点） ──
                query = """
                MATCH (s:Song) WHERE elementId(s) = $eid
                SET s.title = $title,
                    s.music_id = $music_id,
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
                    s.audio_retention = $audio_retention,
                    s.audio_status = 'cached',
                    s.catalog_tier = $catalog_tier,
                    s.is_trial = $is_trial,
                    s.source = CASE WHEN coalesce(s.source, '') = '' THEN $source ELSE s.source END,
                    s.acquired_at = $acquired_at,
                    s.updated_at = timestamp()
                """
                logger.info(f"Neo4j 更新已有歌曲: {title} - {artist}")
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
                    s.audio_retention = $audio_retention,
                    s.audio_status = 'cached',
                    s.catalog_tier = $catalog_tier,
                    s.is_trial = $is_trial,
                    s.source = $source,
                    s.cache_import_run_id = CASE
                        WHEN $cache_import_run_id <> '' THEN $cache_import_run_id
                        ELSE s.cache_import_run_id END,
                    s.acquired_at = $acquired_at,
                    s.updated_at = timestamp()

                MERGE (a:Artist {name: $artist_name})
                MERGE (s)-[:PERFORMED_BY]->(a)
                """
                logger.info(f"Neo4j 秒级入库: {title} - {artist}")

            normalized_meta = normalize_acquisition_metadata(
                {
                    "musicId": song.get("song_id"),
                    "musicName": title,
                    "artist": [[artist, 0]],
                    "album": song.get("album", "Unknown"),
                    "album_id": song.get("album_id", ""),
                    "duration": song.get("duration", 0),
                    "format": song.get("ext", "mp3"),
                    "source": source,
                    "source_platform": song.get("platform") or "netease",
                    "metadata_source": song.get("metadata_source") or "netease",
                    "release_year": song.get("release_year"),
                    "cover_url": song.get("cover_url", ""),
                    "lrc_url": song.get("lrc_url", ""),
                }
            )
            params = {
                "eid": existing[0].get("eid") if existing else "",
                "music_id": music_id,
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
                "audio_retention": song.get("audio_retention") or "temporary",
                "catalog_tier": catalog_tier,
                "is_trial": bool(song.get("is_trial")),
                "acquired_at": datetime.now().isoformat(),
                "source": source,
                "cache_import_run_id": cache_import_run_id,
            }
            client.execute_query(query, params)
            _promote_external_candidate_feedback(client, title=title, artist=artist)

    except Exception as e:
        logger.error(f"Neo4j 快速入库失败: {e}")


def _song_identity_parameters(song: Dict[str, Any]) -> Dict[str, str]:
    return {
        "music_id": _normalize_identity_text(song.get("song_id") or song.get("source_id")),
        "title": _normalize_identity_text(song.get("title")),
        "artist_name": _normalize_identity_text(song.get("artist")),
    }


_SONG_IDENTITY_MATCH = """
MATCH (s:Song)
WHERE ($music_id <> '' AND
       (toString(s.music_id) = $music_id OR toString(s.source_id) = $music_id))
   OR (s.title = $title AND coalesce(s.artist, '') = $artist_name)
WITH s,
     CASE WHEN $music_id <> '' AND
                    (toString(s.music_id) = $music_id OR toString(s.source_id) = $music_id)
          THEN 0 ELSE 1 END AS identity_rank
ORDER BY identity_rank
LIMIT 1
"""


def _ingest_embedding_families() -> tuple[str, ...]:
    """Return the only vector families required by the selected runtime profile.

    The normal CUDA/ROCm ingestion path is MuQ + OMAR. M2D is deliberately
    isolated to the explicit CPU compatibility profile so a GPU worker never
    downloads or runs it merely to satisfy a stale completion check.
    """

    explicit = os.getenv("MUSIC_INGEST_EMBEDDING_PROFILE", "").strip().casefold()
    if explicit in {"cpu", "m2d", "m2d-cpu"}:
        return ("m2d2_embedding",)
    if explicit in {"gpu", "muq", "muq+omar", "cuda", "rocm"}:
        return ("muq_embedding", "omar_embedding")
    backend = os.getenv("DENSE_TEXT_AUDIO_BACKEND", "muq").strip().casefold()
    if backend == "m2d":
        return ("m2d2_embedding",)
    return ("muq_embedding", "omar_embedding")


async def _background_flywheel(songs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Enrich queued songs and fail the job when required retrieval vectors are absent."""
    logger.info("[后台飞轮] 开始处理 %s 首歌...", len(songs))

    from retrieval.neo4j_client import get_neo4j_client

    client = get_neo4j_client()
    embedding_families = _ingest_embedding_families()
    required_embeddings = set(embedding_families)
    logger.info("[后台飞轮] 向量档位: %s", ",".join(embedding_families))
    failures: list[str] = []
    warnings: list[str] = []
    song_results: list[dict[str, Any]] = []

    for song in songs:
        basename = song["file_basename"]
        ext = song["ext"]
        audio_path, lrc_path = _resolve_enrichment_paths(song, basename, ext)
        identity = _song_identity_parameters(song)
        song_result: dict[str, Any] = {
            "music_id": identity["music_id"],
            "title": identity["title"],
            "tagged": False,
            "tagging_status": "pending",
            "embedding_dimensions": {},
            "warnings": [],
        }

        tagging_mode = str(song.get("tagging_mode") or "api").strip().lower()
        if tagging_mode == "deferred":
            song_result["tagging_status"] = "deferred"
        elif os.path.exists(lrc_path):
            try:
                tags = await _extract_lyrics_tags(basename, lrc_path)
                if tags:
                    enriched_tags = prepare_tag_enrichment(tags, source="llm_lyrics")
                    tag_query = _SONG_IDENTITY_MATCH + """
                    SET s.title = $title,
                        s.artist = $artist_name,
                        s.vibe = $vibe,
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
                    RETURN elementId(s) AS eid
                    """
                    tag_rows = client.execute_query(tag_query, {
                        **identity,
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
                    if not tag_rows:
                        raise RuntimeError("catalog node was not found for tag update")
                    song_result["tagged"] = True
                    song_result["tagging_status"] = "completed"
                    logger.info("[后台飞轮] 歌词标签入库: %s", identity["title"])
            except Exception as exc:
                song_result["tagging_status"] = "failed"
                message = f"tags: {type(exc).__name__}: {exc}"
                song_result["warnings"].append(message)
                warnings.append(f"{identity['music_id'] or identity['title']}: {message}")
                logger.warning("[后台飞轮] 歌词标签提取失败 %s: %s", identity["title"], exc)
        else:
            song_result["tagging_status"] = "no_lyrics"

        if not os.path.exists(audio_path):
            failures.append(f"{identity['music_id'] or identity['title']}: audio file missing")
            song_results.append(song_result)
            continue

        extraction = await _extract_embeddings(audio_path, families=embedding_families)
        vectors = extraction.vectors
        embedding_query = _SONG_IDENTITY_MATCH + """
        SET s.m2d2_embedding = CASE WHEN size($m2d2_embedding) > 0
                                    THEN $m2d2_embedding ELSE s.m2d2_embedding END,
            s.omar_embedding = CASE WHEN size($omar_embedding) > 0
                                    THEN $omar_embedding ELSE s.omar_embedding END,
            s.muq_embedding = CASE WHEN size($muq_embedding) > 0
                                   THEN $muq_embedding ELSE s.muq_embedding END,
            s.enrichment_status = $enrichment_status,
            s.enrichment_error = $enrichment_error,
            s.updated_at = timestamp()
        RETURN elementId(s) AS eid,
               size(coalesce(s.m2d2_embedding, [])) AS m2d2,
               size(coalesce(s.omar_embedding, [])) AS omar,
               size(coalesce(s.muq_embedding, [])) AS muq
        """
        missing_required = sorted(required_embeddings - {key for key, value in vectors.items() if value})
        error_text = "; ".join(f"{name}: {error}" for name, error in sorted(extraction.errors.items()))
        embed_rows = client.execute_query(embedding_query, {
            **identity,
            "m2d2_embedding": vectors.get("m2d2_embedding", []),
            "omar_embedding": vectors.get("omar_embedding", []),
            "muq_embedding": vectors.get("muq_embedding", []),
            "enrichment_status": "failed" if missing_required else "ready",
            "enrichment_error": error_text[:1000],
        })
        if not embed_rows:
            failures.append(f"{identity['music_id'] or identity['title']}: catalog node not found")
        else:
            dimensions = embed_rows[0]
            song_result["embedding_dimensions"] = {
                "m2d2": int(dimensions.get("m2d2") or 0),
                "omar": int(dimensions.get("omar") or 0),
                "muq": int(dimensions.get("muq") or 0),
            }
        if missing_required:
            failures.append(
                f"{identity['music_id'] or identity['title']}: missing {','.join(missing_required)}"
            )
        for name, error in sorted(extraction.errors.items()):
            message = f"{name}: {error}"
            song_result["warnings"].append(message)
            warnings.append(f"{identity['music_id'] or identity['title']}: {message}")
        song_results.append(song_result)

    result = {
        "song_count": len(songs),
        "songs": song_results,
        "warnings": warnings,
        "failure_count": len(failures),
    }
    if failures:
        raise EnrichmentIncompleteError("; ".join(failures)[:1000])
    logger.info("[后台飞轮] 全部完成，%s 首歌已完成必要增强", len(songs))
    return result


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


async def _extract_embeddings(
    audio_path: str,
    *,
    families: tuple[str, ...] | None = None,
) -> EmbeddingExtraction:
    """Extract independent embeddings outside the event loop."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        None,
        lambda: _sync_extract_embeddings(audio_path, families=families),
    )


def _sync_extract_embeddings(
    audio_path: str,
    *,
    families: tuple[str, ...] | None = None,
) -> EmbeddingExtraction:
    """Extract each model independently so one optional anchor cannot erase the others."""
    import librosa

    selected = tuple(dict.fromkeys(families or _ingest_embedding_families()))
    supported = {"muq_embedding", "m2d2_embedding", "omar_embedding"}
    unknown = set(selected) - supported
    if unknown:
        raise ValueError(f"unsupported embedding families: {sorted(unknown)}")

    MAX_SECONDS = 300
    file_duration = librosa.get_duration(path=audio_path)
    load_duration = MAX_SECONDS if file_duration > MAX_SECONDS else None

    audio_np, sr = librosa.load(audio_path, sr=None, mono=True, duration=load_duration)

    result = EmbeddingExtraction()
    extractors: list[tuple[str, Any]] = []
    if "muq_embedding" in selected:
        from retrieval.muq_embedder import encode_audio_to_muq

        audio_24k = librosa.resample(audio_np, orig_sr=sr, target_sr=24000)
        extractors.append(
            ("muq_embedding", lambda: encode_audio_to_muq(audio_24k, sample_rate=24000))
        )
    if {"m2d2_embedding", "omar_embedding"} & set(selected):
        from retrieval import audio_embedder

        audio_16k = librosa.resample(audio_np, orig_sr=sr, target_sr=16000)
        if "m2d2_embedding" in selected:
            extractors.append(
                (
                    "m2d2_embedding",
                    lambda: audio_embedder.encode_audio_to_embedding(audio_16k, sample_rate=16000),
                )
            )
        if "omar_embedding" in selected:
            extractors.append(
                (
                    "omar_embedding",
                    lambda: audio_embedder.extract_audio_representation(audio_16k, sample_rate=16000),
                )
            )
    for name, extractor in extractors:
        try:
            vector = extractor()
            result.vectors[name] = list(vector or [])
        except Exception as exc:
            result.vectors[name] = []
            result.errors[name] = f"{type(exc).__name__}: {exc}"[:500]
            logger.warning("%s 提取失败: %s", name, exc)
    return result


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
    logger.info(f"开始联网获取 {len(song_queries)} 首歌曲")

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
