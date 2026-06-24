import json
import logging
import os
import re
import asyncio
import concurrent.futures
import time
from typing import List, Dict, Any, Optional

from tools.semantic_search import semantic_search
from tools.web_search_aggregator import _federated_search_async
from retrieval.recall_sources import (
    cold_start_recall,
    graph_candidate_recall,
    lexical_bm25_recall,
    personalized_recall,
)
from retrieval.retrieval_fusion import (
    apply_hard_filters,
    normalize_song_key,
    recall_weights_for_intent,
    weighted_rrf,
)
from retrieval.neo4j_client import get_neo4j_client
from config.logging_config import get_logger
from config.settings import settings
from schemas.music_state import ToolOutput


logger = get_logger(__name__)

# ---- 检索结果融合常量 ----
BASELINE_SIMILARITY_SCORE = 0.85      # 单引擎命中的基础相似度分（仅作 fallback）
WEB_RESULT_PRIORITY_SCORE = 9.9       # 网络资讯结果的优先级分数
MIN_DIVERSE_RESULTS = 6               # 多样性过滤后的最少结果数

# ---- Neo4j 图距离加权参数（从 settings 读取，此处为 fallback 默认值） ----
GRAPH_AFFINITY_USER_ID = "local_admin" # 图距离计算的用户 ID

# ---- 用户偏好缓存（启动时加载一次，避免每次请求都查 Neo4j） ----
_user_pref_cache: dict = {}  # {user_id: {genres: set, moods: set, themes: set, scenarios: set, expanded_genres: set}}

def _load_user_preferences(user_id: str = GRAPH_AFFINITY_USER_ID) -> dict:
    """
    从 Neo4j 加载用户偏好并缓存。
    首次调用时查询数据库，后续直接返回缓存。
    """
    if user_id in _user_pref_cache:
        return _user_pref_cache[user_id]

    import json as _json
    empty_prefs = {"genres": set(), "moods": set(), "themes": set(), "scenarios": set(), "expanded_genres": set()}
    
    try:
        from retrieval.neo4j_client import get_neo4j_client
        neo4j = get_neo4j_client()
        if not neo4j or not neo4j.driver:
            _user_pref_cache[user_id] = empty_prefs
            return empty_prefs
    except Exception:
        _user_pref_cache[user_id] = empty_prefs
        return empty_prefs

    try:
        pref_query = """
        MATCH (u:User {id: $uid})
        OPTIONAL MATCH (u)-[:LIKES|LISTENED_TO]->(s:Song)
        OPTIONAL MATCH (s)-[:HAS_MOOD]->(m:Mood)
        OPTIONAL MATCH (s)-[:HAS_THEME]->(t:Theme)
        OPTIONAL MATCH (s)-[:FITS_SCENARIO]->(sc:Scenario)
        RETURN u.preferred_genres AS pg,
               collect(DISTINCT m.name) AS moods,
               collect(DISTINCT t.name) AS themes,
               collect(DISTINCT sc.name) AS scenarios
        """
        pref_result = neo4j.execute_query(pref_query, {"uid": user_id})

        prefs = dict(empty_prefs)  # copy
        if pref_result and pref_result[0]:
            row = pref_result[0]
            raw_pg = row.get("pg")
            if raw_pg:
                try:
                    parsed = _json.loads(raw_pg)
                    prefs["genres"] = {g.strip().lower() for g in parsed if g.strip()}
                except (ValueError, TypeError):
                    pass
            prefs["moods"] = {x.strip().lower() for x in (row.get("moods") or []) if x and x.strip()}
            prefs["themes"] = {x.strip().lower() for x in (row.get("themes") or []) if x and x.strip()}
            prefs["scenarios"] = {x.strip().lower() for x in (row.get("scenarios") or []) if x and x.strip()}

        # 展开 Genre 偏好（中文 → 英文别名映射）
        try:
            from tools.graphrag_search import GENRE_TAG_MAP
        except ImportError:
            GENRE_TAG_MAP = {}

        expanded: set = set()
        for pref in prefs["genres"]:
            for key, aliases in GENRE_TAG_MAP.items():
                if key.lower() == pref or pref in key.lower():
                    expanded.update(a.lower() for a in aliases)
                    break
            else:
                expanded.add(pref)
        prefs["expanded_genres"] = expanded

        _user_pref_cache[user_id] = prefs
        logger.info(
            f"[PrefCache] 用户偏好已缓存: genre={len(prefs['genres'])}, mood={len(prefs['moods'])}, "
            f"theme={len(prefs['themes'])}, scenario={len(prefs['scenarios'])}"
        )
        return prefs
    except Exception as e:
        logger.warning(f"[PrefCache] 加载用户偏好失败: {e}")
        _user_pref_cache[user_id] = empty_prefs
        return empty_prefs


def invalidate_user_pref_cache(user_id: str = GRAPH_AFFINITY_USER_ID):
    """当用户偏好更新时（如 LIKES 新歌），调用此函数清除缓存。"""
    _user_pref_cache.pop(user_id, None)
    logger.info(f"[PrefCache] 已清除用户 {user_id} 的偏好缓存")


class MusicHybridRetrieval:
    """
    音乐检索管理器。
    接收上层的 Query，根据检索计划分发给图谱检索和向量检索引擎执行，
    最后汇总排版返回给大模型。
    """
    
    def __init__(self, llm_client=None):
        # 保存 llm_client 引用（预留，供未来扩展使用）
        self.llm_client = llm_client
        self._disliked_cache: set = None  # 同一请求内缓存

    def _get_disliked_titles(self, user_id: str = GRAPH_AFFINITY_USER_ID) -> set:
        """查询用户 DISLIKES 的歌曲标题集合（同一实例内缓存）"""
        if self._disliked_cache is not None:
            return self._disliked_cache
        try:
            from retrieval.neo4j_client import get_neo4j_client
            client = get_neo4j_client()
            query = """
            MATCH (u:User {id: $uid})-[:DISLIKES]->(s:Song)
            RETURN collect(s.title) AS titles
            """
            result = client.execute_query(query, {"uid": user_id})
            self._disliked_cache = set(result[0]["titles"]) if result and result[0].get("titles") else set()
            if self._disliked_cache:
                logger.info(f"[DislikeFilter] 加载到 {len(self._disliked_cache)} 首不喜欢的歌")
            return self._disliked_cache
        except Exception as e:
            logger.warning(f"[DislikeFilter] 查询失败: {e}")
            self._disliked_cache = set()
            return self._disliked_cache

    async def retrieve(self, query: str, limit: int = 5, precomputed_plan: dict = None) -> ToolOutput:
        """
        主检索入口：五路并行召回 → RRF → 硬过滤 → 统一排序。
        
        Args:
            query: 用户查询
            limit: 返回结果数量
            precomputed_plan: 来自上游统一 Prompt 的预计算检索计划（dict 格式的 RetrievalPlan）。
                              layered 字段优先；legacy 字段仅用于历史调用兼容。
        """
        retrieval_started = time.perf_counter()
        timings: Dict[str, float] = {}
        logger.info(f"[Retrieval] 开始处理请求: {query}")
        if os.getenv("MUSIC_MOCK_MODE", "0").lower() in {"1", "true", "yes"}:
            from retrieval.mock_retrieval import mock_retrieve
            logger.info("[Retrieval] MUSIC_MOCK_MODE enabled")
            return mock_retrieve(query, limit)

        from config.settings import settings as _s
        recall_limit = max(
            _s.graph_search_limit,
            _s.semantic_search_limit,
            _s.mixed_retrieval_limit,
            limit,
        )

        # 1. 直接消费分层计划。legacy 字段只在 layered 字段缺失时补位。
        plan = precomputed_plan or {}
        hard_constraints = dict(plan.get("hard_constraints") or {})
        soft_intent = dict(plan.get("soft_intent") or {})
        hints = dict(plan.get("hints") or {})

        if not hard_constraints:
            legacy_language = plan.get("graph_language_filter")
            hard_constraints = {
                "artist_entities": list(plan.get("graph_artist_entities") or []),
                "song_entities": list(plan.get("graph_song_entities") or []),
                "language": legacy_language,
                "region": plan.get("graph_region_filter"),
                "instrumental": str(legacy_language or "").casefold() == "instrumental",
            }
        if not hints:
            hints = {
                "genres": [plan["graph_genre_filter"]] if plan.get("graph_genre_filter") else [],
                "mood": plan.get("graph_mood_filter"),
                "scenario": plan.get("graph_scenario_filter"),
            }

        graph_artist_entities = list(hard_constraints.get("artist_entities") or [])
        graph_song_entities = list(hard_constraints.get("song_entities") or [])
        graph_entities = list(dict.fromkeys(graph_artist_entities + graph_song_entities))
        intent_type = str(plan.get("_intent_type") or "hybrid_search")
        recall_weights = recall_weights_for_intent(intent_type)
        need_web_search = bool(plan.get("use_web_search"))
        search_keyword = str(plan.get("web_search_keywords") or query)

        vector_desc = str(plan.get("vector_acoustic_query") or "").strip()
        if not vector_desc:
            soft_parts = [
                soft_intent.get("goal", ""),
                soft_intent.get("trajectory", ""),
                soft_intent.get("vibe", ""),
                "avoid: " + ", ".join(soft_intent.get("avoid", []))
                if soft_intent.get("avoid")
                else "",
                "genres: " + ", ".join(hints.get("genres", []))
                if hints.get("genres")
                else "",
                f"mood: {hints.get('mood')}" if hints.get("mood") else "",
                f"scenario: {hints.get('scenario')}" if hints.get("scenario") else "",
            ]
            vector_desc = "; ".join(str(part) for part in soft_parts if part)
        if not vector_desc:
            vector_desc = query

        lexical_query = (
            " ".join(graph_artist_entities + graph_song_entities)
            if graph_entities
            else query
        )

        similarity_seed_terms = (
            "类似",
            "相似",
            "听感",
            "像",
            "同类",
            "similar",
            "same vibe",
            "sounds like",
            "like this",
        )
        similarity_context = " ".join(
            [
                query,
                str(soft_intent.get("goal") or ""),
                str(soft_intent.get("vibe") or ""),
                vector_desc,
            ]
        ).casefold()
        filter_hard_constraints = dict(hard_constraints)
        if graph_song_entities and any(term in similarity_context for term in similarity_seed_terms):
            filter_hard_constraints["song_entities"] = []
            logger.info(
                "[Retrieval] song_entities=%s 作为相似听感参考种子，不进入最终硬过滤",
                graph_song_entities,
            )

        logger.info(
            "[Retrieval] 分层计划: intent=%s | hard=%s | soft=%s | hints=%s | weights=%s",
            intent_type,
            {
                "artists": graph_artist_entities,
                "songs": graph_song_entities,
                "language": hard_constraints.get("language"),
                "region": hard_constraints.get("region"),
                "instrumental": bool(hard_constraints.get("instrumental")),
            },
            {
                "goal": bool(soft_intent.get("goal")),
                "trajectory": bool(soft_intent.get("trajectory")),
                "avoid": len(soft_intent.get("avoid") or []),
                "vibe": bool(soft_intent.get("vibe")),
            },
            hints,
            recall_weights,
        )

        loop = asyncio.get_running_loop()

        async def run_sync_in_executor(func, *args, **kwargs):
            return await loop.run_in_executor(None, lambda: func(*args, **kwargs))

        recall_source_timeout = float(os.getenv("RECALL_SOURCE_TIMEOUT_SECONDS", "45"))

        async def timed_recall(source: str, awaitable):
            started = time.perf_counter()
            try:
                return await asyncio.wait_for(awaitable, timeout=recall_source_timeout)
            except asyncio.TimeoutError:
                logger.warning(
                    "[Recall:%s] 超时 %.1fs，跳过本路召回（其它召回继续参与 RRF）",
                    source,
                    recall_source_timeout,
                )
                return ""
            finally:
                timings[f"recall_{source}_ms"] = round(
                    (time.perf_counter() - started) * 1000,
                    3,
                )

        # 2. 五路召回永远一起运行；intent_type 只改变 RRF 权重。
        recall_tasks = {
            "graph": timed_recall("graph", run_sync_in_executor(
                graph_candidate_recall,
                hard_constraints,
                hints,
                limit=recall_limit,
            )),
            "dense": timed_recall("dense", run_sync_in_executor(
                semantic_search.invoke,
                {"query": vector_desc, "limit": recall_limit},
            )),
            "lexical": timed_recall("lexical", run_sync_in_executor(
                lexical_bm25_recall,
                lexical_query,
                limit=recall_limit,
            )),
            "personal": timed_recall("personal", run_sync_in_executor(
                personalized_recall,
                GRAPH_AFFINITY_USER_ID,
                limit=recall_limit,
            )),
            "cold": timed_recall("cold", run_sync_in_executor(
                cold_start_recall,
                limit=recall_limit,
            )),
        }
        recall_results = await asyncio.gather(*recall_tasks.values(), return_exceptions=True)
        source_raw: Dict[str, str] = {}
        for source, result in zip(recall_tasks, recall_results):
            if isinstance(result, Exception):
                logger.error("[Recall:%s] 异常: %s: %s", source, type(result).__name__, result)
                source_raw[source] = ""
            else:
                source_raw[source] = result or ""
                try:
                    count = len(json.loads(source_raw[source])) if source_raw[source] else 0
                except (json.JSONDecodeError, TypeError):
                    count = 0
                logger.info("[Recall:%s] 返回 %d 条", source, count)

        async def _extract_and_fetch_web_songs(web_text: str) -> List[dict]:
            if not web_text or "未能找到" in web_text or not self.llm_client:
                return []

            try:
                from pydantic import BaseModel, Field

                class WebSongTarget(BaseModel):
                    title: str = Field(description="歌曲名称")
                    artist: str = Field(description="歌手名")

                class WebSongExtraction(BaseModel):
                    songs: List[WebSongTarget] = Field(description="从文字中提取出的推荐歌曲列表，最多3首")

                structured_llm = self.llm_client.with_structured_output(WebSongExtraction)
                prompt = (
                    "请从以下全网搜索资讯中提取最多3首代表性歌曲及歌手，"
                    "并严格返回符合 schema 的 JSON；没有明确歌曲时返回空 songs 列表。"
                    f"\n\n资讯文本:\n{web_text}"
                )

                result = await structured_llm.ainvoke(prompt)
                if not result or not result.songs:
                    return []

                from tools.music_fetch_tool import execute_search_online_music

                tasks = [
                    execute_search_online_music(f"{song.artist} {song.title}")
                    for song in result.songs[:3]
                ]
                fetch_results = await asyncio.gather(*tasks, return_exceptions=True)

                playable_songs = []
                for index, fetched in enumerate(fetch_results):
                    if isinstance(fetched, Exception):
                        continue
                    if getattr(fetched, "success", False) and fetched.data:
                        target = result.songs[index]
                        top_hit = fetched.data[0]
                        playable_songs.append(
                            {
                                "song": {
                                    "title": top_hit.get("title", target.title),
                                    "artist": top_hit.get("artist", target.artist),
                                    "preview_url": top_hit.get("play_url")
                                    or top_hit.get("preview_url"),
                                    "cover_url": top_hit.get("cover_url"),
                                    "album": top_hit.get("album", "未知"),
                                    "genre": "Web Trends",
                                    "source": "online_search",
                                    "recall_sources": ["web"],
                                    "recall_source_labels": ["联网"],
                                },
                                "reason": "🌐 全网最新发掘",
                                "similarity_score": 9.5 - (index * 0.1),
                                "_recall_sources": ["web"],
                                "_recall_source_labels": ["联网"],
                            }
                        )
                return playable_songs
            except Exception as exc:
                logger.error("提取全网歌曲失败: %s", exc)
                return []

        # 3. 联网只作为显式需求或实体本地零召回后的补充，不参与本地引擎开关。
        web_raw = ""
        web_started = time.perf_counter()
        if os.environ.get("MUSIC_WEB_SEARCH_ENABLED", "1") != "0":
            graph_empty = source_raw.get("graph") in ("", "[]")
            lexical_empty = source_raw.get("lexical") in ("", "[]")
            if need_web_search:
                logger.info("⚡ 意图明确要求联网: '%s'", search_keyword)
                web_raw = await _federated_search_async(search_keyword)
            elif graph_entities and graph_empty and lexical_empty:
                logger.warning("本地实体召回为空，触发联网补充: '%s'", query)
                web_raw = await _federated_search_async(query)

        web_playable = await _extract_and_fetch_web_songs(web_raw)
        timings["retrieval_web_ms"] = round((time.perf_counter() - web_started) * 1000, 3)
        self._current_query = query
        self._current_hyde_text = vector_desc
        self._current_hard_constraints = filter_hard_constraints

        result = self._format_results(
            source_raw=source_raw,
            recall_weights=recall_weights,
            hard_constraints=filter_hard_constraints,
            web_res=web_raw,
            web_playable=web_playable,
            graph_entities=graph_entities,
            final_limit=limit,
            timings=timings,
        )
        result.metadata.setdefault("timings", timings)
        result.metadata["timings"]["retrieval_total_ms"] = round(
            (time.perf_counter() - retrieval_started) * 1000,
            3,
        )
        return result


    @staticmethod
    def _normalize_key(title: str, artist: str) -> str:
        """生成标准化的去重 key，消除全角/半角、标点、空格差异。"""
        return normalize_song_key(title, artist)

    @staticmethod
    def _parse_engine_results(res_str: str, engine_name: str) -> List[dict]:
        """将引擎原始 JSON 字符串解析为标准化的歌曲列表，保留原始排名。"""
        def _clean_list(value):
            if not value:
                return []
            if isinstance(value, list):
                return [x for x in value if x]
            if isinstance(value, str):
                return [value] if value and value != "Unknown" else []
            return []

        if not res_str:
            return []
        try:
            items = json.loads(res_str)
        except json.JSONDecodeError:
            logger.error(f"Failed to decode JSON from {engine_name}: {res_str[:200]}")
            return []

        results = []
        seen_keys = set()  # 引擎内部去重
        for rank, item in enumerate(items):
            if "error" in item:
                logger.warning(f"Engine {engine_name} returned error: {item}")
                continue

            title = item.get("title", "未知标题")
            artist = item.get("artist", "未知艺术家")
            genre = item.get("genre", "")
            if genre == "Unknown":
                genre = ""

            # 标准化 key（消除全角/半角、标点差异）
            key = MusicHybridRetrieval._normalize_key(title, artist)
            if key in seen_keys:
                logger.info(f"[{engine_name}] 引擎内部去重: '{title}' - '{artist}'")
                continue
            seen_keys.add(key)

            # 提取原始相似度（RRF 不直接使用，但保留用于日志和 MMR）
            raw_distance = item.get("distance", None)
            raw_similarity = item.get("similarity_score", None)
            if raw_distance is not None:
                raw_score = 1.0 / (1.0 + float(raw_distance))
            elif raw_similarity is not None:
                raw_score = float(raw_similarity)
            else:
                raw_score = BASELINE_SIMILARITY_SCORE

            results.append({
                "key": key,
                "rank": rank,          # 该引擎中的排名（0-based）
                "raw_score": raw_score,
                "engine": engine_name,
                "song": {
                    "title": title,
                    "artist": artist,
                    "album": item.get("album", "未知"),
                    "genre": genre,
                    "genres": _clean_list(item.get("genres")),
                    "moods": _clean_list(item.get("moods")),
                    "themes": _clean_list(item.get("themes")),
                    "scenarios": _clean_list(item.get("scenarios")),
                    "language": item.get("language", "Unknown"),
                    "region": item.get("region", "Unknown"),
                    "vector_backend": item.get("vector_backend"),
                    "preview_url": item.get("preview_url", None),
                    "cover_url": item.get("cover_url", None),
                    "lrc_url": item.get("lrc_url", None),
                },
            })
        return results

    @staticmethod
    def _fuse_recall_sources(
        source_items: Dict[str, List[dict]],
        recall_weights: Dict[str, float],
    ) -> List[dict]:
        """Fuse all ranked recall lists with weighted Reciprocal Rank Fusion."""
        merged = weighted_rrf(source_items, recall_weights)
        multi_source_count = sum(
            1 for item in merged if len(item.get("_recall_sources", [])) > 1
        )
        logger.info(
            "[RRF] 五路融合完成: %d 首，多路交叉命中=%d",
            len(merged),
            multi_source_count,
        )
        return merged

    # ================================================================
    # 三锚归一化精排（替代双锚，语义+声学+个性化三维融合）
    # ================================================================
    @staticmethod
    def _tri_anchor_rerank(
        candidates: List[dict],
        query_text: str,
    ) -> List[dict]:
        """
        三锚归一化精排 = 语义相关性 + 声学风格 + 个性化偏好

        所有分数归一化到 [0, 1] 后加权融合：
        final_score = α×semantic + β×acoustic + γ×personalize
        
        - semantic:   M2D-CLAP cosine(song_emb, query_text_emb) → (x+1)/2
        - acoustic:   OMAR-RQ cosine(song_emb, centroid) → (x+1)/2
        - personalize: graph_affinity (已在 Step 4 计算) → MinMax 归一化
        
        权重从 settings 读取，自动归一化使 α+β+γ=1。
        """
        if not candidates:
            return candidates

        from config.settings import settings as _s
        w_sem = _s.tri_anchor_w_semantic
        w_aco = _s.tri_anchor_w_acoustic
        w_per = _s.tri_anchor_w_personal

        # 自动归一化权重
        w_total = w_sem + w_aco + w_per
        if w_total > 0:
            w_sem, w_aco, w_per = w_sem / w_total, w_aco / w_total, w_per / w_total
        else:
            w_sem, w_aco, w_per = 0.45, 0.30, 0.25

        try:
            import numpy as np
            from retrieval.neo4j_client import get_neo4j_client
            from retrieval.audio_embedder import encode_text_to_embedding

            neo4j = get_neo4j_client()
            if not neo4j or not neo4j.driver:
                logger.warning("[TriAnchor] Neo4j 不可用，跳过三锚精排")
                return candidates

            # ── 语义锚：query → text embedding ──
            # 首次启动时 HuggingFace 文本编码器可能仍在下载，不能让精排阻塞整条推荐链。
            tri_text_timeout = float(os.getenv("TRI_ANCHOR_TEXT_TIMEOUT_SECONDS", "8"))
            query_emb = None
            executor = None
            try:
                logger.info("[TriAnchor] 编码 query text embedding...")
                executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
                future = executor.submit(encode_text_to_embedding, query_text)
                query_emb = np.array(future.result(timeout=tri_text_timeout))
            except concurrent.futures.TimeoutError:
                logger.warning(
                    "[TriAnchor] query text embedding 超时 %.1fs，跳过语义锚，保留声学/个性化排序",
                    tri_text_timeout,
                )
            finally:
                if executor is not None:
                    executor.shutdown(wait=False, cancel_futures=True)

            # ── 批量获取候选歌曲的 M2D + OMAR embedding ──
            titles = [c["song"]["title"] for c in candidates if c.get("song", {}).get("title")]
            emb_cypher = """
            UNWIND $titles AS t
            MATCH (s:Song {title: t})
            RETURN s.title AS title,
                   s.m2d2_embedding AS m2d_emb,
                   s.omar_embedding AS omar_emb
            """
            emb_rows = neo4j.execute_query(emb_cypher, {"titles": titles})

            m2d_map, omar_map = {}, {}
            for row in (emb_rows or []):
                t = row.get("title", "")
                if row.get("m2d_emb"):
                    m2d_map[t] = np.array(row["m2d_emb"])
                if row.get("omar_emb"):
                    omar_map[t] = np.array(row["omar_emb"])

            logger.info(
                f"[TriAnchor] embedding 命中: M2D={len(m2d_map)}/{len(titles)}, "
                f"OMAR={len(omar_map)}/{len(titles)}"
            )

            # ── 声学锚：OMAR 质心 ──
            omar_centroid = None
            if omar_map:
                omar_centroid = np.mean(list(omar_map.values()), axis=0)

            def _cosine(a, b):
                dot = np.dot(a, b)
                norm = np.linalg.norm(a) * np.linalg.norm(b)
                return float(dot / norm) if norm > 0 else 0.0

            def _normalize_cosine(score: float) -> float:
                """cosine [-1, 1] → [0, 1]"""
                return (score + 1.0) / 2.0

            # ── 收集 personalize 原始值用于 MinMax 归一化 ──
            raw_personalize = [c.get("_graph_affinity", 0.0) for c in candidates]
            p_min = min(raw_personalize) if raw_personalize else 0
            p_max = max(raw_personalize) if raw_personalize else 0
            p_range = p_max - p_min if p_max > p_min else 1.0

            # ── 三锚融合评分 ──
            for c in candidates:
                title = c.get("song", {}).get("title", "")

                # 维度 1: 语义分（归一化到 [0,1]）
                m2d_emb = m2d_map.get(title)
                if query_emb is not None and m2d_emb is not None:
                    raw_semantic = _cosine(m2d_emb, query_emb)
                    semantic = _normalize_cosine(raw_semantic)
                else:
                    semantic = 0.5  # 无 embedding 时给中位分

                # 维度 2: 声学分（归一化到 [0,1]）
                omar_emb = omar_map.get(title)
                if omar_emb is not None and omar_centroid is not None:
                    raw_acoustic = _cosine(omar_emb, omar_centroid)
                    acoustic = _normalize_cosine(raw_acoustic)
                else:
                    acoustic = 0.5

                # 维度 3: 个性化分（MinMax 归一化到 [0,1]）
                raw_p = c.get("_graph_affinity", 0.0)
                personalize = (raw_p - p_min) / p_range if p_range > 0 else 0.5

                # 动态权重（缺 OMAR 时重分配声学权重）
                if omar_emb is None or omar_centroid is None:
                    _w_sem = w_sem / (w_sem + w_per) if (w_sem + w_per) > 0 else 0.6
                    _w_per = w_per / (w_sem + w_per) if (w_sem + w_per) > 0 else 0.4
                    final = _w_sem * semantic + _w_per * personalize
                else:
                    final = w_sem * semantic + w_aco * acoustic + w_per * personalize

                c["similarity_score"] = round(final, 6)
                c["_semantic_score"] = round(semantic, 4)
                c["_acoustic_score"] = round(acoustic, 4)
                c["_personal_score"] = round(personalize, 4)

            candidates.sort(key=lambda x: x["similarity_score"], reverse=True)

            logger.info(
                f"[TriAnchor] 三锚精排完成 (sem={w_sem:.2f}, aco={w_aco:.2f}, per={w_per:.2f}) | "
                f"Top3: {[(c['song']['title'], round(c['similarity_score'], 4)) for c in candidates[:3]]}"
            )
            return candidates

        except Exception as e:
            logger.warning(f"[TriAnchor] 三锚精排异常（降级保持原排序）: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return candidates

    # ================================================================
    # 【P0 升级 + 优化】Neo4j 图距离亲和力评分
    # 原始版本：3 次 Neo4j 查询（图距离 + 用户偏好 + 候选标签）
    # 优化版本：1 次合并 Cypher 查询（图距离 + 候选标签）+ 缓存用户偏好
    #          延迟从 ~100-150ms 降至 ~40-60ms
    # ================================================================

    @staticmethod
    def _compute_graph_affinity(
        candidates: List[dict],
        user_id: str = GRAPH_AFFINITY_USER_ID,
        max_hops: int = None,
    ) -> tuple:
        """
        为每首候选歌曲计算与用户的图距离亲和力 + 用户画像偏好加分。

        Returns:
            (candidates, cand_tag_map)
            - candidates: 加上 _graph_affinity 字段的候选列表
            - cand_tag_map: {title: {moods: set, themes: set, scenarios: set}}
              供下游 MMR 多维多样性重排使用
        """
        from config.settings import settings as _s
        if max_hops is None:
            max_hops = _s.graph_affinity_max_hops

        empty_tag_map = {}
        if not candidates:
            return candidates, empty_tag_map

        try:
            from retrieval.neo4j_client import get_neo4j_client
            neo4j = get_neo4j_client()
            if not neo4j or not neo4j.driver:
                logger.warning("[GraphAffinity] Neo4j 不可用，跳过图距离计算")
                for c in candidates:
                    c["_graph_affinity"] = 0.0
                return candidates, empty_tag_map
        except Exception:
            logger.warning("[GraphAffinity] 无法导入 Neo4j 客户端，跳过")
            for c in candidates:
                c["_graph_affinity"] = 0.0
            return candidates, empty_tag_map

        titles = [c["song"]["title"] for c in candidates if c.get("song", {}).get("title")]
        if not titles:
            for c in candidates:
                c["_graph_affinity"] = 0.0
            return candidates, empty_tag_map

        # ── Step A: 用户偏好（从缓存读取，首次自动加载） ──
        user_prefs = _load_user_preferences(user_id)
        user_pref_genres = user_prefs["genres"]
        user_pref_moods = user_prefs["moods"]
        user_pref_themes = user_prefs["themes"]
        user_pref_scenarios = user_prefs["scenarios"]
        expanded_genre_prefs = user_prefs["expanded_genres"]
        has_any_pref = user_pref_genres or user_pref_moods or user_pref_themes or user_pref_scenarios

        # ── Step B: 合并查询（图距离 + 候选歌曲标签，1 次 Neo4j round-trip） ──
        combined_query = """
        MATCH (u:User {id: $user_id})
        UNWIND $titles AS candidate_title
        OPTIONAL MATCH (s:Song)
          WHERE s.title = candidate_title
        OPTIONAL MATCH path = shortestPath(
          (u)-[*1..""" + str(max_hops) + """]->(s)
        )
        OPTIONAL MATCH (s)-[:HAS_MOOD]->(m:Mood)
        OPTIONAL MATCH (s)-[:HAS_THEME]->(th:Theme)
        OPTIONAL MATCH (s)-[:FITS_SCENARIO]->(sc:Scenario)
        RETURN candidate_title AS title,
               CASE WHEN path IS NOT NULL THEN length(path) ELSE -1 END AS distance,
               collect(DISTINCT m.name) AS moods,
               collect(DISTINCT th.name) AS themes,
               collect(DISTINCT sc.name) AS scenarios
        """

        cand_tag_map = {}
        try:
            results = neo4j.execute_query(combined_query, {"user_id": user_id, "titles": titles})

            distance_map = {}
            for r in results:
                t = r.get("title", "")
                distance_map[t] = r.get("distance", -1)
                cand_tag_map[t] = {
                    "moods": {x.strip().lower() for x in (r.get("moods") or []) if x and x.strip()},
                    "themes": {x.strip().lower() for x in (r.get("themes") or []) if x and x.strip()},
                    "scenarios": {x.strip().lower() for x in (r.get("scenarios") or []) if x and x.strip()},
                }

            for c in candidates:
                title = c.get("song", {}).get("title", "")
                dist = distance_map.get(title, -1)
                if dist == 1:
                    # 直接交互过的歌（LIKES/LISTENED_TO, 1 hop）
                    # 轻度降权：语义/声学高度匹配时仍可翻盘
                    c["_graph_affinity"] = -0.2
                    c["_affinity_reason"] = "已知歌曲(轻度降权)"
                elif dist > 1:
                    # 间接关联（共享标签/类似偏好的新歌）→ 加权
                    # dist=2 为最佳发现候选（加分最高），距离越远加分越少
                    c["_graph_affinity"] = 1.0 / (dist - 1)  # dist=2→1.0, dist=3→0.5, dist=4→0.33
                    c["_affinity_reason"] = f"发现候选({dist}hop)"
                else:
                    # 无图谱关联 → 中性
                    c["_graph_affinity"] = 0.0
                    c["_affinity_reason"] = "无关联"

            known_cnt = sum(1 for c in candidates if c.get("_graph_affinity", 0) < 0)
            discovery_cnt = sum(1 for c in candidates if c.get("_graph_affinity", 0) > 0)
            logger.info(
                f"[GraphAffinity] 合并查询完成（1 次 Neo4j）: "
                f"已知歌曲(降权)={known_cnt}, 发现候选(加分)={discovery_cnt}, "
                f"无关联={len(candidates) - known_cnt - discovery_cnt}"
            )
        except Exception as e:
            logger.warning(f"[GraphAffinity] 合并查询失败（降级为不加权）: {e}")
            for c in candidates:
                c["_graph_affinity"] = 0.0
            cand_tag_map = {}

        # ── Step C: 四维 Jaccard 偏好加分（纯内存计算，无 DB 调用） ──
        if has_any_pref:
            def _jaccard(set_a: set, set_b: set) -> float:
                if not set_a or not set_b:
                    return 0.0
                return len(set_a & set_b) / len(set_a | set_b)

            PREF_BOOST_WEIGHT = 0.3
            DIM_WEIGHTS = {
                "genre": 0.30, "mood": 0.30,
                "scenario": 0.25, "theme": 0.15,
            }

            pref_hits = 0
            for c in candidates:
                song = c.get("song", {})
                title = song.get("title", "")

                genre_str = song.get("genre", "")
                cand_genre_tags = {t.strip().lower() for t in genre_str.replace(",", "/").split("/") if t.strip()} if genre_str else set()
                j_genre = _jaccard(expanded_genre_prefs, cand_genre_tags)

                cand_tags = cand_tag_map.get(title, {})
                j_mood = _jaccard(user_pref_moods, cand_tags.get("moods", set()))
                j_theme = _jaccard(user_pref_themes, cand_tags.get("themes", set()))
                j_scenario = _jaccard(user_pref_scenarios, cand_tags.get("scenarios", set()))

                weighted_jaccard = (
                    DIM_WEIGHTS["genre"] * j_genre
                    + DIM_WEIGHTS["mood"] * j_mood
                    + DIM_WEIGHTS["theme"] * j_theme
                    + DIM_WEIGHTS["scenario"] * j_scenario
                )
                boost = PREF_BOOST_WEIGHT * weighted_jaccard
                c["_graph_affinity"] += boost
                c["_pref_boost"] = round(boost, 4)
                c["_pref_detail"] = {
                    "genre": round(j_genre, 3), "mood": round(j_mood, 3),
                    "theme": round(j_theme, 3), "scenario": round(j_scenario, 3),
                }
                if boost > 0:
                    pref_hits += 1

            logger.info(
                f"[GraphAffinity] 偏好加分(缓存+Jaccard): {pref_hits}/{len(candidates)} 首命中 | "
                f"用户偏好维度: genre={len(user_pref_genres)}, mood={len(user_pref_moods)}, "
                f"theme={len(user_pref_themes)}, scenario={len(user_pref_scenarios)}"
            )
        else:
            logger.info("[GraphAffinity] 用户未设置画像偏好且无历史行为，跳过 Jaccard 加分")

        return candidates, cand_tag_map

    def _format_results(
        self,
        *,
        source_raw: Dict[str, str],
        recall_weights: Dict[str, float],
        hard_constraints: Dict[str, Any],
        web_res: str = "",
        web_playable: List[dict] = None,
        graph_entities: List[str] = None,
        final_limit: int = 15,
        timings: Dict[str, float] = None,
    ) -> ToolOutput:
        """
        合并各召回源并执行统一过滤与排序。

        排序管线（R1）:
          1. 解析五路召回结果并保留各路排名
          2. 加权 RRF 融合
          3. hard_constraints + DISLIKES 唯一硬过滤
          3. Artist 多样性初筛（每个歌手最多 N 首）
          4. Graph Affinity（图距离 + Jaccard 偏好 → 个性化微调）→ 产出 cand_tag_map
          5. 三锚精排（M2D-CLAP 语义锚 + OMAR-RQ 声学锚 → 核心排序）
          6. MMR 多维多样性重排（genre + mood + theme + scenario）
          7. 最终安全去重 + FinalCut
        """
        from config.settings import settings as _settings
        timings = timings if timings is not None else {}
        fusion_started = time.perf_counter()

        # ---- Step 1: 解析各召回源结果 ----
        source_items = {
            source: self._parse_engine_results(raw, source)
            for source, raw in source_raw.items()
        }
        logger.info(
            "[FusionInput] %s",
            {source: len(items) for source, items in source_items.items()},
        )

        # ---- Step 2: 按各路原始排名做加权 RRF ----
        final_list = self._fuse_recall_sources(source_items, recall_weights)

        # ---- Step 3: 唯一硬过滤（请求 hard_constraints + DISLIKES）----
        disliked_titles = self._get_disliked_titles()
        before_filter = len(final_list)
        final_list = apply_hard_filters(
            final_list,
            hard_constraints,
            disliked_titles,
            # final_limit 是内部过召回数量（通常 30），兜底只保证最小可用结果，
            # 避免有效过滤结果被过早放宽。
            limit=min(final_limit or 8, 8),
            logger=logger,
        )
        logger.info(
            "[HardFilter] hard_constraints + DISLIKES: %d → %d",
            before_filter,
            len(final_list),
        )

        # 联网歌曲也遵守同一硬过滤，再进入后续统一排序。
        if web_playable:
            web_items = []
            for rank, item in enumerate(web_playable):
                song = item.get("song") or {}
                web_items.append(
                    {
                        "key": self._normalize_key(
                            song.get("title", ""),
                            song.get("artist", ""),
                        ),
                        "rank": rank,
                        "raw_score": item.get("similarity_score", 0.0),
                        "engine": "web",
                        "song": song,
                    }
                )
            filtered_web = apply_hard_filters(
                [
                    {
                        "song": item["song"],
                        "reason": "🌐 全网最新发掘",
                        "similarity_score": item["raw_score"],
                    }
                    for item in web_items
                ],
                hard_constraints,
                disliked_titles,
            )
            existing_keys = {
                self._normalize_key(
                    item.get("song", {}).get("title", ""),
                    item.get("song", {}).get("artist", ""),
                )
                for item in final_list
            }
            for item in filtered_web:
                song = item.get("song", {})
                item["recall_sources"] = ["web"]
                item["recall_source_labels"] = ["联网"]
                song["recall_sources"] = ["web"]
                song["recall_source_labels"] = ["联网"]
                key = self._normalize_key(song.get("title", ""), song.get("artist", ""))
                if key not in existing_keys:
                    final_list.append(item)
                    existing_keys.add(key)

        timings["fusion_filter_ms"] = round(
            (time.perf_counter() - fusion_started) * 1000,
            3,
        )
        ranking_started = time.perf_counter()

        # ---- Step 4: Artist 多样性初筛（提前执行，减轻后续计算负担）----
        max_per_artist = _settings.max_songs_per_artist
        exempt_artists: set = set()
        if graph_entities:
            exempt_artists = {e.lower().strip() for e in graph_entities if e}
        artist_count: Dict[str, int] = {}
        diverse_list = []
        overflow_list = []
        for item in final_list:
            artist = item.get("song", {}).get("artist", "")
            if not artist or artist in ("互联网最新情报",):
                diverse_list.append(item)
                continue
            artist_lower = artist.lower().strip()
            # 指定歌手豁免多样性限制
            if any(ea in artist_lower or artist_lower in ea for ea in exempt_artists):
                diverse_list.append(item)
                artist_count[artist_lower] = artist_count.get(artist_lower, 0) + 1
                continue
            count = artist_count.get(artist_lower, 0)
            if count < max_per_artist:
                diverse_list.append(item)
                artist_count[artist_lower] = count + 1
            else:
                overflow_list.append(item)
        if len(diverse_list) < MIN_DIVERSE_RESULTS:
            diverse_list.extend(overflow_list[:MIN_DIVERSE_RESULTS - len(diverse_list)])
        final_list = diverse_list
        if exempt_artists:
            logger.info(f"[ArtistDiversity] 指定歌手豁免: {exempt_artists}")
        logger.info(f"[ArtistDiversity] 初筛完成 (max={max_per_artist}/artist)，剩余 {len(final_list)} 首，艺术家分布: {dict(artist_count)}")

        # ---- Step 4: Graph Affinity 粗排 + Thompson Sampling 探索槽 ----
        cand_tag_map = {}
        if _settings.graph_affinity_enabled and final_list:
            total_before = len(final_list)
            final_list, cand_tag_map = self._compute_graph_affinity(final_list)

            # ── Phase A: 按 graph_affinity 排序并截断（粗排）──
            final_list.sort(key=lambda x: x.get("_graph_affinity", 0), reverse=True)

            coarse_cut = max(int(total_before * _settings.coarse_cut_ratio), 10)
            coarse_cut = min(coarse_cut, len(final_list))  # 不能超过实际数量

            main_candidates = final_list[:coarse_cut]
            tail_candidates = final_list[coarse_cut:]

            # ── Phase B: Thompson Sampling 探索槽 ──
            # 从尾部捞回冷门歌，用 TS 采样决定哪些冷门歌有机会进入精排
            import random
            import math
            n_explore = max(int(coarse_cut * _settings.exploration_ratio), 1)

            explore_picks = []
            if tail_candidates:
                try:
                    import numpy as np
                    from retrieval.neo4j_client import get_neo4j_client
                    neo4j = get_neo4j_client()

                    # 批量读取尾部歌曲的 TS 参数 (ts_alpha, ts_beta)
                    tail_titles = [c.get("song", {}).get("title", "") for c in tail_candidates]
                    ts_query = """
                    UNWIND $titles AS t
                    MATCH (s:Song {title: t})
                    RETURN s.title AS title,
                           coalesce(s.ts_alpha, 1) AS alpha,
                           coalesce(s.ts_beta, 1) AS beta
                    """
                    ts_rows = neo4j.execute_query(ts_query, {"titles": tail_titles}) if neo4j and neo4j.driver else []
                    ts_map = {r["title"]: (r["alpha"], r["beta"]) for r in (ts_rows or [])}

                    # TS 采样：为尾部每首歌采样一个分数
                    ts_scores = []
                    for c in tail_candidates:
                        title = c.get("song", {}).get("title", "")
                        alpha, beta = ts_map.get(title, (1, 1))
                        ts_score = float(np.random.beta(alpha, beta))
                        c["_ts_score"] = round(ts_score, 4)
                        ts_scores.append(ts_score)

                    # 按 TS 采样分数排序，取 Top N 作为探索槽
                    tail_with_scores = sorted(
                        zip(tail_candidates, ts_scores),
                        key=lambda x: x[1], reverse=True
                    )
                    explore_picks = [item for item, _ in tail_with_scores[:n_explore]]
                except Exception as e:
                    logger.warning(f"[TS] Thompson Sampling 失败，降级随机探索: {e}")
                    random.shuffle(tail_candidates)
                    explore_picks = tail_candidates[:n_explore]

            for ep in explore_picks:
                ep["_is_exploration"] = True
                reason = ep.get("reason", "")
                if "🆕" not in reason:
                    ep["reason"] = reason + " 🆕探索发现"

            main_candidates.extend(explore_picks)
            final_list = main_candidates

            n_explore_actual = sum(1 for x in final_list if x.get("_is_exploration"))
            logger.info(
                f"[CoarseRank] 粗排: {total_before} → {len(final_list)} 首 "
                f"(主力={len(final_list) - n_explore_actual}, 探索槽={n_explore_actual}, "
                f"cut_ratio={_settings.coarse_cut_ratio}, explore_ratio={_settings.exploration_ratio})"
            )

        # ---- Step 5: 三锚归一化精排（语义 + 声学 + 个性化）----
        # ★ 语义锚使用 HyDE 声学描述（而非原始中文 query）
        # 原因：M2D-CLAP 在英文声学描述上的对齐质量远高于中文情绪词，
        #       HyDE 存在的意义就是弥合用户自然语言与模型训练分布的 gap。
        #       同时，semantic_search 已经编码过相同的 HyDE 文本，
        #       embedding 缓存会命中，节省 ~100ms 重复推理。
        hyde_text = getattr(self, '_current_hyde_text', '') or ''
        rerank_query = hyde_text if hyde_text else (getattr(self, '_current_query', '') or '')
        if rerank_query and final_list:
            final_list = self._tri_anchor_rerank(final_list, rerank_query)

        # ---- Step 5.5: Cross-Encoder 精排（可选，默认关闭）----
        if _settings.reranker_enabled and final_list:
            try:
                from retrieval.cross_encoder_reranker import CrossEncoderReranker
                reranker = CrossEncoderReranker()
                rerank_query = getattr(self, "_current_query", "") or "music recommendation"
                final_list = reranker.rerank(rerank_query, final_list)
            except Exception as e:
                logger.warning(f"[Reranker] Cross-Encoder 精排异常（降级跳过）: {e}")

        # ---- Step 6: MMR 多维多样性重排（genre + mood + theme + scenario）----
        def _build_rich_tags(item: dict, tag_map: dict) -> set:
            """
            构建丰富的多维标签集合（替代旧版仅用 genre 字段）。
            合并 genre 字段 + cand_tag_map 中的 mood/theme/scenario。
            """
            song = item.get("song", {})
            title = song.get("title", "")
            tags = set()
            # genre 字段（如果有）
            genre_str = song.get("genre", "")
            if genre_str:
                tags.update(t.strip().lower() for t in genre_str.replace(",", "/").split("/") if t.strip())
            # 从 cand_tag_map 获取 mood/theme/scenario
            ct = tag_map.get(title, {})
            tags.update(ct.get("moods", set()))
            tags.update(ct.get("themes", set()))
            tags.update(ct.get("scenarios", set()))
            return tags

        def _jaccard(set_a: set, set_b: set) -> float:
            if not set_a or not set_b:
                return 0.0
            return len(set_a & set_b) / len(set_a | set_b)

        if len(final_list) > 2:
            mmr_lambda = _settings.mmr_lambda
            selected = [final_list[0]]
            mmr_candidates = final_list[1:]

            # 预计算每首歌的多维标签集合
            tag_cache: Dict[int, set] = {}
            for item in final_list:
                tag_cache[id(item)] = _build_rich_tags(item, cand_tag_map)

            while mmr_candidates and len(selected) < len(final_list):
                best_score = -float('inf')
                best_idx = 0

                selected_tag_sets = [tag_cache[id(s)] for s in selected]

                for i, cand in enumerate(mmr_candidates):
                    relevance = cand.get("similarity_score", 0)
                    cand_tags = tag_cache[id(cand)]

                    # 与已选集合中 Jaccard 最大的作为重叠度
                    max_overlap = 0.0
                    if cand_tags:
                        for sel_tags in selected_tag_sets:
                            j = _jaccard(cand_tags, sel_tags)
                            if j > max_overlap:
                                max_overlap = j

                    mmr_score = mmr_lambda * relevance - (1 - mmr_lambda) * max_overlap
                    if mmr_score > best_score:
                        best_score = mmr_score
                        best_idx = i

                selected.append(mmr_candidates.pop(best_idx))

            final_list = selected
            # 日志：展示前 5 首的多维标签
            top5_tags = []
            for item in final_list[:5]:
                tags = _build_rich_tags(item, cand_tag_map)
                top5_tags.append(sorted(tags)[:4] if tags else ["(无标签)"])
            logger.info(f"[MMR-MultiDim] 多维多样性重排序完成，前5首标签: {top5_tags}")

        # ---- Step 7: 最终安全去重 + FinalCut ----
        seen_final = set()
        deduped_list = []
        for item in final_list:
            s = item.get("song", {})
            fk = MusicHybridRetrieval._normalize_key(s.get("title", ""), s.get("artist", ""))
            if fk not in seen_final:
                seen_final.add(fk)
                deduped_list.append(item)
            else:
                logger.info(f"[Dedup-Final] 最终去重: {s.get('title', '')} - {s.get('artist', '')}")
        if len(deduped_list) < len(final_list):
            logger.info(f"[Dedup-Final] 去重前={len(final_list)}, 去重后={len(deduped_list)}")
        final_list = deduped_list

        if final_limit and len(final_list) > final_limit:
            logger.info(f"[FinalCut] 精排后截断: {len(final_list)} → {final_limit} 首")
            final_list = final_list[:final_limit]

        timings["ranking_ms"] = round((time.perf_counter() - ranking_started) * 1000, 3)

        # 如果有全网聚合结果，强行塞一条纯文本作为上下文给大模型
        if web_res and "未能找到相关有效信息" not in web_res:
            final_list.insert(0, {
                "_raw_markdown": web_res,
                "song": {
                    "title": "🌐 全网资讯补充",
                    "artist": "互联网最新情报",
                    "genre": "News",
                    "recall_sources": ["web"],
                    "recall_source_labels": ["联网"],
                },
                "reason": "包含通过多源聚合引擎获取的最新的互联网关联资讯，用于补充音乐库之外的信息。",
                "recall_sources": ["web"],
                "recall_source_labels": ["联网"],
                "similarity_score": WEB_RESULT_PRIORITY_SCORE
            })

        # 终端日志：打印每一首入选歌曲及来源
        logger.info(f"=== 🎵 检索引擎合并完毕，共找到 {len(final_list)} 条结果 (包含资讯) ===")
        for i, item in enumerate(final_list):
            logger.info(f"  [{i+1}] {item['song']['title']} - {item['reason']}")

        # 构建 raw_markdown 供大模型参考
        markdown_lines: List[str] = []
        if web_res and "未能找到" not in web_res:
            markdown_lines.append(web_res.strip())
            markdown_lines.append("")
        if final_list:
            markdown_lines.append("**推荐结果**")
            for idx, item in enumerate(final_list, 1):
                song = item.get("song", {}) if isinstance(item, dict) else {}
                title = song.get("title", "未知") if isinstance(song, dict) else "未定"
                artist = song.get("artist", "未知") if isinstance(song, dict) else "未定"
                genre = song.get("genre", "") if isinstance(song, dict) else ""
                reason = item.get("reason", "") if isinstance(item, dict) else ""

                if title == "🎪 全网资讯补充":
                    continue

                line = f"{idx}. **{title}** - {artist}"
                if genre:
                    line += f" ({genre})"
                markdown_lines.append(line)
                if reason:
                    markdown_lines.append(f"   推荐理由: {reason}")

        raw_markdown = "\n".join(markdown_lines).strip()
        if not raw_markdown:
            raw_markdown = "抱歉，没有找到合适的音乐推荐。"

        # ---- Step 8: 异步更新 Thompson Sampling 曝光衰减 ----
        # 每推荐一次，ts_beta += 1.0，使该歌在未来探索槽中的采样分数自然降低
        # Beta(alpha, beta) 中 beta 越大，采样值越倾向于低值
        recommended_titles = [
            item.get("song", {}).get("title", "")
            for item in final_list
            if item.get("song", {}).get("title") and item.get("song", {}).get("title") != "🌐 全网资讯补充"
        ]
        if recommended_titles:
            try:
                import asyncio
                async def _update_ts_exposure(titles):
                    try:
                        from retrieval.neo4j_client import get_neo4j_client
                        neo4j = get_neo4j_client()
                        if neo4j and neo4j.driver:
                            ts_update_query = """
                            UNWIND $titles AS t
                            MATCH (s:Song {title: t})
                            SET s.ts_beta = coalesce(s.ts_beta, 1) + 0.3
                            """
                            neo4j.execute_query(ts_update_query, {"titles": titles})
                            logger.info(f"[TS] 曝光衰减更新: {len(titles)} 首歌 ts_beta += 0.3")
                    except Exception as e:
                        logger.warning(f"[TS] 曝光衰减更新失败（不影响推荐）: {e}")

                # fire-and-forget: 异步更新，不阻塞返回
                try:
                    loop = asyncio.get_running_loop()
                    loop.create_task(_update_ts_exposure(recommended_titles))
                except RuntimeError:
                    # 如果没有运行中的事件循环，同步执行
                    import threading
                    threading.Thread(target=lambda: asyncio.run(_update_ts_exposure(recommended_titles)), daemon=True).start()
            except Exception:
                pass  # TS 更新失败不影响主流程

        return ToolOutput(
            success=len(final_list) > 0,
            data=final_list,
            raw_markdown=raw_markdown,
            error_message=None if final_list else "Not found",
            metadata={"timings": timings},
        )

    def _generate_hyde_description(
        self,
        query: str,
        graphzep_facts: str = "",
        intent_type: str = "",
    ) -> str:
        """
        HyDE 声学描述生成（架构分离后的专用模块）。
        
        仅在 use_vector=true 时调用。接收用户输入和 GraphZep 记忆，
        通过 HYDE_ACOUSTIC_GENERATOR_PROMPT 生成纯英文声学描述。
        GraphZep 记忆只影响声学描述内容，不影响路由决策（路由已由 Planner 完成）。
        
        Args:
            query: 用户原始输入
            graphzep_facts: GraphZep 长期记忆文本（可为空）
            intent_type: Planner 识别的意图类型

        Returns:
            英文声学描述文本，供 M2D-CLAP 编码
        """
        try:
            from llms.prompts import HYDE_ACOUSTIC_GENERATOR_PROMPT
            from langchain_core.prompts import ChatPromptTemplate
            from langchain_core.output_parsers import StrOutputParser
            from llms.multi_llm import get_intent_chat_model as get_intent_llm
            import traceback
            
            llm = self.llm_client or get_intent_llm()
            chain = (
                ChatPromptTemplate.from_template(HYDE_ACOUSTIC_GENERATOR_PROMPT)
                | llm
                | StrOutputParser()
            )
            
            invoke_params = {
                "user_input": query,
                "graphzep_facts": graphzep_facts or "暂无",
                "intent_type": intent_type or "unknown",
            }
            
            # 使用同步 invoke() 来调用 HyDE chain
            # _generate_hyde_description 在 async retrieve() 中被调用，
            # 但 LangChain chain.invoke() 本身是同步的，这里无需 await
            result = chain.invoke(invoke_params)
            
            result = result.strip()
            word_count = len(result.split())
            logger.info(
                f"[HyDE] 生成声学描述 ({word_count} words): {result[:100]}..."
            )
            return result
            
        except Exception as e:
            logger.warning(f"[HyDE] 声学描述生成失败，降级使用原始查询: {e}\n{traceback.format_exc()}")
            return query
