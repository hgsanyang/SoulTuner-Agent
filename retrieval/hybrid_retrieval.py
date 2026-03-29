import json
import logging
import os
import re
import asyncio
import unicodedata
from typing import List, Dict, Any, Optional

from tools.graphrag_search import graphrag_search
from tools.semantic_search import semantic_search
from tools.web_search_aggregator import _federated_search_async
from retrieval.neo4j_client import get_neo4j_client
from config.logging_config import get_logger
from config.settings import settings
from schemas.music_state import ToolOutput


logger = get_logger(__name__)

# ---- 检索结果融合常量 ----
BASELINE_SIMILARITY_SCORE = 0.85      # 单引擎命中的基础相似度分（仅作 fallback）
WEB_RESULT_PRIORITY_SCORE = 9.9       # 网络资讯结果的优先级分数
MAX_SONGS_PER_ARTIST = 3              # 多样性过滤：每个艺术家最多占的歌曲数
MIN_DIVERSE_RESULTS = 6               # 多样性过滤后的最少结果数

# ---- 加权 RRF 融合参数 ----
RRF_K = 60                            # RRF 平滑常数（标准值 60）
RRF_WEIGHT_VECTOR = 0.7               # 向量检索路的 RRF 权重（主导）
RRF_WEIGHT_GRAPH = 0.3                # 图谱检索路的 RRF 权重（辅助）

# ---- Neo4j 图距离加权参数 ----
GRAPH_AFFINITY_ENABLED = True         # 是否启用图距离加权
GRAPH_AFFINITY_WEIGHT = 0.15          # 图亲和力分数在最终排序中的权重
GRAPH_AFFINITY_MAX_HOPS = 4           # 最大跳数（超过视为无关联）
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

    def retrieve(self, query: str, limit: int = 5, precomputed_plan: dict = None) -> ToolOutput:
        """
        主检索入口
        
        Args:
            query: 用户查询
            limit: 返回结果数量
            precomputed_plan: 来自上游统一 Prompt 的预计算检索计划（dict 格式的 RetrievalPlan）。
                              如果提供，则使用预计算计划；否则默认启用图谱+向量双引擎。
        """
        logger.info(f"[Retrieval] 开始处理请求: {query}")
        
        # 1. 确定检索策略：优先用预计算计划，否则安全默认（双引擎）
        if precomputed_plan:
            logger.info("[Retrieval] 使用上游预计算的检索计划")
            use_graph = precomputed_plan.get("use_graph", False)
            use_vector = precomputed_plan.get("use_vector", False)
            use_web = precomputed_plan.get("use_web_search", False)
            graph_entities = precomputed_plan.get("graph_entities", [])
            genre_filter = precomputed_plan.get("graph_genre_filter")
            scenario_filter = precomputed_plan.get("graph_scenario_filter")
            mood_filter = precomputed_plan.get("graph_mood_filter")
            language_filter = precomputed_plan.get("graph_language_filter")
            region_filter = precomputed_plan.get("graph_region_filter")
            web_keywords = precomputed_plan.get("web_search_keywords", "")
            need_web_search = use_web
            search_keyword = web_keywords
            vector_desc = ""

            # ── 调试日志：确认过滤字段最终值 ──
            logger.info(
                f"[Retrieval] 过滤字段最终值: genre='{genre_filter}' | scenario='{scenario_filter}' | "
                f"mood='{mood_filter}' | language='{language_filter}' | region='{region_filter}' | "
                f"entities={graph_entities} | query='{query[:50]}'"
            )

            # ── HyDE 声学描述：双模式分支 ──
            # API 模式：LLM 已在 retrieval_plan 中内联填写 vector_acoustic_query，直接使用
            # 本地模式：vector_acoustic_query 为空，调用独立 HyDE 模块生成
            if use_vector:
                vector_acoustic_query = precomputed_plan.get("vector_acoustic_query", "") or ""
                if vector_acoustic_query:
                    vector_desc = vector_acoustic_query
                    logger.info(f"[HyDE] API 模式：使用 LLM 内联声学描述 ({len(vector_desc.split())} words)")
                else:
                    graphzep_for_hyde = precomputed_plan.get("_graphzep_facts", "")
                    intent_type = precomputed_plan.get("_intent_type", "")
                    logger.info("[HyDE] 本地模式：调用独立 HyDE 模块生成声学描述")
                    vector_desc = self._generate_hyde_description(
                        query=query,
                        graphzep_facts=graphzep_for_hyde,
                        intent_type=intent_type,
                    )
        else:
            # 安全默认：同时启用图谱和向量检索
            logger.info("[Retrieval] 无预计算计划，使用默认双引擎检索")
            use_graph = True
            use_vector = True
            use_web = False
            graph_entities = [query]
            genre_filter = None
            scenario_filter = None
            mood_filter = None
            language_filter = None
            region_filter = None
            vector_desc = self._generate_hyde_description(query=query, graphzep_facts="", intent_type="")
            need_web_search = False
            search_keyword = ""
        
        graph_result = ""
        vector_result = ""
        
        # 2. 根据策略分发执行 (加入并发机制和全网搜索支持)
        # 建立异步运行环境
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
        import nest_asyncio
        nest_asyncio.apply()
        
        # 定义任务包装器
        async def run_sync_in_executor(func, *args, **kwargs):
            return await loop.run_in_executor(None, lambda: func(*args, **kwargs))
            
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
                prompt = f"请从以下全网搜索的资讯文本中，提取出最具代表性的最多3首歌曲名称和歌手。如果没有明确提到新歌，请返回空列表。\n\n资讯文本:\n{web_text}"
                
                result = await structured_llm.ainvoke(prompt)
                if not result or not result.songs:
                    return []
                    
                from tools.music_fetch_tool import execute_search_online_music
                
                tasks = []
                for s in result.songs[:3]:
                    q = f"{s.artist} {s.title}"
                    tasks.append(execute_search_online_music(q))
                    
                fetch_results = await asyncio.gather(*tasks, return_exceptions=True)
                
                playable_songs = []
                for i, res in enumerate(fetch_results):
                    if isinstance(res, Exception):
                        continue
                    if getattr(res, "success", False) and res.data:
                        top_hit = res.data[0]
                        playable_songs.append({
                            "song": {
                                "title": top_hit.get("title", result.songs[i].title),
                                "artist": top_hit.get("artist", result.songs[i].artist),
                                "preview_url": top_hit.get("play_url") or top_hit.get("preview_url"),
                                "cover_url": top_hit.get("cover_url"),
                                "album": top_hit.get("album", "未知"),
                                "genre": "Web Trends"
                            },
                            "reason": "🌐 全网最新发掘",
                            "similarity_score": 9.5 - (i * 0.1),
                            "_vector_score": 0.0,
                            "_graph_score": 0.0
                        })
                return playable_songs
            except Exception as e:
                logger.error(f"提取全网歌曲失败: {e}")
                return []
            
        async def execute_retrieval():
            local_tasks = []
            
            # 根据统一的 use_graph/use_vector 标志分发任务
            if use_graph and not use_vector:
                # 仅图谱检索
                graph_query_dict = {"tags": graph_entities, "genre": genre_filter,
                                    "scenario": scenario_filter, "mood": mood_filter,
                                    "language": language_filter, "region": region_filter}
                if not graph_entities and not genre_filter and not scenario_filter and not mood_filter and not language_filter and not region_filter:
                    graph_query_dict["tags"] = [query]
                search_term = json.dumps(graph_query_dict, ensure_ascii=False)
                local_tasks.append(run_sync_in_executor(graphrag_search.invoke, {"query": search_term, "limit": limit}))
                local_tasks.append(asyncio.sleep(0))  # 占位 vector
            elif use_vector and not use_graph:
                # 仅向量检索（带 language/region 硬过滤）
                local_tasks.append(asyncio.sleep(0))  # 占位 graph
                search_term = vector_desc if vector_desc else query
                local_tasks.append(run_sync_in_executor(semantic_search.invoke, {
                    "query": search_term, "limit": limit,
                    "language_filter": language_filter or "",
                    "region_filter": region_filter or "",
                }))
            else:
                # 混合检索（同时启用 graph + vector，或两者都未指定时的默认行为）
                graph_query_dict = {"tags": graph_entities, "genre": genre_filter,
                                    "scenario": scenario_filter, "mood": mood_filter,
                                    "language": language_filter, "region": region_filter}
                if not graph_entities and not genre_filter and not scenario_filter and not mood_filter and not language_filter and not region_filter:
                    graph_query_dict["tags"] = [query]
                graph_term = json.dumps(graph_query_dict, ensure_ascii=False)
                vector_term = vector_desc if vector_desc else query
                
                logger.info(f"[Hybrid] 混合检索：向子引擎传递 limit={limit}")
                local_tasks.append(run_sync_in_executor(graphrag_search.invoke, {"query": graph_term, "limit": limit}))
                local_tasks.append(run_sync_in_executor(semantic_search.invoke, {
                    "query": vector_term, "limit": limit,
                    "language_filter": language_filter or "",
                    "region_filter": region_filter or "",
                }))
                
            # 并发执行本地数据库检索
            local_results = await asyncio.gather(*local_tasks, return_exceptions=True)
            
            # ── 诊断日志：检查各引擎原始返回（同时写文件防丢失） ──
            diag_lines = [f"=== 诊断时间: {__import__('datetime').datetime.now()} ===\n"]
            diag_lines.append(f"use_graph={use_graph}, use_vector={use_vector}\n")
            diag_lines.append(f"vector_desc(HyDE输出)={repr(vector_desc)[:300]}\n")
            for idx, label in enumerate(["Graph", "Vector"]):
                r = local_results[idx] if idx < len(local_results) else None
                if isinstance(r, Exception):
                    msg = f"[诊断] {label} 引擎抛出异常: {type(r).__name__}: {r}"
                    logger.error(msg)
                    diag_lines.append(msg + "\n")
                    # 打印完整异常栈
                    import traceback
                    tb = "".join(traceback.format_exception(type(r), r, r.__traceback__))
                    diag_lines.append(f"[诊断] {label} 异常栈:\n{tb}\n")
                elif r is None or r == "" or r == 0:
                    msg = f"[诊断] {label} 引擎返回空值: repr={repr(r)[:200]}"
                    logger.warning(msg)
                    diag_lines.append(msg + "\n")
                else:
                    msg = f"[诊断] {label} 引擎返回: 长度={len(str(r))}, 前200字符={str(r)[:200]}"
                    logger.info(msg)
                    diag_lines.append(msg + "\n")
            
            # 写入诊断文件
            try:
                with open("/tmp/hyde_diag.txt", "w", encoding="utf-8") as f:
                    f.writelines(diag_lines)
            except Exception:
                pass
            
            graph_raw = local_results[0] if not isinstance(local_results[0], Exception) and local_results[0] else ""
            vector_raw = local_results[1] if not isinstance(local_results[1], Exception) and local_results[1] else ""
            
            web_raw = ""
            
            _web_search_globally_enabled = os.environ.get("MUSIC_WEB_SEARCH_ENABLED", "1") != "0"
            
            if _web_search_globally_enabled:
                # 情况A: 预计算计划或路由器明确要求联网搜索
                if need_web_search and search_keyword:
                    logger.info(f"⚡ 意图明确要求联网: '{search_keyword}'")
                    web_raw = await _federated_search_async(search_keyword)
                else:
                    # 情况B: Fallback 逻辑 - 本地数据库未找到结果
                    graph_empty = not graph_raw or graph_raw == "[]" or "error" in graph_raw.lower()
                    vector_empty = not vector_raw or vector_raw == "[]" or "error" in vector_raw.lower()
                    
                    needs_fallback = False
                    if graph_entities and graph_empty:
                        needs_fallback = True
                    elif graph_empty and vector_empty:
                        needs_fallback = True
                        
                    if needs_fallback:
                        logger.warning(f"本地数据库未能找到核心实体或结果太少，触发联网保底搜索 (Fallback): '{query}'")
                        search_kw = search_keyword if search_keyword else query
                        web_raw = await _federated_search_async(search_kw)
                        
            web_playable = await _extract_and_fetch_web_songs(web_raw)
                        
            return graph_raw, vector_raw, web_raw, web_playable
            
        # 同步等待结果
        graph_raw, vector_raw, web_raw, web_playable = loop.run_until_complete(execute_retrieval())
        
        # 确定策略名称（用于日志和结果格式化）
        if use_graph and use_vector:
            strategy_name = "hybrid_balanced"
        elif use_graph:
            strategy_name = "graph_only"
        elif use_vector:
            strategy_name = "vector_only"
        else:
            strategy_name = "hybrid_balanced"
        
        
        # 保存当前 query 供 _format_results 中的 Cross-Encoder 精排使用
        self._current_query = query
        
        return self._format_results(strategy_name, graph_raw, vector_raw, web_raw, web_playable, graph_entities)


    # ================================================================
    # 【P0 升级】加权 RRF (Reciprocal Rank Fusion) 排序融合
    # 替代旧版硬编码 v×0.7 + g×0.3 的原始分数加权。
    # RRF 基于排名（非原始分数）融合，对不同尺度的引擎更公平。
    # ================================================================

    @staticmethod
    def _normalize_key(title: str, artist: str) -> str:
        """生成标准化的去重 key，消除全角/半角、标点、空格差异。"""
        def _clean(s: str) -> str:
            s = unicodedata.normalize("NFKC", s)  # 全角→半角
            s = s.lower().strip()
            s = re.sub(r"[,，、/\\\s()（）【】\[\]]+", "", s)  # 去掉标点和空格
            return s
        return f"{_clean(title)}_{_clean(artist)}"

    @staticmethod
    def _parse_engine_results(res_str: str, engine_name: str) -> List[dict]:
        """将引擎原始 JSON 字符串解析为标准化的歌曲列表，保留原始排名。"""
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
            key = HybridRetriever._normalize_key(title, artist)
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
                    "preview_url": item.get("preview_url", None),
                    "cover_url": item.get("cover_url", None),
                    "lrc_url": item.get("lrc_url", None),
                },
            })
        return results

    @staticmethod
    def _weighted_rrf_fusion(
        graph_items: List[dict],
        vector_items: List[dict],
        k: int = RRF_K,
        w_vector: float = RRF_WEIGHT_VECTOR,
        w_graph: float = RRF_WEIGHT_GRAPH,
    ) -> List[dict]:
        """
        加权 RRF 融合两路检索结果。

        公式:
            RRF_Score(d) = w_vector / (k + rank_vector(d))
                         + w_graph  / (k + rank_graph(d))

        如果某首歌只出现在一路中，另一路的贡献为 0。
        双路都命中的歌曲天然获得更高分数。
        """
        # 以 key 为索引收集各路排名
        song_data: Dict[str, dict] = {}   # key → 歌曲元数据
        graph_ranks: Dict[str, int] = {}   # key → 图谱排名
        vector_ranks: Dict[str, int] = {}  # key → 向量排名

        for item in graph_items:
            key = item["key"]
            graph_ranks[key] = item["rank"]
            if key not in song_data:
                song_data[key] = item

        for item in vector_items:
            key = item["key"]
            vector_ranks[key] = item["rank"]
            if key not in song_data:
                song_data[key] = item

        # 计算每首歌的加权 RRF 分数
        rrf_scores: Dict[str, float] = {}
        all_keys = set(graph_ranks.keys()) | set(vector_ranks.keys())

        for key in all_keys:
            score = 0.0
            if key in vector_ranks:
                score += w_vector / (k + vector_ranks[key])
            if key in graph_ranks:
                score += w_graph / (k + graph_ranks[key])
            rrf_scores[key] = score

        # 构建融合结果列表
        fused = []
        for key in all_keys:
            item = song_data[key]
            both_hit = key in graph_ranks and key in vector_ranks
            engines = []
            if key in graph_ranks:
                engines.append("知识图谱(GraphRAG)")
            if key in vector_ranks:
                engines.append("语义向量(Neo4j Vector)")

            reason = "引擎检索来源: " + " + ".join(engines)
            if both_hit:
                reason += " 🔥双引擎交叉命中"

            fused.append({
                "song": item["song"],
                "reason": reason,
                "similarity_score": rrf_scores[key],
                "_rrf_score": rrf_scores[key],
                "_graph_rank": graph_ranks.get(key),
                "_vector_rank": vector_ranks.get(key),
                "_both_engines": both_hit,
            })

        # 按 RRF 分数降序排列
        fused.sort(key=lambda x: x["similarity_score"], reverse=True)
        logger.info(
            f"[RRF] 加权融合完成: {len(fused)} 首 "
            f"(双引擎命中: {sum(1 for f in fused if f['_both_engines'])} 首) "
            f"| 权重 vector={w_vector} graph={w_graph} k={k}"
        )
        return fused

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
        max_hops: int = GRAPH_AFFINITY_MAX_HOPS,
    ) -> List[dict]:
        """
        为每首候选歌曲计算与用户的图距离亲和力 + 用户画像偏好加分。

        优化点（相比旧版）：
          - 用户偏好从缓存读取（_load_user_preferences），不再每次查 Neo4j
          - 图距离 + 候选歌曲标签合并为 1 条 Cypher（原来是 2 条）
          - 总共从 3 次 Neo4j round-trip → 1 次（+ 首次启动时的偏好缓存查询）
        """
        if not candidates:
            return candidates

        try:
            from retrieval.neo4j_client import get_neo4j_client
            neo4j = get_neo4j_client()
            if not neo4j or not neo4j.driver:
                logger.warning("[GraphAffinity] Neo4j 不可用，跳过图距离计算")
                for c in candidates:
                    c["_graph_affinity"] = 0.0
                return candidates
        except Exception:
            logger.warning("[GraphAffinity] 无法导入 Neo4j 客户端，跳过")
            for c in candidates:
                c["_graph_affinity"] = 0.0
            return candidates

        titles = [c["song"]["title"] for c in candidates if c.get("song", {}).get("title")]
        if not titles:
            for c in candidates:
                c["_graph_affinity"] = 0.0
            return candidates

        # ── Step A: 用户偏好（从缓存读取，首次自动加载） ──
        user_prefs = _load_user_preferences(user_id)
        user_pref_genres = user_prefs["genres"]
        user_pref_moods = user_prefs["moods"]
        user_pref_themes = user_prefs["themes"]
        user_pref_scenarios = user_prefs["scenarios"]
        expanded_genre_prefs = user_prefs["expanded_genres"]
        has_any_pref = user_pref_genres or user_pref_moods or user_pref_themes or user_pref_scenarios

        # ── Step B: 合并查询（图距离 + 候选歌曲标签，1 次 Neo4j round-trip） ──
        # 将原来的 2 条独立 Cypher 合并为 1 条，同时返回 distance 和 tags
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

        try:
            results = neo4j.execute_query(combined_query, {"user_id": user_id, "titles": titles})

            # 解析合并结果：同时提取 distance 和 tags
            distance_map = {}
            cand_tag_map = {}
            for r in results:
                t = r.get("title", "")
                distance_map[t] = r.get("distance", -1)
                cand_tag_map[t] = {
                    "moods": {x.strip().lower() for x in (r.get("moods") or []) if x and x.strip()},
                    "themes": {x.strip().lower() for x in (r.get("themes") or []) if x and x.strip()},
                    "scenarios": {x.strip().lower() for x in (r.get("scenarios") or []) if x and x.strip()},
                }

            # 计算图距离亲和力
            for c in candidates:
                title = c.get("song", {}).get("title", "")
                dist = distance_map.get(title, -1)
                if dist > 0:
                    c["_graph_affinity"] = 1.0 / (1.0 + dist)
                else:
                    c["_graph_affinity"] = 0.0

            affinity_hits = sum(1 for c in candidates if c["_graph_affinity"] > 0)
            logger.info(
                f"[GraphAffinity] 合并查询完成（1 次 Neo4j）: "
                f"图距离 {affinity_hits}/{len(candidates)} 首有亲和关系"
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

                # Genre Jaccard
                genre_str = song.get("genre", "")
                cand_genre_tags = {t.strip().lower() for t in genre_str.replace(",", "/").split("/") if t.strip()} if genre_str else set()
                j_genre = _jaccard(expanded_genre_prefs, cand_genre_tags)

                # Mood/Theme/Scenario Jaccard（从合并查询结果）
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

        return candidates

    def _format_results(self, strategy_name: str, graph_res: str, vector_res: str, web_res: str = "", web_playable: List[dict] = None, graph_entities: List[str] = None) -> ToolOutput:
        """
        合并各检索引擎的结果，使用 **加权 RRF** 排序融合 + **Cross-Encoder 精排** + **Neo4j 图距离加权**。

        排序管线:
          1. 解析各引擎原始 JSON → 标准化列表（含排名）
          2. 加权 RRF 融合两路排名 → 统一分数
          2.5. Cross-Encoder 精排（bge-reranker-v2-m3 深度语义匹配）
          3. Neo4j 图距离亲和力评分 → 微调排序
          4. Artist 多样性过滤
          5. MMR genre-aware 多样性重排序
        """
        # ---- Step 1: 解析各引擎结果 ----
        graph_items = self._parse_engine_results(graph_res, "知识图谱(GraphRAG)")
        vector_items = self._parse_engine_results(vector_res, "语义向量(Neo4j Vector)")
        logger.info(
            f"[诊断-融合入口] graph_items={len(graph_items)}, vector_items={len(vector_items)} | "
            f"graph_res长度={len(graph_res)}, vector_res长度={len(vector_res)} | "
            f"vector_res前100字符={vector_res[:100] if vector_res else '(空)'}"
        )

        # ---- Step 2: 加权 RRF 融合 ----
        if graph_items and vector_items:
            # 双引擎混合检索 → RRF 融合
            final_list = self._weighted_rrf_fusion(graph_items, vector_items)
        elif graph_items:
            # 仅图谱 → 保留图谱原始排序
            final_list = [{
                "song": item["song"],
                "reason": f"引擎检索来源: {item['engine']}",
                "similarity_score": item["raw_score"],
                "_graph_affinity": 0.0,
            } for item in graph_items]
        elif vector_items:
            # 仅向量 → 保留向量原始排序
            final_list = [{
                "song": item["song"],
                "reason": f"引擎检索来源: {item['engine']}",
                "similarity_score": item["raw_score"],
                "_graph_affinity": 0.0,
            } for item in vector_items]
        else:
            final_list = []

        # 将从全网转化来的真实可播歌曲加入（去重后）
        if web_playable:
            existing_keys = set()
            for item in final_list:
                s = item.get("song", {})
                existing_keys.add(HybridRetriever._normalize_key(s.get("title", ""), s.get("artist", "")))
            for wp in web_playable:
                wp["_graph_affinity"] = 0.0
                s = wp.get("song", {})
                wp_key = HybridRetriever._normalize_key(s.get("title", ""), s.get("artist", ""))
                if wp_key not in existing_keys:
                    final_list.append(wp)
                    existing_keys.add(wp_key)
                else:
                    logger.info(f"[Dedup] web_playable 重复跳过: {s.get('title', '')} - {s.get('artist', '')}")

        # ---- Step 2.5: Cross-Encoder 精排 ----
        from config.settings import settings as _settings
        if _settings.reranker_enabled and final_list:
            try:
                from retrieval.cross_encoder_reranker import CrossEncoderReranker
                reranker = CrossEncoderReranker()
                # 使用 retrieve() 传入的原始 query（从 graph_entities 或策略推断）
                rerank_query = ""
                if graph_entities:
                    rerank_query = " ".join(graph_entities)
                if not rerank_query:
                    # fallback: 从第一个候选的 reason 中无法获取 query，使用空字符串
                    rerank_query = ""
                # 注意: rerank_query 可能为空，此时由调用方在 retrieve() 中传入
                # 这里我们通过检查 _format_results 的调用上下文来获取 query
                # 由于 _format_results 没有 query 参数，我们需要通过实例变量传递
                if hasattr(self, '_current_query') and self._current_query:
                    rerank_query = self._current_query
                elif rerank_query:
                    pass  # 使用 graph_entities 拼接
                else:
                    rerank_query = strategy_name  # 最终 fallback

                final_list = reranker.rerank(rerank_query, final_list)
            except Exception as e:
                logger.warning(f"[Reranker] Cross-Encoder 精排异常（降级跳过）: {e}")

        # ---- Step 3: Neo4j 图距离亲和力加权 ----
        if GRAPH_AFFINITY_ENABLED and final_list:
            final_list = self._compute_graph_affinity(final_list)
            # 将亲和力分数融入最终排序分:
            # final_score = (1 - α) × rrf_score + α × graph_affinity
            for item in final_list:
                rrf = item.get("similarity_score", 0)
                affinity = item.get("_graph_affinity", 0)
                item["similarity_score"] = (1 - GRAPH_AFFINITY_WEIGHT) * rrf + GRAPH_AFFINITY_WEIGHT * affinity
            # 重新排序
            final_list.sort(key=lambda x: x["similarity_score"], reverse=True)
            logger.info(
                f"[GraphAffinity] 亲和力加权后 Top3: "
                f"{[(r['song']['title'], round(r['similarity_score'], 4)) for r in final_list[:3]]}"
            )

        # ---- Step 4: Artist 多样性过滤 ----
        # 如果用户指定了某个歌手（graph_entities 中包含该歌手名），则该歌手不受限制
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
            if count < MAX_SONGS_PER_ARTIST:
                diverse_list.append(item)
                artist_count[artist_lower] = count + 1
            else:
                overflow_list.append(item)
        if len(diverse_list) < MIN_DIVERSE_RESULTS:
            diverse_list.extend(overflow_list[:MIN_DIVERSE_RESULTS - len(diverse_list)])
        final_list = diverse_list
        if exempt_artists:
            logger.info(f"[多样性过滤] 指定歌手豁免: {exempt_artists}")
        logger.info(f"多样性过滤完成，艺术家分布: {dict(artist_count)}")

        # ---- Step 5: MMR genre-aware 多样性重排序 (Jaccard 集合相似度) ----
        def _genre_tags(genre_str: str) -> set:
            """将 'Rock/Hard Rock/Energetic' 拆分为 {'rock', 'hard rock', 'energetic'}"""
            if not genre_str:
                return set()
            return {t.strip().lower() for t in genre_str.replace(",", "/").split("/") if t.strip()}

        def _jaccard(set_a: set, set_b: set) -> float:
            """计算 Jaccard 相似度: |A ∩ B| / |A ∪ B|"""
            if not set_a or not set_b:
                return 0.0
            intersection = len(set_a & set_b)
            union = len(set_a | set_b)
            return intersection / union if union > 0 else 0.0

        if len(final_list) > 2:
            mmr_lambda = 0.7
            selected = [final_list[0]]
            candidates = final_list[1:]

            # 预计算每首歌的 genre 标签集合
            genre_cache: Dict[int, set] = {}
            for idx, item in enumerate(final_list):
                genre_cache[id(item)] = _genre_tags(item.get("song", {}).get("genre", ""))

            while candidates and len(selected) < len(final_list):
                best_score = -1
                best_idx = 0

                # 收集已选歌曲的所有 genre 标签集合
                selected_tag_sets = [genre_cache[id(s)] for s in selected]

                for i, cand in enumerate(candidates):
                    relevance = cand.get("similarity_score", 0)
                    cand_tags = genre_cache[id(cand)]

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

                selected.append(candidates.pop(best_idx))

            final_list = selected
            logger.info(f"[MMR-Jaccard] 多样性重排序完成，流派分布: {[r.get('song', {}).get('genre', '?') for r in final_list[:5]]}")
        
        # ---- 最终安全去重 ----
        seen_final = set()
        deduped_list = []
        for item in final_list:
            s = item.get("song", {})
            fk = HybridRetriever._normalize_key(s.get("title", ""), s.get("artist", ""))
            if fk not in seen_final:
                seen_final.add(fk)
                deduped_list.append(item)
            else:
                logger.info(f"[Dedup-Final] 最终去重: {s.get('title', '')} - {s.get('artist', '')}")
        if len(deduped_list) < len(final_list):
            logger.info(f"[Dedup-Final] 去重前={len(final_list)}, 去重后={len(deduped_list)}")
        final_list = deduped_list

        # 如果有全网聚合结果，强行塞一条纯文本作为上下文给大模型，防止丢了
        if web_res and "未能找到相关有效信息" not in web_res:
            final_list.insert(0, {
                "_raw_markdown": web_res,  # 特殊标记位，供音乐推荐管线解析
                "song": {"title": "🌐 全网资讯补充", "artist": "互联网最新情报", "genre": "News"},
                "reason": "包含通过多源聚合引擎获取的最新的互联网关联资讯，用于补充音乐库之外的信息。",
                "similarity_score": WEB_RESULT_PRIORITY_SCORE
            })
            
        # 为了让终端观测更清晰，打印每一首入选的歌曲及其来源引擎
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
                
                # 避免重复展示已包含在 web_res 的资讯补充
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
        
        return ToolOutput(
            success=len(final_list) > 0,
            data=final_list,
            raw_markdown=raw_markdown,
            error_message=None if final_list else "Not found"
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
            
            # 使用同步 invoke()，避免 async 上下文冲突
            # retrieve() 在 nest_asyncio 环境中通过 run_in_executor 调用，
            # 此处直接用同步调用最为稳健
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
