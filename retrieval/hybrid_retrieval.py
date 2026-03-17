import json
import logging
import os
from typing import Dict, Any, List
import asyncio

from tools.graphrag_search import graphrag_search
# 【V2 升级】替换旧版 Milvus 向量检索为 Neo4j 原生图向量语义搜索
from tools.semantic_search import semantic_search
from tools.web_search_aggregator import _federated_search_async
from config.logging_config import get_logger
from schemas.music_state import ToolOutput


logger = get_logger(__name__)

# ---- 检索结果融合常量 ----
BASELINE_SIMILARITY_SCORE = 0.85      # 单引擎命中的基础相似度分
DUAL_ENGINE_BONUS = 0.1               # 双引擎同时命中时的加分
WEB_RESULT_PRIORITY_SCORE = 9.9       # 网络资讯结果的优先级分数
MAX_SONGS_PER_ARTIST = 2              # 多样性过滤：每个艺术家最多占的歌曲数
MIN_DIVERSE_RESULTS = 3               # 多样性过滤后的最少结果数

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
            vector_desc = precomputed_plan.get("vector_acoustic_query", "")
            web_keywords = precomputed_plan.get("web_search_keywords", "")
            need_web_search = use_web
            search_keyword = web_keywords
            
            # ── 确定性后处理兜底：扫描用户原文补充 LLM 漏填的过滤字段 ──
            from tools.graphrag_search import SCENARIO_TAG_MAP, MOOD_TAG_MAP, GENRE_TAG_MAP
            if not scenario_filter:
                for keyword in SCENARIO_TAG_MAP:
                    if keyword in query:
                        scenario_filter = keyword
                        logger.info(f"[Retrieval] 确定性兜底：从用户输入补充 scenario_filter='{keyword}'")
                        break
            if not mood_filter:
                for keyword in MOOD_TAG_MAP:
                    if keyword in query:
                        mood_filter = keyword
                        logger.info(f"[Retrieval] 确定性兜底：从用户输入补充 mood_filter='{keyword}'")
                        break
            if not genre_filter:
                for keyword in GENRE_TAG_MAP:
                    if keyword in query:
                        genre_filter = keyword
                        logger.info(f"[Retrieval] 确定性兜底：从用户输入补充 genre_filter='{keyword}'")
                        break
            
            # ── 确定性兜底：语言/地区扫描 ──
            from tools.graphrag_search import LANGUAGE_ALIAS_MAP, REGION_ALIAS_MAP
            if not language_filter:
                for keyword, lang in LANGUAGE_ALIAS_MAP.items():
                    if keyword in query:
                        language_filter = lang
                        logger.info(f"[Retrieval] 确定性兜底：从用户输入补充 language_filter='{keyword}' → '{lang}'")
                        break
            if not region_filter:
                for keyword, reg in REGION_ALIAS_MAP.items():
                    if keyword in query:
                        region_filter = reg
                        logger.info(f"[Retrieval] 确定性兜底：从用户输入补充 region_filter='{keyword}' → '{reg}'")
                        break
            
            # ── 调试日志：确认三维过滤字段最终值 ──
            logger.info(
                f"[Retrieval] 过滤字段最终值: genre='{genre_filter}' | scenario='{scenario_filter}' | "
                f"mood='{mood_filter}' | language='{language_filter}' | region='{region_filter}' | "
                f"entities={graph_entities} | query='{query[:50]}'"
            )
            
            # vector_acoustic_query 已在 Planner LLM 中生成完整的 HyDE 声学描述
            # 无需二次 LLM 调用（已合并为零延迟优化）
            if vector_desc:
                logger.info(f"[Vector] 使用 Planner 生成的声学描述 ({len(vector_desc)} chars): {vector_desc[:80]}...")
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
            vector_desc = query
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
                
                local_tasks.append(run_sync_in_executor(graphrag_search.invoke, {"query": graph_term, "limit": limit//2 + 1}))
                local_tasks.append(run_sync_in_executor(semantic_search.invoke, {
                    "query": vector_term, "limit": limit//2 + 1,
                    "language_filter": language_filter or "",
                    "region_filter": region_filter or "",
                }))
                
            # 并发执行本地数据库检索
            local_results = await asyncio.gather(*local_tasks, return_exceptions=True)
            
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
        
        return self._format_results(strategy_name, graph_raw, vector_raw, web_raw, web_playable)


    def _format_results(self, strategy_name: str, graph_res: str, vector_res: str, web_res: str = "", web_playable: List[dict] = None) -> ToolOutput:
        """
        合并各个模块返回的数据，统一构建结构化的 Recommendation 列表。
        如果包含 Web 搜索结果，会额外注入一条特殊的聚合信息供大模型最后组织答案。
        """
        combined_results = {}
        
        def _add_results(res_str, engine_name):
            if not res_str:
                return
            try:
                # 尝试解析底层传上来的 JSON
                items = json.loads(res_str)
                for item in items:
                    if "error" in item:
                        logger.warning(f"Engine {engine_name} returned error: {item}")
                        continue
                        
                    title = item.get("title", "未知标题")
                    artist = item.get("artist", "未知艺术家")
                    
                    key = f"{title}_{artist}"
                    if key not in combined_results:
                        genre = item.get("genre", "")
                        if genre == "Unknown":
                            genre = ""
                            
                        # ============================================================
                        # 【升级】提取底层引擎返回的真实相似度分数
                        # 来源：《第八章 记忆与检索》混合评分公式建议
                        # 用于后续的科学化加权排序，替代原来的固定基准分。
                        # ============================================================
                        raw_distance = item.get("distance", None)
                        raw_similarity = item.get("similarity_score", None)
                        # 向量引擎返回 L2 distance → 转换为 0~1 相似度
                        if raw_distance is not None:
                            normalized_score = 1.0 / (1.0 + float(raw_distance))
                        elif raw_similarity is not None:
                            normalized_score = float(raw_similarity)
                        else:
                            normalized_score = BASELINE_SIMILARITY_SCORE
                            
                        combined_results[key] = {
                            "song": {
                                "title": title,
                                "artist": artist,
                                "album": item.get("album", "未知"),
                                "genre": genre,
                                "preview_url": item.get("preview_url", None),
                                "cover_url": item.get("cover_url", None),
                                "lrc_url": item.get("lrc_url", None)
                            },
                            "reason": f"引擎检索来源: {engine_name}",
                            "similarity_score": normalized_score,
                            # 【升级】分别记录各引擎的原始得分，用于混合加权
                            "_vector_score": normalized_score if "Vector" in engine_name else 0.0,
                            "_graph_score": normalized_score if "Graph" in engine_name else 0.0,
                        }
                    else:
                        combined_results[key]["reason"] += f" 同时被 {engine_name} 引擎捕捉增强！"
                        # 【升级】双引擎命中时，记录另一个引擎的得分
                        if "Vector" in engine_name:
                            combined_results[key]["_vector_score"] = normalized_score
                        elif "Graph" in engine_name:
                            combined_results[key]["_graph_score"] = normalized_score
                        # 【升级】应用混合评分公式: (向量×0.7 + 图谱×0.3)
                        v_score = combined_results[key].get("_vector_score", 0.0)
                        g_score = combined_results[key].get("_graph_score", 0.0)
                        combined_results[key]["similarity_score"] = v_score * 0.7 + g_score * 0.3
            except json.JSONDecodeError:
                logger.error(f"Failed to decode JSON from {engine_name}: {res_str}")
        
        _add_results(graph_res, "知识图谱(GraphRAG)")
        _add_results(vector_res, "语义向量(Neo4j Vector)")
        
        final_list = list(combined_results.values())
        
        # 将从全网转化来的真实可播歌曲库加进去
        if web_playable:
            final_list.extend(web_playable)
            
        final_list.sort(key=lambda x: x["similarity_score"], reverse=True)
        
        # 【多样性过滤】限制同一艺术家最多占 MAX_SONGS_PER_ARTIST 首，避免被单一艺术家垄断
        artist_count: Dict[str, int] = {}
        diverse_list = []
        overflow_list = []  # 超出 max_per_artist 的条目放到末尾备用
        for item in final_list:
            artist = item.get("song", {}).get("artist", "")
            if not artist or artist in ("互联网最新情报",):
                diverse_list.append(item)  # 特殊条目直接保留
                continue
            artist_lower = artist.lower().strip()
            count = artist_count.get(artist_lower, 0)
            if count < MAX_SONGS_PER_ARTIST:
                diverse_list.append(item)
                artist_count[artist_lower] = count + 1
            else:
                overflow_list.append(item)
        # 如果过滤后结果太少（<3首），把溢出条目补充进来
        if len(diverse_list) < MIN_DIVERSE_RESULTS:
            diverse_list.extend(overflow_list[:MIN_DIVERSE_RESULTS - len(diverse_list)])
        final_list = diverse_list
        logger.info(f"多样性过滤完成，艺术家分布: {dict(artist_count)}")
        
        # ---- P2-2: MMR 多样性重排序（genre-aware） ----
        # 在保持相关性的同时，减少同流派歌曲过度集中
        # lambda=0.7 偏向相关性，0.3 偏向多样性
        if len(final_list) > 2:
            mmr_lambda = 0.7
            selected = [final_list[0]]  # 第一首直接选（最相关的）
            candidates = final_list[1:]
            
            while candidates and len(selected) < len(final_list):
                best_score = -1
                best_idx = 0
                selected_genres = set()
                for s in selected:
                    g = s.get("song", {}).get("genre", "").lower().strip()
                    if g:
                        selected_genres.add(g)
                
                for i, cand in enumerate(candidates):
                    relevance = cand.get("similarity_score", 0)
                    cand_genre = cand.get("song", {}).get("genre", "").lower().strip()
                    # 计算冗余度：该流派已有的歌曲越多，冗余越高
                    genre_overlap = 1.0 if cand_genre and cand_genre in selected_genres else 0.0
                    
                    mmr_score = mmr_lambda * relevance - (1 - mmr_lambda) * genre_overlap
                    if mmr_score > best_score:
                        best_score = mmr_score
                        best_idx = i
                
                selected.append(candidates.pop(best_idx))
            
            final_list = selected
            logger.info(f"[MMR] 多样性重排序完成，流派分布: {[r.get('song', {}).get('genre', '?') for r in final_list[:5]]}")
        
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
