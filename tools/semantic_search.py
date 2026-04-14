# ============================================================
# 【V2 升级】语义搜索工具 ── 替代旧版 vector_search.py
# 来源：V2 架构重构方案 → Phase 4
#
# 核心变更：
#   - 废弃 Milvus 向量数据库，改用 Neo4j Native Vector Search
#   - 废弃 LAION-CLAP 占位编码，改用真实 M2D-CLAP 跨模态编码
#   - 支持"硬过滤 + 软排序"一体化 Cypher 查询
#     （如：锁定"周杰伦"的歌，再按"悲伤"语义向量排序）
# ============================================================

import json
import logging
import os
from typing import List, Dict, Any, Optional
from langchain_core.tools import tool

from retrieval.neo4j_client import get_neo4j_client
from retrieval.audio_embedder import encode_text_to_embedding
from config.logging_config import get_logger
from config.settings import settings

logger = get_logger(__name__)

# 【V2 升级】常见中文情绪词 → 英文声学描述缓存
# 命中缓存时直接用英文去编码向量，跳过 LLM 翻译调用
_TRANSLATION_CACHE = {
    "开心": "happy, joyful, upbeat, bright, cheerful",
    "快乐": "happy, joyful, upbeat, bright, cheerful",
    "悲伤": "sad, melancholic, sorrowful, slow, minor key",
    "伤心": "sad, heartbroken, melancholic, emotional",
    "难过": "sad, painful, sorrowful, slow",
    "丧": "depressed, gloomy, dark, slow, lo-fi",
    "疗愈": "healing, soothing, gentle, warm, acoustic",
    "放松": "relaxing, chill, calm, ambient, mellow",
    "舒缓": "soothing, gentle, slow, relaxing, ambient",
    "平静": "peaceful, calm, serene, quiet, ambient",
    "安静": "quiet, peaceful, still, ambient, soft",
    "怀旧": "nostalgic, retro, vintage, warm, classic",
    "浪漫": "romantic, warm, gentle, love, acoustic",
    "甜蜜": "sweet, lovely, warm, happy, soft",
    "兴奋": "excited, energetic, upbeat, fast, pumping",
    "激昂": "thrilling, intense, powerful, fast, epic",
    "学习": "focus, concentration, lo-fi, ambient, calm",
    "专注": "focused, deep concentration, minimal, ambient",
    "运动": "workout, energetic, high tempo, pumping, electronic",
    "健身": "gym, workout, high energy, fast beat, electronic",
    "睡觉": "sleep, lullaby, quiet, ambient, very slow",
    "派对": "party, dance, upbeat, electronic, fun",
    # 新增缺失的情绪词
    "深情": "emotional, sentimental, romantic, heartfelt, slow, ballad",
    "温柔": "gentle, soft, warm, romantic, acoustic, slow",
    "感动": "emotional, touching, heartfelt, warm, hopeful",
    "温暖": "warm, gentle, hopeful, comforting, acoustic",
    "深沉": "dark, deep, melancholic, slow, atmospheric",
    "激昂": "intense, powerful, epic, fast, energetic",
    "壮阔": "epic, grand, orchestral, powerful, atmospheric",
    "柔情": "soft, romantic, gentle, emotional, slow",
    "抹情": "emotional, romantic, melancholic, slow, ballad",
    "沉醉": "dreamy, atmospheric, romantic, ambient, slow",
    "丧": "energetic, intense, powerful, fast, heavy",
    "带感": "groovy, rhythmic, energetic, raw, powerful",

    # ── 场景/亚文化词（对齐 M2D-CLAP 训练分布，使用感知描述词汇）──
    "蹦迪": "high energy dance music, heavy bass drops, fast electronic beat, pumping club",
    "夜店": "club dance music, heavy bass, electronic, high energy, fast beat",
    "仙气": "ethereal, floating, dreamy, soft ambient, celestial, gentle",
    "仙气飘飘": "ethereal, floating, dreamy, soft ambient, celestial, gentle and airy",
    "赛博朋克": "dark electronic, futuristic synths, heavy bass, industrial, atmospheric",
    "复古": "retro, vintage, warm analog, classic, old school",
    "氛围感": "atmospheric, ambient, spacious, dreamy, layered textures",
    "电影感": "cinematic, orchestral, dramatic, sweeping, emotional strings",
    "文艺": "indie, acoustic, soft, intimate, folk, gentle vocals",
    "小清新": "light, fresh, acoustic guitar, bright, cheerful, gentle",
    "暗黑": "dark, heavy, distorted, aggressive, low and menacing",
    "空灵": "ethereal, airy, reverberant, floating, delicate, ambient",
    "炸裂": "explosive, heavy distortion, powerful drums, intense, loud",
    "上头": "catchy, addictive, repetitive hook, energetic, groovy",
    "催泪": "tearful, deeply emotional, slow strings, sorrowful, heartbreaking",

    # ── 身体活动/生理状态词 ──
    "瑜伽": "peaceful, meditative, slow ambient, gentle, flowing, calm",
    "冥想": "meditative, quiet, ambient drone, still, peaceful, minimal",
    "散步": "light, gentle, moderate tempo, pleasant, acoustic, easy-going",
    "跑步": "fast tempo, energetic, driving beat, pumping, high energy",
    "发呆": "ambient, dreamy, floating, slow, spacious, minimal",
    "失眠": "quiet, ambient, very slow, gentle, soothing, soft piano",
    "起床": "bright, uplifting, moderate tempo, cheerful, fresh, acoustic",

    # ── 情绪细粒度扩展 ──
    "思念": "nostalgic, longing, emotional, slow, warm, bittersweet",
    "释然": "peaceful, relieved, hopeful, gentle, warm, uplifting",
    "窒息": "intense, oppressive, heavy, dark, slow, suffocating atmosphere",
    "炸裂": "explosive, heavy distortion, powerful drums, intense, loud",
    "躁动": "restless, agitated, fast, distorted, raw, energetic",
    "迷幻": "psychedelic, swirling, trippy, reverberant, spacious, dreamy",
    "震撼": "epic, powerful, grand, orchestral, intense, dramatic",
    "甜美": "sweet, bright, light, cheerful, soft, lovely melody",
    "苦涩": "bittersweet, melancholic, slow, minor key, emotional, subdued",
    "孤独": "lonely, solitary, quiet, sparse arrangement, intimate, somber",
    "愤怒": "angry, aggressive, fast, heavy, distorted, loud, intense",
    "慵懒": "lazy, laid-back, slow groove, mellow, relaxed, lo-fi",
    "忧郁": "melancholic, gloomy, dark, slow, minor key, somber",
    "惆怅": "wistful, melancholic, nostalgic, slow, gentle, bittersweet",
    "热烈": "passionate, intense, fast, energetic, powerful, bright",
}


def _translate_query(query: str) -> str:
    """
    【V2 升级】查询预处理：中文情绪词命中缓存则直接翻译，
    否则保持原文（由 M2D-CLAP 多语言能力直接理解）。
    """
    query_stripped = query.strip()
    if query_stripped in _TRANSLATION_CACHE:
        translated = _TRANSLATION_CACHE[query_stripped]
        logger.info(f"[SemanticSearch] 命中翻译缓存: '{query_stripped}' → '{translated}'")
        return translated
    return query


@tool
def semantic_search(query: str, limit: int = 0, artist_filter: str = "", genre_filter: str = "",
                    language_filter: str = "", region_filter: str = "") -> str:
    """
    【V2 升级】Neo4j 原生图向量语义搜索工具
    根据用户的自然语言描述，使用 M2D-CLAP 编码为向量，
    在 Neo4j 图中检索语义最相似的歌曲。
    支持可选的硬过滤条件（歌手/流派/语言/地区），实现:
    "先通过图谱关系精准圈定候选池，再通过向量排序找到最匹配的"

    Args:
        query: 用户的音乐描述（如 "适合雨天听的悲伤钢琴曲"）
        limit: 返回结果数量（默认读取 settings.semantic_search_limit）
        artist_filter: 可选，按歌手名过滤（如 "周杰伦"）
        genre_filter: 可选，按流派过滤（如 "Pop"）
        language_filter: 可选，按语言过滤（如 "Chinese"）
        region_filter: 可选，按地区过滤（如 "Mainland China"）
    """
    if limit <= 0:
        limit = settings.semantic_search_limit
    logger.info(f"[SemanticSearch] 实际使用 limit={limit} | 查询: '{query}' | 歌手过滤: '{artist_filter}' | 流派过滤: '{genre_filter}' | 语言过滤: '{language_filter}' | 地区过滤: '{region_filter}'")

    try:
        # 1. 文本预处理
        search_text = _translate_query(query)

        # 2. M2D-CLAP 编码：文本 → query_vector
        logger.info(f"[SemanticSearch] 正在用 M2D-CLAP 编码查询文本...")
        query_vector = encode_text_to_embedding(search_text)
        logger.info(f"[SemanticSearch] 编码完成，向量维度: {len(query_vector)}")

        # 3. 构建 Neo4j 原生向量检索 Cypher
        client = get_neo4j_client()

        if artist_filter or genre_filter or language_filter or region_filter:
            # ============================================================
            # 【V2 核心创新】联合查询：硬过滤 + 向量软排序
            # 先用 MATCH 锁定歌手/流派候选池，再 vector.queryNodes 排序
            # ============================================================
            where_clauses = []
            params = {"query_vector": query_vector, "limit": limit}

            match_pattern = "(s:Song)"
            if artist_filter:
                match_pattern = "(s:Song)-[:PERFORMED_BY]->(a:Artist)"
                where_clauses.append("toLower(a.name) CONTAINS toLower($artist_filter)")
                params["artist_filter"] = artist_filter
            if genre_filter:
                if "-[:PERFORMED_BY]->" in match_pattern:
                    match_pattern = "(s:Song)-[:PERFORMED_BY]->(a:Artist), (s)-[:BELONGS_TO_GENRE]->(g:Genre)"
                else:
                    match_pattern = "(s:Song)-[:BELONGS_TO_GENRE]->(g:Genre)"
                where_clauses.append("toLower(g.name) CONTAINS toLower($genre_filter)")
                params["genre_filter"] = genre_filter

            # 语言过滤：使用 Song 节点属性（不是关系节点）
            lang_match = ""
            if language_filter:
                where_clauses.append("toLower(s.language) = toLower($language_filter)")
                params["language_filter"] = language_filter

            # 地区过滤：使用 Song 节点属性（不是关系节点）
            region_match = ""
            if region_filter:
                where_clauses.append("toLower(s.region) = toLower($region_filter)")
                params["region_filter"] = region_filter

            where_str = " AND ".join(where_clauses)

            # 两阶段查询：先硬过滤收集候选 Song ID，再在候选中做向量排序
            cypher = f"""
            MATCH {match_pattern}{lang_match}{region_match}
            WHERE {where_str} AND s.m2d2_embedding IS NOT NULL
            WITH s, 
                 vector.similarity.cosine(s.m2d2_embedding, $query_vector) AS score
            ORDER BY score DESC
            LIMIT $limit
            OPTIONAL MATCH (s)-[:PERFORMED_BY]->(art:Artist)
            RETURN s.title AS title, art.name AS artist, 
                   s.album AS album, s.audio_url AS audio_url,
                   s.cover_url AS cover_url, s.lrc_url AS lrc_url,
                   score AS similarity_score
            """

            logger.info(f"[SemanticSearch] 执行联合图向量查询（硬过滤模式，含 language={language_filter}, region={region_filter}）")
            results = client.execute_query(cypher, params)

        else:
            # ============================================================
            # 【V2 + OMAR 双向量融合】纯向量检索 + 双模型加权融合
            # M2D-CLAP: 跨模态语义匹配（文本↔音频语义）权重 0.7
            # OMAR-RQ:  纯声学特征匹配（乐器/节奏/调性）权重 0.3
            # 当 Song 节点无 omar_embedding 时自动退回单模型 M2D-CLAP
            # ============================================================
            
            # 先检查 omar_embedding 向量索引是否存在
            _has_omar_index = False
            try:
                check_cypher = "SHOW INDEXES YIELD name WHERE name = 'song_omar_index' RETURN count(*) AS cnt"
                check_result = client.execute_query(check_cypher, {})
                if check_result and check_result[0].get("cnt", 0) > 0:
                    _has_omar_index = True
                    logger.info("[SemanticSearch] OMAR-RQ 向量索引存在，启用双向量融合")
            except Exception:
                logger.info("[SemanticSearch] OMAR 索引检查跳过，使用单模型 M2D-CLAP")
            
            if _has_omar_index:
                # ============================================================
                # 正确的两阶段 OMAR 双向量融合
                # Phase 1: M2D-CLAP KNN（文本 → 语义匹配）→ 广泛候选集
                # Phase 2: 取候选集的 OMAR 向量求质心 → OMAR KNN → 声学近邻
                # Phase 3: 两路结果按 song 合并加权排序
                # ============================================================
                wide = limit * 3  # 广泛候选                
                # ---- Phase 1: M2D-CLAP KNN ----
                m2d_cypher = """
                CALL db.index.vector.queryNodes('song_m2d2_index', $wide, $query_vector)
                YIELD node AS song, score
                OPTIONAL MATCH (song)-[:PERFORMED_BY]->(art:Artist)
                RETURN song.title AS title, art.name AS artist,
                       song.album AS album, song.audio_url AS audio_url,
                       song.cover_url AS cover_url, song.lrc_url AS lrc_url,
                       score AS similarity_score,
                       elementId(song) AS _eid
                """
                m2d_results = client.execute_query(m2d_cypher, {"query_vector": query_vector, "wide": wide})
                logger.info(f"[SemanticSearch] Phase 1 M2D-CLAP KNN: {len(m2d_results)} 候选")
                
                # ---- Phase 2: OMAR 质心 + OMAR KNN ----
                omar_scores = {}  # eid → omar_score
                try:
                    if m2d_results:
                        # 取 M2D top 候选的 OMAR embedding，求质心
                        top_eids = [r["_eid"] for r in m2d_results[:limit]]
                        centroid_cypher = """
                        UNWIND $eids AS eid
                        MATCH (s) WHERE elementId(s) = eid AND s.omar_embedding IS NOT NULL
                        RETURN s.omar_embedding AS emb
                        """
                        emb_rows = client.execute_query(centroid_cypher, {"eids": top_eids})
                        
                        if emb_rows and len(emb_rows) >= 2:
                            import numpy as np
                            embeddings = [row["emb"] for row in emb_rows]
                            centroid = np.mean(embeddings, axis=0).tolist()
                            
                            # 用质心查 OMAR KNN
                            omar_cypher = """
                            CALL db.index.vector.queryNodes('song_omar_index', $wide, $centroid)
                            YIELD node AS song, score
                            RETURN elementId(song) AS _eid, score AS omar_score
                            """
                            omar_results = client.execute_query(omar_cypher, {"centroid": centroid, "wide": wide})
                            for r in (omar_results or []):
                                omar_scores[r["_eid"]] = r["omar_score"]
                            logger.info(f"[SemanticSearch] Phase 2 OMAR KNN: {len(omar_scores)} 声学近邻")
                        else:
                            logger.info("[SemanticSearch] OMAR 候选不足（<2），跳过 Phase 2")
                except Exception as e:
                    logger.warning(f"[SemanticSearch] OMAR Phase 2 失败，退回 M2D-only: {e}")
                
                # ---- Phase 3: 融合排序 ----
                fused = []
                for r in m2d_results:
                    m2d_s = r["similarity_score"]
                    omar_s = omar_scores.get(r["_eid"], 0.0)
                    # 有 OMAR 分数时加权融合，否则只用 M2D
                    if omar_s > 0:
                        score = 0.7 * m2d_s + 0.3 * omar_s
                    else:
                        score = m2d_s
                    fused.append({**r, "similarity_score": score})
                
                fused.sort(key=lambda x: x["similarity_score"], reverse=True)
                results = fused[:limit]
                logger.info(f"[SemanticSearch] Phase 3 融合排序完成，返回 {len(results)} 首(limit={limit}, OMAR加权: {sum(1 for r in fused[:limit] if omar_scores.get(r.get('_eid'), 0) > 0)})")
            else:
                # 单模型 M2D-CLAP KNN（无 OMAR 索引时）
                cypher = """
                CALL db.index.vector.queryNodes('song_m2d2_index', $limit, $query_vector)
                YIELD node AS song, score
                OPTIONAL MATCH (song)-[:PERFORMED_BY]->(art:Artist)
                RETURN song.title AS title, art.name AS artist,
                       song.album AS album, song.audio_url AS audio_url,
                       song.cover_url AS cover_url, song.lrc_url AS lrc_url,
                       score AS similarity_score
                """
                params = {"query_vector": query_vector, "limit": limit}
                logger.info(f"[SemanticSearch] 执行单模型 M2D-CLAP KNN 检索")
                results = client.execute_query(cypher, params)

        # 4. 格式化结果
        BASE_API_URL = settings.api_base_url
        structured_results = []
        for record in results:
            title = record.get("title", "未知")
            artist = record.get("artist") or "未知"
            album = record.get("album") or "未知"
            # Neo4j 存储的是相对路径（如 /static/audio/xxx.flac），需拼接 base URL
            audio_url = record.get("audio_url", "") or ""
            cover_url = record.get("cover_url", "") or ""
            lrc_url = record.get("lrc_url", "") or ""
            preview_url = f"{BASE_API_URL}{audio_url}" if audio_url else ""
            cover_full = f"{BASE_API_URL}{cover_url}" if cover_url else ""
            lrc_full = f"{BASE_API_URL}{lrc_url}" if lrc_url else ""
            similarity = record.get("similarity_score", 0.0)

            structured_results.append({
                "title": title,
                "artist": artist,
                "album": album,
                "source": "Neo4j SemanticSearch (M2D-CLAP)",
                "similarity_score": float(similarity),
                "preview_url": preview_url,
                "cover_url": cover_full,
                "lrc_url": lrc_full,
            })

        logger.info(f"[SemanticSearch] 返回 {len(structured_results)} 条结果")

        if not structured_results:
            return json.dumps([{"error": "Neo4j 语义搜索未找到匹配结果"}], ensure_ascii=False)

        return json.dumps(structured_results, ensure_ascii=False)

    except Exception as e:
        logger.error(f"[SemanticSearch] 检索失败: {e}")
        return json.dumps([{"error": f"语义搜索异常: {str(e)}"}], ensure_ascii=False)
