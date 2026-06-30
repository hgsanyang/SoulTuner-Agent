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
import os
from typing import List, Dict, Any
from langchain_core.tools import tool

from retrieval.neo4j_client import get_neo4j_client
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
    "壮阔": "epic, grand, orchestral, powerful, atmospheric",
    "柔情": "soft, romantic, gentle, emotional, slow",
    "抹情": "emotional, romantic, melancholic, slow, ballad",
    "沉醉": "dreamy, atmospheric, romantic, ambient, slow",
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


def _dense_backend() -> str:
    backend = str(getattr(settings, "dense_text_audio_backend", "muq") or "muq").strip().lower()
    if backend not in {"muq", "m2d", "both"}:
        logger.warning("[SemanticSearch] 未知 dense backend=%s，回退到 muq", backend)
        return "muq"
    return backend


def _backend_spec(backend: str) -> dict:
    if backend == "muq":
        return {
            "name": "MuQ-MuLan",
            "index": "song_muq_index",
            "property": "muq_embedding",
            "source": "Neo4j SemanticSearch (MuQ-MuLan)",
        }
    return {
        "name": "M2D-CLAP",
        "index": "song_m2d2_index",
        "property": "m2d2_embedding",
        "source": "Neo4j SemanticSearch (M2D-CLAP)",
    }


def _dense_query_variants_enabled() -> bool:
    return str(os.getenv("MUSIC_DENSE_QUERY_VARIANTS", "0")).strip().lower() in {"1", "true", "yes", "on"}


def build_dense_query_variants(text: str) -> List[str]:
    """Return deterministic HyDE-style views for one music query."""
    base = str(text or "").strip()
    if not base:
        return []
    return [
        base,
        f"{base}\nFocus on instrumentation, tempo, timbre, vocals, production texture, and acoustic energy.",
        f"{base}\nFocus on mood trajectory, listening context, language, genre cues, and emotional atmosphere.",
    ]


def _normalize_vector(vector: List[float]) -> List[float]:
    norm = sum(float(x) * float(x) for x in vector) ** 0.5
    return [float(x) / norm for x in vector] if norm > 0 else vector


def _mean_vectors(vectors: List[List[float]]) -> List[float]:
    if not vectors:
        return []
    dim = len(vectors[0])
    if any(len(v) != dim for v in vectors):
        return vectors[0]
    mean = [sum(float(v[i]) for v in vectors) / len(vectors) for i in range(dim)]
    return _normalize_vector(mean)


def _encode_single_query_for_backend(text: str, backend: str) -> List[float]:
    if backend == "muq":
        from retrieval.muq_embedder import encode_text_to_muq

        return encode_text_to_muq(text)
    from retrieval.audio_embedder import encode_text_to_embedding

    return encode_text_to_embedding(text)


def _encode_query_for_backend(text: str, backend: str) -> List[float]:
    variants = build_dense_query_variants(text) if _dense_query_variants_enabled() else [text]
    vectors = [_normalize_vector(_encode_single_query_for_backend(variant, backend)) for variant in variants if variant]
    query_vector = _mean_vectors(vectors) if len(vectors) > 1 else vectors[0]
    if len(vectors) > 1:
        logger.info("[SemanticSearch] 多描述 HyDE 集成启用: %d variants", len(vectors))
    from retrieval.alignment_calibration import apply_alignment_calibration

    return apply_alignment_calibration(query_vector, backend)


def _has_vector_index(client, index_name: str) -> bool:
    try:
        check_result = client.execute_query(
            "SHOW INDEXES YIELD name, state WHERE name = $name RETURN count(*) AS cnt, collect(state) AS states",
            {"name": index_name},
        )
        return bool(check_result and check_result[0].get("cnt", 0) > 0)
    except Exception as exc:
        logger.warning("[SemanticSearch] 向量索引检查失败 %s: %s", index_name, exc)
        return False


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

        # 2. 构建 Neo4j 原生向量检索 Cypher
        client = get_neo4j_client()

        def _run_backend(backend: str) -> List[Dict[str, Any]]:
            spec = _backend_spec(backend)
            if not _has_vector_index(client, spec["index"]):
                logger.warning("[SemanticSearch] %s 索引不存在，跳过该后端", spec["index"])
                return []

            logger.info("[SemanticSearch] 正在用 %s 编码查询文本...", spec["name"])
            query_vector = _encode_query_for_backend(search_text, backend)
            logger.info("[SemanticSearch] %s 编码完成，向量维度: %d", spec["name"], len(query_vector))

            if artist_filter or genre_filter or language_filter or region_filter:
                # ============================================================
                # 联合查询：硬过滤 + 向量软排序
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

                if language_filter:
                    where_clauses.append("toLower(s.language) = toLower($language_filter)")
                    params["language_filter"] = language_filter

                if region_filter:
                    where_clauses.append("toLower(s.region) = toLower($region_filter)")
                    params["region_filter"] = region_filter

                where_str = " AND ".join(where_clauses)
                vector_property = spec["property"]
                cypher = f"""
                MATCH {match_pattern}
                WHERE {where_str} AND s.{vector_property} IS NOT NULL
                WITH s,
                     vector.similarity.cosine(s.{vector_property}, $query_vector) AS score
                ORDER BY score DESC
                LIMIT $limit
                OPTIONAL MATCH (s)-[:PERFORMED_BY]->(art:Artist)
                OPTIONAL MATCH (s)-[:BELONGS_TO_GENRE]->(genre:Genre)
                OPTIONAL MATCH (s)-[:HAS_MOOD]->(mood:Mood)
                OPTIONAL MATCH (s)-[:HAS_THEME]->(theme:Theme)
                OPTIONAL MATCH (s)-[:FITS_SCENARIO]->(scenario:Scenario)
                RETURN s.title AS title, art.name AS artist,
                       s.album AS album, s.audio_url AS audio_url,
                       s.cover_url AS cover_url, s.lrc_url AS lrc_url,
                       coalesce(s.language, 'Unknown') AS language,
                       coalesce(s.region, 'Unknown') AS region,
                       collect(DISTINCT genre.name) AS genres,
                       collect(DISTINCT mood.name) AS moods,
                       collect(DISTINCT theme.name) AS themes,
                       collect(DISTINCT scenario.name) AS scenarios,
                       score AS similarity_score
                """
                logger.info(
                    "[SemanticSearch] 执行 %s 联合图向量查询（language=%s, region=%s）",
                    spec["name"],
                    language_filter,
                    region_filter,
                )
                rows = client.execute_query(cypher, params)
                for row in rows:
                    row["_vector_backend"] = backend
                    row["_vector_backend_name"] = spec["name"]
                return rows

            # ============================================================
            # 纯向量检索 + OMAR 声学质心二阶段融合
            # MuQ/M2D 负责文本→音频语义匹配，OMAR 保持声学辅助。
            # ============================================================
            _has_omar_index = False
            try:
                _has_omar_index = _has_vector_index(client, "song_omar_index")
                if _has_omar_index:
                    logger.info("[SemanticSearch] OMAR-RQ 向量索引存在，启用声学融合")
            except Exception:
                logger.info("[SemanticSearch] OMAR 索引检查跳过，使用单文本音频后端")

            if _has_omar_index:
                wide = limit * 3
                base_cypher = f"""
                CALL db.index.vector.queryNodes('{spec["index"]}', $wide, $query_vector)
                YIELD node AS song, score
                OPTIONAL MATCH (song)-[:PERFORMED_BY]->(art:Artist)
                OPTIONAL MATCH (song)-[:BELONGS_TO_GENRE]->(genre:Genre)
                OPTIONAL MATCH (song)-[:HAS_MOOD]->(mood:Mood)
                OPTIONAL MATCH (song)-[:HAS_THEME]->(theme:Theme)
                OPTIONAL MATCH (song)-[:FITS_SCENARIO]->(scenario:Scenario)
                RETURN song.title AS title, art.name AS artist,
                       song.album AS album, song.audio_url AS audio_url,
                       song.cover_url AS cover_url, song.lrc_url AS lrc_url,
                       coalesce(song.language, 'Unknown') AS language,
                       coalesce(song.region, 'Unknown') AS region,
                       collect(DISTINCT genre.name) AS genres,
                       collect(DISTINCT mood.name) AS moods,
                       collect(DISTINCT theme.name) AS themes,
                       collect(DISTINCT scenario.name) AS scenarios,
                       score AS similarity_score,
                       elementId(song) AS _eid
                """
                base_results = client.execute_query(base_cypher, {"query_vector": query_vector, "wide": wide})
                logger.info("[SemanticSearch] Phase 1 %s KNN: %d 候选", spec["name"], len(base_results))

                # ---- Phase 2: OMAR 质心 + OMAR KNN ----
                omar_scores = {}  # eid → omar_score
                try:
                    if base_results:
                        # 取文本音频 top 候选的 OMAR embedding，求质心
                        top_eids = [r["_eid"] for r in base_results[:limit]]
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
                    logger.warning("[SemanticSearch] OMAR Phase 2 失败，退回 %s-only: %s", spec["name"], e)

                # ---- Phase 3: 融合排序 ----
                fused = []
                for r in base_results:
                    base_s = r["similarity_score"]
                    omar_s = omar_scores.get(r["_eid"], 0.0)
                    # 有 OMAR 分数时加权融合，否则只用 M2D
                    if omar_s > 0:
                        score = 0.7 * base_s + 0.3 * omar_s
                    else:
                        score = base_s
                    fused.append({**r, "similarity_score": score, "_vector_backend": backend, "_vector_backend_name": spec["name"]})

                fused.sort(key=lambda x: x["similarity_score"], reverse=True)
                rows = fused[:limit]
                logger.info(
                    "[SemanticSearch] Phase 3 融合排序完成，返回 %d 首(limit=%d, OMAR加权=%d)",
                    len(rows),
                    limit,
                    sum(1 for r in rows if omar_scores.get(r.get("_eid"), 0) > 0),
                )
                return rows

            cypher = f"""
            CALL db.index.vector.queryNodes('{spec["index"]}', $limit, $query_vector)
            YIELD node AS song, score
            OPTIONAL MATCH (song)-[:PERFORMED_BY]->(art:Artist)
            OPTIONAL MATCH (song)-[:BELONGS_TO_GENRE]->(genre:Genre)
            OPTIONAL MATCH (song)-[:HAS_MOOD]->(mood:Mood)
            OPTIONAL MATCH (song)-[:HAS_THEME]->(theme:Theme)
            OPTIONAL MATCH (song)-[:FITS_SCENARIO]->(scenario:Scenario)
            RETURN song.title AS title, art.name AS artist,
                   song.album AS album, song.audio_url AS audio_url,
                   song.cover_url AS cover_url, song.lrc_url AS lrc_url,
                   coalesce(song.language, 'Unknown') AS language,
                   coalesce(song.region, 'Unknown') AS region,
                   collect(DISTINCT genre.name) AS genres,
                   collect(DISTINCT mood.name) AS moods,
                   collect(DISTINCT theme.name) AS themes,
                   collect(DISTINCT scenario.name) AS scenarios,
                   score AS similarity_score
            """
            logger.info("[SemanticSearch] 执行单模型 %s KNN 检索", spec["name"])
            rows = client.execute_query(cypher, {"query_vector": query_vector, "limit": limit})
            for row in rows:
                row["_vector_backend"] = backend
                row["_vector_backend_name"] = spec["name"]
            return rows

        configured_backend = _dense_backend()
        backend_order = ["muq", "m2d"] if configured_backend == "both" else [configured_backend]
        if configured_backend == "muq":
            backend_order.append("m2d")

        results = []
        used_backends = []
        seen_backend = set()
        for backend in backend_order:
            if backend in seen_backend:
                continue
            seen_backend.add(backend)
            try:
                backend_results = _run_backend(backend)
            except Exception as exc:
                logger.warning("[SemanticSearch] %s 后端失败: %s", _backend_spec(backend)["name"], exc)
                backend_results = []
            if backend_results:
                used_backends.append(backend)
                if configured_backend == "both":
                    results.extend(backend_results)
                    continue
                results = backend_results
                break
            if backend == "muq":
                logger.warning("[SemanticSearch] MuQ-MuLan 无结果，尝试 M2D-CLAP fallback")

        if configured_backend == "both" and results:
            deduped = {}
            for row in results:
                key = f"{row.get('title') or ''}::{row.get('artist') or ''}".casefold()
                existing = deduped.get(key)
                if existing is None or float(row.get("similarity_score") or 0.0) > float(existing.get("similarity_score") or 0.0):
                    deduped[key] = row
            results = sorted(deduped.values(), key=lambda row: float(row.get("similarity_score") or 0.0), reverse=True)[:limit]
        logger.info("[SemanticSearch] dense backend configured=%s used=%s", configured_backend, used_backends or ["none"])

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
            genres = [x for x in (record.get("genres") or []) if x]
            moods = [x for x in (record.get("moods") or []) if x]
            themes = [x for x in (record.get("themes") or []) if x]
            scenarios = [x for x in (record.get("scenarios") or []) if x]
            vector_backend = record.get("_vector_backend") or configured_backend
            vector_backend_name = record.get("_vector_backend_name") or _backend_spec("muq" if vector_backend == "both" else vector_backend)["name"]
            genre_display_parts = []
            if genres:
                genre_display_parts.append("/".join(genres[:2]))
            if moods:
                genre_display_parts.append(moods[0])
            if scenarios:
                genre_display_parts.append(scenarios[0])

            structured_results.append({
                "title": title,
                "artist": artist,
                "album": album,
                "genre": "/".join(genre_display_parts),
                "genres": genres,
                "moods": moods,
                "themes": themes,
                "scenarios": scenarios,
                "language": record.get("language", "Unknown"),
                "region": record.get("region", "Unknown"),
                "source": f"Neo4j SemanticSearch ({vector_backend_name})",
                "vector_backend": vector_backend,
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
