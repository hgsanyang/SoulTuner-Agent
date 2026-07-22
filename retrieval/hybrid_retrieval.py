import json
import os
import re
import asyncio
import concurrent.futures
import time
from typing import List, Dict, Any

from tools.semantic_search import semantic_search
from tools.web_search_aggregator import _federated_search_async
from retrieval.recall_sources import (
    graph_candidate_recall,
)
from retrieval.retrieval_fusion import (
    apply_hard_filters,
    normalize_song_key,
    recall_weights_for_intent,
    weighted_rrf,
)
from retrieval.post_recall_adjustments import (
    DEFAULT_CONFIG as DEFAULT_POST_RECALL_CONFIG,
    PostRecallAdjustmentConfig,
    apply_post_recall_adjustments,
)
from config.logging_config import get_logger
from schemas.music_state import ToolOutput


logger = get_logger(__name__)

# ---- 检索结果融合常量 ----
BASELINE_SIMILARITY_SCORE = 0.85      # 单引擎命中的基础相似度分（仅作 fallback）
WEB_RESULT_PRIORITY_SCORE = 9.9       # 网络资讯结果的优先级分数
MIN_DIVERSE_RESULTS = 6               # 多样性过滤后的最少结果数

# ---- Neo4j 图距离加权参数（从 settings 读取，此处为 fallback 默认值） ----
GRAPH_AFFINITY_USER_ID = "local_admin" # 图距离计算的用户 ID

USER_EXPOSURE_UPDATE_QUERY = """
UNWIND $songs AS row
MERGE (u:User {id: $user_id})
ON CREATE SET u.created_at = timestamp()
WITH u, row
MATCH (s:Song)
WHERE s.title = row.title
  AND (row.artist = '' OR coalesce(s.artist, '') = row.artist)
MERGE (u)-[e:EXPOSED]->(s)
ON CREATE SET e.ts_alpha = 1.0, e.ts_beta = 1.0, e.count = 0
SET e.ts_beta = coalesce(e.ts_beta, 1.0) + 0.3,
    e.count = coalesce(e.count, 0) + 1,
    e.last_exposed_at = timestamp()
"""


def local_recall_tools_from_plan(
    plan: Dict[str, Any] | None,
    *,
    execution_enabled: bool,
) -> tuple[bool, bool, bool]:
    """Return graph/audio switches and whether ToolPlan is actively controlling them."""

    tool_plan = dict((plan or {}).get("_tool_plan") or {})
    if not execution_enabled or not tool_plan:
        return True, True, False
    names = {
        str(call.get("name") or "")
        for call in (tool_plan.get("tool_calls") or [])
        if isinstance(call, dict)
    }
    return "search_graph" in names, "search_audio" in names, True


def _post_recall_config_for_user(user_id: str = GRAPH_AFFINITY_USER_ID) -> PostRecallAdjustmentConfig:
    """Apply only bounded, validation-gated feedback multipliers."""
    multipliers: Dict[str, float] = {}
    try:
        from services.ranking_policy import runtime_policy_for_user

        policy = runtime_policy_for_user(user_id)
        multipliers.update((policy or {}).get("post_recall_multipliers") or {})
    except Exception:
        multipliers = {}
    try:
        from services.policy_memory import policy_runtime_payload_for_user

        memory_payload = policy_runtime_payload_for_user(user_id) or {}
        memory_multipliers = memory_payload.get("post_recall_multipliers") or {}
        for key, value in memory_multipliers.items():
            multipliers[key] = float(multipliers.get(key, 1.0)) * float(value)
    except Exception:
        pass
    return PostRecallAdjustmentConfig(
        personal_weight=DEFAULT_POST_RECALL_CONFIG.personal_weight
        * float(multipliers.get("personal", 1.0)),
        freshness_weight=DEFAULT_POST_RECALL_CONFIG.freshness_weight
        * float(multipliers.get("freshness", 1.0)),
        longtail_weight=DEFAULT_POST_RECALL_CONFIG.longtail_weight
        * float(multipliers.get("longtail", 1.0)),
        exposure_penalty_weight=DEFAULT_POST_RECALL_CONFIG.exposure_penalty_weight
        * float(multipliers.get("exposure_penalty", 1.0)),
        semantic_preference_weight=DEFAULT_POST_RECALL_CONFIG.semantic_preference_weight
        * float(multipliers.get("semantic_preference", 1.0)),
        semantic_conflict_weight=DEFAULT_POST_RECALL_CONFIG.semantic_conflict_weight
        * float(multipliers.get("semantic_conflict", 1.0)),
        delta_limit=DEFAULT_POST_RECALL_CONFIG.delta_limit,
        freshness_half_life_days=DEFAULT_POST_RECALL_CONFIG.freshness_half_life_days,
        exposure_half_life_days=DEFAULT_POST_RECALL_CONFIG.exposure_half_life_days,
        exposure_penalty_pivot=DEFAULT_POST_RECALL_CONFIG.exposure_penalty_pivot,
    )

def _norm_token(value: Any) -> str:
    return str(value or "").strip().casefold()


def _iter_terms(value: Any) -> List[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if value is None:
        return []
    text = str(value).strip()
    return [text] if text else []


def _expand_query_terms(values: List[str]) -> List[str]:
    terms: List[str] = []
    for value in values:
        for part in re.split(r"[/|,，;；]+", str(value or "")):
            text = part.strip()
            if text and text not in terms:
                terms.append(text)
    return terms


def _song_objective_tokens(song: Dict[str, Any]) -> set[str]:
    tokens: set[str] = set()
    for field in ("genre", "genres", "moods", "themes", "scenarios", "language", "region", "energy_level"):
        value = song.get(field)
        values = value if isinstance(value, list) else [value]
        for item in values:
            normalized = _norm_token(item)
            if not normalized:
                continue
            tokens.add(normalized)
            for part in re.split(r"[/|,，\s]+", normalized):
                if part:
                    tokens.add(part)
    if song.get("is_instrumental") or song.get("instrumental"):
        tokens.update({"instrumental", "withoutvocals", "no vocals"})
    if song.get("has_vocal") is False:
        tokens.update({"instrumental", "withoutvocals", "no vocals"})
    elif song.get("has_vocal") is True:
        tokens.update({"vocal", "vocals"})
    if song.get("has_drums") is False:
        tokens.update({"nodrums", "no drums", "withoutdrums"})
    elif song.get("has_drums") is True:
        tokens.add("drums")
    if _norm_token(song.get("energy_level")) in {"low", "lowenergy", "quiet", "calm"}:
        tokens.update({"lowenergy", "low energy", "quiet"})
    return tokens


def _contains_token(tokens: set[str], wanted: str) -> bool:
    wanted_norm = _norm_token(wanted)
    if not wanted_norm:
        return False
    for token in tokens:
        if wanted_norm == token:
            return True
        # Avoid treating broad parent tags as a hit for specific labels:
        # "pop" must not count as "k-pop", and "r&b" must not count as a
        # longer neighboring label.
        if token in wanted_norm and len(token) < 4:
            continue
        if wanted_norm in token or token in wanted_norm:
            return True
    return False


def rerank_with_soft_constraints(
    candidates: List[dict],
    soft_intent: Dict[str, Any] | None,
    hints: Dict[str, Any] | None = None,
    query_text: str = "",
    *,
    min_keep: int = 8,
) -> List[dict]:
    """Demote objective-tag conflicts from the LLM-produced plan.

    Hard constraints are handled by ``apply_hard_filters``. This step is
    intentionally LLM-first: it does not parse the raw user query and does not
    infer "quiet means avoid dance" in Python. If a turn should avoid Dance,
    Driving, Party, etc., the planner must express that in ``soft_intent.avoid``.
    """
    if not candidates:
        return candidates

    soft = soft_intent or {}
    avoid_terms = _expand_query_terms(_iter_terms(soft.get("avoid")))
    positive_terms = _expand_query_terms(
        _iter_terms((hints or {}).get("mood"))
        + _iter_terms((hints or {}).get("scenario"))
        + _iter_terms((hints or {}).get("genres"))
    )

    if not avoid_terms and not positive_terms:
        return candidates

    adjusted: List[dict] = []
    for item in candidates:
        song = item.get("song") or {}
        tokens = _song_objective_tokens(song)
        negative_hits = {_norm_token(term) for term in avoid_terms if _contains_token(tokens, term)}
        positive_hits = {_norm_token(term) for term in positive_terms if _contains_token(tokens, term)}

        penalty = min(0.45, 0.16 * len(negative_hits))
        bonus = min(0.12, 0.03 * len(positive_hits))

        clone = dict(item)
        base = float(clone.get("similarity_score") or clone.get("_rrf_score") or 0.0)
        clone["_soft_avoid_penalty"] = round(penalty, 4)
        clone["_soft_positive_bonus"] = round(bonus, 4)
        clone["_soft_negative_hits"] = sorted(negative_hits)
        clone["_soft_conflict_hits"] = sorted(negative_hits)
        clone["_soft_positive_hits"] = sorted(positive_hits)
        clone["similarity_score"] = base - penalty + bonus
        adjusted.append(clone)

    adjusted.sort(
        key=lambda item: (
            float(item.get("similarity_score") or 0.0),
            float(item.get("_rrf_score") or 0.0),
        ),
        reverse=True,
    )

    non_conflicting = [
        item
        for item in adjusted
        if float(item.get("_soft_avoid_penalty") or 0.0) < 0.16
    ]
    if len(non_conflicting) >= max(3, min_keep):
        return non_conflicting
    return adjusted

# ---- 用户偏好缓存（启动时加载一次，避免每次请求都查 Neo4j） ----
_user_pref_cache: dict = {}  # {user_id: structured hot-path preference sets}


def _as_pref_set(values) -> set:
    if isinstance(values, str):
        values = values.replace("，", ",").replace("/", ",").split(",")
    return {str(x).strip().lower() for x in (values or []) if str(x or "").strip()}

def _load_user_preferences(user_id: str = GRAPH_AFFINITY_USER_ID) -> dict:
    """
    从 Neo4j 加载用户偏好并缓存。
    首次调用时查询数据库，后续直接返回缓存。
    """
    if user_id in _user_pref_cache:
        return _user_pref_cache[user_id]

    empty_prefs = {
        "genres": set(),
        "moods": set(),
        "themes": set(),
        "scenarios": set(),
        "avoid_genres": set(),
        "avoid_moods": set(),
        "avoid_scenarios": set(),
        "expanded_genres": set(),
        "expanded_avoid_genres": set(),
        "activity_contexts": set(),
    }

    try:
        prefs = dict(empty_prefs)  # copy
        try:
            from services.memory_gateway import get_memory_gateway

            profile = get_memory_gateway().get_user_profile(user_id)
        except Exception as gateway_error:
            logger.warning("[PrefCache] MemoryGateway 读取失败，偏好降级为空: %s", gateway_error)
            profile = {}

        prefs["genres"] = (
            _as_pref_set(profile.get("preferred_genres"))
            | _as_pref_set(profile.get("favorite_genres"))
            | _as_pref_set(profile.get("preferred_genres_explicit"))
        )
        prefs["moods"] = (
            _as_pref_set(profile.get("preferred_moods"))
            | _as_pref_set(profile.get("favorite_moods"))
            | _as_pref_set(profile.get("add_moods"))
            | _as_pref_set(profile.get("mood_tendency"))
        )
        prefs["themes"] = _as_pref_set(profile.get("favorite_themes"))
        prefs["scenarios"] = (
            _as_pref_set(profile.get("preferred_scenarios"))
            | _as_pref_set(profile.get("favorite_scenarios"))
            | _as_pref_set(profile.get("add_scenarios"))
            | _as_pref_set(profile.get("activity_contexts"))
        )
        prefs["avoid_genres"] = _as_pref_set(profile.get("avoid_genres"))
        prefs["avoid_moods"] = _as_pref_set(profile.get("avoid_moods"))
        prefs["avoid_scenarios"] = _as_pref_set(profile.get("avoid_scenarios"))
        prefs["activity_contexts"] = _as_pref_set(profile.get("activity_contexts"))

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

        expanded_avoid: set = set()
        for pref in prefs["avoid_genres"]:
            for key, aliases in GENRE_TAG_MAP.items():
                if key.lower() == pref or pref in key.lower():
                    expanded_avoid.update(a.lower() for a in aliases)
                    break
            else:
                expanded_avoid.add(pref)
        prefs["expanded_avoid_genres"] = expanded_avoid

        _user_pref_cache[user_id] = prefs
        logger.info(
            f"[PrefCache] 用户偏好已缓存: genre={len(prefs['genres'])}, mood={len(prefs['moods'])}, "
            f"theme={len(prefs['themes'])}, scenario={len(prefs['scenarios'])}, "
            f"avoid_genre={len(prefs['avoid_genres'])}, avoid_mood={len(prefs['avoid_moods'])}"
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
        self._disliked_cache_by_user: Dict[str, set[str]] = {}

    def _get_disliked_titles(self, user_id: str = GRAPH_AFFINITY_USER_ID) -> set:
        """查询用户 DISLIKES 的歌曲标题集合（同一实例内缓存）"""
        if user_id in self._disliked_cache_by_user:
            return self._disliked_cache_by_user[user_id]
        try:
            from retrieval.neo4j_client import get_neo4j_client
            client = get_neo4j_client()
            query = """
            MATCH (u:User {id: $uid})-[:DISLIKES]->(s:Song)
            RETURN collect(s.title) AS titles
            """
            result = client.execute_query(query, {"uid": user_id})
            titles = set(result[0]["titles"]) if result and result[0].get("titles") else set()
            self._disliked_cache_by_user[user_id] = titles
            if titles:
                logger.info("[DislikeFilter] 用户 %s 加载到 %d 首不喜欢的歌", user_id, len(titles))
            return titles
        except Exception as e:
            logger.warning(f"[DislikeFilter] 查询失败: {e}")
            self._disliked_cache_by_user[user_id] = set()
            return self._disliked_cache_by_user[user_id]

    async def retrieve(self, query: str, limit: int = 5, precomputed_plan: dict = None) -> ToolOutput:
        """
        主检索入口：内容三路并行召回 → RRF → 硬过滤 → 统一排序。

        Args:
            query: 用户查询
            limit: 返回结果数量
            precomputed_plan: 来自上游统一 Prompt 的预计算检索计划（dict 格式的 RetrievalPlan）。
                              layered 字段优先；legacy 字段仅用于历史调用兼容。
        """
        retrieval_started = time.perf_counter()
        timings: Dict[str, float] = {}
        logger.info(f"[Retrieval] 开始处理请求: {query}")
        self._skip_tri_anchor_for_entity_graph = False
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
        tool_plan = dict(plan.get("_tool_plan") or {})
        planned_tools = {
            str(call.get("name") or "")
            for call in (tool_plan.get("tool_calls") or [])
            if isinstance(call, dict)
        }
        run_graph_tool, run_audio_tool, tool_plan_active = local_recall_tools_from_plan(
            plan,
            execution_enabled=_s.tool_plan_execution_enabled,
        )
        hard_constraints = dict(plan.get("hard_constraints") or {})
        soft_intent = dict(plan.get("soft_intent") or {})
        hints = dict(plan.get("hints") or {})

        if not hard_constraints:
            legacy_language = plan.get("graph_language_filter")
            legacy_instrumental = str(legacy_language or "").strip().casefold() in {
                "instrumental",
                "纯音乐",
                "器乐",
            }
            hard_constraints = {
                "artist_entities": list(plan.get("graph_artist_entities") or []),
                "song_entities": list(plan.get("graph_song_entities") or []),
                "language": None if legacy_instrumental else legacy_language,
                "region": plan.get("graph_region_filter"),
                "instrumental": legacy_instrumental,
            }
        elif str(hard_constraints.get("language") or "").strip().casefold() in {"instrumental", "纯音乐", "器乐"}:
            hard_constraints["language"] = None
            hard_constraints["instrumental"] = True
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
        user_id = str(plan.get("_user_id") or GRAPH_AFFINITY_USER_ID)
        recall_weights = recall_weights_for_intent(
            intent_type,
            query=query,
            hard_constraints=hard_constraints,
            soft_intent=soft_intent,
            hints=hints,
        )
        try:
            from services.ranking_policy import apply_multipliers, runtime_policy_for_user

            runtime_policy = runtime_policy_for_user(user_id)
            if runtime_policy:
                recall_weights = apply_multipliers(
                    recall_weights,
                    runtime_policy.get("rrf_multipliers"),
                    normalise=False,
                )
                logger.info("[A3] 使用已验证反馈策略调整三路 RRF 权重: %s", recall_weights)
        except Exception as policy_error:
            logger.debug("[A3] 反馈策略不可用，保留默认 RRF 权重: %s", policy_error)
        need_web_search = bool(plan.get("use_web_search"))
        search_keyword = str(plan.get("web_search_keywords") or query)

        # ── 双路线：联网补充与本地召回并行 ──
        # 第二路线由 planner 模型的原生联网搜索（DashScope enable_search）发现
        # 有证据支撑的补充歌曲，超时/失败 fail-soft 为空，不影响本地链路。
        web_supplement_task: "asyncio.Task | None" = None
        if not tool_plan_active:
            from retrieval.web_supplement import get_web_supplement, supplement_enabled

            if supplement_enabled():
                web_supplement_task = asyncio.create_task(
                    get_web_supplement().discover(
                        query=query,
                        plan_summary={
                            "intent_type": intent_type,
                            "soft_intent": soft_intent,
                            "hints": hints,
                            "hard_constraints": hard_constraints,
                        },
                        avoid=list(soft_intent.get("avoid") or []),
                    )
                )

        vector_descs: List[str] = []
        for item in plan.get("vector_acoustic_queries") or []:
            text = str(item or "").strip()
            if text and text not in vector_descs:
                vector_descs.append(text)
        vector_desc = str(plan.get("vector_acoustic_query") or "").strip()
        if vector_desc and vector_desc not in vector_descs:
            vector_descs.insert(0, vector_desc)
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
        if not vector_descs:
            vector_descs = [vector_desc]

        filter_hard_constraints = dict(hard_constraints)
        reference_song_entities = []
        uses_song_as_acoustic_seed = bool(
            graph_song_entities
            and str(intent_type or "") in {"hybrid_search", "vector_search"}
            and (
                soft_intent.get("goal")
                or soft_intent.get("trajectory")
                or soft_intent.get("vibe")
                or vector_desc != query
            )
        )
        if uses_song_as_acoustic_seed:
            reference_song_entities = list(graph_song_entities)
            filter_hard_constraints["song_entities"] = []
            logger.info(
                "[Retrieval] song_entities=%s 由 LLM plan 判定为声学参考种子，不进入最终硬过滤",
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

        def _count_recall_rows(raw: str) -> int:
            try:
                return len(json.loads(raw)) if raw else 0
            except (json.JSONDecodeError, TypeError):
                return 0

        # 2. 内容召回：场景/听感类保持 graph+dense 并行；明确实体类先跑图谱。
        # 如果歌手/歌名硬约束已经由图谱给出足够候选，就跳过全库 dense，
        # 避免 MuQ 冷启动/文本编码拖慢精确实体查询。
        source_raw: Dict[str, str] = {}
        entity_constrained = bool(graph_artist_entities or (graph_song_entities and not uses_song_as_acoustic_seed))
        enough_entity_graph = False
        if tool_plan_active:
            if run_graph_tool and not run_audio_tool:
                self._skip_tri_anchor_for_entity_graph = True
            recall_tasks = {}
            if run_graph_tool:
                recall_tasks["graph"] = timed_recall("graph", run_sync_in_executor(
                    graph_candidate_recall,
                    hard_constraints,
                    hints,
                    limit=recall_limit,
                ))
            if run_audio_tool:
                recall_tasks["dense"] = timed_recall("dense", run_sync_in_executor(
                    semantic_search.invoke,
                    {"query": vector_desc, "query_variants": vector_descs, "limit": recall_limit},
                ))
            recall_results = await asyncio.gather(*recall_tasks.values(), return_exceptions=True)
            for source, result_value in zip(recall_tasks, recall_results):
                if isinstance(result_value, Exception):
                    logger.error("[ToolPlan:%s] 异常: %s: %s", source, type(result_value).__name__, result_value)
                    source_raw[source] = ""
                else:
                    source_raw[source] = result_value or ""
                    logger.info("[ToolPlan:%s] 返回 %d 条", source, _count_recall_rows(source_raw[source]))
            logger.info("[ToolPlan] active local tools=%s", sorted(planned_tools))
        elif entity_constrained:
            graph_result = await timed_recall("graph", run_sync_in_executor(
                graph_candidate_recall,
                hard_constraints,
                hints,
                limit=recall_limit,
            ))
            if isinstance(graph_result, Exception):
                logger.error("[Recall:graph] 异常: %s: %s", type(graph_result).__name__, graph_result)
                source_raw["graph"] = ""
            else:
                source_raw["graph"] = graph_result or ""
            graph_count = _count_recall_rows(source_raw["graph"])
            logger.info("[Recall:graph] 返回 %d 条", graph_count)
            enough_entity_graph = graph_count >= 3

        if not tool_plan_active and entity_constrained and enough_entity_graph:
            source_raw["dense"] = ""
            timings["recall_dense_ms"] = 0.0
            self._skip_tri_anchor_for_entity_graph = True
            logger.info("[Recall:dense] 明确实体图谱候选已足够，跳过全库 dense 召回")
        elif not tool_plan_active:
            recall_tasks = {}
            if "graph" not in source_raw:
                recall_tasks["graph"] = timed_recall("graph", run_sync_in_executor(
                    graph_candidate_recall,
                    hard_constraints,
                    hints,
                    limit=recall_limit,
                ))
            recall_tasks["dense"] = timed_recall("dense", run_sync_in_executor(
                semantic_search.invoke,
                {"query": vector_desc, "query_variants": vector_descs, "limit": recall_limit},
            ))
            recall_results = await asyncio.gather(*recall_tasks.values(), return_exceptions=True)
            for source, result in zip(recall_tasks, recall_results):
                if isinstance(result, Exception):
                    logger.error("[Recall:%s] 异常: %s: %s", source, type(result).__name__, result)
                    source_raw[source] = ""
                else:
                    source_raw[source] = result or ""
                    logger.info("[Recall:%s] 返回 %d 条", source, _count_recall_rows(source_raw[source]))

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

        # 3. 联网歌曲来自并行的补充路线（LLM 原生联网 + 证据），
        #    旧的 federated+extract 路径仅在补充路线关闭时兜底。
        web_raw = ""
        web_started = time.perf_counter()
        if web_supplement_task is not None:
            web_playable = await web_supplement_task
            if web_playable:
                logger.info("[WebSupplement] 联网补充 %d 首（证据驱动）", len(web_playable))
        else:
            if not tool_plan_active and os.environ.get("MUSIC_WEB_SEARCH_ENABLED", "1") != "0":
                graph_empty = source_raw.get("graph") in ("", "[]")
                if need_web_search:
                    logger.info("⚡ 意图明确要求联网: '%s'", search_keyword)
                    web_raw = await _federated_search_async(search_keyword)
                elif graph_entities and graph_empty:
                    logger.warning("本地实体召回为空，触发联网补充: '%s'", query)
                    web_raw = await _federated_search_async(query)
            web_playable = await _extract_and_fetch_web_songs(web_raw)
        timings["retrieval_web_ms"] = round((time.perf_counter() - web_started) * 1000, 3)
        self._current_query = query
        self._current_hyde_text = vector_desc
        self._current_hard_constraints = filter_hard_constraints
        self._current_reference_song_entities = reference_song_entities
        self._current_soft_intent = soft_intent

        result = self._format_results(
            source_raw=source_raw,
            recall_weights=recall_weights,
            hard_constraints=filter_hard_constraints,
            soft_intent=soft_intent,
            hints=hints,
            web_res=web_raw,
            web_playable=web_playable,
            graph_entities=graph_entities,
            final_limit=limit,
            timings=timings,
            user_id=user_id,
        )
        result.metadata.setdefault("timings", timings)
        if tool_plan:
            tool_observations = []
            source_mapping = {"search_graph": "graph", "search_audio": "dense"}
            for call in tool_plan.get("tool_calls") or []:
                if not isinstance(call, dict):
                    continue
                tool_name = str(call.get("name") or "")
                source = source_mapping.get(tool_name)
                if not source:
                    continue
                count = _count_recall_rows(source_raw.get(source, ""))
                tool_observations.append({
                    "call_id": str(call.get("id") or tool_name),
                    "tool_name": tool_name,
                    "success": source in source_raw,
                    "status": "success" if count else "empty",
                    "data": {"candidate_count": count, "source": source},
                    "error": "",
                    "duration_ms": float(timings.get(f"recall_{source}_ms") or 0.0),
                    "metadata": {"shadow": not tool_plan_active},
                })
            result.metadata["tool_observations"] = tool_observations
            result.metadata["tool_plan_active"] = tool_plan_active
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
            "[RRF] 内容召回融合完成: %d 首，多路交叉命中=%d",
            len(merged),
            multi_source_count,
        )
        return merged

    # ================================================================
    # 内容双锚精排（语义 + 声学）；个性化只在限幅 post-recall 层生效
    # ================================================================
    def _tri_anchor_rerank(
        self,
        candidates: List[dict],
        query_text: str,
        user_id: str = GRAPH_AFFINITY_USER_ID,
    ) -> List[dict]:
        """
        内容双锚精排 = 语义相关性 + 声学风格

        所有分数归一化到 [0, 1] 后加权融合：
        final_score = α×semantic + β×acoustic

        - semantic:   主文搜音锚 cosine(song_emb, query_text_emb) → (x+1)/2
        - acoustic:   OMAR-RQ 仅在有参考种子歌时对齐种子，否则保持中性

        个性化、新鲜度、长尾和过曝只通过 +/-0.08 的召回后校正层影响
        已召回候选，避免用户画像压过当前查询的内容相关性。
        """
        if not candidates:
            return candidates

        from config.settings import settings as _s
        w_sem = _s.tri_anchor_w_semantic
        w_aco = _s.tri_anchor_w_acoustic
        try:
            from services.ranking_policy import apply_multipliers, runtime_policy_for_user

            runtime_policy = runtime_policy_for_user(user_id)
            if runtime_policy:
                content_weights = apply_multipliers(
                    {"semantic": w_sem, "acoustic": w_aco},
                    runtime_policy.get("content_anchor_multipliers"),
                    normalise=True,
                )
                w_sem = content_weights["semantic"]
                w_aco = content_weights["acoustic"]
                logger.info(
                    "[ContentAnchor] 使用已验证反馈策略: sem=%.2f, aco=%.2f",
                    w_sem,
                    w_aco,
                )
        except Exception as e:
            logger.debug("[ContentAnchor] 反馈策略不可用，使用配置权重: %s", e)

        # 自动归一化权重
        w_total = w_sem + w_aco
        if w_total > 0:
            w_sem, w_aco = w_sem / w_total, w_aco / w_total
        else:
            w_sem, w_aco = 0.6, 0.4

        try:
            import numpy as np
            from retrieval.neo4j_client import get_neo4j_client

            neo4j = get_neo4j_client()
            if not neo4j or not neo4j.driver:
                logger.warning("[TriAnchor] Neo4j 不可用，跳过三锚精排")
                return candidates

            configured_backend = str(getattr(_s, "dense_text_audio_backend", "muq") or "muq").lower()
            semantic_backend = "muq" if configured_backend in {"muq", "both"} else "m2d"

            def _encode_query(backend: str):
                if backend == "muq":
                    from retrieval.muq_embedder import encode_text_to_muq
                    from retrieval.alignment_calibration import apply_alignment_calibration

                    return apply_alignment_calibration(encode_text_to_muq(query_text), backend)
                from retrieval.audio_embedder import encode_text_to_embedding
                from retrieval.alignment_calibration import apply_alignment_calibration

                return apply_alignment_calibration(encode_text_to_embedding(query_text), backend)

            # ── 语义锚：query → text embedding ──
            # 首次启动时 HuggingFace 文本编码器可能仍在下载，不能让精排阻塞整条推荐链。
            tri_text_timeout = float(os.getenv("TRI_ANCHOR_TEXT_TIMEOUT_SECONDS", "8"))
            query_emb = None
            executor = None
            try:
                logger.info("[TriAnchor] 编码 query text embedding backend=%s...", semantic_backend)
                executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
                future = executor.submit(_encode_query, semantic_backend)
                query_emb = np.array(future.result(timeout=tri_text_timeout))
            except concurrent.futures.TimeoutError:
                logger.warning(
                    "[TriAnchor] query text embedding 超时 %.1fs，跳过语义锚，保留声学/个性化排序",
                    tri_text_timeout,
                )
            except Exception as encode_error:
                if semantic_backend == "muq":
                    logger.warning("[TriAnchor] MuQ 精排编码失败，尝试 M2D fallback: %s", encode_error)
                    try:
                        future = executor.submit(_encode_query, "m2d") if executor else None
                        query_emb = np.array(future.result(timeout=tri_text_timeout)) if future else None
                        semantic_backend = "m2d"
                    except Exception as fallback_error:
                        logger.warning("[TriAnchor] M2D fallback 也失败，跳过语义锚: %s", fallback_error)
                else:
                    logger.warning("[TriAnchor] query text embedding 失败，跳过语义锚: %s", encode_error)
            finally:
                if executor is not None:
                    executor.shutdown(wait=False, cancel_futures=True)

            # ── 批量获取候选歌曲的 MuQ/M2D + OMAR embedding ──
            titles = [c["song"]["title"] for c in candidates if c.get("song", {}).get("title")]
            emb_cypher = """
            UNWIND $titles AS t
            MATCH (s:Song {title: t})
            RETURN s.title AS title,
                   s.muq_embedding AS muq_emb,
                   s.m2d2_embedding AS m2d_emb,
                   s.omar_embedding AS omar_emb
            """
            emb_rows = neo4j.execute_query(emb_cypher, {"titles": titles})

            muq_map, m2d_map, omar_map = {}, {}, {}
            for row in (emb_rows or []):
                t = row.get("title", "")
                if row.get("muq_emb"):
                    muq_map[t] = np.array(row["muq_emb"])
                if row.get("m2d_emb"):
                    m2d_map[t] = np.array(row["m2d_emb"])
                if row.get("omar_emb"):
                    omar_map[t] = np.array(row["omar_emb"])

            logger.info(
                f"[TriAnchor] embedding 命中: MuQ={len(muq_map)}/{len(titles)}, M2D={len(m2d_map)}/{len(titles)}, "
                f"OMAR={len(omar_map)}/{len(titles)}"
            )

            # ── 声学锚：只在有明确相似种子歌时使用种子 OMAR，不再用候选集质心 ──
            acoustic_anchor = None
            reference_songs = list(getattr(self, "_current_reference_song_entities", []) or [])
            if reference_songs:
                seed_rows = neo4j.execute_query(
                    """
                    UNWIND $titles AS t
                    MATCH (s:Song {title: t})
                    WHERE s.omar_embedding IS NOT NULL
                    RETURN s.omar_embedding AS omar_emb
                    """,
                    {"titles": reference_songs},
                )
                seed_vectors = [np.array(row["omar_emb"]) for row in (seed_rows or []) if row.get("omar_emb")]
                if seed_vectors:
                    acoustic_anchor = np.mean(seed_vectors, axis=0)
                    logger.info("[TriAnchor] OMAR 声学锚使用参考种子歌: %s", reference_songs)

            def _cosine(a, b):
                dot = np.dot(a, b)
                norm = np.linalg.norm(a) * np.linalg.norm(b)
                return float(dot / norm) if norm > 0 else 0.0

            def _normalize_cosine(score: float) -> float:
                """cosine [-1, 1] → [0, 1]"""
                return (score + 1.0) / 2.0

            # ── 内容双锚融合评分 ──
            for c in candidates:
                title = c.get("song", {}).get("title", "")

                # 维度 1: 语义分（归一化到 [0,1]）
                semantic_emb = muq_map.get(title) if semantic_backend == "muq" else m2d_map.get(title)
                if query_emb is not None and semantic_emb is not None:
                    raw_semantic = _cosine(semantic_emb, query_emb)
                    semantic = _normalize_cosine(raw_semantic)
                else:
                    semantic = 0.5  # 无 embedding 时给中位分

                # 维度 2: 声学分（归一化到 [0,1]）
                omar_emb = omar_map.get(title)
                if omar_emb is not None and acoustic_anchor is not None:
                    raw_acoustic = _cosine(omar_emb, acoustic_anchor)
                    acoustic = _normalize_cosine(raw_acoustic)
                else:
                    acoustic = 0.5

                # 没有可靠声学锚时只使用语义锚；个性化不在这里重复计分。
                if omar_emb is None or acoustic_anchor is None:
                    final = semantic
                else:
                    final = w_sem * semantic + w_aco * acoustic

                c["similarity_score"] = round(final, 6)
                c["_semantic_score"] = round(semantic, 4)
                c["_acoustic_score"] = round(acoustic, 4)
                c["_semantic_backend"] = semantic_backend
                c["_personal_score"] = round(float(c.get("_post_personal_score", 0.5)), 4)

            candidates.sort(key=lambda x: x["similarity_score"], reverse=True)

            logger.info(
                f"[ContentAnchor] 双锚精排完成 (sem={w_sem:.2f}, aco={w_aco:.2f}) | "
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
    def _fetch_post_recall_metadata(
        candidates: List[dict],
        *,
        user_id: str = GRAPH_AFFINITY_USER_ID,
    ) -> Dict[str, dict]:
        """Fetch score-adjustment metadata for already-recalled candidates."""
        titles = [
            item.get("song", {}).get("title", "")
            for item in candidates
            if item.get("song", {}).get("title")
        ]
        if not titles:
            return {}
        try:
            from retrieval.neo4j_client import get_neo4j_client
            neo4j = get_neo4j_client()
            if not neo4j or not neo4j.driver:
                return {}
            query = """
            UNWIND $titles AS t
            MATCH (s:Song {title: t})
            OPTIONAL MATCH (u:User {id: $user_id})-[e:EXPOSED]->(s)
            WITH s, e, properties(s) AS props
            RETURN s.title AS title,
                   coalesce(s.updated_at, 0) AS updated_at,
                   coalesce(e.ts_alpha, 1) AS ts_alpha,
                   coalesce(e.ts_beta, 1) AS ts_beta,
                   coalesce(e.last_exposed_at, 0) AS ts_last_exposed_at,
                   props['acoustic_vocalness'] AS acoustic_vocalness,
                   props['acoustic_drumness'] AS acoustic_drumness,
                   props['acoustic_energy'] AS acoustic_energy,
                   props['acoustic_probe_version'] AS acoustic_probe_version
            """
            rows = neo4j.execute_query(
                query,
                {"titles": titles, "user_id": user_id},
            ) or []
            return {
                str(row.get("title") or ""): {
                    "updated_at": row.get("updated_at", 0),
                    "ts_alpha": row.get("ts_alpha", 1),
                    "ts_beta": row.get("ts_beta", 1),
                    "ts_last_exposed_at": row.get("ts_last_exposed_at", 0),
                    "acoustic_vocalness": row.get("acoustic_vocalness"),
                    "acoustic_drumness": row.get("acoustic_drumness"),
                    "acoustic_energy": row.get("acoustic_energy"),
                    "acoustic_probe_version": row.get("acoustic_probe_version"),
                }
                for row in rows
                if row.get("title")
            }
        except Exception as e:
            logger.warning(f"[PostRecallAdjust] 元数据读取失败（降级为中性加权）: {e}")
            return {}

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
        avoid_genre_prefs = user_prefs.get("expanded_avoid_genres", set())
        avoid_mood_prefs = user_prefs.get("avoid_moods", set())
        avoid_scenario_prefs = user_prefs.get("avoid_scenarios", set())
        has_any_pref = (
            user_pref_genres
            or user_pref_moods
            or user_pref_themes
            or user_pref_scenarios
            or avoid_genre_prefs
            or avoid_mood_prefs
            or avoid_scenario_prefs
        )

        # ── Step B: 合并查询（图距离 + 候选歌曲标签，1 次 Neo4j round-trip） ──
        combined_query = """
        MATCH (u:User {id: $user_id})
        UNWIND $titles AS candidate_title
        OPTIONAL MATCH (s:Song)
          WHERE s.title = candidate_title
        OPTIONAL MATCH path = shortestPath(
          (u)-[*1..""" + str(max_hops) + """]->(s)
        )
        OPTIONAL MATCH (s)-[:BELONGS_TO_GENRE]->(g:Genre)
        OPTIONAL MATCH (s)-[:HAS_MOOD]->(m:Mood)
        OPTIONAL MATCH (s)-[:HAS_THEME]->(th:Theme)
        OPTIONAL MATCH (s)-[:FITS_SCENARIO]->(sc:Scenario)
        RETURN candidate_title AS title,
               CASE WHEN path IS NOT NULL THEN length(path) ELSE -1 END AS distance,
               collect(DISTINCT g.name) AS genres,
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
                    "genres": {x.strip().lower() for x in (r.get("genres") or []) if x and x.strip()},
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
            AVOID_PENALTY_WEIGHT = 0.38
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
                j_avoid_genre = _jaccard(avoid_genre_prefs, cand_genre_tags)
                j_avoid_mood = _jaccard(avoid_mood_prefs, cand_tags.get("moods", set()))
                j_avoid_scenario = _jaccard(avoid_scenario_prefs, cand_tags.get("scenarios", set()))

                weighted_jaccard = (
                    DIM_WEIGHTS["genre"] * j_genre
                    + DIM_WEIGHTS["mood"] * j_mood
                    + DIM_WEIGHTS["theme"] * j_theme
                    + DIM_WEIGHTS["scenario"] * j_scenario
                )
                avoid_jaccard = (
                    DIM_WEIGHTS["genre"] * j_avoid_genre
                    + DIM_WEIGHTS["mood"] * j_avoid_mood
                    + DIM_WEIGHTS["scenario"] * j_avoid_scenario
                )
                boost = PREF_BOOST_WEIGHT * weighted_jaccard
                avoid_penalty = AVOID_PENALTY_WEIGHT * avoid_jaccard
                c["_graph_affinity"] += boost - avoid_penalty
                c["_pref_boost"] = round(boost, 4)
                c["_pref_avoid_penalty"] = round(avoid_penalty, 4)
                c["_pref_detail"] = {
                    "genre": round(j_genre, 3), "mood": round(j_mood, 3),
                    "theme": round(j_theme, 3), "scenario": round(j_scenario, 3),
                    "avoid_genre": round(j_avoid_genre, 3),
                    "avoid_mood": round(j_avoid_mood, 3),
                    "avoid_scenario": round(j_avoid_scenario, 3),
                }
                if boost > 0 or avoid_penalty > 0:
                    pref_hits += 1

            logger.info(
                f"[GraphAffinity] 偏好加分(缓存+Jaccard): {pref_hits}/{len(candidates)} 首命中 | "
                f"用户偏好维度: genre={len(user_pref_genres)}, mood={len(user_pref_moods)}, "
                f"theme={len(user_pref_themes)}, scenario={len(user_pref_scenarios)}, "
                f"avoid_genre={len(avoid_genre_prefs)}, avoid_mood={len(avoid_mood_prefs)}"
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
        soft_intent: Dict[str, Any] | None = None,
        hints: Dict[str, Any] | None = None,
        web_res: str = "",
        web_playable: List[dict] = None,
        graph_entities: List[str] = None,
        final_limit: int = 15,
        timings: Dict[str, float] = None,
        user_id: str = GRAPH_AFFINITY_USER_ID,
    ) -> ToolOutput:
        """
        合并各召回源并执行统一过滤与排序。

        排序管线（R1）:
          1. 解析内容召回结果并保留各路排名
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
        disliked_titles = self._get_disliked_titles(user_id)
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
            from retrieval.web_supplement import is_duplicate_song

            filtered_web = apply_hard_filters(
                [
                    {
                        "song": item.get("song") or {},
                        "reason": item.get("reason") or "🌐 全网最新发掘",
                        "similarity_score": item.get("similarity_score", 0.0),
                    }
                    for item in web_playable
                ],
                hard_constraints,
                disliked_titles,
            )
            # 模糊去重：归一化歌名/歌手 + 相似度，防止同一首歌因
            # 括号后缀、feat、大小写、全半角差异重复出现。
            existing_pairs = [
                (
                    item.get("song", {}).get("title", ""),
                    item.get("song", {}).get("artist", ""),
                )
                for item in final_list
            ]
            for item in filtered_web:
                song = item.get("song", {})
                item["recall_sources"] = ["web"]
                item["recall_source_labels"] = ["联网"]
                song["recall_sources"] = ["web"]
                song["recall_source_labels"] = ["联网"]
                title, artist = song.get("title", ""), song.get("artist", "")
                if is_duplicate_song(title, artist, existing_pairs):
                    logger.info("[WebSupplement] 去重跳过: %s - %s", title, artist)
                    continue
                final_list.append(item)
                existing_pairs.append((title, artist))

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

        # ---- Step 4: 召回后加分/减分 + 粗排 + Thompson Sampling 探索槽 ----
        cand_tag_map = {}
        post_metadata_by_title = {}
        if _settings.graph_affinity_enabled and final_list:
            total_before = len(final_list)
            final_list, cand_tag_map = self._compute_graph_affinity(final_list, user_id=user_id)
            post_metadata_by_title = self._fetch_post_recall_metadata(
                final_list,
                user_id=user_id,
            )
            final_list = apply_post_recall_adjustments(
                final_list,
                metadata_by_title=post_metadata_by_title,
                query_text=getattr(self, "_current_query", "") or "",
                soft_intent=soft_intent or getattr(self, "_current_soft_intent", {}) or {},
                hints=hints or {},
                score_field="_rrf_score",
                output_score_field="_post_coarse_score",
                config=_post_recall_config_for_user(user_id),
                enable_acoustic_probe=_settings.acoustic_probe_ranking_enabled,
            )

            # ── Phase A: 内容 RRF 分 + 召回后小幅加减分（粗排）──
            final_list.sort(key=lambda x: x.get("_post_coarse_score", 0), reverse=True)

            coarse_cut = max(int(total_before * _settings.coarse_cut_ratio), 10)
            coarse_cut = min(coarse_cut, len(final_list))  # 不能超过实际数量

            main_candidates = final_list[:coarse_cut]
            tail_candidates = final_list[coarse_cut:]

            # ── Phase B: Thompson Sampling 探索槽 ──
            # 从尾部捞回冷门/新歌，同时压制近期过曝歌曲。
            import random
            fallback_rng = random.Random(0) if _settings.eval_disable_side_effects else random
            n_explore = max(int(coarse_cut * _settings.exploration_ratio), 1)

            explore_picks = []
            if tail_candidates:
                try:
                    import numpy as np
                    ts_rng = np.random.default_rng(0 if _settings.eval_disable_side_effects else None)

                    # TS 采样：为尾部每首歌采样一个分数
                    ts_scores = []
                    for c in tail_candidates:
                        alpha = max(float(c.get("_post_ts_alpha", 1.0)), 0.1)
                        beta = max(1.0 + float(c.get("_post_effective_exposure", 0.0)), 0.1)
                        ts_sample = float(ts_rng.beta(alpha, beta))
                        ts_score = (
                            ts_sample
                            + 0.12 * float(c.get("_post_freshness_score", 0.0))
                            + 0.08 * float(c.get("_post_longtail_score", 0.0))
                            - 0.10 * float(c.get("_post_exposure_penalty", 0.0))
                        )
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
                    fallback_rng.shuffle(tail_candidates)
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
        if rerank_query and final_list and not getattr(self, "_skip_tri_anchor_for_entity_graph", False):
            final_list = self._tri_anchor_rerank(final_list, rerank_query, user_id=user_id)
        elif final_list and getattr(self, "_skip_tri_anchor_for_entity_graph", False):
            logger.info("[ContentAnchor] 明确实体图谱候选已足够，跳过 tri-anchor 文本精排")

        if cand_tag_map:
            for item in final_list:
                song = item.get("song") or {}
                title = song.get("title", "")
                tag_entry = cand_tag_map.get(title) or {}
                for song_field, tag_key in (
                    ("genres", "genres"),
                    ("moods", "moods"),
                    ("themes", "themes"),
                    ("scenarios", "scenarios"),
                ):
                    incoming = {str(tag) for tag in (tag_entry.get(tag_key) or set()) if tag}
                    if not incoming:
                        continue
                    existing = set(_iter_terms(song.get(song_field)))
                    song[song_field] = sorted(existing | incoming)
                if not song.get("genre"):
                    display_parts = (
                        _iter_terms(song.get("genres"))[:2]
                        + _iter_terms(song.get("moods"))[:1]
                        + _iter_terms(song.get("scenarios"))[:1]
                    )
                    if display_parts:
                        song["genre"] = "/".join(display_parts)

        before_soft = len(final_list)
        final_list = rerank_with_soft_constraints(
            final_list,
            soft_intent or getattr(self, "_current_soft_intent", {}) or {},
            hints or {},
            " ".join(
                part
                for part in (
                    getattr(self, "_current_query", "") or "",
                    getattr(self, "_current_hyde_text", "") or "",
                )
                if part
            ),
            min_keep=3,
        )
        if len(final_list) != before_soft:
            logger.info(
                "[SoftAvoidRank] soft_intent.avoid 保守降权/裁剪: %d → %d",
                before_soft,
                len(final_list),
            )

        # ---- Step 5.5: Cross-Encoder 精排（可选，默认关闭）----
        if _settings.reranker_enabled and final_list:
            try:
                from retrieval.cross_encoder_reranker import CrossEncoderReranker
                reranker = CrossEncoderReranker()
                rerank_query = getattr(self, "_current_query", "") or "music recommendation"
                final_list = reranker.rerank(rerank_query, final_list)
            except Exception as e:
                logger.warning(f"[Reranker] Cross-Encoder 精排异常（降级跳过）: {e}")

        if final_list:
            if not post_metadata_by_title:
                post_metadata_by_title = self._fetch_post_recall_metadata(
                    final_list,
                    user_id=user_id,
                )
            final_list = apply_post_recall_adjustments(
                final_list,
                metadata_by_title=post_metadata_by_title,
                query_text=" ".join(
                    part
                    for part in (
                        getattr(self, "_current_query", "") or "",
                        getattr(self, "_current_hyde_text", "") or "",
                    )
                    if part
                ),
                soft_intent=soft_intent or getattr(self, "_current_soft_intent", {}) or {},
                hints=hints or {},
                score_field="similarity_score",
                output_score_field="_post_final_score",
                apply_to_similarity=True,
                config=_post_recall_config_for_user(user_id),
                enable_acoustic_probe=_settings.acoustic_probe_ranking_enabled,
            )
            final_list.sort(key=lambda x: x.get("similarity_score", 0), reverse=True)
            logger.info(
                "[PostRecallAdjust] 完成最终加减分: Top3=%s",
                [
                    (
                        item.get("song", {}).get("title", ""),
                        round(item.get("similarity_score", 0), 4),
                        item.get("_post_recall_delta", 0),
                    )
                    for item in final_list[:3]
                ],
            )

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
                    avoid_conflict = float(cand.get("_post_semantic_conflict_score") or 0.0)
                    cand_tags = tag_cache[id(cand)]

                    # 与已选集合中 Jaccard 最大的作为重叠度
                    max_overlap = 0.0
                    if cand_tags:
                        for sel_tags in selected_tag_sets:
                            j = _jaccard(cand_tags, sel_tags)
                            if j > max_overlap:
                                max_overlap = j

                    # MMR should diversify within the user's current intent, not
                    # re-promote candidates that the LLM plan explicitly marked
                    # as avoid/conflict.  The conflict score is derived only
                    # from soft_intent.avoid vs catalog tags in post-recall
                    # adjustment, not from fixed query phrase triggers.
                    mmr_score = (
                        mmr_lambda * relevance
                        - (1 - mmr_lambda) * max_overlap
                        - 0.18 * avoid_conflict
                    )
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

        # ---- Step 8: 异步更新按用户隔离的曝光疲劳参数 ----
        # 离线评测必须只读；真实请求只更新 (User)-[:EXPOSED]->(Song)。
        recommended_songs = []
        for item in final_list:
            song = item.get("song", {})
            title = str(song.get("title") or "").strip()
            if title and title != "🌐 全网资讯补充":
                recommended_songs.append(
                    {
                        "title": title,
                        "artist": str(song.get("artist") or "").strip(),
                    }
                )
        if recommended_songs and not _settings.eval_disable_side_effects:
            try:
                import asyncio
                async def _update_ts_exposure(songs):
                    try:
                        from retrieval.neo4j_client import get_neo4j_client
                        neo4j = get_neo4j_client()
                        if neo4j and neo4j.driver:
                            neo4j.execute_query(
                                USER_EXPOSURE_UPDATE_QUERY,
                                {"songs": songs, "user_id": user_id},
                            )
                            logger.info(
                                "[Exposure] 用户 %s 曝光疲劳更新: %d 首歌",
                                user_id,
                                len(songs),
                            )
                    except Exception as e:
                        logger.warning(f"[Exposure] 曝光疲劳更新失败（不影响推荐）: {e}")

                # fire-and-forget: 异步更新，不阻塞返回
                try:
                    loop = asyncio.get_running_loop()
                    loop.create_task(_update_ts_exposure(recommended_songs))
                except RuntimeError:
                    # 如果没有运行中的事件循环，同步执行
                    import threading
                    threading.Thread(
                        target=lambda: asyncio.run(_update_ts_exposure(recommended_songs)),
                        daemon=True,
                    ).start()
            except Exception:
                pass  # TS 更新失败不影响主流程
        elif recommended_songs:
            logger.debug("[EvalMode] 跳过按用户曝光疲劳写入")

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
            try:
                from config.settings import settings as _settings
                from services.teacher_log import log_teacher_example

                log_teacher_example(
                    "hyde",
                    inputs=invoke_params,
                    output={"acoustic_caption": result},
                    metadata={
                        "intent_type": intent_type or "unknown",
                        "model": getattr(llm, "model_name", ""),
                        "planner_quality_mode": getattr(_settings, "planner_quality_mode", "teacher"),
                        "prompt_version": "hyde_acoustic_2026_07_10",
                    },
                )
            except Exception:
                pass
            return result

        except Exception as e:
            logger.warning(f"[HyDE] 声学描述生成失败，降级使用原始查询: {e}\n{traceback.format_exc()}")
            return query
