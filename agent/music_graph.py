"""
音乐推荐Agent的工作流图
"""

import asyncio
import json
import os
from datetime import date
from typing import Dict, Any

from langgraph.graph import StateGraph, END
from langgraph.graph.state import CompiledStateGraph

# MemorySaver: 内存级 Checkpoint，支持对话状态持久化
# 生产环境可替换为 SqliteSaver / PostgresSaver
try:
    from langgraph.checkpoint.memory import MemorySaver
    _CHECKPOINTER_AVAILABLE = True
except ImportError:
    _CHECKPOINTER_AVAILABLE = False

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from config.logging_config import get_logger
from config.settings import settings
from agent.catalog_gap import CatalogGapDecision, analyze_catalog_gap, interleave_online_results, unwrap_recommendation_items
from agent.explanation import emit_fast_explanation
from agent.intent.delta_planner import IntentDeltaPlanner
from agent.intent.planner import IntentPlanner
from agent.netease_query import (
    artist_matches,
    build_netease_query_plan,
    extract_artist_id,
    fetch_json_with_retry,
    normalize_artist_catalog_songs,
    parse_play_url_payload,
)
from agent.retrieval_fallback import (
    avoid_terms,
    decide_online_fallback,
    fallback_query,
    filter_results_by_avoid,
    filter_results_by_requested_language,
)
from agent.web_discovery import build_web_discovery_query, extract_song_candidates
from llms.multi_llm import get_chat_model, get_intent_chat_model, get_explain_chat_model

from schemas.music_state import MusicAgentState, ToolOutput
from tools.graphrag_search import graphrag_search
# 【V2 升级】替换旧版 vector_search 为 Neo4j 原生语义搜索
from tools.semantic_search import semantic_search
from tools.acquire_music import acquire_online_music
from retrieval.hybrid_retrieval import MusicHybridRetrieval
from retrieval.user_memory import UserMemoryManager
from retrieval.history import MusicContextManager
from llms.prompts import (
    MUSIC_CHAT_RESPONSE_PROMPT,
    MUSIC_RECOMMENDATION_EXPLAINER_PROMPT,
    MUSIC_TUNER_RESPONSE_PROMPT,
)
from schemas.query_plan import MusicQueryPlan, RetrievalPlan
from schemas.tool_plan import ToolPlan, tool_plan_alignment_issues
from schemas.dialog_state import (
    ClarificationRequest,
    apply_dialog_state_to_plan,
    apply_plan_delta_operations,
    apply_plan_delta_with_report,
    clarification_from_delta,
    coerce_followup_general_chat_to_retrieval,
    compile_dialog_state_to_plan,
    is_followup_turn,
    load_dialog_state,
    update_dialog_result_anchors,
)
from schemas.refinement import build_refinement_suggestions
from services.llm_feedback_logger import build_planning_feedback, log_planning_feedback

logger = get_logger(__name__)


def _state_user_id(state: MusicAgentState) -> str:
    """Return the request user consistently across old and new state payloads."""
    metadata = state.get("metadata") or {}
    return str(
        state.get("user_id")
        or metadata.get("user_id")
        or settings.default_user_id
    ).strip() or settings.default_user_id


def _schedule_recommended_knowledge_backfill(recommendations: Any, *, context: str) -> None:
    """Queue missing knowledge-card enrichment for songs that were actually shown."""

    if settings.eval_disable_side_effects or not recommendations:
        return
    try:
        from services.recommendation_knowledge_backfill import schedule_recommendation_knowledge_backfill

        meta = schedule_recommendation_knowledge_backfill(recommendations)
        if meta.get("scheduled"):
            logger.info(
                "[KnowledgeBackfill] %s queued %s recommended songs",
                context,
                meta.get("scheduled"),
            )
    except Exception as exc:
        logger.debug("[KnowledgeBackfill] %s scheduling skipped: %s", context, exc)


def _web_search_enabled() -> bool:
    return os.environ.get("MUSIC_WEB_SEARCH_ENABLED", "1").lower() not in {"0", "false", "no", "off"}


def _record_timing(state: MusicAgentState, name: str, elapsed_seconds: float) -> Dict[str, float]:
    timings = dict(state.get("timings") or {})
    timings[name] = round(max(0.0, elapsed_seconds) * 1000, 3)
    return timings


def _song_field(song: Dict[str, Any], *keys: str) -> Any:
    for key in keys:
        value = song.get(key)
        if value not in (None, "", []):
            return value
    return None


def _list_field(song: Dict[str, Any], *keys: str, limit: int = 4) -> list[str]:
    values: list[str] = []
    for key in keys:
        raw = song.get(key)
        if isinstance(raw, str):
            parts = [part.strip() for part in raw.replace("|", "/").split("/") if part.strip()]
        elif isinstance(raw, list):
            parts = [str(part).strip() for part in raw if str(part).strip()]
        else:
            parts = []
        for part in parts:
            if part and part not in values:
                values.append(part)
            if len(values) >= limit:
                return values
    return values


def _build_tuner_recommendation_overview(recommendations: list[Dict[str, Any]], limit: int = 8) -> str:
    rows: list[str] = []
    for index, rec in enumerate(recommendations[:limit], 1):
        song = rec.get("song", rec) if isinstance(rec, dict) else rec
        if not isinstance(song, dict):
            continue
        title = _song_field(song, "title") or "未知歌曲"
        artist = _song_field(song, "artist", "artist_name") or "未知歌手"
        tags = _list_field(song, "genres", "moods", "themes", "scenarios", "tags", "genre", limit=5)
        language = _song_field(song, "language")
        source_labels = _list_field(song, "retrieval_sources", "source", limit=3)
        details = []
        if language:
            details.append(f"language={language}")
        if tags:
            details.append("tags=" + "/".join(tags))
        if source_labels:
            details.append("sources=" + "/".join(source_labels))
        detail_text = f" ({'; '.join(details)})" if details else ""
        rows.append(f"{index}. {title} - {artist}{detail_text}")
    return "\n".join(rows) or "本轮已生成可播放推荐歌单。"

# 延迟初始化 llm，避免在模块导入时配置未加载
_llm = None

def get_llm():
    """获取LLM实例（延迟初始化）"""
    global _llm
    if _llm is None:
        _llm = get_chat_model(settings.llm_default_provider, settings.llm_default_model)
    return _llm

def set_llm(new_llm):
    """覆盖全局 LLM 实例（由 server.py 在每次请求时调用，实现动态切换）"""
    global _llm
    _llm = new_llm
    logger.info(f"[music_graph] LLM 已切换为: {getattr(new_llm, 'model_name', str(new_llm))}")

# 意图分析专用 LLM（可配置更快/更小的模型）
_intent_llm = None

def get_intent_llm():
    """获取意图分析专用 LLM 实例（延迟初始化，从 settings 读取配置）"""
    global _intent_llm
    if _intent_llm is None:
        _intent_llm = get_intent_chat_model()
        logger.info(f"[music_graph] 意图分析 LLM 初始化: {getattr(_intent_llm, 'model_name', str(_intent_llm))}")
    return _intent_llm

def set_intent_llm(new_llm):
    """覆盖意图分析 LLM 实例"""
    global _intent_llm
    _intent_llm = new_llm
    logger.info(f"[music_graph] 意图分析 LLM 已切换为: {getattr(new_llm, 'model_name', str(new_llm))}")


# 解释生成专用 LLM（generate_explanation 节点，负责流式生成推荐理由）
_explain_llm = None

def get_explain_llm():
    """获取解释生成专用 LLM 实例（延迟初始化，从 settings 读取配置）"""
    global _explain_llm
    if _explain_llm is None:
        _explain_llm = get_explain_chat_model()
        logger.info(f"[music_graph] 解释生成 LLM 初始化: {getattr(_explain_llm, 'model_name', str(_explain_llm))}")
    return _explain_llm

def set_explain_llm(new_llm):
    """覆盖解释生成 LLM 实例"""
    global _explain_llm
    _explain_llm = new_llm
    logger.info(f"[music_graph] 解释生成 LLM 已切换为: {getattr(new_llm, 'model_name', str(new_llm))}")


# _clean_json_from_llm 已被 with_structured_output 替代，不再需要手动正则解析


class MusicRecommendationGraph:
    """音乐推荐工作流图
    
    支持 LangGraph MemorySaver Checkpoint：
    - 编译时注入 checkpointer，每次 ainvoke 传入 thread_id
    - 同一 thread_id 的对话共享状态（chat_history 自动累积）
    - 内存级实现，重启进程后状态丢失
    - 生产环境可替换为 SqliteSaver / PostgresSaver 实现持久化
    """
    
    def __init__(self, enable_checkpoint: bool = True):
        self.enable_checkpoint = enable_checkpoint and _CHECKPOINTER_AVAILABLE
        self.checkpointer = MemorySaver() if self.enable_checkpoint else None
        # 并发安全的流式队列注册表：{request_id: asyncio.Queue}
        # 每个请求创建独立的 queue，避免并发请求间的数据交叉污染
        self._explanation_queues: dict = {}
        self.intent_planner = IntentPlanner(get_intent_llm)
        self.intent_delta_planner = IntentDeltaPlanner(get_intent_llm)
        self.workflow = self._build_graph()
    
    def get_app(self) -> CompiledStateGraph:
        """获取编译后的应用"""
        return self.workflow
    
    def _load_user_profile_for_prompt(self, user_id: str = "local_admin") -> str:
        """
        从动态画像 / Neo4j User 节点加载用户画像，格式化为简洁文本。
        优先级：动态画像（Profile Synthesizer）> 静态标签（用户手动设置）
        """
        # ① 优先尝试从 Profile Synthesizer 缓存获取动态画像
        try:
            from services.profile_synthesizer import get_profile_synthesizer
            synth = get_profile_synthesizer(user_id)
            portrait_text = synth.get_portrait_for_prompt()
            if portrait_text:
                logger.info(f"[UserProfile] 动态画像加载成功: {portrait_text[:80]}")
                return portrait_text
        except Exception as e:
            logger.warning(f"[UserProfile] 动态画像加载失败，退回静态标签: {e}")
        
        # ② Fallback：从 Neo4j 读取用户手动设置的静态偷好标签
        try:
            from retrieval.neo4j_client import get_neo4j_client
            import json as _json
            client = get_neo4j_client()
            if not client or not client.driver:
                return ""
            result = client.execute_query("""
            MATCH (u:User {id: $uid})
            RETURN u.preferred_genres AS genres,
                   u.preferred_moods AS moods,
                   u.preferred_scenarios AS scenarios,
                   u.preferred_languages AS languages,
                   u.profile_free_text AS free_text
            """, {"uid": user_id})
            
            if not result or not result[0]:
                return ""
            
            row = result[0]
            parts = []
            for field, label in [
                ("genres", "偏好流派"),
                ("moods", "情绪偏向"),
                ("scenarios", "常听场景"),
                ("languages", "语言偏好"),
            ]:
                raw = row.get(field)
                if raw:
                    try:
                        values = _json.loads(raw)
                        if values:
                            parts.append(f"{label}: {', '.join(values)}")
                    except (ValueError, TypeError):
                        pass
            
            free_text = row.get("free_text") or ""
            if free_text.strip():
                parts.append(f"自述: {free_text.strip()}")
            
            profile_text = "；".join(parts) if parts else ""
            if profile_text:
                logger.info(f"[UserProfile] 静态标签加载成功: {profile_text[:80]}")
            return profile_text
        except Exception as e:
            logger.warning(f"[UserProfile] 画像加载失败: {e}")
            return ""
    
    async def warmup_kv_cache(self):
        """启动时预热 KV Prefix Cache（后台异步执行，不阻塞启动）
        
        原理：向 API 发送一个包含完整 system prompt 的轻量请求，
        让服务商（SiliconFlow/DeepSeek）计算并缓存 system prompt 的 KV 状态。
        后续真实请求的相同 system prefix 会自动命中缓存，跳过 Prefill 阶段。
        
        预期效果：首次用户请求从 8-12s 降低到 2-4s。
        """
        import time as _time
        _t0 = _time.time()
        try:
            _intent_llm = get_intent_llm()
            _provider = (settings.intent_llm_provider or settings.llm_default_provider or "").lower()
            _local_providers = {"sglang", "vllm", "ollama"}
            
            if _provider in _local_providers:
                logger.info("[Warmup] 本地模型无需预热 KV Cache，跳过")
                return
            
            from llms.prompts import UNIFIED_PLANNER_SYSTEM, UNIFIED_PLANNER_HUMAN
            from langchain_core.prompts import ChatPromptTemplate
            
            # 用最简单的输入触发一次完整的 system prompt 计算
            _warmup_model_name = getattr(_intent_llm, 'model_name', '') or ''
            _is_qwen3_warmup = any(kw in _warmup_model_name.lower() for kw in ['qwen3', 'qwen-3'])
            _is_dashscope_warmup = _provider == 'dashscope'
            _bound_llm = _intent_llm
            if _is_qwen3_warmup or _is_dashscope_warmup:
                _bound_llm = _intent_llm.bind(
                    extra_body={"enable_thinking": False},
                    response_format={"type": "json_object"},
                )
                structured_llm = _bound_llm.with_structured_output(MusicQueryPlan, include_raw=True, method="json_mode")
            else:
                structured_llm = _bound_llm.with_structured_output(MusicQueryPlan, include_raw=True)
            
            # DashScope 显式缓存：warmup 时用 content 数组格式创建缓存条目
            _intent_provider = (settings.intent_llm_provider or settings.llm_default_provider or "").lower()
            if _intent_provider == "dashscope":
                from langchain_core.messages import SystemMessage
                _sys_msg = SystemMessage(
                    content=[{
                        "type": "text",
                        "text": UNIFIED_PLANNER_SYSTEM,
                        "cache_control": {"type": "ephemeral"}
                    }]
                )
                _prompt = ChatPromptTemplate.from_messages([_sys_msg, ("human", UNIFIED_PLANNER_HUMAN)])
            else:
                _prompt = ChatPromptTemplate.from_messages([
                    ("system", UNIFIED_PLANNER_SYSTEM),
                    ("human", UNIFIED_PLANNER_HUMAN),
                ])
            
            chain = _prompt | structured_llm
            _raw = await chain.ainvoke({
                "user_input": "你好",
                "user_preferences": "无",
                "chat_history": "",
                "previous_plan": "",
                "current_date": str(date.today()),
            })
            _elapsed = _time.time() - _t0
            
            # 检查缓存状态
            _raw_msg = _raw.get("raw")
            _cache_info = ""
            if _raw_msg and hasattr(_raw_msg, "usage_metadata") and _raw_msg.usage_metadata:
                _usage = _raw_msg.usage_metadata
                _hit = (
                    _usage.get("prompt_cache_hit_tokens", 0)
                    or _usage.get("cache_read_input_tokens", 0)
                    or (_usage.get("input_token_details") or {}).get("cache_read", 0)
                )
                _cache_info = f" | cache_hit={_hit}"
            
            logger.info(f"[Warmup] ✅ KV Cache 预热完成, 耗时 {_elapsed:.1f}s{_cache_info}")
        except Exception as e:
            logger.warning(f"[Warmup] ⚠️ KV Cache 预热失败（不影响正常使用）: {e}")
    
    async def analyze_intent(self, state: MusicAgentState) -> Dict[str, Any]:
        """
        节点1: 统一意图分析 + 检索规划
        使用 with_structured_output 直接输出类型安全的 MusicQueryPlan 对象，
        彻底消除手动正则 + json.loads 的脆弱解析。
        """
        import time as _time
        _t0 = _time.time()
        
        user_input = state.get("input", "")
        chat_history = state.get("chat_history", [])
        user_id = _state_user_id(state)
        
        try:
            # Full chat history is interpreted by the LLM Planner. Do not infer
            # semantic state from regexes when an explicit session state is absent.
            previous_dialog_state = state.get("dialog_state") or {}
            _profile_text = self._load_user_profile_for_prompt(user_id)
            if is_followup_turn(user_input, previous_dialog_state):
                try:
                    plan_delta = await self.intent_delta_planner.plan(
                        user_input=user_input,
                        dialog_state=previous_dialog_state,
                    )
                    delta_clarification = clarification_from_delta(
                        plan_delta,
                        confidence_threshold=settings.dst_clarification_confidence_threshold,
                    )
                    if delta_clarification.required:
                        pending_state, _ = apply_plan_delta_operations(
                            previous_dialog_state,
                            plan_delta,
                            user_input,
                        )
                        pending_state.pending_clarification = delta_clarification
                        return {
                            "intent_type": "clarification",
                            "intent_parameters": {"query": user_input},
                            "intent_context": delta_clarification.reason,
                            "clarification": delta_clarification.model_dump(),
                            "clarification_options": list(delta_clarification.options),
                            "dialog_state": pending_state.model_dump(),
                            "dialog_delta": pending_state.last_delta.model_dump(),
                            "intent_confidence": plan_delta.confidence,
                            "refinement_options": [],
                            "final_response": delta_clarification.question,
                            "step_count": state.get("step_count", 0) + 1,
                            "timings": _record_timing(state, "intent_ms", _time.time() - _t0),
                        }
                    if plan_delta.ambiguity_reasons:
                        logger.info(
                            "[DST] Delta 存在歧义，回退 full planner: %s",
                            "; ".join(plan_delta.ambiguity_reasons[:2]),
                        )
                    elif plan_delta.operations:
                        dialog_state, dialog_delta = apply_plan_delta_operations(
                            previous_dialog_state,
                            plan_delta,
                            user_input,
                        )
                        plan = compile_dialog_state_to_plan(dialog_state, user_input)
                        dialog_state.last_complete_plan = plan.model_dump()
                        refinement = build_refinement_suggestions(
                            user_input=user_input,
                            plan=plan,
                            dialog_state=dialog_state,
                        )
                        retrieval_plan_dict = plan.retrieval_plan.model_dump()
                        tool_plan_dict = plan.tool_plan.model_dump(mode="json") if plan.tool_plan else {}
                        retrieval_plan_dict["_tool_plan"] = tool_plan_dict
                        retrieval_plan_dict["_tool_plan_alignment_issues"] = tool_plan_alignment_issues(plan)
                        retrieval_plan_dict["_intent_type"] = plan.intent_type
                        retrieval_plan_dict["_graphzep_facts"] = state.get("graphzep_facts", "")
                        retrieval_plan_dict["_user_profile"] = _profile_text
                        retrieval_plan_dict["_user_id"] = user_id
                        logger.info(
                            "[DST] PlanDelta 已确定性应用: mode=%s operations=%d",
                            plan_delta.planner_mode,
                            len(plan_delta.operations),
                        )
                        return {
                            "intent_type": plan.intent_type,
                            "intent_parameters": plan.parameters,
                            "intent_context": plan.context,
                            "retrieval_plan": retrieval_plan_dict,
                            "tool_plan": tool_plan_dict,
                            "dialog_state": dialog_state.model_dump(),
                            "dialog_delta": dialog_delta.model_dump(),
                            "intent_confidence": refinement.confidence,
                            "refinement_options": [
                                option.model_dump() for option in refinement.options
                            ],
                            "step_count": state.get("step_count", 0) + 1,
                            "timings": _record_timing(state, "intent_ms", _time.time() - _t0),
                        }
                    logger.info("[DST] Delta 无有效操作，回退 full planner")
                except Exception as delta_error:
                    logger.warning("[DST] Delta planner 失败，回退 full planner: %s", delta_error)

            # 格式化对话历史
            context_manager = MusicContextManager()
            history_text = context_manager.format_chat_history(chat_history)
            
            # ✅ 【DST】构建上轮检索计划文本，供 Planner 做多轮标签继承
            # 区分两种继承模式：
            #   graph/hybrid → 继承离散标签（mood/genre/language...）
            #   vector       → 继承声学语义（acoustic_query），不继承粗粒度标签
            _prev_plan = state.get("retrieval_plan")
            _prev_intent = state.get("intent_type", "")
            _previous_plan_text = ""
            if _prev_plan and isinstance(_prev_plan, dict):
                if _prev_intent == "vector_search":
                    # ── 上轮是纯向量检索：继承声学语义，避免标签趋同化 ──
                    _acoustic = _prev_plan.get("vector_acoustic_query", "")
                    if _acoustic:
                        _previous_plan_text = (
                            f"上轮为纯向量检索(vector_search)，声学语义: \"{_acoustic[:150]}\"\n"
                            f"注意：追问时应使用 hybrid_search（graph筛新标签 + vector继承声学语义），"
                            f"不要将上轮的情绪降级为粗粒度标签(如mood=悲伤)来做图谱硬筛选"
                        )
                        logger.info(f"[DST] 上轮为 vector_search，继承声学语义(前80字): {_acoustic[:80]}")
                else:
                    # ── 上轮是图谱/混合检索：继承离散标签 + 声学语义（如有）──
                    _tag_parts = []
                    for _tag_key, _tag_label in [
                        ("graph_mood_filter", "mood"),
                        ("graph_scenario_filter", "scenario"),
                        ("graph_genre_filter", "genre"),
                        ("graph_language_filter", "language"),
                        ("graph_region_filter", "region"),
                    ]:
                        _val = _prev_plan.get(_tag_key)
                        if _val:
                            _tag_parts.append(f"{_tag_label}={_val}")
                    
                    _parts = []
                    if _tag_parts:
                        _parts.append(f"上轮检索策略: {_prev_intent}，标签: {', '.join(_tag_parts)}")
                    # hybrid_search 时同时继承声学描述，避免追问时丢失氛围上下文
                    _acoustic = _prev_plan.get("vector_acoustic_query", "")
                    if _acoustic:
                        _parts.append(f"上轮声学描述: \"{_acoustic[:150]}\"")
                        _parts.append("注意：用户追问时应继承上轮的检索策略和声学描述，不可降级为纯 graph_search")
                    
                    if _parts:
                        _previous_plan_text = "\n".join(_parts)
                        logger.info(f"[DST] 上轮: {_prev_intent}, 标签={_tag_parts}, acoustic={'有' if _acoustic else '无'}")
            
            # ✅ with_structured_output：让模型直接输出 MusicQueryPlan Pydantic 对象
            # 底层自动处理 json_schema 约束，无需任何正则或 json.loads
            _intent_llm_instance = get_intent_llm()
            _intent_model_name = getattr(_intent_llm_instance, 'model_name', '?')
            _intent_provider = (settings.intent_llm_provider or settings.llm_default_provider or '?').lower()
            logger.info(f"--- [步骤 1] 统一意图分析与检索规划 (Structured Output) | 🤖 {_intent_provider} / {_intent_model_name} ---")
            
            # ── 统一构建用户偏好上下文（用户画像 + MemoryGateway 长期记忆）──
            _graphzep = state.get("graphzep_facts", "")
            _pref_parts = []
            if _profile_text:
                _pref_parts.append(f"【用户画像】{_profile_text}")
            if _graphzep and _graphzep != "暂无用户长期记忆":
                _pref_parts.append(f"【长期记忆】{_graphzep}")
            _combined_preferences = "\n".join(_pref_parts) if _pref_parts else "无"
            
            plan = await self.intent_planner.plan(
                user_input=user_input,
                user_preferences=_combined_preferences,
                chat_history=history_text,
                previous_plan=_previous_plan_text,
                graphzep_facts=state.get("graphzep_facts", ""),
                user_id=user_id,
            )
            if plan.intent_type == "clarification":
                params = plan.parameters or {}
                options = params.get("options") or params.get("clarification_options") or []
                if isinstance(options, str):
                    options = [options]
                clarification = ClarificationRequest(
                    required=True,
                    reason=str(params.get("reason") or plan.reasoning or "llm_clarification"),
                    question=str(
                        params.get("question")
                        or plan.context
                        or "我还不能可靠判断你想保留哪种音乐方向。你想按哪个方向继续？"
                    ),
                    options=[str(option) for option in options if str(option).strip()][:6],
                    unresolved_paths=[
                        str(path)
                        for path in (params.get("unresolved_paths") or [])
                        if str(path).strip()
                    ],
                )
                if not clarification.options:
                    clarification.options = ["告诉我一首参考歌", "描述想保留的氛围", "只按这句话重新推荐"]
                pending_state = load_dialog_state(previous_dialog_state).model_copy(deep=True)
                pending_state.pending_clarification = clarification
                clarification_delta = {
                    "followup": pending_state.turn_count > 0,
                    "topic_shift": False,
                    "confidence": 0.0,
                    "reason": clarification.reason,
                    "inherited": [],
                    "added": {},
                    "replaced": {},
                    "removed": [],
                    "planner_mode": "full_planner",
                }
                return {
                    "intent_type": "clarification",
                    "intent_parameters": {"query": user_input},
                    "intent_context": clarification.reason,
                    "clarification": clarification.model_dump(),
                    "clarification_options": list(clarification.options),
                    "dialog_state": pending_state.model_dump(),
                    "dialog_delta": clarification_delta,
                    "intent_confidence": 0.0,
                    "refinement_options": [],
                    "final_response": clarification.question,
                    "step_count": state.get("step_count", 0) + 1,
                    "timings": _record_timing(state, "intent_ms", _time.time() - _t0),
                }
            dialog_state, dialog_delta = apply_plan_delta_with_report(previous_dialog_state, plan, user_input)
            plan = apply_dialog_state_to_plan(plan, dialog_state)
            plan = coerce_followup_general_chat_to_retrieval(plan, dialog_state, user_input)
            dialog_state.last_complete_plan = plan.model_dump()
            refinement = build_refinement_suggestions(
                user_input=user_input,
                plan=plan,
                dialog_state=dialog_state,
            )
            # 直接通过属性访问，完全类型安全，字段缺失会有 Pydantic 默认值兜底
            logger.info(
                f"识别到意图: {plan.intent_type} | "
                f"检索规划: graph={plan.retrieval_plan.use_graph}, "
                f"vector={plan.retrieval_plan.use_vector}, "
                f"web={plan.retrieval_plan.use_web_search}"
            )
            logger.info(f"决策理由: {plan.reasoning}")
            logger.info(f"[⏱ 意图分析] 耗时 {_time.time()-_t0:.1f}s")
            
            # ============================================================
            # 【升级】将 intent_type 和 graphzep_facts 注入 retrieval_plan
            # intent_type: 供 HyDE 根据意图类型调整描述风格
            # _graphzep_facts: 供 HyDE 参考用户偏好生成声学描述
            # ============================================================
            retrieval_plan_dict = plan.retrieval_plan.model_dump()
            tool_plan_dict = plan.tool_plan.model_dump(mode="json") if plan.tool_plan else {}
            retrieval_plan_dict["_tool_plan"] = tool_plan_dict
            retrieval_plan_dict["_tool_plan_alignment_issues"] = tool_plan_alignment_issues(plan)
            retrieval_plan_dict["_intent_type"] = plan.intent_type
            retrieval_plan_dict["_graphzep_facts"] = state.get("graphzep_facts", "")
            retrieval_plan_dict["_user_profile"] = _profile_text  # 画像文本供 HyDE 参考
            retrieval_plan_dict["_user_id"] = user_id

            try:
                log_planning_feedback(
                    build_planning_feedback(
                        user_input=user_input,
                        plan=plan,
                        retrieval_plan=retrieval_plan_dict,
                        provider=_intent_provider,
                        model=_intent_model_name,
                        user_id=user_id,
                        dialog_delta=dialog_delta.model_dump(),
                        refinement_options=[option.model_dump() for option in refinement.options],
                    )
                )
            except Exception as feedback_error:
                logger.debug("[LLMFeedback] 规划审计日志写入失败，已跳过: %s", feedback_error)
            
            return {
                "intent_type": plan.intent_type,
                "intent_parameters": plan.parameters,
                "intent_context": plan.context,
                "retrieval_plan": retrieval_plan_dict,
                "tool_plan": tool_plan_dict,
                "dialog_state": dialog_state.model_dump(),
                "dialog_delta": dialog_delta.model_dump(),
                "intent_confidence": refinement.confidence,
                "refinement_options": [option.model_dump() for option in refinement.options],
                "step_count": state.get("step_count", 0) + 1,
                "timings": _record_timing(state, "intent_ms", _time.time() - _t0),
            }
            
        except Exception as e:
            # 【可观测降级】意图分析失败时，不再静默退化为 general_chat。
            # 原因：(1) 用户多数是来"求歌"的，退闲聊=答非所问；
            #       (2) 若失败本就源于 LLM 不可用，general_chat 仍需调 LLM 生成闲聊 → 二次失败。
            # 改为保守的纯向量检索：用原始输入直接作声学查询（MuQ 主锚可编码），
            # 该路径不依赖 LLM，至少能返回语义相近的音乐；并打 _intent_degraded 标记供监控/离线评测统计真实失败率。
            import traceback as _tb
            logger.error(f"意图分析失败，降级为保守 vector_search: {e}\n{_tb.format_exc()}")
            _fallback_plan = {
                "use_graph": False,
                "graph_entities": [], "graph_artist_entities": [], "graph_song_entities": [],
                "graph_genre_filter": None, "graph_scenario_filter": None,
                "graph_mood_filter": None, "graph_language_filter": None, "graph_region_filter": None,
                "use_vector": True,
                "vector_acoustic_query": user_input,
                "vector_acoustic_queries": [user_input],
                "use_web_search": False, "web_search_keywords": "",
                "_intent_type": "vector_search",
                "_graphzep_facts": state.get("graphzep_facts", ""),
                "_user_profile": "",
                "_intent_degraded": True,
            }
            _fallback_tool_plan = ToolPlan.model_validate({
                "origin": "legacy_compiler",
                "request_mode": "recommendation",
                "tool_calls": [{
                    "id": "audio_recall",
                    "name": "search_audio",
                    "arguments": {"acoustic_queries": [user_input], "limit": 30},
                    "reason": "planner failure fallback",
                }],
                "confidence": 0.0,
                "decision_summary": "planner unavailable; bounded audio fallback",
                "max_replans": 0,
            }).model_dump(mode="json")
            _fallback_plan["_tool_plan"] = _fallback_tool_plan
            return {
                "intent_type": "vector_search",
                "intent_parameters": {"query": user_input, "entities": []},
                "intent_context": user_input,
                "retrieval_plan": _fallback_plan,
                "tool_plan": _fallback_tool_plan,
                "step_count": state.get("step_count", 0) + 1,
                "error_log": state.get("error_log", []) + [
                    {"node": "analyze_intent", "error": str(e), "degraded_to": "vector_search"}
                ],
                "timings": _record_timing(state, "intent_ms", _time.time() - _t0),
            }
    
    def route_by_intent(self, state: MusicAgentState) -> str:
        """
        路由函数: 根据意图类型决定下一步（5 类检索策略 + 2 类功能性意图）
        """
        intent_type = state.get("intent_type", "general_chat")
        logger.info(f"根据意图 '{intent_type}' 进行路由")

        # 3 类检索策略意图 → 统一走 search_songs 节点（retrieval_plan 已明确 use_graph/use_vector）
        if intent_type in ["graph_search", "hybrid_search", "vector_search"]:
            return "search_songs"
        elif intent_type == "web_search":
            if not _web_search_enabled():
                logger.info("[route_by_intent] web_search 意图被前端开关关闭")
                return "web_disabled"
            # web_search 直接走 web_fallback（网易云 API 搜可播放歌曲）
            # 不走 MusicHybridRetrieval 的纯文本联网搜索（只返回资讯不返回音频）
            return "web_fallback"
        elif intent_type == "recommend_by_favorites":
            # 查用户收藏：路由到 generate_recommendations，内部有专门的收藏召回逻辑
            return "generate_recommendations"
        elif intent_type == "acquire_music":
            return "acquire_online_music"
        elif intent_type == "clarification":
            return "clarification"
        elif intent_type.startswith("create_playlist"):
            return "analyze_user_preferences"
        else:
            return "general_chat"

    async def clarification_node(self, state: MusicAgentState) -> Dict[str, Any]:
        """Return a deterministic clarification question instead of guessing."""
        clarification = state.get("clarification") or {}
        question = clarification.get("question") or state.get("final_response") or "你想保留上一轮的哪种音乐感觉？"
        options = clarification.get("options") or state.get("clarification_options") or []
        request_id = (state.get("metadata") or {}).get("request_id")
        explanation_queue = self._explanation_queues.get(request_id) if request_id else None
        if explanation_queue is not None:
            await explanation_queue.put(
                {
                    "__clarification__": {
                        "question": question,
                        "options": options,
                        "reason": clarification.get("reason"),
                    }
                }
            )
            await explanation_queue.put(None)
        return {
            "final_response": question,
            "recommendations": [],
            "search_results": [],
            "clarification_options": options,
            "retrieval_meta": {
                **(state.get("retrieval_meta") or {}),
                "source": "clarification",
                "degraded": False,
                "clarification_required": True,
                "clarification_reason": clarification.get("reason"),
            },
            "step_count": state.get("step_count", 0) + 1,
        }

    async def web_disabled_node(self, state: MusicAgentState) -> Dict[str, Any]:
        """Respect the UI web-search switch and explain the local-catalog boundary."""
        message = (
            "我现在按你的设置只使用本地曲库，不会联网搜索。这个请求需要外部候选或实时资料，"
            "本地曲库暂时无法可靠满足。你可以打开下方“联网搜索”，我再为你查找更多可播放候选。"
        )
        request_id = (state.get("metadata") or {}).get("request_id")
        explanation_queue = self._explanation_queues.get(request_id) if request_id else None
        if explanation_queue is not None:
            await explanation_queue.put(message)
            await explanation_queue.put(None)
        return {
            "search_results": [],
            "recommendations": [],
            "final_response": message,
            "retrieval_meta": {
                **(state.get("retrieval_meta") or {}),
                "source": "local",
                "degraded": True,
                "degraded_reason": "web_search_disabled",
                "web_search_blocked": True,
            },
            "_web_action": "blocked",
            "_catalog_gap": {
                "action": "blocked",
                "reasons": ["web_search_disabled"],
                "message": message,
            },
            "step_count": state.get("step_count", 0) + 1,
        }
    
    async def search_songs_node(self, state: MusicAgentState) -> Dict[str, Any]:
        """
        节点2a: 搜索歌曲
        """
        import time as _time
        _t0 = _time.time()
        logger.info("--- [步骤 2a] 搜索歌曲 ---")
        
        parameters = state.get("intent_parameters", {})
        query = parameters.get("query", "")
        genre = parameters.get("genre")
        
        try:
            retriever = MusicHybridRetrieval(llm_client=get_llm())
            
            # 将可用的参数合并为一句话供路由分析
            search_intent = f"查询:{query} 流派:{genre}" if genre else query
            logger.info(f"调用检索引擎执行歌曲搜索: {search_intent}")
            
            # 传递上游统一规划的 retrieval_plan，避免二次 LLM 调用
            retrieval_plan = state.get("retrieval_plan")
            raw_hybrid_result = await retriever.retrieve(search_intent, limit=settings.hybrid_retrieval_limit, precomputed_plan=retrieval_plan)
            
            # 直接使用标准的 ToolOutput
            if raw_hybrid_result and raw_hybrid_result.success:
                search_results = raw_hybrid_result.data
            else:
                search_results = []
            
            logger.info(f"搜索到 {len(search_results)} 首歌曲, 耗时 {_time.time()-_t0:.1f}s")
            timings = dict(state.get("timings") or {})
            if raw_hybrid_result and getattr(raw_hybrid_result, "metadata", None):
                timings.update(raw_hybrid_result.metadata.get("timings") or {})
            timings["search_node_ms"] = round((_time.time() - _t0) * 1000, 3)

            for item in search_results:
                if not isinstance(item, dict):
                    continue
                song = item.get("song", item)
                if isinstance(song, dict):
                    song.setdefault("source", "local")

            tool_plan = state.get("tool_plan") or {}
            planned_tools = {
                str(call.get("name") or "")
                for call in (tool_plan.get("tool_calls") or [])
                if isinstance(call, dict)
            }
            run_gap_tool = bool(
                not settings.tool_plan_execution_enabled
                or not tool_plan
                or "inspect_catalog_gap" in planned_tools
                or not search_results
            )
            if run_gap_tool:
                fallback_decision = decide_online_fallback(search_results, retrieval_plan, query)
                gap_decision = analyze_catalog_gap(
                    search_results,
                    retrieval_plan,
                    query,
                    web_enabled=_web_search_enabled(),
                    fallback_decision=fallback_decision,
                    normal_mix_count=getattr(settings, "web_mix_in_count", 4),
                    fallback_count=getattr(settings, "web_fallback_count", 10),
                    min_local_results=getattr(settings, "catalog_gap_min_local_results", 8),
                )
            else:
                gap_decision = CatalogGapDecision(
                    action="none",
                    inventory_count=len(search_results),
                    details={"tool_plan_skipped_gap": True},
                )
            if gap_decision.action == "fallback":
                logger.warning(
                    "[search_songs] Catalog gap 触发联网兜底: reasons=%s, inventory=%d",
                    ",".join(gap_decision.reasons),
                    gap_decision.inventory_count,
                )
            elif gap_decision.action == "mix_in":
                logger.info(
                    "[search_songs] 联网开启，计划少量穿插候选: inventory=%d, target=%d",
                    gap_decision.inventory_count,
                    gap_decision.target_web_count,
                )
            elif gap_decision.action == "blocked":
                logger.info(
                    "[search_songs] 本地缺口但联网关闭: reasons=%s",
                    ",".join(gap_decision.reasons),
                )
            dialog_state = update_dialog_result_anchors(
                state.get("dialog_state"),
                search_results,
            )
            if gap_decision.action != "blocked":
                _schedule_recommended_knowledge_backfill(search_results, context="search_songs")
            recommendations_payload = (
                []
                if gap_decision.action == "blocked"
                else raw_hybrid_result if raw_hybrid_result and raw_hybrid_result.success else []
            )
            result_count = 0 if gap_decision.action == "blocked" else len(search_results)
            tool_observations = list(
                (getattr(raw_hybrid_result, "metadata", None) or {}).get("tool_observations")
                or []
            )
            gap_calls = [
                call for call in (tool_plan.get("tool_calls") or [])
                if isinstance(call, dict) and call.get("name") == "inspect_catalog_gap"
            ]
            if (
                settings.tool_plan_execution_enabled
                and tool_plan
                and not gap_calls
                and not search_results
            ):
                gap_calls = [{"id": "catalog_gap_recovery", "name": "inspect_catalog_gap"}]
            for call in gap_calls:
                tool_observations.append({
                    "call_id": str(call.get("id") or "catalog_gap"),
                    "tool_name": "inspect_catalog_gap",
                    "success": True,
                    "status": "success",
                    "data": gap_decision.model_dump(),
                    "error": "",
                    "duration_ms": 0.0,
                    "metadata": {"needs_replan": gap_decision.needs_online},
                })

            return {
                "search_results": [] if gap_decision.action == "blocked" else search_results,
                "recommendations": recommendations_payload,
                "_need_web_fallback": gap_decision.needs_online,
                "_web_fallback_query": fallback_query(retrieval_plan, query),
                "_web_action": gap_decision.action,
                "_web_target_count": gap_decision.target_web_count,
                "_web_discovery_required": gap_decision.discovery_required,
                "_catalog_gap": gap_decision.model_dump(),
                "retrieval_meta": {
                    "inventory_count": gap_decision.inventory_count,
                    "result_count": result_count,
                    "source": "local",
                    "degraded": gap_decision.action in {"fallback", "blocked"},
                    "degraded_reason": ",".join(gap_decision.reasons) or None,
                    "web_action": gap_decision.action,
                    "web_search_blocked": gap_decision.action == "blocked",
                    "catalog_gap": gap_decision.model_dump(),
                },
                "tool_observations": tool_observations,
                "dialog_state": dialog_state.model_dump(),
                "step_count": state.get("step_count", 0) + 1,
                "timings": timings,
            }
            
        except Exception as e:
            logger.error(f"搜索歌曲失败: {str(e)}")
            retrieval_plan = state.get("retrieval_plan") or {}
            fallback_decision = decide_online_fallback(
                [],
                retrieval_plan,
                state.get("intent_parameters", {}).get("query", state.get("input", "")),
            )
            gap_decision = analyze_catalog_gap(
                [],
                retrieval_plan,
                state.get("intent_parameters", {}).get("query", state.get("input", "")),
                web_enabled=_web_search_enabled(),
                fallback_decision=fallback_decision,
                normal_mix_count=getattr(settings, "web_mix_in_count", 4),
                fallback_count=getattr(settings, "web_fallback_count", 10),
                min_local_results=getattr(settings, "catalog_gap_min_local_results", 8),
            )
            return {
                "search_results": [],
                "recommendations": [],
                "_need_web_fallback": gap_decision.needs_online,
                "_web_fallback_query": fallback_query(
                    retrieval_plan,
                    state.get("intent_parameters", {}).get("query", state.get("input", "")),
                ),
                "_web_action": gap_decision.action,
                "_web_target_count": gap_decision.target_web_count,
                "_web_discovery_required": gap_decision.discovery_required,
                "_catalog_gap": gap_decision.model_dump(),
                "retrieval_meta": {
                    "inventory_count": 0,
                    "result_count": 0,
                    "source": "local",
                    "degraded": gap_decision.action in {"fallback", "blocked"},
                    "degraded_reason": "local_retrieval_error" if gap_decision.action in {"fallback", "blocked"} else None,
                    "web_action": gap_decision.action,
                    "web_search_blocked": gap_decision.action == "blocked",
                    "catalog_gap": gap_decision.model_dump(),
                },
                "step_count": state.get("step_count", 0) + 1,
                "error_log": state.get("error_log", []) + [
                    {"node": "search_songs", "error": str(e)}
                ],
                "timings": _record_timing(state, "search_node_ms", _time.time() - _t0),
            }


    def route_after_search(self, state: MusicAgentState) -> str:
        """搜索后路由：按 Catalog Gap Detector 决定联网兜底/混入/本地解释。"""
        if state.get("_web_action") == "blocked":
            logger.info("[route_after_search] 本地缺口但联网关闭 → generate_explanation")
            return "generate_explanation"
        if state.get("_need_web_fallback"):
            logger.info("[route_after_search] Catalog gap / mix-in → web_fallback")
            return "web_fallback"
        return "generate_explanation"

    async def web_fallback_node(self, state: MusicAgentState) -> Dict[str, Any]:
        """
        节点：本地库未命中或 web_search 意图时，从网易云 API 联网搜索。
        不下载，只返回流媒体 URL，供前端即时播放。
        支持从 _web_fallback_query / intent_parameters / graph_entities / input 多级获取查询词。
        """
        import time as _time
        _t0 = _time.time()
        logger.info("--- [步骤] 联网搜索（网易云 API）---")

        # ── 多级查询词提取（Netease 搜索需要中文原文，不能用英文翻译）──
        user_input = state.get("input", "")
        fallback_query = state.get("_web_fallback_query", "")
        retrieval_plan = state.get("retrieval_plan") or {}
        prior_retrieval_meta = dict(state.get("retrieval_meta") or {})
        excluded_by_avoid = 0
        excluded_by_language = 0
        web_action = str(state.get("_web_action") or "fallback")
        target_count = int(state.get("_web_target_count") or getattr(settings, "web_fallback_count", 10))
        if web_action == "mix_in":
            target_count = int(state.get("_web_target_count") or getattr(settings, "web_mix_in_count", 4))
        target_count = max(1, min(target_count, 20))
        discovery_required = bool(state.get("_web_discovery_required"))
        catalog_gap = dict(state.get("_catalog_gap") or {})
        tool_plan = state.get("tool_plan") or {}

        def _web_tool_observations(status: str, count: int = 0, error: str = "") -> list[dict[str, Any]]:
            observations = list(state.get("tool_observations") or [])
            external_calls = [
                call for call in (tool_plan.get("tool_calls") or [])
                if isinstance(call, dict) and call.get("name") == "search_external_music"
            ]
            if (
                settings.tool_plan_execution_enabled
                and tool_plan
                and not external_calls
                and state.get("_need_web_fallback")
            ):
                external_calls = [{"id": "external_recovery_1", "name": "search_external_music"}]
            for call in external_calls:
                observations.append({
                    "call_id": str(call.get("id") or "external_discovery"),
                    "tool_name": "search_external_music",
                    "success": status in {"success", "empty"},
                    "status": status,
                    "data": {"candidate_count": count, "source": "external"},
                    "error": error[:500],
                    "duration_ms": round((_time.time() - _t0) * 1000, 3),
                    "metadata": {},
                })
            return observations

        def _local_preserve_payload(reason: str) -> Dict[str, Any] | None:
            local_items = unwrap_recommendation_items(state.get("recommendations", []))
            if not local_items:
                return None
            from schemas.music_state import ToolOutput

            logger.info(
                "[web_fallback] 联网未补到结果，保留本地候选: reason=%s, local=%d",
                reason,
                len(local_items),
            )
            local_meta = {
                **prior_retrieval_meta,
                "inventory_count": int(prior_retrieval_meta.get("inventory_count") or len(local_items)),
                "result_count": len(local_items),
                "source": "local",
                "degraded": False,
                "degraded_reason": prior_retrieval_meta.get("degraded_reason"),
                "web_failure_reason": reason,
                "online_result_count": 0,
                "local_result_count": len(local_items),
                "web_action": web_action,
                "web_target_count": target_count,
                "web_discovery_required": discovery_required,
                "catalog_gap": catalog_gap,
            }
            return {
                "search_results": [r.get("song", r) if isinstance(r, dict) else r for r in local_items],
                "recommendations": ToolOutput(success=True, data=local_items, raw_markdown=""),
                "_need_web_fallback": False,
                "retrieval_meta": local_meta,
                "tool_observations": _web_tool_observations("empty", 0, reason),
                "step_count": state.get("step_count", 0) + 1,
                "timings": _record_timing(state, "web_fallback_ms", _time.time() - _t0),
            }

        def _web_meta(result_count: int, failure_reason: str | None = None) -> Dict[str, Any]:
            degraded = bool(prior_retrieval_meta.get("degraded")) or bool(failure_reason)
            return {
                "inventory_count": int(prior_retrieval_meta.get("inventory_count") or 0),
                "result_count": result_count,
                "source": "web",
                "degraded": degraded,
                "degraded_reason": failure_reason or prior_retrieval_meta.get("degraded_reason"),
                "excluded_by_avoid": excluded_by_avoid,
                "excluded_by_language": excluded_by_language,
                "web_action": web_action,
                "web_target_count": target_count,
                "web_discovery_required": discovery_required,
                "catalog_gap": catalog_gap,
            }

        params = state.get("intent_parameters", {})
        netease_plan = build_netease_query_plan(
            user_input=user_input,
            fallback_query=fallback_query,
            retrieval_plan=retrieval_plan,
            intent_parameters=params,
        )
        query = netease_plan.query
        query_candidates = netease_plan.query_candidates()
        logger.info(
            "[web_fallback] 查询词: '%s' | mode=%s | candidates=%s",
            query,
            netease_plan.mode,
            list(query_candidates),
        )

        try:
            import aiohttp
            from config.settings import settings as _cfg
            api_base = _cfg.netease_api_base
            timeout = aiohttp.ClientTimeout(total=max(15, _cfg.netease_api_timeout))

            async with aiohttp.ClientSession() as session:
                def _log_search_retry(attempt: int, exc: Exception) -> None:
                    logger.warning(
                        "[web_fallback] 搜索请求第 %d 次失败，将重试: %s",
                        attempt,
                        type(exc).__name__,
                    )

                # 1) 搜索 / 外部候选发现
                import re as _re
                from urllib.parse import quote as _url_quote
                clean_query = _re.sub(r'[《》\[\]【】]', ' ', query).strip()
                songs = []

                if discovery_required:
                    try:
                        from tools.web_search_aggregator import _federated_search_async

                        discovery_query = build_web_discovery_query(
                            user_input,
                            retrieval_plan,
                            catalog_gap,
                        )
                        logger.info("[web_fallback] 外部候选发现: %s", discovery_query)
                        discovery_docs = await _federated_search_async(discovery_query)
                        candidates = extract_song_candidates(
                            discovery_docs,
                            max_candidates=max(target_count * 2, 8),
                        )
                        logger.info("[web_fallback] 外部资料抽取候选 %d 个", len(candidates))

                        async def _resolve_candidate(candidate):
                            if not candidate.query:
                                return []
                            url = f"{api_base}/search?keywords={_url_quote(candidate.query)}&limit=3"
                            payload = await fetch_json_with_retry(
                                session,
                                url,
                                timeout=timeout,
                                attempts=1,
                            )
                            resolved = payload.get("result", {}).get("songs", [])
                            for row in resolved:
                                row["_discovery_evidence"] = candidate.evidence
                                row["_discovery_query"] = candidate.query
                            return resolved

                        resolved_batches = await asyncio.gather(
                            *(_resolve_candidate(candidate) for candidate in candidates),
                            return_exceptions=True,
                        )
                        seen_ids = set()
                        for batch in resolved_batches:
                            if isinstance(batch, Exception):
                                continue
                            for row in batch:
                                sid = row.get("id")
                                if sid and sid not in seen_ids:
                                    seen_ids.add(sid)
                                    songs.append(row)
                                if len(songs) >= target_count:
                                    break
                            if len(songs) >= target_count:
                                break
                    except Exception as exc:
                        logger.warning("[web_fallback] 外部候选发现失败，回退网易云直搜: %s", type(exc).__name__)

                if songs:
                    logger.info("[web_fallback] 使用外部资料候选解析出 %d 首", len(songs))
                elif netease_plan.mode == "new_songs":
                    search_url = f"{api_base}/top/song?type=7"
                    data = await fetch_json_with_retry(
                        session,
                        search_url,
                        timeout=timeout,
                        attempts=2,
                        on_retry=_log_search_retry,
                    )
                    raw_songs = data.get("data", [])[:20]
                    songs = [
                        {
                            "id": s.get("id"),
                            "name": s.get("name", "Unknown"),
                            "artists": s.get("artists") or s.get("ar") or [],
                            "album": s.get("album") or s.get("al") or {},
                        }
                        for s in raw_songs
                        if s.get("id")
                    ]
                else:
                    search_limit = max(
                        target_count,
                        20 if netease_plan.artist_terms and not netease_plan.song_terms else settings.netease_search_limit,
                    )
                    for candidate_query in query_candidates:
                        clean_query = _re.sub(r'[《》\[\]【】]', ' ', candidate_query).strip()
                        if not clean_query:
                            continue
                        search_url = f"{api_base}/search?keywords={_url_quote(clean_query)}&limit={search_limit}"
                        data = await fetch_json_with_retry(
                            session,
                            search_url,
                            timeout=timeout,
                            attempts=2,
                            on_retry=_log_search_retry,
                        )
                        candidate_songs = data.get("result", {}).get("songs", [])
                        if netease_plan.artist_terms and not netease_plan.song_terms:
                            candidate_songs = [
                                s for s in candidate_songs
                                if artist_matches("、".join(a.get("name", "") for a in s.get("artists", [])), netease_plan.artist_terms)
                            ]
                        if candidate_songs:
                            if clean_query != query:
                                logger.info("[web_fallback] 查询候选命中: %s", clean_query)
                            songs = candidate_songs
                            break
                        logger.info("[web_fallback] 查询候选无结果: %s", clean_query)

                    if not songs and netease_plan.artist_terms and not netease_plan.song_terms:
                        artist_search_limit = max(search_limit, 20)

                        async def _fetch_artist_catalog():
                            for artist_query in netease_plan.artist_query_candidates():
                                clean_artist_query = _re.sub(r'[《》\[\]【】]', ' ', artist_query).strip()
                                if not clean_artist_query:
                                    continue

                                artist_id = ""
                                for endpoint in ("cloudsearch", "search"):
                                    artist_url = (
                                        f"{api_base}/{endpoint}?keywords={_url_quote(clean_artist_query)}"
                                        f"&type=100&limit=5"
                                    )
                                    try:
                                        artist_payload = await fetch_json_with_retry(
                                            session,
                                            artist_url,
                                            timeout=timeout,
                                            attempts=1,
                                        )
                                    except Exception as exc:
                                        logger.info(
                                            "[web_fallback] 歌手搜索端点失败: endpoint=%s query=%s error=%s",
                                            endpoint,
                                            clean_artist_query,
                                            type(exc).__name__,
                                        )
                                        continue
                                    artist_id = extract_artist_id(
                                        artist_payload,
                                        netease_plan.artist_terms,
                                        allow_top_result=clean_artist_query in netease_plan.artist_terms,
                                    )
                                    if artist_id:
                                        break

                                if not artist_id:
                                    logger.info("[web_fallback] 歌手候选未命中: %s", clean_artist_query)
                                    continue

                                for endpoint in (
                                    f"artist/songs?id={artist_id}&limit={artist_search_limit}",
                                    f"artist/top/song?id={artist_id}",
                                ):
                                    catalog_url = f"{api_base}/{endpoint}"
                                    try:
                                        catalog_payload = await fetch_json_with_retry(
                                            session,
                                            catalog_url,
                                            timeout=timeout,
                                            attempts=1,
                                        )
                                    except Exception as exc:
                                        logger.info(
                                            "[web_fallback] 歌手曲库端点失败: %s error=%s",
                                            endpoint,
                                            type(exc).__name__,
                                        )
                                        continue
                                    catalog_rows = catalog_payload.get("songs") or catalog_payload.get("hotSongs") or []
                                    catalog_songs = normalize_artist_catalog_songs(catalog_rows)
                                    if catalog_songs:
                                        for row in catalog_songs:
                                            row["_artist_catalog_query"] = clean_artist_query
                                        logger.info(
                                            "[web_fallback] 歌手曲库兜底命中: query=%s artist_id=%s songs=%d",
                                            clean_artist_query,
                                            artist_id,
                                            len(catalog_songs),
                                        )
                                        return catalog_songs
                            return []

                        songs = await _fetch_artist_catalog()

                songs, excluded_by_avoid = filter_results_by_avoid(
                    songs,
                    avoid_terms(retrieval_plan, user_input),
                )
                if excluded_by_avoid:
                    logger.info(
                        "[web_fallback] 联网结果应用否定约束，排除 %d 首",
                        excluded_by_avoid,
                    )

                requested_language = (retrieval_plan.get("hard_constraints") or {}).get("language")
                songs, excluded_by_language = filter_results_by_requested_language(
                    songs,
                    requested_language,
                )
                if excluded_by_language:
                    logger.info(
                        "[web_fallback] 联网结果应用语言确认，排除 %d 首",
                        excluded_by_language,
                    )

                if not songs:
                    logger.warning(f"[web_fallback] 联网搜索无结果: {query}")
                    preserved = _local_preserve_payload("web_search_empty")
                    if preserved:
                        return preserved
                    return {"search_results": [], "recommendations": [],
                            "_need_web_fallback": False,
                            "retrieval_meta": _web_meta(0, "web_search_empty"),
                            "tool_observations": _web_tool_observations("empty"),
                            "step_count": state.get("step_count", 0) + 1,
                            "timings": _record_timing(state, "web_fallback_ms", _time.time() - _t0)}

                # 收集 song_ids 用于批量获取详情
                song_ids = [str(s["id"]) for s in songs[:target_count]]

                # 2) 批量获取详情 (封面 + 更准确的元数据)
                detail_url = f"{api_base}/song/detail?ids={','.join(song_ids)}"
                detail_map = {}
                try:
                    async with session.get(detail_url, timeout=timeout) as dresp:
                        ddata = await dresp.json()
                    for ds in ddata.get("songs", []):
                        detail_map[str(ds["id"])] = ds
                except Exception:
                    pass  # 详情获取失败不影响主流程

                # 3) 批量获取播放链接；缺失项并发单曲重试，抵御代理的瞬时空响应。
                play_url_map = {}
                trial_info_map = {}
                try:
                    url_api = f"{api_base}/song/url?id={','.join(song_ids)}&level=exhigh"
                    async with session.get(url_api, timeout=timeout) as uresp:
                        udata = await uresp.json()
                    play_url_map, trial_info_map = parse_play_url_payload(udata)
                except Exception as exc:
                    logger.warning("[web_fallback] 批量播放链接获取失败，将尝试单曲补偿: %s", type(exc).__name__)

                missing_ids = [sid for sid in song_ids if sid not in play_url_map]
                if missing_ids:
                    logger.info("[web_fallback] %d 个播放链接缺失，启动单曲并发补偿", len(missing_ids))

                    async def _fetch_single_play_url(song_id: str):
                        single_url = f"{api_base}/song/url?id={song_id}&level=exhigh"
                        async with session.get(single_url, timeout=timeout) as single_resp:
                            return await single_resp.json()

                    single_payloads = await asyncio.gather(
                        *(_fetch_single_play_url(sid) for sid in missing_ids),
                        return_exceptions=True,
                    )
                    failed_retries = 0
                    for payload in single_payloads:
                        if isinstance(payload, Exception):
                            failed_retries += 1
                            continue
                        retry_urls, retry_trials = parse_play_url_payload(payload)
                        play_url_map.update(retry_urls)
                        trial_info_map.update(retry_trials)
                    if failed_retries:
                        logger.warning("[web_fallback] %d 个单曲播放链接补偿请求失败", failed_retries)

                for sid, is_trial in trial_info_map.items():
                    if is_trial:
                        logger.warning(f"[web_fallback] 歌曲 {sid} 为 30s 试听版")

                # 4) 组装结果 —— 必须包含 preview_url (前端播放用) + cover_url
                results = []
                for s in songs[:target_count]:
                    sid = str(s["id"])
                    title = s.get("name", "Unknown")
                    artists = [a["name"] for a in s.get("artists", [])]
                    artist_str = "、".join(artists)

                    # 从 detail 获取封面
                    detail = detail_map.get(sid, {})
                    cover_url = (detail.get("al", {}).get("picUrl", "")
                                 or s.get("album", {}).get("picUrl", ""))
                    album = detail.get("al", {}).get("name", "") or s.get("album", {}).get("name", "")

                    play_url = play_url_map.get(sid, "")
                    is_trial = trial_info_map.get(sid, False)

                    results.append({
                        "song": {
                            "title": title,
                            "artist": artist_str,
                            "album": album,
                            "song_id": sid,
                            "preview_url": play_url,   # 前端用 preview_url 播放
                            "audio_url": play_url,      # 兼容
                            "cover_url": cover_url,
                            "source": "online_search",
                            "recall_sources": ["web"],
                            "recall_source_labels": ["联网"],
                            "platform": "netease",
                            "is_trial": is_trial,       # 标记是否 30s 试听
                            "language": s.get("_inferred_language"),
                            "web_evidence": s.get("_discovery_evidence", ""),
                            "web_discovery_query": s.get("_discovery_query", ""),
                        }
                    })

            matched = sum(1 for r in results if r["song"]["preview_url"])
            trial_count = sum(1 for r in results if r["song"].get("is_trial"))
            logger.info(f"[web_fallback] 联网返回 {len(results)} 首歌曲，{matched} 首可播放，{trial_count} 首为试听版")

            final_results = results
            final_meta = _web_meta(len(results))
            if web_action == "mix_in":
                local_items = unwrap_recommendation_items(state.get("recommendations", []))
                if local_items:
                    final_results = interleave_online_results(
                        local_items,
                        results,
                        target_len=len(local_items),
                    )
                    final_meta = {
                        **final_meta,
                        "source": "local+web",
                        "result_count": len(final_results),
                        "online_result_count": len(results),
                        "local_result_count": len(local_items),
                        "degraded": False,
                    }

            auto_flywheel_candidates = 0
            try:
                from services.online_audio_flywheel import schedule_online_recommendation_flywheel

                auto_flywheel_candidates = schedule_online_recommendation_flywheel(final_results)
            except Exception as exc:
                logger.debug("[web_fallback] online auto flywheel scheduling skipped: %s", exc)
            if auto_flywheel_candidates:
                final_meta = {
                    **final_meta,
                    "online_auto_flywheel_candidates": auto_flywheel_candidates,
                    "online_audio_retention": "temporary_until_user_save",
                }
            _schedule_recommended_knowledge_backfill(final_results, context="web_fallback")

            from schemas.music_state import ToolOutput
            dialog_state = update_dialog_result_anchors(state.get("dialog_state"), final_results)
            return {
                "search_results": [r.get("song", r) for r in final_results],
                "recommendations": ToolOutput(
                    success=True,
                    data=final_results,
                    raw_markdown="",
                ),
                "_need_web_fallback": False,
                "retrieval_meta": final_meta,
                "tool_observations": _web_tool_observations("success", len(results)),
                "dialog_state": dialog_state.model_dump(),
                "step_count": state.get("step_count", 0) + 1,
                "timings": _record_timing(state, "web_fallback_ms", _time.time() - _t0),
            }

        except Exception as e:
            logger.error(f"[web_fallback] 联网搜索失败: {e}")
            preserved = _local_preserve_payload("web_search_error")
            if preserved:
                return preserved
            return {
                "search_results": [], "recommendations": [],
                "_need_web_fallback": False,
                "retrieval_meta": _web_meta(0, "web_search_error"),
                "tool_observations": _web_tool_observations("error", 0, str(e)),
                "step_count": state.get("step_count", 0) + 1,
                "timings": _record_timing(state, "web_fallback_ms", _time.time() - _t0),
            }

    async def acquire_online_music_node(self, state: MusicAgentState) -> Dict[str, Any]:
        """
        节点：下载音频/歌词/封面到本地待入库目录。
        不写入 Neo4j，用户需在前端待入库页面确认后才入库。
        """
        logger.info("--- [步骤] 联网获取音乐（下载到待入库）---")

        parameters = state.get("intent_parameters", {})
        # song_queries 从 parameters 中取，LLM 应该填入类似 ["歌名 歌手", ...]
        song_queries = parameters.get("song_queries", [])

        # 如果 LLM 没有提供 song_queries，按优先级从其他字段提取
        if not song_queries:
            # 优先从 graph_entities 提取（LLM 识别到的歌手/歌名实体，最干净）
            retrieval_plan = state.get("retrieval_plan") or {}
            graph_entities = retrieval_plan.get("graph_entities", [])
            entities = parameters.get("entities", [])

            if graph_entities:
                song_queries = [" ".join(graph_entities[:2])]
                logger.info(f"[acquire] 从 graph_entities 提取搜索词: {song_queries}")
            elif entities:
                song_queries = [" ".join(entities[:2])]
                logger.info(f"[acquire] 从 entities 提取搜索词: {song_queries}")
            else:
                # 最后兜底：清洗掉动作动词后使用 query
                import re
                raw_query = parameters.get("query", state.get("input", ""))
                if raw_query:
                    clean = re.sub(
                        r'^(帮我|请帮我|帮忙|麻烦|能不能|可以)?\s*'
                        r'(下载|获取|帮我下载|帮我获取|下载获取|搜索|找一下|找到)\s*',
                        '', raw_query, flags=re.IGNORECASE
                    ).strip()
                    clean = re.sub(r'(这首歌|这首歌曲|歌曲|这首)[\s。.]*$', '', clean).strip()
                    song_queries = [clean] if clean else [raw_query]
                    logger.info(f"[acquire] 清洗后搜索词: {raw_query!r} → {song_queries}")

        if not song_queries:
            return {
                "recommendations": ToolOutput(
                    success=False,
                    data=[],
                    raw_markdown="❌ 未指定要获取的歌曲名称",
                    error_message="No song queries",
                ),
                "step_count": state.get("step_count", 0) + 1,
            }

        try:
            result = await acquire_online_music.ainvoke({"song_queries": song_queries})
            return {
                "recommendations": result,
                "step_count": state.get("step_count", 0) + 1,
            }
        except Exception as e:
            logger.error(f"联网获取音乐失败: {str(e)}")
            return {
                "error_log": state.get("error_log", []) + [
                    {"node": "acquire_online_music", "error": str(e)}
                ],
                "step_count": state.get("step_count", 0) + 1,
            }

    def _build_preference_query(
        self,
        seed_songs: list,
        graphzep_facts: str = "",
        *,
        user_id: str | None = None,
    ) -> str:
        """
        从种子歌曲标签 + 用户 Neo4j 画像 + MemoryGateway 记忆中提炼偏好文本。
        零 LLM 调用，纯结构化数据拼装。
        """
        import re
        tags = set()

        # 1. 从种子歌曲收集标签
        for song_item in seed_songs:
            s = song_item.get("song", {})
            moods = s.get("moods", [])
            themes = s.get("themes", [])
            genre = s.get("genre", "")
            if moods:
                tags.update(m.strip() for m in moods if m and m.strip())
            if themes:
                tags.update(t.strip() for t in themes if t and t.strip())
            if genre:
                # genre 可能是 "Pop/Indie/Driving" 格式
                tags.update(t.strip() for t in genre.replace(",", "/").split("/") if t.strip())

        # 2. 从 Neo4j 用户画像补充（行为推导的偏好）
        try:
            from retrieval.user_memory import UserMemoryManager
            mem = UserMemoryManager()
            profile = mem.get_user_preferences(user_id or settings.default_user_id)
            if profile:
                for g in profile.get("favorite_genres", []):
                    if g:
                        tags.add(g.strip())
                mood_tendency = profile.get("mood_tendency", "")
                if mood_tendency:
                    tags.update(m.strip() for m in mood_tendency.replace(",", "，").split("，") if m.strip())
        except Exception as e:
            logger.warning(f"[Favorites] 加载行为画像失败: {e}")

        # 2b. 从用户画像面板的显式设置补充（preferred_genres / preferred_moods）
        try:
            from retrieval.neo4j_client import get_neo4j_client
            import json as _json
            _client = get_neo4j_client()
            if _client and _client.driver:
                _profile_row = _client.execute_query(
                    "MATCH (u:User {id: $uid}) RETURN u.preferred_genres AS pg, u.preferred_moods AS pm",
                    {"uid": user_id or settings.default_user_id}
                )
                if _profile_row and _profile_row[0]:
                    for field in ["pg", "pm"]:
                        raw = _profile_row[0].get(field)
                        if raw:
                            try:
                                parsed = _json.loads(raw)
                                tags.update(t.strip() for t in parsed if t and t.strip())
                            except (ValueError, TypeError):
                                pass
        except Exception as e:
            logger.warning(f"[Favorites] 加载画像标签失败: {e}")

        # 3. 从长期记忆文本提取场景/情绪关键词
        if graphzep_facts and graphzep_facts != "暂无用户长期记忆":
            # 提取场景标签（如"开车"、"深夜"、"学习"）
            scene_matches = re.findall(r'场景[：:]\s*(\S+)', graphzep_facts)
            tags.update(scene_matches)
            # 提取情绪关键词
            mood_matches = re.findall(r'情绪偏好[：:]\s*([^；\n]+)', graphzep_facts)
            for match in mood_matches:
                tags.update(m.strip() for m in match.replace(",", "，").split("，") if m.strip())
            # 提取流派
            genre_matches = re.findall(r'流派[：:]\s*([^；\n]+)', graphzep_facts)
            for match in genre_matches:
                tags.update(g.strip() for g in match.replace(",", "，").split("，") if g.strip())

        # 清理无效标签
        tags.discard("")
        tags.discard("Unknown")
        tags.discard("未知")

        result = " ".join(sorted(tags)) if tags else "relaxing chill indie folk"
        logger.info(f"[Favorites] 偏好标签集合({len(tags)}个): {tags}")
        return result

    async def generate_recommendations_node(self, state: MusicAgentState) -> Dict[str, Any]:
        """
        节点2b: 生成推荐
        根据不同的意图类型调用不同的推荐方法
        """
        logger.info("--- [步骤 2b] 生成音乐推荐 ---")
        
        intent_type = state.get("intent_type")
        parameters = state.get("intent_parameters", {})
        user_id = _state_user_id(state)
        
        try:
            # ── 特殊意图：recommend_by_favorites（两层智能推荐）──
            if intent_type == "recommend_by_favorites":
                logger.info("检测到 recommend_by_favorites 意图，启动两层智能推荐")
                memory = UserMemoryManager()
                if not settings.eval_disable_side_effects:
                    memory.ensure_user_exists(user_id)
                all_liked = memory.get_liked_songs(user_id=user_id, limit=20)

                if not all_liked:
                    logger.info("用户暂无点赞/收藏记录，退回常规推荐")
                    # fallthrough 到常规检索
                else:
                    from config.settings import settings as _fav_settings

                    # ── Tier 1: Seeds（可播放的收藏歌曲，最多 N 首）──
                    seed_limit = _fav_settings.favorites_seed_limit
                    discovery_limit = _fav_settings.favorites_discovery_limit
                    playable_seeds = [
                        s for s in all_liked
                        if s.get("song", {}).get("audio_url")
                    ][:seed_limit]
                    logger.info(f"[Favorites] Seeds: {len(playable_seeds)} 首可播放收藏 (总收藏 {len(all_liked)})")

                    # ── 构建偏好查询文本（零 LLM 调用）──
                    preference_query = self._build_preference_query(
                        seed_songs=playable_seeds or all_liked[:seed_limit],
                        graphzep_facts=state.get("graphzep_facts", ""),
                        user_id=user_id,
                    )
                    logger.info(f"[Favorites] 偏好查询文本: {preference_query}")

                    # ── Tier 2: Discoveries（向量检索发现新歌）──
                    retriever = MusicHybridRetrieval(llm_client=get_llm())
                    discovery_plan = {
                        "use_graph": False,
                        "use_vector": True,
                        "use_web_search": False,
                        "_intent_type": "recommend_by_favorites",
                        "_user_id": user_id,
                        "_graphzep_facts": state.get("graphzep_facts", ""),
                    }
                    discovery_result = await retriever.retrieve(
                        preference_query,
                        limit=discovery_limit + 5,  # 多取一些以备去重
                        precomputed_plan=discovery_plan,
                    )

                    # 排除已在种子中的歌曲
                    seed_titles = {s["song"]["title"] for s in playable_seeds}
                    discoveries = []
                    if discovery_result and discovery_result.success:
                        for item in discovery_result.data:
                            t = item.get("song", {}).get("title", "")
                            if t and t not in seed_titles and t != "🌐 全网资讯补充":
                                item["reason"] = f"基于你的品味发现 🔍 {item.get('reason', '')}"
                                discoveries.append(item)
                            if len(discoveries) >= discovery_limit:
                                break
                    logger.info(f"[Favorites] Discoveries: {len(discoveries)} 首新发现")

                    # ── 合并 Seeds + Discoveries ──
                    for s in playable_seeds:
                        s["reason"] = f"❤️ {s.get('reason', '你喜欢的歌')}"
                    merged = playable_seeds + discoveries

                    # 构建 raw_markdown
                    md_lines = []
                    if playable_seeds:
                        md_lines.append("**🎵 你的收藏**")
                        for i, s in enumerate(playable_seeds, 1):
                            song = s["song"]
                            md_lines.append(f"{i}. **{song['title']}** - {song['artist']}")
                    if discoveries:
                        md_lines.append("")
                        md_lines.append("**🔍 猜你可能喜欢**")
                        for i, d in enumerate(discoveries, len(playable_seeds) + 1):
                            song = d["song"]
                            md_lines.append(f"{i}. **{song['title']}** - {song.get('artist', '未知')}")

                    result = ToolOutput(
                        success=True,
                        data=merged,
                        raw_markdown="\n".join(md_lines),
                    )
                    logger.info(f"[Favorites] 两层推荐完成: Seeds={len(playable_seeds)}, Discoveries={len(discoveries)}, Total={len(merged)}")
                    _schedule_recommended_knowledge_backfill(merged, context="favorites")
                    return {
                        "recommendations": result,
                        "step_count": state.get("step_count", 0) + 1
                    }

            retriever = MusicHybridRetrieval(llm_client=get_llm())
            recommendations = []
            
            # 直接使用用户的原始输入，保留所有的语义和情绪标签（如：带感、激情），而不是使用写死的模板
            search_query = state.get("input", "")
            if not search_query:
                # 兜底：如果意外没有 input，才从意图回退
                search_query = intent_type
                
            logger.info(f"调用检索引擎执行生成推荐: {search_query}")
            
            # 传递上游统一规划的 retrieval_plan，避免二次 LLM 调用
            retrieval_plan = state.get("retrieval_plan")
            raw_hybrid_result = await retriever.retrieve(search_query, limit=settings.hybrid_retrieval_limit, precomputed_plan=retrieval_plan)
            
            # 直接使用标准的 ToolOutput
            if raw_hybrid_result and raw_hybrid_result.success:
                recommendations = raw_hybrid_result.data
            else:
                recommendations = []
                
            logger.info(f"生成了 {len(recommendations)} 条推荐")
            _schedule_recommended_knowledge_backfill(recommendations, context="generate_recommendations")
            
            return {
                "recommendations": raw_hybrid_result if raw_hybrid_result and raw_hybrid_result.success else [], # 完整保存 ToolOutput 对象供解释节点用
                "step_count": state.get("step_count", 0) + 1
            }
            
        except Exception as e:
            logger.error(f"生成推荐失败: {str(e)}")
            return {
                "recommendations": [],
                "step_count": state.get("step_count", 0) + 1,
                "error_log": state.get("error_log", []) + [
                    {"node": "generate_recommendations", "error": str(e)}
                ]
            }
    
    async def general_chat_node(self, state: MusicAgentState) -> Dict[str, Any]:
        """
        节点2c: 通用聊天
        处理一般性的音乐话题聊天
        """
        _main_llm = get_llm()
        _main_model_name = getattr(_main_llm, 'model_name', '?')
        _main_provider = (settings.llm_default_provider or '?').lower()
        logger.info(f"--- [步骤 2c] 通用音乐聊天 | 🤖 {_main_provider} / {_main_model_name} ---")
        
        user_message = state.get("input", "")
        chat_history = state.get("chat_history", [])
        
        try:
            # 格式化对话历史
            context_manager = MusicContextManager()
            history_text = context_manager.format_chat_history(chat_history)
            
            # [LCEL 1.2 优化] 使用 LCEL 链统一调度通用聊天任务
            # StrOutputParser 会自动提取大模型回复消息中的文本内容，省去手动获取 .content。
            chain = (
                ChatPromptTemplate.from_template(MUSIC_CHAT_RESPONSE_PROMPT)
                | get_llm()
                | StrOutputParser()
            )
            # [P3] GSSC Token budget management
            from retrieval.gssc_context_builder import build_context
            _ctx = await build_context(
                graphzep_facts=state.get("graphzep_facts", "暂无用户长期记忆"),
                chat_history=history_text,
                total_budget=0,
                user_id=_state_user_id(state),
            )
            
            response_content = await chain.ainvoke({
                "chat_history": _ctx["chat_history"],
                "user_message": user_message,
                "graphzep_facts": _ctx["graphzep_facts"],
            })
            
            logger.info("生成聊天回复")
            
            # ★ 将回复推送到流式队列，否则 music_agent 的 SSE 会永远卡住
            _req_id = state.get("metadata", {}).get("request_id")
            _chat_queue = self._explanation_queues.get(_req_id) if _req_id else None
            if _chat_queue:
                await _chat_queue.put(response_content)  # 推送完整文本
                await _chat_queue.put(None)              # 终止信号
            
            return {
                "final_response": response_content,
                "step_count": state.get("step_count", 0) + 1
            }
            
        except Exception as e:
            logger.error(f"生成聊天回复失败: {str(e)}")
            # 也要推送终止信号，否则异常时也会卡住
            _req_id = state.get("metadata", {}).get("request_id")
            _err_queue = self._explanation_queues.get(_req_id) if _req_id else None
            if _err_queue:
                try:
                    await _err_queue.put(None)
                except Exception:
                    pass
            return {
                "final_response": "抱歉，我现在遇到了一些问题。不过我很乐意和你聊音乐！你可以告诉我你喜欢什么类型的音乐吗？",
                "step_count": state.get("step_count", 0) + 1,
                "error_log": state.get("error_log", []) + [
                    {"node": "general_chat", "error": str(e)}
                ]
            }
    
    async def generate_explanation(self, state: MusicAgentState) -> Dict[str, Any]:
        """
        节点3: 生成推荐解释
        为搜索结果或推荐结果生成友好的解释文本
        """
        import time as _time
        _t0 = _time.time()

        # 兼容处理 ToolOutput 对象或列表
        raw_recommendations = state.get("recommendations", [])
        recommendations = getattr(raw_recommendations, "data", raw_recommendations)
        
        user_query = state.get("input", "")
        request_id = state.get("metadata", {}).get("request_id", "")
        explanation_queue = self._explanation_queues.get(request_id) if request_id else None

        async def _push_song_cards() -> None:
            if not explanation_queue or not recommendations:
                return
            songs_payload = []
            for i, rec in enumerate(recommendations):
                song = rec.get("song", rec) if isinstance(rec, dict) else rec
                if isinstance(song, dict) and song.get("title"):
                    songs_payload.append({"song": song, "index": i})
            if songs_payload:
                await explanation_queue.put({"__songs__": songs_payload})

        async def _finish_queue(response: str = "") -> None:
            if not explanation_queue:
                return
            if response:
                await explanation_queue.put(response)
            await explanation_queue.put(None)
        
        # 判断是否有真实内容
        has_real_content = False
        if recommendations:
            if hasattr(raw_recommendations, "success"): # ToolOutput instance
               has_real_content = len(recommendations) > 0
            else:
                has_real_content = any(
                    isinstance(r, dict) and ("_raw_markdown" in r or r.get("song", {}).get("title", "") not in ["", "🌐 全网资讯补充"])
                    for r in recommendations
                )
                
        if not recommendations or not has_real_content:
            logger.warning("没有推荐结果，跳过解释生成")
            retrieval_meta = state.get("retrieval_meta") or {}
            catalog_gap = retrieval_meta.get("catalog_gap") or state.get("_catalog_gap") or {}
            gap_message = catalog_gap.get("message") if isinstance(catalog_gap, dict) else ""
            response = (
                gap_message
                if retrieval_meta.get("web_search_blocked") and gap_message
                else "抱歉，没有找到符合你要求的音乐。你可以换个方式描述你的需求，或者告诉我你喜欢的歌手和风格？"
            )
            await _finish_queue(response)
            return {
                "explanation": "抱歉，没有找到合适的音乐推荐。",
                "final_response": response,
                "step_count": state.get("step_count", 0) + 1,
                "timings": _record_timing(state, "explanation_ms", _time.time() - _t0),
            }

        if os.getenv("MUSIC_MOCK_MODE", "0").lower() in {"1", "true", "yes"}:
            response = "Mock 模式推荐已完成，检索、路由与流式响应链路工作正常。"
            await _push_song_cards()
            await _finish_queue(response)
            return {
                "explanation": response,
                "final_response": response,
                "step_count": state.get("step_count", 0) + 1,
                "timings": _record_timing(state, "explanation_ms", _time.time() - _t0),
            }

        explanation_mode = str(getattr(settings, "explanation_mode", "tuner_async") or "tuner_async").strip().lower()
        if settings.explanation_fast_mode:
            explanation_mode = "off"

        try:
            await _push_song_cards()
        except Exception as e:
            logger.warning(f"推送歌曲到队列失败: {e}")

        if explanation_mode in {"off", "none", "disabled", "fast"}:
            response = await emit_fast_explanation(recommendations, None)
            await _finish_queue(response)
            logger.info("[Explanation] mode=off 跳过解释 LLM")
            return {
                "explanation": response,
                "final_response": response,
                "step_count": state.get("step_count", 0) + 1,
                "timings": _record_timing(state, "explanation_ms", _time.time() - _t0),
            }

        _explain = get_explain_llm()
        _explain_model_name = getattr(_explain, 'model_name', '?')
        _explain_provider = (settings.explain_llm_provider or settings.llm_default_provider or '?').lower()
        logger.info(
            f"--- [步骤 3] 生成推荐文本({explanation_mode}) | 🤖 "
            f"{_explain_provider} / {_explain_model_name} ---"
        )
        
        try:
            # 格式化推荐结果 (ToolOutput 已提供 raw_markdown)
            songs_text = ""
            if hasattr(raw_recommendations, "raw_markdown"):
                songs_text = getattr(raw_recommendations, "raw_markdown", "")
                
                # ✅ 推荐结果已通过 raw_markdown 传递，无需额外处理
                # 注意：不在这里记录“收听”历史，推荐 ≠ 收听，应由前端播放时触发
            else:
                # 兼容旧代码分支
                for i, rec in enumerate(recommendations, 1):
                    # 兼容旧的方法
                    if isinstance(rec, dict) and "_raw_markdown" in rec:
                        # 如果是由 search_songs_node 直接返回的检索引擎 markdown
                        songs_text += f"\n【检索详情报告 {i}】\n{rec['_raw_markdown']}\n"
                        continue
                        
                    song = rec.get("song", rec)  # 可能是搜索结果或推荐结果
                    
                    # 如果是 enhanced_recommendations 或 generate_recommendations 的检索结果
                    reason = rec.get("reason", "")
                    if reason and "混合引擎检索报告" in reason:
                        songs_text += f"\n【混合 RAG 综合分析】\n{reason}\n"
                        continue
                        
                    title = song.get("title", "未知") if isinstance(song, dict) else getattr(song, "title", "未知")
                    artist = song.get("artist", "未知") if isinstance(song, dict) else getattr(song, "artist", "未知")
                    genre = song.get("genre", "未知") if isinstance(song, dict) else getattr(song, "genre", "未知")
                    
                    
                    # ✅ 不在推荐阶段记录“收听”历史，等用户实际播放时再记录
                    
                    songs_text += f"{i}. 《{title}》 - {artist} ({genre})\n"
                    if reason:
                        songs_text += f"   推荐理由: {reason}\n"
            
            if explanation_mode == "song_detail":
                prompt_template = MUSIC_RECOMMENDATION_EXPLAINER_PROMPT
                prompt_payload = {
                    "user_query": user_query,
                    "recommended_songs": songs_text,
                }
            else:
                if explanation_mode != "tuner_async":
                    logger.warning(f"[Explanation] 未知模式 {explanation_mode!r}，降级为 tuner_async")
                retrieval_plan = state.get("retrieval_plan", {}) or {}
                refinement_options = state.get("refinement_options", []) or []
                prompt_template = MUSIC_TUNER_RESPONSE_PROMPT
                prompt_payload = {
                    "user_query": user_query,
                    "intent_context": state.get("intent_context", ""),
                    "retrieval_plan": json.dumps(retrieval_plan, ensure_ascii=False, default=str)[:2400],
                    "recommendation_overview": _build_tuner_recommendation_overview(recommendations),
                    "refinement_options": json.dumps(refinement_options, ensure_ascii=False, default=str),
                }

            chain = ChatPromptTemplate.from_template(prompt_template) | get_explain_llm() | StrOutputParser()
            
            explanation = ""
            async for chunk in chain.astream(prompt_payload):
                explanation += chunk
                if explanation_queue:
                    try:
                        await explanation_queue.put(chunk)
                    except Exception:
                        pass
            
            # 通知队列流式结束
            if explanation_queue:
                try:
                    await explanation_queue.put(None)  # 哨兵值
                except Exception:
                    pass
            
            # 构建完整的最终回复
            final_response = explanation
            
            logger.info(f"成功生成推荐文本({explanation_mode}), 耗时 {_time.time()-_t0:.1f}s")
            try:
                from services.teacher_log import log_teacher_example

                log_teacher_example(
                    "explain",
                    inputs={
                        "mode": explanation_mode,
                        "prompt_payload": prompt_payload,
                    },
                    output={"final_response": final_response},
                    metadata={
                        "provider": _explain_provider,
                        "model": _explain_model_name,
                        "prompt_version": f"explain_{explanation_mode}_2026_07_10",
                    },
                )
            except Exception:
                pass
            
            # 偏好提取已解耦为独立节点 extract_preferences_node
            
            return {
                "explanation": explanation,
                "final_response": final_response,
                "step_count": state.get("step_count", 0) + 1,
                "timings": _record_timing(state, "explanation_ms", _time.time() - _t0),
            }
            
        except Exception as e:
            logger.error(f"生成解释失败: {str(e)}")
            
            # 确保队列收到终止信号，防止前端消费者永久阻塞
            if explanation_queue:
                try:
                    await explanation_queue.put(None)
                except Exception:
                    pass
            
            # 生成简单的备用回复
            songs_list = "\n".join([
                f"{i}. 《{rec.get('song', rec).get('title', '未知')}》 - {rec.get('song', rec).get('artist', '未知')}"
                for i, rec in enumerate(recommendations, 1)
            ])
            
            return {
                "explanation": "为你找到了以下歌曲：",
                "final_response": f"为你找到了以下歌曲：\n\n{songs_list}",
                "step_count": state.get("step_count", 0) + 1,
                "error_log": state.get("error_log", []) + [
                    {"node": "generate_explanation", "error": str(e)}
                ],
                "timings": _record_timing(state, "explanation_ms", _time.time() - _t0),
            }

    async def analyze_user_preferences_node(self, state: MusicAgentState) -> Dict[str, Any]:
        """
        节点: 分析用户偏好 ⭐ NEW
        从 Neo4j 图谱记忆中获取用户偏好数据
        """
        logger.info("--- [步骤] 分析用户偏好 ---")
        
        try:
            from schemas.music_state import UserPreferences
            
            # 目前系统是一个单用户/本地演示型系统，默认给定一个 userID
            user_id = _state_user_id(state)
            
            logger.info("向 Neo4j 查询本地用户图谱记忆...")
            memory_manager = UserMemoryManager()
            
            # 真实请求可初始化用户；评测必须保持数据库只读。
            if not settings.eval_disable_side_effects:
                memory_manager.ensure_user_exists(user_id, "本地用户")
            
            # 读取历史偏好
            graph_prefs = memory_manager.get_user_preferences(user_id, limit=settings.user_preference_limit)
            
            favorite_artists = graph_prefs.get("favorite_artists", [])
            favorite_genres = graph_prefs.get("favorite_genres", [])
            
            # 此处获取的 favorite_songs 只是 title 数组
            favorite_songs_titles = graph_prefs.get("favorite_songs", [])
            
            # 为了适配下方的推荐流，将纯字符串简单封装一下
            top_tracks_mock = [{"title": t, "artist": "未知", "genre": "未知"} for t in favorite_songs_titles]
            
            # 默认偏好只服务首次本地体验，不得污染独立评测用户。
            if not favorite_artists and not settings.eval_disable_side_effects:
                favorite_artists = ["周杰伦", "林俊杰"]
            if not favorite_genres and not settings.eval_disable_side_effects:
                favorite_genres = ["Pop", "R&B"]
            if not top_tracks_mock and not settings.eval_disable_side_effects:
                top_tracks_mock = [
                    {"title": "七里香", "artist": "周杰伦", "genre": "Pop"},
                    {"title": "夜曲", "artist": "周杰伦", "genre": "R&B"}
                ]
            
            favorite_decades = ["2000s"]
            
            preferences: UserPreferences = {
                "favorite_genres": favorite_genres,
                "favorite_artists": favorite_artists,
                "favorite_decades": favorite_decades,
                "avoid_genres": [],
                "mood_preferences": [],
                "activity_contexts": [],
                "language_preference": "mixed"
            }
            
            logger.info(f"分析完成: 偏好流派={favorite_genres}, 偏好艺术家={favorite_artists[:3]}")
            
            return {
                "user_preferences": preferences,
                "favorite_songs": top_tracks_mock,
                "step_count": state.get("step_count", 0) + 1
            }
            
        except Exception as e:
            logger.error(f"分析用户偏好失败: {str(e)}", exc_info=True)
            # 如果失败，返回空偏好，继续执行
            return {
                "user_preferences": {},
                "favorite_songs": [],
                "step_count": state.get("step_count", 0) + 1,
                "error_log": state.get("error_log", []) + [
                    {"node": "analyze_user_preferences", "error": str(e)}
                ]
            }
    
    async def enhanced_recommendations_node(self, state: MusicAgentState) -> Dict[str, Any]:
        """
        节点: 增强推荐 ⭐ NEW
        结合用户偏好生成推荐
        """
        logger.info("--- [步骤] 生成增强推荐 ---")
        
        try:
            # 去除了对 MCP Adapter 的依赖
            user_preferences = state.get("user_preferences", {})
            parameters = state.get("intent_parameters", {})
            
            recommendations = []
            
            # 根据意图类型生成推荐
            if intent_type.startswith("create_playlist"):
                # 创建歌单：结合用户偏好和意图参数
                activity = parameters.get("activity", "")
                mood = parameters.get("mood", "")
                
                # 使用用户 top tracks 作为种子
                favorite_songs = state.get("favorite_songs", [])
                seed_tracks = []
                if favorite_songs:
                    for song in favorite_songs[:5]:
                        if isinstance(song, dict) and song.get("spotify_id"):
                            seed_tracks.append(song["spotify_id"])
                
                # 使用用户偏好流派
                favorite_genres = user_preferences.get("favorite_genres", [])
                seed_genres = favorite_genres[:3] if favorite_genres else ["pop"]
                
                # 如果指定了活动或心情，调整流派
                if activity:
                    activity_genre_map = {
                        "运动": ["electronic", "rock"],
                        "健身": ["electronic", "rock"],
                        "学习": ["acoustic", "jazz"],
                        "工作": ["acoustic", "jazz"],
                    }
                    for key, genres in activity_genre_map.items():
                        if key in activity:
                            seed_genres = genres[:3]
                            break
                
                # 使用本地检索系统获取推荐 (替代原 Spotify 调用)
                retriever = MusicHybridRetrieval(llm_client=get_llm())
                query = f"流派:{','.join(seed_genres)} 活动:{activity} 心情:{mood}"
                
                logger.info(f"调用检索引擎进行增强推荐: {query}")
                raw_hybrid_result = await retriever.retrieve(query, limit=settings.graph_search_limit)
                
                # 直接扩展到推荐列表
                recommendations.extend(raw_hybrid_result)
            else:
                # 其他推荐类型，走统一检索管线
                retriever = MusicHybridRetrieval(llm_client=get_llm())
                fallback_query = state.get("input", intent_type)
                logger.info(f"调用检索引擎进行增强推荐(fallback): {fallback_query}")
                raw_hybrid_result = retriever.retrieve(fallback_query, limit=settings.graph_search_limit)
                if raw_hybrid_result and hasattr(raw_hybrid_result, 'data'):
                    recommendations = raw_hybrid_result.data if raw_hybrid_result.data else []
                else:
                    recommendations = []
            
            logger.info(f"生成了 {len(recommendations)} 条增强推荐")
            _schedule_recommended_knowledge_backfill(recommendations, context="enhanced_recommendations")
            
            return {
                "recommendations": recommendations,
                "step_count": state.get("step_count", 0) + 1
            }
            
        except Exception as e:
            logger.error(f"生成增强推荐失败: {str(e)}", exc_info=True)
            return {
                "recommendations": [],
                "step_count": state.get("step_count", 0) + 1,
                "error_log": state.get("error_log", []) + [
                    {"node": "enhanced_recommendations", "error": str(e)}
                ]
            }
    
    def route_after_preferences(self, state: MusicAgentState) -> str:
        """
        路由函数: 分析用户偏好后的路由
        """
        intent_type = state.get("intent_type", "")
        if intent_type.startswith("create_playlist"):
            return "enhanced_recommendations"
        else:
            return "generate_recommendations"
    
    async def create_playlist_node(self, state: MusicAgentState) -> Dict[str, Any]:
        """
        节点: 创建播放列表 ⭐ NEW
        """
        logger.info("--- [步骤] 创建播放列表 ---")
        
        try:
            # 彻底摒弃 Spotify 建单功能
            # 直接将现有 recommendation 格式化打包返回给前端即可
            
            # 获取推荐结果
            recommendations = state.get("recommendations", [])
            if not recommendations:
                logger.warning("没有推荐结果，无法创建播放列表")
                return {
                    "playlist": None,
                    "step_count": state.get("step_count", 0) + 1,
                    "error_log": state.get("error_log", []) + [
                        {"node": "create_playlist", "error": "没有推荐结果"}
                    ]
                }
            
            # 提取歌曲
            songs = []
            for rec in recommendations:
                song_data = rec.get("song", rec)
                if isinstance(song_data, dict):
                    title = str(song_data.get("title") or "").strip()
                    if title and title != "未知" and "集合" not in title:
                        songs.append(
                            {
                                "title": title,
                                "artist": str(song_data.get("artist") or "未知").strip(),
                                "album": song_data.get("album"),
                                "genre": song_data.get("genre"),
                                "year": song_data.get("year"),
                                "duration": song_data.get("duration"),
                                "preview_url": song_data.get("preview_url") or song_data.get("audio_url"),
                            }
                        )
                    
            
            if not songs:
                logger.warning("无法提取歌曲信息")
                return {
                    "playlist": None,
                    "step_count": state.get("step_count", 0) + 1
                }
            
            # 生成播放列表名称和描述
            intent_type = state.get("intent_type", "")
            parameters = state.get("intent_parameters", {})
            
            if "activity" in parameters:
                playlist_name = f"适合{parameters['activity']}的歌单"
                description = f"AI 为你推荐的适合{parameters['activity']}时听的音乐"
            elif "mood" in parameters:
                playlist_name = f"{parameters['mood']}心情歌单"
                description = f"AI 为你推荐的适合{parameters['mood']}心情的音乐"
            else:
                playlist_name = "AI 推荐歌单"
                description = "AI 为你推荐的个性化音乐歌单"
            
            # 创建播放列表 (已停用 Spotify API)
            # 由于已封锁 Spotify，直接返回本地生成的虚拟播放列表结构
            playlist_dict = {
                "id": "local_playlist_123",
                "name": playlist_name,
                "url": "local_only",
                "description": description,
                "track_count": len(songs)
            }
            
            logger.info(f"本地虚拟播放列表创建成功: {playlist_name}")
            return {
                "playlist": playlist_dict,
                "step_count": state.get("step_count", 0) + 1
            }
                
        except Exception as e:
            logger.error(f"创建播放列表失败: {str(e)}", exc_info=True)
            return {
                "playlist": None,
                "step_count": state.get("step_count", 0) + 1,
                "error_log": state.get("error_log", []) + [
                    {"node": "create_playlist", "error": str(e)}
                ]
            }
    
    def route_after_recommendations(self, state: MusicAgentState) -> str:
        """
        路由函数: 生成推荐后的路由
        """
        intent_type = state.get("intent_type", "")
        if intent_type.startswith("create_playlist"):
            return "create_playlist"
        else:
            return "generate_explanation"
    
    async def recall_graphzep_memory(self, state: MusicAgentState) -> Dict[str, Any]:
        """
        MemoryGateway 长期记忆召回。

        节点名保持 recall_graphzep_memory 以兼容既有 LangGraph 拓扑，
        但内部已经不直接依赖 GraphZep。Neo4j 用户画像是热路径；
        GraphZep/Mem0 只是可选 episodic sidecar。
        """
        import time as _time
        _t0 = _time.time()
        if os.getenv("MUSIC_MOCK_MODE", "0").lower() in {"1", "true", "yes"}:
            return {
                "graphzep_facts": "",
                "graphzep_group_id": "mock",
                "timings": _record_timing(state, "graphzep_ms", _time.time() - _t0),
            }
        logger.info("--- [MemoryGateway] 长期记忆召回 ---")

        memory_total_timeout = max(0.5, float(settings.graphzep_total_timeout_seconds))

        async def _do_recall() -> Dict[str, Any]:
            user_input = state.get("input", "")
            user_id = _state_user_id(state)
            from services.memory_gateway import get_memory_gateway

            # 场景来自上一轮结构化 dialog_state（planner 尚未运行），
            # 原样作为自由文本传入，检索侧用语义相关性判断场景适用性。
            _prev_dialog = state.get("dialog_state") or {}
            _prev_hints = _prev_dialog.get("hints") if isinstance(_prev_dialog, dict) else {}
            scene = str((_prev_hints or {}).get("scenario") or "").strip()

            context = await get_memory_gateway().retrieve_context(
                query=user_input,
                user_id=user_id,
                max_facts=8,
                scene=scene,
            )
            facts = context.get("episodic") or "暂无用户长期记忆"
            logger.info(
                "[MemoryGateway] 召回完成: hot_profile=%s, sidecars=%s, chars=%s",
                bool(context.get("profile")),
                list((context.get("episodic_backends") or {}).keys()),
                len(facts),
            )
            return {
                "graphzep_facts": facts,
                "memory_context": context,
            }

        try:
            result = await asyncio.wait_for(_do_recall(), timeout=memory_total_timeout)
            _elapsed = _time.time() - _t0
            logger.info(f"[MemoryGateway] ✅ 记忆召回完成, 总耗时 {_elapsed:.1f}s")
            return {
                **result,
                "timings": _record_timing(state, "graphzep_ms", _elapsed),
            }
        except asyncio.TimeoutError:
            _elapsed = _time.time() - _t0
            logger.warning(
                f"[MemoryGateway] ⚠️ 记忆召回超时 ({_elapsed:.1f}s > {memory_total_timeout}s)，"
                f"降级为空记忆以保证推荐流程不阻塞"
            )
            return {
                "graphzep_facts": "暂无用户长期记忆",
                "timings": _record_timing(state, "graphzep_ms", _elapsed),
            }
        except Exception as e:
            logger.warning(f"[MemoryGateway] 记忆召回失败（降级为空）: {e}")
            return {
                "graphzep_facts": "暂无用户长期记忆",
                "timings": _record_timing(state, "graphzep_ms", _time.time() - _t0),
            }

    async def extract_preferences_node(self, state: MusicAgentState) -> Dict[str, Any]:
        """Record user-only evidence and debounce LLM memory consolidation."""
        logger.info("--- [MemoryV2] 记录用户证据并检查记忆归纳 ---")

        if settings.eval_disable_side_effects:
            logger.info("[EvalMode] 跳过记忆证据写入与 GSSC 预压缩")
            return {}

        user_query = state.get("input", "")
        user_id = _state_user_id(state)
        raw_recommendations = state.get("recommendations", [])
        recommendations = getattr(raw_recommendations, "data", raw_recommendations)

        if not user_query:
            logger.info("[MemoryV2] 无用户输入，跳过证据写入")
            return {}

        try:
            from datetime import datetime as _dt
            from services.memory_gateway import get_memory_gateway

            retrieval_plan = state.get("retrieval_plan", {})
            scene_ctx = (
                getattr(retrieval_plan, "graph_scenario_filter", None)
                or (retrieval_plan.get("graph_scenario_filter") if isinstance(retrieval_plan, dict) else None)
                or "未知"
            )

            hour = _dt.now().hour
            if hour < 6:
                time_label = "凌晨"
            elif hour < 9:
                time_label = "早晨"
            elif hour < 12:
                time_label = "上午"
            elif hour < 14:
                time_label = "中午"
            elif hour < 18:
                time_label = "下午"
            elif hour < 21:
                time_label = "傍晚"
            else:
                time_label = "深夜"

            rec_titles: list[str] = []
            for item in recommendations[:5] if isinstance(recommendations, list) else []:
                song = item.get("song", item) if isinstance(item, dict) else {}
                title = str(song.get("title") or "").strip()
                if title:
                    rec_titles.append(title)
            result = get_memory_gateway().remember_conversation_evidence(
                user_id=user_id,
                user_text=user_query,
                scene=scene_ctx,
                time_label=time_label,
                recommended_songs=rec_titles,
            )
            logger.info(
                "[MemoryV2] 用户证据已记录，consolidation_scheduled=%s",
                result.consolidation_scheduled,
            )

            try:
                from retrieval.gssc_context_builder import pre_compress_and_cache
                from retrieval.history import MusicContextManager as _HisMgr
                _ctx_mgr = _HisMgr()
                _raw_history = state.get("chat_history", [])
                _history_str = _ctx_mgr.format_chat_history(_raw_history)
                asyncio.create_task(pre_compress_and_cache(user_id, _history_str))
                logger.info("[GSSC-Cache] 历史预压缩任务已投递到后台")
            except Exception as _cache_e:
                logger.warning(f"[GSSC-Cache] 投递预压缩任务失败（不影响主流程）: {_cache_e}")
        except Exception as pref_e:
            logger.warning(f"[MemoryV2] 用户证据写入异常（不影响主流程）: {pref_e}")

        return {}

    async def persist_to_graphzep(self, state: MusicAgentState) -> Dict[str, Any]:
        """
        出口旁路节点：将本轮完整对话异步送入 MemoryGateway sidecars。

        函数名保持 persist_to_graphzep 以兼容现有图结构；实际后端由
        MEMORY_EPISODIC_BACKENDS 配置控制，可为 graphzep/mem0/noop。
        """
        logger.info("--- [MemoryGateway] 异步持久化对话 ---")

        if settings.eval_disable_side_effects:
            logger.info("[EvalMode] 跳过长期记忆持久化与画像刷新")
            return {}
        
        user_input = state.get("input", "")
        bot_response = state.get("final_response", "")
        user_id = _state_user_id(state)
        
        if not user_input or not bot_response:
            return {}
        
        try:
            from datetime import datetime as _dt
            from services.memory_gateway import get_memory_gateway

            # 携带场景上下文，让长期记忆后端可提取出带场景的事实
            retrieval_plan = state.get("retrieval_plan", {})
            scene_ctx = (
                getattr(retrieval_plan, "graph_scenario_filter", None)
                or (retrieval_plan.get("graph_scenario_filter") if isinstance(retrieval_plan, dict) else None)
                or ""
            )
            hour = _dt.now().hour
            time_label = "凌晨" if hour < 6 else "早晨" if hour < 9 else "上午" if hour < 12 else "中午" if hour < 14 else "下午" if hour < 18 else "傍晚" if hour < 21 else "深夜"
            
            # 将场景标签注入用户消息，让长期记忆后端提取事实时能感知场景
            enriched_user_msg = user_input
            if scene_ctx:
                enriched_user_msg = f"[场景: {scene_ctx} | 时间: {time_label}] {user_input}"

            description = f"用户: {enriched_user_msg}\n助手: {bot_response}"
            asyncio.create_task(
                get_memory_gateway().remember_text(
                    description=description,
                    user_id=user_id,
                    extra={
                        "source": "agent_dialog",
                        "scene": scene_ctx,
                        "time": time_label,
                        "user_text": enriched_user_msg,
                    },
                )
            )
            logger.info(f"[MemoryGateway] 对话已投递到长期记忆旁路 (scene={scene_ctx or '无'})")
            
        except Exception as e:
            logger.warning(f"[MemoryGateway] 持久化投递失败（不影响用户）: {e}")
        
        # ★ Profile Synthesizer: 对话计数 + 自动触发画像刷新
        try:
            from services.profile_synthesizer import get_profile_synthesizer, trigger_portrait_refresh
            synth = get_profile_synthesizer(user_id)
            if synth.increment_conversation():
                logger.info("[ProfileSynth] 达到刷新阈值，后台异步刷新用户画像...")
                asyncio.create_task(trigger_portrait_refresh(user_id))
        except Exception as synth_err:
            logger.warning(f"[ProfileSynth] 画像刷新触发失败（不影响主流程）: {synth_err}")
        
        return {}

    def _build_graph(self) -> CompiledStateGraph:
        """构建工作流图"""
        logger.info("开始构建音乐推荐工作流图...")
        
        workflow = StateGraph(MusicAgentState)
        
        # ==== MemoryGateway 记忆节点（节点名保留 graphzep 兼容旧拓扑）====
        workflow.add_node("recall_graphzep_memory", self.recall_graphzep_memory)
        workflow.add_node("persist_to_graphzep", self.persist_to_graphzep)
        
        # ==== 偏好提取节点（从 generate_explanation 解耦） ====
        workflow.add_node("extract_preferences", self.extract_preferences_node)
        
        # 添加节点
        workflow.add_node("analyze_intent", self.analyze_intent)
        workflow.add_node("acquire_online_music", self.acquire_online_music_node)  # 数据飞轮
        workflow.add_node("search_songs", self.search_songs_node)
        workflow.add_node("web_fallback", self.web_fallback_node)  # 本地未命中 → 联网降级
        workflow.add_node("generate_recommendations", self.generate_recommendations_node)
        workflow.add_node("analyze_user_preferences", self.analyze_user_preferences_node)
        workflow.add_node("enhanced_recommendations", self.enhanced_recommendations_node)
        workflow.add_node("create_playlist", self.create_playlist_node)
        workflow.add_node("general_chat", self.general_chat_node)
        workflow.add_node("clarification", self.clarification_node)
        workflow.add_node("web_disabled", self.web_disabled_node)
        workflow.add_node("generate_explanation", self.generate_explanation)
        
        # 设置入口点为 MemoryGateway 记忆召回
        workflow.set_entry_point("recall_graphzep_memory")
        
        # 召回完成后 → 意图分析
        workflow.add_edge("recall_graphzep_memory", "analyze_intent")
        
        # 条件边：根据意图路由
        workflow.add_conditional_edges(
            "analyze_intent",
            self.route_by_intent,
            {
                "acquire_online_music": "acquire_online_music",
                "search_songs": "search_songs",
                "web_fallback": "web_fallback",  # web_search 意图直达联网搜索
                "generate_recommendations": "generate_recommendations",
                "analyze_user_preferences": "analyze_user_preferences",
                "general_chat": "general_chat",
                "clarification": "clarification",
                "web_disabled": "web_disabled",
            }
        )
        
        # 用户偏好分析后的路由
        workflow.add_conditional_edges(
            "analyze_user_preferences",
            self.route_after_preferences,
            {
                "enhanced_recommendations": "enhanced_recommendations",
                "generate_recommendations": "generate_recommendations"
            }
        )
        
        # 增强推荐后的路由
        workflow.add_conditional_edges(
            "enhanced_recommendations",
            self.route_after_recommendations,
            {
                "create_playlist": "create_playlist",
                "generate_explanation": "generate_explanation"
            }
        )
        
        # 搜索和推荐后生成解释
        workflow.add_edge("acquire_online_music", "generate_explanation")
        # search_songs 后根据是否需要降级联网进行条件路由
        workflow.add_conditional_edges(
            "search_songs",
            self.route_after_search,
            {
                "web_fallback": "web_fallback",
                "generate_explanation": "generate_explanation",
            }
        )
        workflow.add_edge("web_fallback", "generate_explanation")
        workflow.add_edge("generate_recommendations", "generate_explanation")
        
        # 创建播放列表后生成解释
        workflow.add_edge("create_playlist", "generate_explanation")
        
        # ======================================================================
        # 出口管线（V2 解耦版）:
        #   generate_explanation → extract_preferences → persist_to_graphzep → END
        #   general_chat → persist_to_graphzep → END（闲聊无推荐，跳过偏好提取）
        # ======================================================================
        workflow.add_edge("generate_explanation", "extract_preferences")
        workflow.add_edge("extract_preferences", "persist_to_graphzep")
        workflow.add_edge("general_chat", "persist_to_graphzep")
        workflow.add_edge("clarification", "persist_to_graphzep")
        workflow.add_edge("web_disabled", "persist_to_graphzep")
        workflow.add_edge("persist_to_graphzep", END)
        
        # 编译图（注入 checkpointer 实现状态持久化）
        if self.checkpointer:
            app = workflow.compile(checkpointer=self.checkpointer)
            logger.info("音乐推荐工作流图构建完成 (✅ MemorySaver Checkpoint 已启用)")
        else:
            app = workflow.compile()
            logger.info("音乐推荐工作流图构建完成 (⚠️ 无 Checkpoint，每次对话独立)")
        
        return app

