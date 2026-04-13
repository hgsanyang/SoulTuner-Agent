"""
音乐推荐Agent的工作流图
"""

import asyncio
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
    UNIFIED_MUSIC_QUERY_PLANNER_PROMPT,
    LOCAL_PLANNER_PROMPT,
    MUSIC_RECOMMENDATION_EXPLAINER_PROMPT,
    MUSIC_CHAT_RESPONSE_PROMPT
)
from schemas.query_plan import MusicQueryPlan, RetrievalPlan

logger = get_logger(__name__)

# 延迟初始化 llm，避免在模块导入时配置未加载
_llm = None

def get_llm():
    """获取LLM实例（延迟初始化）"""
    global _llm
    if _llm is None:
        _llm = get_chat_model("siliconflow")
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
        self.workflow = self._build_graph()
    
    def get_app(self) -> CompiledStateGraph:
        """获取编译后的应用"""
        return self.workflow
    
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
            structured_llm = _intent_llm.with_structured_output(MusicQueryPlan, include_raw=True)
            
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
        
        try:
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
            
            # ✅ 根据 provider 类型选择提示词模板和 structured output 方法：
            # 本地小模型（sglang/vllm/ollama）→ 正交化 6 示例的 LOCAL_PLANNER_PROMPT
            #   + method='json_mode'（SGLang 不支持 function_calling 的 tools 参数，会返回 400）
            # 云端大模型（API）→ 完整 12 示例的 UNIFIED_MUSIC_QUERY_PLANNER_PROMPT
            #   + 默认 function_calling 模式
            _local_providers = {"sglang", "vllm", "ollama"}
            _intent_provider = (settings.intent_llm_provider or settings.llm_default_provider or "").lower()
            if _intent_provider in _local_providers:
                _planner_prompt = LOCAL_PLANNER_PROMPT
                
                if _intent_provider == "sglang":
                    # ── SGLang + Qwen3：完全绕过 LangChain ChatOpenAI ──
                    # LangChain 的 .bind(extra_body=...) 不可靠：首次请求能传 chat_template_kwargs，
                    # 后续请求会丢失，导致 Qwen3 thinking 模式重新激活（0.1 tok/s vs 10 tok/s）。
                    # 解决方案：用 httpx 直接构造 HTTP 请求体，完全控制每个字段。
                    import httpx
                    import json as _json
                    
                    from retrieval.gssc_context_builder import build_context
                    _ctx = await build_context(
                        graphzep_facts=state.get("graphzep_facts", ""),
                        chat_history=history_text,
                        total_budget=0,
                    )
                    
                    # 渲染提示词模板
                    _prompt = ChatPromptTemplate.from_template(_planner_prompt)
                    _messages = _prompt.format_messages(
                        user_input=user_input,
                        chat_history=_ctx["chat_history"],
                        previous_plan=_previous_plan_text,
                    )
                    
                    # 构造 OpenAI 兼容请求体
                    _api_messages = []
                    for m in _messages:
                        _api_messages.append({"role": getattr(m, "type", "user"), "content": m.content})
                    # LangChain 的 type 是 "human"/"ai"，需要映射
                    for m in _api_messages:
                        if m["role"] == "human":
                            m["role"] = "user"
                        elif m["role"] == "ai":
                            m["role"] = "assistant"
                    
                    _base_url = str(_intent_llm_instance.openai_api_base).rstrip("/")
                    _sglang_url = f"{_base_url}/chat/completions"
                    _request_body = {
                        "model": _intent_llm_instance.model_name,
                        "messages": _api_messages,
                        "max_tokens": 1024,
                        "temperature": _intent_llm_instance.temperature,
                        "response_format": {"type": "json_object"},
                        "chat_template_kwargs": {"enable_thinking": False},
                    }
                    
                    logger.info(f"[SGLang] POST → {_sglang_url} (model={_intent_llm_instance.model_name}, prompt_msgs={len(_api_messages)})")
                    import time as _time
                    _t0 = _time.time()
                    try:
                        async with httpx.AsyncClient(timeout=60.0) as _client:
                            _resp = await _client.post(_sglang_url, json=_request_body)
                    except httpx.TimeoutException as _te:
                        _elapsed = _time.time() - _t0
                        raise RuntimeError(f"SGLang 请求超时 ({_elapsed:.1f}s): {_sglang_url}") from _te
                    except Exception as _he:
                        _elapsed = _time.time() - _t0
                        raise RuntimeError(f"SGLang HTTP 错误 ({_elapsed:.1f}s): {type(_he).__name__}: {_he}") from _he
                    _elapsed = _time.time() - _t0
                    
                    if _resp.status_code != 200:
                        raise RuntimeError(f"SGLang API 返回 {_resp.status_code}: {_resp.text[:300]}")
                    
                    _resp_json = _resp.json()
                    raw_json_str = _resp_json["choices"][0]["message"]["content"]
                    logger.info(f"[SGLang] 意图分析完成, 耗时 {_elapsed:.1f}s")
                    
                    # 手动 Pydantic 解析（兼容 <think>...</think> 残留）
                    _clean = raw_json_str.strip()
                    if "<think>" in _clean:
                        _think_end = _clean.find("</think>")
                        if _think_end > 0:
                            _clean = _clean[_think_end + 8:].strip()
                    if "```json" in _clean:
                        _clean = _clean.split("```json")[-1].split("```")[0].strip()
                    elif "```" in _clean:
                        _clean = _clean.split("```")[1].strip()
                    
                    logger.debug(f"[SGLang] 原始 JSON 输出: {_clean[:500]}")
                    logger.info(f"[SGLang] 原始 JSON 输出 (首100字符): {_clean[:100]}")
                    plan = MusicQueryPlan.model_validate_json(_clean)
                    
                else:
                    # vLLM / Ollama: 保留 with_structured_output 路径
                    structured_llm = _intent_llm_instance.with_structured_output(MusicQueryPlan, method="json_mode")
                    chain = (
                        ChatPromptTemplate.from_template(_planner_prompt)
                        | structured_llm
                    )
                    from retrieval.gssc_context_builder import build_context
                    _ctx = await build_context(
                        graphzep_facts=state.get("graphzep_facts", ""),
                        chat_history=history_text,
                        total_budget=0,
                    )
                    plan: MusicQueryPlan = await chain.ainvoke({
                        "user_input": user_input,
                        "chat_history": _ctx["chat_history"],
                        "previous_plan": _previous_plan_text,
                    })
            else:
                # ── API 大模型路径（SiliconFlow / Volcengine / Gemini / DeepSeek 等）──
                # ★ 关键优化：拆分 system + human 消息，启用 KV Prefix Cache
                #   - system 消息 = UNIFIED_PLANNER_SYSTEM（能力画像+示例，~600 token）
                #     → 服务商自动缓存，后续请求只需重算 human 部分
                #   - human 消息  = UNIFIED_PLANNER_HUMAN（偏好+历史+输入，~100-300 token）
                #     → 每次都算，但量很小
                # 预期效果：首次 6-10s（冷启动），后续 2-4s（缓存命中）
                from llms.prompts import UNIFIED_PLANNER_SYSTEM, UNIFIED_PLANNER_HUMAN
                structured_llm = _intent_llm_instance.with_structured_output(MusicQueryPlan, include_raw=True)
                
                # ★ DashScope 显式缓存：用 content 数组格式 + cache_control
                #   百炼要求 cache_control 嵌套在 content 数组的 text 对象里
                #   命中时 input token 仅收 10%，5 分钟有效期（命中后重置）
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
                from retrieval.gssc_context_builder import build_context
                _ctx = await build_context(
                    graphzep_facts=state.get("graphzep_facts", ""),
                    chat_history=history_text,
                    total_budget=0,
                )
            
                _raw_result = await chain.ainvoke({
                    "user_input": user_input,
                    "user_preferences": state.get("graphzep_facts", "无"),
                    "chat_history": _ctx["chat_history"],
                    "previous_plan": _previous_plan_text,
                })
                plan: MusicQueryPlan = _raw_result["parsed"]
                
                # ── Token 消耗追踪 + 缓存命中检测 ──
                _raw_msg = _raw_result.get("raw")
                if _raw_msg and hasattr(_raw_msg, "usage_metadata") and _raw_msg.usage_metadata:
                    _usage = _raw_msg.usage_metadata
                    _p = _usage.get("input_tokens", 0)
                    _c = _usage.get("output_tokens", 0)
                    # 多厂商缓存命中检测：
                    #   SiliconFlow: prompt_cache_hit_tokens
                    #   DashScope/百炼: input_token_details.cache_read (LangChain 映射自 prompt_tokens_details.cached_tokens)
                    _cache_hit = (
                        _usage.get("prompt_cache_hit_tokens", 0)
                        or _usage.get("cache_read_input_tokens", 0)
                        or (_usage.get("input_token_details") or {}).get("cache_read", 0)
                    )
                    _cache_miss = _usage.get("prompt_cache_miss_tokens", 0) or (_p - _cache_hit if _cache_hit else 0)
                    _cache_info = ""
                    if _cache_hit:
                        _cache_info = f" | ✅ KV Cache命中={_cache_hit:,}tok, 未命中={_cache_miss:,}tok"
                    else:
                        _cache_info = " | ❄️ 冷启动(无缓存命中)"
                    logger.info(
                        f"[Token Report] 意图分析: "
                        f"prompt={_p:,} + completion={_c:,} = {_p + _c:,} tokens{_cache_info}"
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
            retrieval_plan_dict["_intent_type"] = plan.intent_type
            retrieval_plan_dict["_graphzep_facts"] = state.get("graphzep_facts", "")
            
            return {
                "intent_type": plan.intent_type,
                "intent_parameters": plan.parameters,
                "intent_context": plan.context,
                "retrieval_plan": retrieval_plan_dict,
                "step_count": state.get("step_count", 0) + 1
            }
            
        except Exception as e:
            logger.error(f"意图分析失败: {str(e)}")
            return {
                "intent_type": "general_chat",
                "intent_parameters": {},
                "intent_context": user_input,
                "retrieval_plan": None,
                "step_count": state.get("step_count", 0) + 1,
                "error_log": state.get("error_log", []) + [
                    {"node": "analyze_intent", "error": str(e)}
                ]
            }
    
    def route_by_intent(self, state: MusicAgentState) -> str:
        """
        路由函数: 根据意图类型决定下一步（5 类检索策略 + 2 类功能性意图）
        """
        intent_type = state.get("intent_type", "general_chat")
        logger.info(f"根据意图 '{intent_type}' 进行路由")

        # 3 类检索策略意图 → 统一走 search_songs 节点（retrieval_plan 已明确 use_graph/use_vector）
        if intent_type in ["graph_search", "hybrid_search", "vector_search", "web_search"]:
            return "search_songs"
        elif intent_type == "recommend_by_favorites":
            # 查用户收藏：路由到 generate_recommendations，内部有专门的收藏召回逻辑
            return "generate_recommendations"
        elif intent_type == "acquire_music":
            return "acquire_online_music"
        elif intent_type.startswith("create_playlist"):
            return "analyze_user_preferences"
        else:
            return "general_chat"
    
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

            # ══ 本地未命中检测：graph_search 有实体但结果不精确时，降级联网 ══
            # Case 1: 完全无结果
            # Case 2: 有结果但没有精确匹配请求的歌名（如搜"痛仰西湖"返回痛仰的其他歌）
            intent_type = state.get("intent_type", "")
            graph_entities = (retrieval_plan or {}).get("graph_entities", [])
            need_web_fallback = False

            if intent_type == "graph_search" and bool(graph_entities):
                if not search_results:
                    # Case 1: 完全无结果
                    need_web_fallback = True
                    logger.warning(
                        f"[search_songs] 本地库完全未命中，将降级联网搜索: entities={graph_entities}"
                    )
                elif len(graph_entities) >= 2:
                    # Case 2: 有多个实体（歌手+歌名），检查返回结果是否精确包含歌名
                    # 实体可能是 ["痛仰乐队", "西湖"] 或 ["周杰伦", "Jay Chou", "稻香"]
                    # 检查 graph 返回的歌名是否与任何 entity 模糊匹配
                    result_titles = set()
                    for r in search_results:
                        t = ""
                        if isinstance(r, dict):
                            t = r.get("title", "") or r.get("song_name", "") or r.get("name", "")
                            song_dict = r.get("song", {})
                            if isinstance(song_dict, dict):
                                t = t or song_dict.get("title", "") or song_dict.get("song_name", "")
                        result_titles.add(t.lower().strip())

                    # 对每个 entity，看它是否出现在某首歌名中（或者歌名出现在它中）
                    entity_matched_in_title = False
                    for ent in graph_entities:
                        ent_lower = ent.lower().strip()
                        if not ent_lower:
                            continue
                        for title in result_titles:
                            if not title:
                                continue
                            if ent_lower in title or title in ent_lower:
                                entity_matched_in_title = True
                                break
                        if entity_matched_in_title:
                            break

                    if not entity_matched_in_title:
                        need_web_fallback = True
                        logger.warning(
                            f"[search_songs] 本地库返回 {len(search_results)} 首但无精确歌名匹配，"
                            f"降级联网搜索: entities={graph_entities}, "
                            f"result_titles={list(result_titles)[:3]}"
                        )

            return {
                "search_results": search_results,
                "recommendations": raw_hybrid_result if raw_hybrid_result and raw_hybrid_result.success else [],
                "_need_web_fallback": need_web_fallback,
                "_web_fallback_query": " ".join(graph_entities[:2]) if graph_entities else query,
                "step_count": state.get("step_count", 0) + 1
            }
            
        except Exception as e:
            logger.error(f"搜索歌曲失败: {str(e)}")
            return {
                "search_results": [],
                "recommendations": [],
                "_need_web_fallback": False,
                "_web_fallback_query": "",
                "step_count": state.get("step_count", 0) + 1,
                "error_log": state.get("error_log", []) + [
                    {"node": "search_songs", "error": str(e)}
                ]
            }


    def route_after_search(self, state: MusicAgentState) -> str:
        """搜索后路由：本地未命中时降级到联网，否则就线生成解释"""
        if state.get("_need_web_fallback"):
            logger.info("[route_after_search] 本地未命中 → web_fallback")
            return "web_fallback"
        return "generate_explanation"

    async def web_fallback_node(self, state: MusicAgentState) -> Dict[str, Any]:
        """
        节点：本地库未命中时自动降级到网易云 API 联网搜索。
        不下载，只返回流媒体 URL，供前端即时播放。
        """
        logger.info("--- [步骤] 本地未命中，降级联网搜索 ---")
        query = state.get("_web_fallback_query") or state.get("input", "")

        try:
            import aiohttp
            from config.settings import settings as _cfg
            api_base = _cfg.netease_api_base
            timeout = aiohttp.ClientTimeout(total=8)

            async with aiohttp.ClientSession() as session:
                # 1) 搜索
                search_url = f"{api_base}/search?keywords={query}&limit=5"
                async with session.get(search_url, timeout=timeout) as resp:
                    data = await resp.json()

                songs = data.get("result", {}).get("songs", [])
                if not songs:
                    logger.warning(f"[web_fallback] 联网搜索无结果: {query}")
                    return {"search_results": [], "recommendations": [],
                            "_need_web_fallback": False,
                            "step_count": state.get("step_count", 0) + 1}

                # 收集 song_ids 用于批量获取详情
                song_ids = [str(s["id"]) for s in songs[:5]]

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

                # 3) 批量获取播放链接
                play_url_map = {}
                try:
                    url_api = f"{api_base}/song/url?id={','.join(song_ids)}&level=exhigh"
                    async with session.get(url_api, timeout=timeout) as uresp:
                        udata = await uresp.json()
                    for item in udata.get("data", []):
                        if item.get("url"):
                            play_url_map[str(item["id"])] = item["url"]
                except Exception:
                    pass

                # 4) 组装结果 —— 必须包含 preview_url (前端播放用) + cover_url
                results = []
                for s in songs[:5]:
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
                            "platform": "netease",
                        }
                    })

            matched = sum(1 for r in results if r["song"]["preview_url"])
            logger.info(f"[web_fallback] 联网返回 {len(results)} 首歌曲，{matched} 首可播放")

            from schemas.music_state import ToolOutput
            return {
                "search_results": [r["song"] for r in results],
                "recommendations": ToolOutput(
                    success=True,
                    data=results,
                    raw_markdown="",
                ),
                "_need_web_fallback": False,
                "step_count": state.get("step_count", 0) + 1,
            }

        except Exception as e:
            logger.error(f"[web_fallback] 联网搜索失败: {e}")
            return {
                "search_results": [], "recommendations": [],
                "_need_web_fallback": False,
                "step_count": state.get("step_count", 0) + 1,
            }

    async def acquire_online_music_node(self, state: MusicAgentState) -> Dict[str, Any]:
        """
        节点：用户确认后，自动下载音频/歌词/封面并入库 Neo4j。
        触发条件：LLM 识别到用户确认「好的，帮我下载/获取这些歌」。
        """
        logger.info("--- [步骤] 联网获取并入库音乐（自动数据飞轮）---")

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

    def _build_preference_query(self, seed_songs: list, graphzep_facts: str = "") -> str:
        """
        从种子歌曲标签 + 用户 Neo4j 画像 + GraphZep 记忆中提炼偏好文本。
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

        # 2. 从 Neo4j 用户画像补充
        try:
            from retrieval.user_memory import UserMemoryManager
            mem = UserMemoryManager()
            profile = mem.get_user_preferences("local_admin")
            if profile:
                for g in profile.get("favorite_genres", []):
                    if g:
                        tags.add(g.strip())
                mood_tendency = profile.get("mood_tendency", "")
                if mood_tendency:
                    tags.update(m.strip() for m in mood_tendency.replace(",", "，").split("，") if m.strip())
        except Exception as e:
            logger.warning(f"[Favorites] 加载用户画像失败: {e}")

        # 3. 从 GraphZep 记忆提取场景/情绪关键词
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
        
        try:
            # ── 特殊意图：recommend_by_favorites（两层智能推荐）──
            if intent_type == "recommend_by_favorites":
                logger.info("检测到 recommend_by_favorites 意图，启动两层智能推荐")
                memory = UserMemoryManager()
                memory.ensure_user_exists("local_admin")
                all_liked = memory.get_liked_songs(user_id="local_admin", limit=20)

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
                    )
                    logger.info(f"[Favorites] 偏好查询文本: {preference_query}")

                    # ── Tier 2: Discoveries（向量检索发现新歌）──
                    retriever = MusicHybridRetrieval(llm_client=get_llm())
                    discovery_plan = {
                        "use_graph": False,
                        "use_vector": True,
                        "use_web_search": False,
                        "_intent_type": "recommend_by_favorites",
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
        _explain = get_explain_llm()
        _explain_model_name = getattr(_explain, 'model_name', '?')
        _explain_provider = (settings.explain_llm_provider or settings.llm_default_provider or '?').lower()
        logger.info(f"--- [步骤 3] 生成推荐解释 | 🤖 {_explain_provider} / {_explain_model_name} ---")
        
        # 兼容处理 ToolOutput 对象或列表
        raw_recommendations = state.get("recommendations", [])
        recommendations = getattr(raw_recommendations, "data", raw_recommendations)
        
        user_query = state.get("input", "")
        
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
            return {
                "explanation": "抱歉，没有找到合适的音乐推荐。",
                "final_response": "抱歉，没有找到符合你要求的音乐。你可以换个方式描述你的需求，或者告诉我你喜欢的歌手和风格？",
                "step_count": state.get("step_count", 0) + 1
            }
        
        try:
            memory_manager = UserMemoryManager()
            default_user_id = settings.default_user_id
            
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
            
            # [LCEL 1.2 优化] 构建 LCEL 执行管道，生成推荐结果的解释
            # 将原来手动格式化字符串和接收 AIMessage 对象的两步操作合并为优雅的链式调用。
            chain = (
                ChatPromptTemplate.from_template(MUSIC_RECOMMENDATION_EXPLAINER_PROMPT)
                | get_explain_llm()
                | StrOutputParser()
            )
            
            # 流式生成推荐解释：通过 astream 逐 chunk 送入队列
            # 并发安全：通过 request_id 从注册表中取出当前请求专属的队列
            _req_id = state.get("metadata", {}).get("request_id", "")
            explanation_queue = self._explanation_queues.get(_req_id) if _req_id else None
            
            # ★ 先把歌曲数据推入队列，让前端立刻渲染歌曲卡片
            if explanation_queue and recommendations:
                try:
                    songs_payload = []
                    for i, rec in enumerate(recommendations):
                        song = rec.get("song", rec) if isinstance(rec, dict) else rec
                        if isinstance(song, dict) and song.get("title"):
                            songs_payload.append({"song": song, "index": i})
                    if songs_payload:
                        await explanation_queue.put({"__songs__": songs_payload})
                except Exception as e:
                    logger.warning(f"推送歌曲到队列失败: {e}")
            
            explanation = ""
            async for chunk in chain.astream({
                "user_query": user_query,
                "recommended_songs": songs_text
            }):
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
            
            logger.info(f"成功生成推荐解释, 耗时 {_time.time()-_t0:.1f}s")
            
            # 偏好提取已解耦为独立节点 extract_preferences_node
            
            return {
                "explanation": explanation,
                "final_response": final_response,
                "step_count": state.get("step_count", 0) + 1
            }
            
        except Exception as e:
            logger.error(f"生成解释失败: {str(e)}")
            
            # 确保队列收到终止信号，防止前端消费者永久阻塞
            _req_id = state.get("metadata", {}).get("request_id", "")
            _err_queue = self._explanation_queues.get(_req_id) if _req_id else None
            if _err_queue:
                try:
                    await _err_queue.put(None)
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
                ]
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
            default_user_id = "local_admin"
            
            logger.info("向 Neo4j 查询本地用户图谱记忆...")
            memory_manager = UserMemoryManager()
            
            # 确保用户节点存在（第一次运行防报错）
            memory_manager.ensure_user_exists(default_user_id, "本地管理员")
            
            # 读取历史偏好
            graph_prefs = memory_manager.get_user_preferences(default_user_id, limit=settings.user_preference_limit)
            
            favorite_artists = graph_prefs.get("favorite_artists", [])
            favorite_genres = graph_prefs.get("favorite_genres", [])
            
            # 此处获取的 favorite_songs 只是 title 数组
            favorite_songs_titles = graph_prefs.get("favorite_songs", [])
            
            # 为了适配下方的推荐流，将纯字符串简单封装一下
            top_tracks_mock = [{"title": t, "artist": "未知", "genre": "未知"} for t in favorite_songs_titles]
            
            # 若没查到（比如刚启动的空库），给点默认值以便链路正常运行
            if not favorite_artists:
                favorite_artists = ["周杰伦", "林俊杰"]
            if not favorite_genres:
                favorite_genres = ["Pop", "R&B"]
            if not top_tracks_mock:
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
            intent_type = state.get("intent_type", "")
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
            
            memory_manager = UserMemoryManager()
            default_user_id = "local_admin"
            
            # 提取歌曲
            songs = []
            for rec in recommendations:
                song_data = rec.get("song", rec)
                if isinstance(song_data, dict):
                    # 从字典创建 Song 对象
                    song = Song(
                        title=song_data.get("title", "未知"),
                        artist=song_data.get("artist", "未知"),
                        album=song_data.get("album"),
                        genre=song_data.get("genre"),
                        year=song_data.get("year"),
                        duration=song_data.get("duration"),
                        popularity=song_data.get("popularity"),
                        preview_url=song_data.get("preview_url"),
                        spotify_id=song_data.get("spotify_id"),
                        external_url=song_data.get("external_url")
                    )
                    songs.append(song)
                    
                    # 记录图谱喜欢/收藏行为
                    if song.title != "未知" and "集合" not in song.title:
                        memory_manager.record_liked_song(default_user_id, song.title, song.artist)
            
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
        【P1-4 双阶段 GraphZep 记忆召回】
        
        Stage 1（粗召回）：search_facts(max_facts=20) — 语义广撒网
        Stage 2（精排序）：get_memory(chat_history) — 结合对话上下文精排，取 top 5
        
        降级策略：
        - 整体 8s 硬超时 → 直接返回空记忆（避免阻塞推荐主流程）
        - Stage 2 失败 → 退回 Stage 1 结果
        - Stage 1 也失败 → 返回空
        """
        import time as _time
        _t0 = _time.time()
        logger.info("--- [GraphZep] 双阶段记忆召回 ---")
        
        # ★ 整体硬超时：GraphZep 服务可能因 LLM 调用而阻塞很久（尤其 Docker 环境）
        # 记忆召回是锦上添花功能，不能因此阻塞推荐主流程
        _GRAPHZEP_TOTAL_TIMEOUT = 8  # 秒
        
        async def _do_recall() -> Dict[str, Any]:
            user_input = state.get("input", "")
            group_id = state.get("graphzep_group_id", "music-agent-memory")
            chat_history = state.get("chat_history", [])
            
            from services.graphzep_client import get_graphzep_client
            client = get_graphzep_client()
            
            # ---- P2-1: 按意图选择 GraphZep 搜索策略 ----
            intent_type = state.get("intent_type", "")
            _INTENT_SEARCH_MAP = {
                "search":                "keyword",    # 精确搜歌手/歌名 → 关键词匹配
                "recommend_by_mood":     "semantic",   # 情绪推荐 → 语义理解
                "recommend_by_activity": "hybrid",     # 场景推荐 → 关键词+语义
                "recommend_by_genre":    "hybrid",     # 流派推荐 → 关键词+语义
                "recommend_by_favorites":"mmr",        # 基于历史 → MMR 多样化
                "create_playlist":       "mmr",        # 歌单生成 → MMR 多样化
            }
            search_type = _INTENT_SEARCH_MAP.get(intent_type, "hybrid")
            logger.info(f"[GraphZep] P2-1 意图路由: intent={intent_type} → search_type={search_type}")
            
            # ---- Stage 1: 粗召回（广撒网，20 条候选） ----
            coarse_facts = await client.search_facts(
                query=user_input,
                group_ids=[group_id],
                max_facts=20,
                search_type=search_type,
            )
            logger.info(f"[GraphZep] Stage 1 粗召回: {len(coarse_facts)}chars ({_time.time()-_t0:.1f}s)")
            
            # ---- Stage 2: 精排序（结合最近对话做上下文感知排序） ----
            fine_facts = coarse_facts  # 默认退回 Stage 1
            try:
                recent_msgs = []
                if chat_history:
                    for msg in chat_history[-3:]:
                        # chat_history 可能是 LangChain Message 对象或 dict
                        if hasattr(msg, 'content'):
                            content = msg.content
                            role = getattr(msg, 'type', 'human')  # 'human' or 'ai'
                            role = 'user' if role == 'human' else 'assistant'
                        else:
                            content = msg.get("content", "")
                            role = msg.get("role", "user")
                        recent_msgs.append({
                            "content": content,
                            "role_type": role,
                        })
                # 追加当前用户输入
                recent_msgs.append({"content": user_input, "role_type": "user"})
                
                fine_facts = await client.get_memory(
                    recent_messages=recent_msgs,
                    group_id=group_id,
                    max_facts=5,
                )
                logger.info(f"[GraphZep] Stage 2 精排序: {len(fine_facts)}chars ({_time.time()-_t0:.1f}s)")
            except Exception as stage2_err:
                logger.warning(f"[GraphZep] Stage 2 失败，退回 Stage 1: {stage2_err}")
            
            # 合并两阶段结果（去重）
            if fine_facts and fine_facts != "暂无用户长期记忆":
                combined = fine_facts
                # 如果 Stage 1 有额外有用信息，追加（但限制总长度）
                if coarse_facts and coarse_facts != fine_facts and coarse_facts != "暂无用户长期记忆":
                    extra_lines = [l for l in coarse_facts.split("\n") if l not in fine_facts]
                    if extra_lines:
                        combined += "\n" + "\n".join(extra_lines[:3])
                logger.info(f"[GraphZep] 最终记忆: {combined[:120]}...")
                return {"graphzep_facts": combined}
            elif coarse_facts and coarse_facts != "暂无用户长期记忆":
                return {"graphzep_facts": coarse_facts}
            else:
                return {"graphzep_facts": "暂无用户长期记忆"}
        
        try:
            result = await asyncio.wait_for(_do_recall(), timeout=_GRAPHZEP_TOTAL_TIMEOUT)
            _elapsed = _time.time() - _t0
            logger.info(f"[GraphZep] ✅ 记忆召回完成, 总耗时 {_elapsed:.1f}s")
            return result
        except asyncio.TimeoutError:
            _elapsed = _time.time() - _t0
            logger.warning(
                f"[GraphZep] ⚠️ 记忆召回超时 ({_elapsed:.1f}s > {_GRAPHZEP_TOTAL_TIMEOUT}s)，"
                f"降级为空记忆以保证推荐流程不阻塞"
            )
            return {"graphzep_facts": "暂无用户长期记忆"}
        except Exception as e:
            logger.warning(f"[GraphZep] 记忆召回失败（降级为空）: {e}")
            return {"graphzep_facts": "暂无用户长期记忆"}

    async def extract_preferences_node(self, state: MusicAgentState) -> Dict[str, Any]:
        """
        独立节点：从本轮对话中提取用户音乐偏好，异步写入 Neo4j。
        
        原本嵌入在 generate_explanation 中（~90 行硬编码），
        现解耦为独立 LangGraph 节点，提升架构可维护性。
        
        执行逻辑（fire-and-forget，不阻塞工作流）：
        1. 收集场景上下文（时间段、场景标签、推荐歌曲）
        2. 调用 LLM 通过 MUSIC_PREFERENCE_EXTRACTOR_PROMPT 提取偏好
        3. 将结果写入 Neo4j 用户画像（UserMemoryManager）
        """
        logger.info("--- [步骤] 提取用户偏好（独立节点） ---")
        
        user_query = state.get("input", "")
        raw_recommendations = state.get("recommendations", [])
        recommendations = getattr(raw_recommendations, "data", raw_recommendations)
        
        if not user_query or not recommendations:
            logger.info("[SemanticMemory] 无用户输入或推荐结果，跳过偏好提取")
            return {}
        
        try:
            from llms.prompts import MUSIC_PREFERENCE_EXTRACTOR_PROMPT
            import json as _json
            from datetime import datetime as _dt
            
            memory_manager = UserMemoryManager()
            
            # ── 收集场景上下文 ──
            retrieval_plan = state.get("retrieval_plan", {})
            scene_ctx = (
                getattr(retrieval_plan, "graph_scenario_filter", None)
                or (retrieval_plan.get("graph_scenario_filter") if isinstance(retrieval_plan, dict) else None)
                or "未知"
            )
            
            # 推断当前时间段
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
            
            # 本轮推荐歌曲摘要
            rec_songs_text = "无" if not recommendations else ", ".join([
                f"《{r.get('song', r).get('title', '?')}》"
                for r in recommendations[:5]
            ])
            
            pref_chain = (
                ChatPromptTemplate.from_template(MUSIC_PREFERENCE_EXTRACTOR_PROMPT)
                | get_llm()
                | StrOutputParser()
            )
            
            # ── 异步 fire-and-forget，不阻塞工作流返回 ──
            async def _bg_extract_preferences():
                try:
                    pref_raw = await pref_chain.ainvoke({
                        "user_message": user_query,
                        "scene_context": scene_ctx,
                        "current_time": time_label,
                        "recommended_songs": rec_songs_text,
                        "user_feedback": "暂无",
                    })
                    
                    if pref_raw and pref_raw.strip():
                        pref_text = pref_raw.strip()
                        if "```json" in pref_text:
                            pref_text = pref_text.split("```json")[-1].split("```")[0].strip()
                        elif "```" in pref_text:
                            pref_text = pref_text.split("```")[1].strip()
                        
                        pref_data = _json.loads(pref_text)
                        
                        # 处理新格式（含 global_preference）
                        global_pref = pref_data.get("global_preference", pref_data)
                        has_content = any(
                            (isinstance(v, list) and len(v) > 0) or
                            (isinstance(v, str) and v.strip())
                            for v in global_pref.values()
                        )
                        if has_content:
                            memory_manager.update_semantic_preferences("local_admin", global_pref)
                            logger.info(f"[SemanticMemory] 偏好提取成功: {global_pref}")
                        
                        # 场景偏好也写入
                        scene_pref = pref_data.get("scene_preference", {})
                        if scene_pref.get("summary"):
                            logger.info(f"[SemanticMemory] 场景偏好: {scene_pref.get('summary')}")
                    else:
                        logger.info("[SemanticMemory] 本轮对话无明确偏好表达，跳过写入")
                except Exception as e:
                    logger.warning(f"[SemanticMemory] 后台偏好提取失败（不影响主流程）: {e}")
            
            asyncio.create_task(_bg_extract_preferences())
            logger.info("[SemanticMemory] 偏好提取任务已投递到后台")
            
            # ★ 同步投递：预压缩对话历史，为下一轮请求消除 17s 阻塞
            # 与偏好提取并行执行，互不干扰，在推荐响应返回之后进行
            try:
                from retrieval.gssc_context_builder import pre_compress_and_cache
                from retrieval.history import MusicContextManager as _HisMgr
                _ctx_mgr = _HisMgr()
                # 获取本次请求携带的 chat_history（已包含当前轮 user query，但不含本轮 bot 回复）
                _raw_history = state.get("chat_history", [])
                _history_str = _ctx_mgr.format_chat_history(_raw_history)
                asyncio.create_task(pre_compress_and_cache("local_admin", _history_str))
                logger.info("[GSSC-Cache] 历史预压缩任务已投递到后台")
            except Exception as _cache_e:
                logger.warning(f"[GSSC-Cache] 投递预压缩任务失败（不影响主流程）: {_cache_e}")

            
        except Exception as pref_e:
            logger.warning(f"[SemanticMemory] 偏好提取节点异常（不影响主流程）: {pref_e}")
        
        return {}

    async def persist_to_graphzep(self, state: MusicAgentState) -> Dict[str, Any]:
        """
        出口旁路节点：将本轮完整对话异步送入 GraphZep。
        
        执行逻辑：
        1. 取用户本轮输入 + Bot 最终回复
        2. 调用 GraphZep POST /messages（fire-and-forget）
        3. GraphZep 内部会异步 LLM 抽取实体/关系并持久化到 Neo4j
        
        使用 asyncio.create_task 确保不阻塞返回流程。
        """
        logger.info("--- [GraphZep] 异步持久化对话 ---")
        
        user_input = state.get("input", "")
        bot_response = state.get("final_response", "")
        group_id = state.get("graphzep_group_id", "music-agent-memory")
        
        if not user_input or not bot_response:
            return {}
        
        try:
            from services.graphzep_client import get_graphzep_client
            from datetime import datetime as _dt
            client = get_graphzep_client()
            
            # P1-3: 携带场景上下文，让 GraphZep 的 LLM 提取出带场景的事实
            retrieval_plan = state.get("retrieval_plan", {})
            scene_ctx = (
                getattr(retrieval_plan, "graph_scenario_filter", None)
                or (retrieval_plan.get("graph_scenario_filter") if isinstance(retrieval_plan, dict) else None)
                or ""
            )
            hour = _dt.now().hour
            time_label = "凌晨" if hour < 6 else "早晨" if hour < 9 else "上午" if hour < 12 else "中午" if hour < 14 else "下午" if hour < 18 else "傍晚" if hour < 21 else "深夜"
            
            # 将场景标签注入用户消息，让 GraphZep 的 LLM 提取事实时能感知场景
            enriched_user_msg = user_input
            if scene_ctx:
                enriched_user_msg = f"[场景: {scene_ctx} | 时间: {time_label}] {user_input}"
            
            # Fire-and-forget：不等 GraphZep 处理完
            asyncio.create_task(
                client.add_messages(
                    user_message=enriched_user_msg,
                    bot_response=bot_response,
                    group_id=group_id,
                )
            )
            logger.info(f"[GraphZep] 对话已投递到异步队列 (scene={scene_ctx or '无'})")
            
        except Exception as e:
            logger.warning(f"[GraphZep] 持久化投递失败（不影响用户）: {e}")
        
        return {}

    def _build_graph(self) -> CompiledStateGraph:
        """构建工作流图"""
        logger.info("开始构建音乐推荐工作流图...")
        
        workflow = StateGraph(MusicAgentState)
        
        # ==== GraphZep 记忆节点 ====
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
        workflow.add_node("generate_explanation", self.generate_explanation)
        
        # 设置入口点为 GraphZep 记忆召回
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
                "generate_recommendations": "generate_recommendations",
                "analyze_user_preferences": "analyze_user_preferences",
                "general_chat": "general_chat"
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
        workflow.add_edge("persist_to_graphzep", END)
        
        # 编译图（注入 checkpointer 实现状态持久化）
        if self.checkpointer:
            app = workflow.compile(checkpointer=self.checkpointer)
            logger.info("音乐推荐工作流图构建完成 (✅ MemorySaver Checkpoint 已启用)")
        else:
            app = workflow.compile()
            logger.info("音乐推荐工作流图构建完成 (⚠️ 无 Checkpoint，每次对话独立)")
        
        return app

