"""
音乐推荐Agent的工作流图
"""

import asyncio
from typing import Dict, Any

from langgraph.graph import StateGraph, END
from langgraph.graph.state import CompiledStateGraph

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from config.logging_config import get_logger
from config.settings import settings
from llms.multi_llm import get_chat_model, get_intent_chat_model

from schemas.music_state import MusicAgentState, ToolOutput
from tools.music_tools import get_music_search_tool, get_music_recommender, Song
from tools.graphrag_search import graphrag_search
# 【V2 升级】替换旧版 vector_search 为 Neo4j 原生语义搜索
from tools.semantic_search import semantic_search
from tools.music_fetch_tool import search_online_music, execute_search_online_music
from tools.acquire_music import acquire_online_music
from retrieval.hybrid_retrieval import MusicHybridRetrieval
from retrieval.user_memory import UserMemoryManager
from retrieval.history import MusicContextManager
from llms.prompts import (
    UNIFIED_MUSIC_QUERY_PLANNER_PROMPT,
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


# _clean_json_from_llm 已被 with_structured_output 替代，不再需要手动正则解析


class MusicRecommendationGraph:
    """音乐推荐工作流图"""
    
    def __init__(self):
        self.workflow = self._build_graph()
    
    def get_app(self) -> CompiledStateGraph:
        """获取编译后的应用"""
        return self.workflow
    
    async def analyze_intent(self, state: MusicAgentState) -> Dict[str, Any]:
        """
        节点1: 统一意图分析 + 检索规划
        使用 with_structured_output 直接输出类型安全的 MusicQueryPlan 对象，
        彻底消除手动正则 + json.loads 的脆弱解析。
        """
        logger.info("--- [步骤 1] 统一意图分析与检索规划 (Structured Output) ---")
        
        user_input = state.get("input", "")
        chat_history = state.get("chat_history", [])
        
        try:
            # 格式化对话历史
            context_manager = MusicContextManager()
            history_text = context_manager.format_chat_history(chat_history)
            
            # ✅ with_structured_output：让模型直接输出 MusicQueryPlan Pydantic 对象
            # 底层自动处理 json_schema 约束，无需任何正则或 json.loads
            structured_llm = get_intent_llm().with_structured_output(MusicQueryPlan)
            chain = (
                ChatPromptTemplate.from_template(UNIFIED_MUSIC_QUERY_PLANNER_PROMPT)
                | structured_llm
            )
            # [P3] GSSC Token budget management
            from retrieval.gssc_context_builder import build_context
            _ctx = build_context(
                graphzep_facts=state.get("graphzep_facts", "暂无用户长期记忆"),
                chat_history=history_text,
                total_budget=3000,
            )
            
            plan: MusicQueryPlan = await chain.ainvoke({
                "user_input": user_input,
                "chat_history": _ctx["chat_history"],
                "graphzep_facts": _ctx["graphzep_facts"],
            })
            
            # 直接通过属性访问，完全类型安全，字段缺失会有 Pydantic 默认值兜底
            logger.info(
                f"识别到意图: {plan.intent_type} | "
                f"检索规划: graph={plan.retrieval_plan.use_graph}, "
                f"vector={plan.retrieval_plan.use_vector}, "
                f"web={plan.retrieval_plan.use_web_search}"
            )
            logger.info(f"决策理由: {plan.reasoning}")
            
            # ============================================================
            # 【升级】将 intent_type 注入 retrieval_plan，供 HyDE 拦截判断
            # 来源：HyDE 检索增强需要根据意图类型决定是否生成虚拟乐评
            # ============================================================
            retrieval_plan_dict = plan.retrieval_plan.model_dump()
            retrieval_plan_dict["_intent_type"] = plan.intent_type
            
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
        路由函数: 根据意图类型决定下一步
        """
        intent_type = state.get("intent_type", "general_chat")
        logger.info(f"根据意图 '{intent_type}' 进行路由")
        
        if intent_type == "play_specific_song_online":
            return "fetch_online_music"
        elif intent_type == "acquire_music":
            return "acquire_online_music"
        elif intent_type == "search":
            return "search_songs"
        elif intent_type.startswith("create_playlist"):
            # 创建歌单意图，先分析用户偏好
            return "analyze_user_preferences"
        elif intent_type in ["recommend_by_mood", "recommend_by_activity", 
                            "recommend_by_genre", "recommend_by_artist", 
                            "recommend_by_favorites"]:
            return "generate_recommendations"
        else:
            return "general_chat"
    
    async def search_songs_node(self, state: MusicAgentState) -> Dict[str, Any]:
        """
        节点2a: 搜索歌曲
        """
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
            raw_hybrid_result = retriever.retrieve(search_intent, limit=settings.graph_search_limit, precomputed_plan=retrieval_plan)
            
            # 直接使用标准的 ToolOutput
            if raw_hybrid_result and raw_hybrid_result.success:
                search_results = raw_hybrid_result.data
            else:
                search_results = []
            
            logger.info(f"搜索到 {len(search_results)} 首歌曲")
            
            return {
                "search_results": search_results,
                "recommendations": raw_hybrid_result if raw_hybrid_result and raw_hybrid_result.success else [],  # 存入完整的 ToolOutput 对象供解释节点读取
                "step_count": state.get("step_count", 0) + 1
            }
            
        except Exception as e:
            logger.error(f"搜索歌曲失败: {str(e)}")
            return {
                "search_results": [],
                "recommendations": [],
                "step_count": state.get("step_count", 0) + 1,
                "error_log": state.get("error_log", []) + [
                    {"node": "search_songs", "error": str(e)}
                ]
            }

    async def fetch_online_music_node(self, state: MusicAgentState) -> Dict[str, Any]:
        """
        节点：通过公共外网大盘获取真实的流媒体播放链接
        """
        logger.info("--- [步骤 2] 联网抓取试听级音乐源 ---")
        
        user_query = state.get("intent_parameters", {}).get("query", state.get("input", ""))
        try:
            result = await execute_search_online_music(user_query)
            
            # 直接使用标准的 ToolOutput 透传给 explanation 节点
            return {
                "recommendations": result,
                "step_count": state.get("step_count", 0) + 1
            }
        except Exception as e:
            logger.error(f"联网抓取音乐源失败: {str(e)}")
            return {
                "error_log": state.get("error_log", []) + [
                    {"node": "fetch_online_music", "error": str(e)}
                ],
                "step_count": state.get("step_count", 0) + 1
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

        # 如果 LLM 没有提供 song_queries，尝试从 query 字段构造
        if not song_queries:
            query = parameters.get("query", state.get("input", ""))
            if query:
                song_queries = [query]

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

    async def generate_recommendations_node(self, state: MusicAgentState) -> Dict[str, Any]:
        """
        节点2b: 生成推荐
        根据不同的意图类型调用不同的推荐方法
        """
        logger.info("--- [步骤 2b] 生成音乐推荐 ---")
        
        intent_type = state.get("intent_type")
        parameters = state.get("intent_parameters", {})
        
        try:
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
            raw_hybrid_result = retriever.retrieve(search_query, limit=settings.hybrid_retrieval_limit, precomputed_plan=retrieval_plan)
            
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
        logger.info("--- [步骤 2c] 通用音乐聊天 ---")
        
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
            _ctx = build_context(
                graphzep_facts=state.get("graphzep_facts", "暂无用户长期记忆"),
                chat_history=history_text,
                total_budget=3000,
            )
            
            response_content = await chain.ainvoke({
                "chat_history": _ctx["chat_history"],
                "user_message": user_message,
                "graphzep_facts": _ctx["graphzep_facts"],
            })
            
            logger.info("生成聊天回复")
            
            return {
                "final_response": response_content,
                "step_count": state.get("step_count", 0) + 1
            }
            
        except Exception as e:
            logger.error(f"生成聊天回复失败: {str(e)}")
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
        logger.info("--- [步骤 3] 生成推荐解释 ---")
        
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
                | get_llm()
                | StrOutputParser()
            )
            
            # 流式生成推荐解释：通过 astream 逐 chunk 送入队列
            explanation_queue = state.get("_explanation_queue")
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
            
            logger.info("成功生成推荐解释")
            
            # ============================================================
            # 【升级】旁路静默执行：后台提取用户语义偏好并持久化
            # 来源：《第八章 记忆与检索》+ memory_hyde_analysis 建议
            # 在每轮对话即将结束时，将用户的发言喂给 LLM 提取偏好，
            # 然后调用 user_memory.update_semantic_preferences 写入 Neo4j。
            # 使用 try/except 包裹确保不影响主流程返回。
            # ============================================================
            try:
                from llms.prompts import MUSIC_PREFERENCE_EXTRACTOR_PROMPT
                import json as _json
                from datetime import datetime as _dt
                
                # 从 state 中收集场景上下文 (P1-1 / P1-3)
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
                
                # 本轮推荐歌曲
                rec_songs_text = "无" if not recommendations else ", ".join([
                    f"《{r.get('song', r).get('title', '?')}》"
                    for r in recommendations[:5]
                ])
                
                pref_chain = (
                    ChatPromptTemplate.from_template(MUSIC_PREFERENCE_EXTRACTOR_PROMPT)
                    | get_llm()
                    | StrOutputParser()
                )
                
                # ---- P1-2: 异步 fire-and-forget，不阻塞主流程返回 ----
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
                            
                            # P1-3: 场景偏好也写入
                            scene_pref = pref_data.get("scene_preference", {})
                            if scene_pref.get("summary"):
                                logger.info(f"[SemanticMemory] 场景偏好: {scene_pref.get('summary')}")
                        else:
                            logger.info("[SemanticMemory] 本轮对话无明确偏好表达，跳过写入")
                    except Exception as e:
                        logger.warning(f"[SemanticMemory] 后台偏好提取失败（不影响主流程）: {e}")
                
                import asyncio
                asyncio.create_task(_bg_extract_preferences())
            except Exception as pref_e:
                logger.warning(f"[SemanticMemory] 后台偏好提取失败（不影响主流程）: {pref_e}")
            
            return {
                "explanation": explanation,
                "final_response": final_response,
                "step_count": state.get("step_count", 0) + 1
            }
            
        except Exception as e:
            logger.error(f"生成解释失败: {str(e)}")
            
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
                raw_hybrid_result = retriever.retrieve(query, limit=settings.graph_search_limit)
                
                # 直接扩展到推荐列表
                recommendations.extend(raw_hybrid_result)
            else:
                # 其他推荐类型，使用原有逻辑
                recommender = get_music_recommender()
                if intent_type == "recommend_by_mood":
                    mood = parameters.get("mood", "开心")
                    recs = await recommender.recommend_by_mood(mood, limit=settings.enhanced_recommend_limit)
                    recommendations = [rec.to_dict() for rec in recs]
                elif intent_type == "recommend_by_activity":
                    activity = parameters.get("activity", "放松")
                    recs = await recommender.recommend_by_activity(activity, limit=settings.enhanced_recommend_limit)
                    recommendations = [rec.to_dict() for rec in recs]
            
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
        
        降级策略：Stage 2 失败 → 退回 Stage 1 结果；Stage 1 也失败 → 返回空
        """
        logger.info("--- [GraphZep] 双阶段记忆召回 ---")
        
        user_input = state.get("input", "")
        group_id = state.get("graphzep_group_id", "music-agent-memory")
        chat_history = state.get("chat_history", [])
        
        try:
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
            logger.info(f"[GraphZep] Stage 1 粗召回: {len(coarse_facts)}chars")
            
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
                logger.info(f"[GraphZep] Stage 2 精排序: {len(fine_facts)}chars")
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
            
        except Exception as e:
            logger.warning(f"[GraphZep] 记忆召回失败（降级为空）: {e}")
            return {"graphzep_facts": "暂无用户长期记忆"}

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
        
        # 添加节点
        workflow.add_node("analyze_intent", self.analyze_intent)
        workflow.add_node("fetch_online_music", self.fetch_online_music_node)
        workflow.add_node("acquire_online_music", self.acquire_online_music_node)  # 🆕 数据飞轮
        workflow.add_node("search_songs", self.search_songs_node)
        workflow.add_node("generate_recommendations", self.generate_recommendations_node)
        workflow.add_node("analyze_user_preferences", self.analyze_user_preferences_node)  # ⭐ NEW
        workflow.add_node("enhanced_recommendations", self.enhanced_recommendations_node)  # ⭐ NEW
        workflow.add_node("create_playlist", self.create_playlist_node)  # ⭐ NEW
        workflow.add_node("general_chat", self.general_chat_node)
        workflow.add_node("generate_explanation", self.generate_explanation)
        
        # 设置入口点为 GraphZep 记忆召回
        workflow.set_entry_point("recall_graphzep_memory")
        
        # 召回完成后 → 意图分析
        workflow.add_edge("recall_graphzep_memory", "analyze_intent")
        
        # 添加条件边：根据意图路由
        workflow.add_conditional_edges(
            "analyze_intent",
            self.route_by_intent,
            {
                "fetch_online_music": "fetch_online_music",
                "acquire_online_music": "acquire_online_music",  # 🆕 数据飞轮
                "search_songs": "search_songs",
                "generate_recommendations": "generate_recommendations",
                "analyze_user_preferences": "analyze_user_preferences",  # ⭐ NEW
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
        workflow.add_edge("fetch_online_music", "generate_explanation")
        workflow.add_edge("acquire_online_music", "generate_explanation")  # 🆕
        workflow.add_edge("search_songs", "generate_explanation")
        workflow.add_edge("generate_recommendations", "generate_explanation")
        
        # 创建播放列表后生成解释
        workflow.add_edge("create_playlist", "generate_explanation")
        
        # 聊天和解释后 → 异步持久化 → 结束
        workflow.add_edge("general_chat", "persist_to_graphzep")
        workflow.add_edge("generate_explanation", "persist_to_graphzep")
        workflow.add_edge("persist_to_graphzep", END)
        
        # 编译图
        app = workflow.compile()
        logger.info("音乐推荐工作流图构建完成")
        
        return app

