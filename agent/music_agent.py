"""
音乐推荐Agent主入口
提供完整的音乐推荐功能
"""

import asyncio
import os
import time
from typing import Dict, Any, Optional, List


from config.logging_config import get_logger
from config.settings import settings
from agent.music_graph import MusicRecommendationGraph
from schemas.music_state import MusicAgentState
from services.feedback_logger import log_exposure
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage

logger = get_logger(__name__)


class MusicRecommendationAgent:
    """音乐推荐智能体主类"""
    
    def __init__(self):
        """初始化智能体"""
        self.graph = MusicRecommendationGraph()
        self.app = self.graph.get_app()
        logger.info("MusicRecommendationAgent 初始化完成")
    
    async def get_recommendations(
        self,
        query: str,
        chat_history: Optional[List[Dict[str, str]]] = None,
        user_preferences: Optional[Dict[str, Any]] = None,
        dialog_state: Optional[Dict[str, Any]] = None,
        user_id: str = "local_admin",
    ) -> Dict[str, Any]:
        """
        获取音乐推荐
        
        Args:
            query: 用户查询/需求
            chat_history: 对话历史
            user_preferences: 用户偏好数据
            
        Returns:
            包含推荐结果的字典
        """
        request_started = time.perf_counter()
        try:
            logger.info(f"开始处理音乐推荐请求: {query}")
            
            # 构建初始状态
            # 将历史记录中的字典转换为 BaseMessage 以适配 LangGraph 规范
            formatted_history: List[BaseMessage] = []
            if chat_history:
                for msg in chat_history:
                    role = msg.get("role", "")
                    content = msg.get("content", "")
                    if role == "user":
                        formatted_history.append(HumanMessage(content=content))
                    elif role == "assistant":
                        formatted_history.append(AIMessage(content=content))
            
            initial_state: MusicAgentState = {
                "user_id": user_id,
                "input": query,
                "chat_history": formatted_history,
                "user_preferences": user_preferences or {},
                "favorite_songs": [],
                "intent_type": "",
                "intent_parameters": {},
                "intent_context": "",
                "search_results": [],
                "recommendations": [],
                "explanation": "",
                "final_response": "",
                "playlist": None,
                "step_count": 0,
                "error_log": [],
                "metadata": {"user_id": user_id},
                "timings": {},
                "retrieval_meta": {},
                "tool_plan": {},
                "tool_observations": [],
                "dialog_state": dialog_state or {},
            }
            
            # 执行工作流
            config = {
                "recursion_limit": 50
            }
            # MemorySaver Checkpoint: 传入 thread_id 实现对话状态持久化
            thread_id = ""
            if getattr(self.graph, 'checkpointer', None):
                import uuid
                thread_id = config.get("configurable", {}).get("thread_id", str(uuid.uuid4()))
                config["configurable"] = {"thread_id": thread_id}
                logger.info(f"[Checkpoint] thread_id={thread_id}")
            result = await self.app.ainvoke(initial_state, config=config)
            timings = dict(result.get("timings") or {})
            timings["agent_total_ms"] = round((time.perf_counter() - request_started) * 1000, 3)
            raw_recommendations = result.get("recommendations", [])
            recommendations_for_log = getattr(raw_recommendations, "data", raw_recommendations)
            try:
                from agent.intent.planner import UNIFIED_PLANNER_PROMPT_VERSION
                from schemas.tool_plan import TOOL_PLAN_VERSION
                from services.teacher_log import log_teacher_example

                tool_plan = result.get("tool_plan") or {}
                if tool_plan:
                    ranked_ids = [
                        str(
                            (item.get("song") or item).get("music_id")
                            or (item.get("song") or item).get("id")
                            or ""
                        )
                        for item in (recommendations_for_log or [])
                        if isinstance(item, dict)
                    ]
                    log_teacher_example(
                        "agent_trajectory",
                        inputs={
                            "query": query,
                            "chat_history": chat_history or [],
                            "user_preferences": user_preferences or {},
                        },
                        output={
                            "tool_plan": tool_plan,
                            "tool_observations": result.get("tool_observations") or [],
                            "ranked_ids": [item for item in ranked_ids if item],
                            "validator_issues": (
                                (result.get("retrieval_plan") or {}).get("_tool_plan_alignment_issues")
                                or []
                            ),
                        },
                        metadata={
                            "provider": settings.intent_llm_provider or settings.llm_default_provider,
                            "model": settings.intent_llm_model or settings.llm_default_model,
                            "planner_quality_mode": settings.planner_quality_mode,
                            "prompt_version": UNIFIED_PLANNER_PROMPT_VERSION,
                            "tool_schema_version": TOOL_PLAN_VERSION,
                            "catalog_snapshot": os.getenv("CATALOG_SNAPSHOT_VERSION", "neo4j-live"),
                            "timings": timings,
                        },
                    )
            except Exception as trajectory_error:
                logger.debug("[Trajectory] 轨迹写入失败，已跳过: %s", trajectory_error)
            if (
                isinstance(recommendations_for_log, list)
                and recommendations_for_log
                and not settings.eval_disable_side_effects
            ):
                try:
                    log_exposure(
                        query=query,
                        user_id=user_id,
                        request_id=thread_id,
                        recommendations=recommendations_for_log,
                        intent_type=result.get("intent_type", ""),
                        retrieval_meta=result.get("retrieval_meta", {}),
                        dialog_state=result.get("dialog_state", {}),
                        timings=timings,
                    )
                except Exception as log_error:
                    logger.warning(f"[Feedback] 曝光日志写入失败: {log_error}")
            
            logger.info("音乐推荐完成")
            
            return {
                "success": True,
                "response": result.get("final_response", ""),
                "recommendations": result.get("recommendations", []),
                "search_results": result.get("search_results", []),
                "intent_type": result.get("intent_type", ""),
                "explanation": result.get("explanation", ""),
                "playlist": result.get("playlist"),
                "errors": result.get("error_log", []),
                "timings": timings,
                "retrieval_meta": result.get("retrieval_meta", {}),
                "tool_plan": result.get("tool_plan", {}),
                "tool_observations": result.get("tool_observations", []),
                "dialog_state": result.get("dialog_state", {}),
                "dialog_delta": result.get("dialog_delta", {}),
                "clarification_options": result.get("clarification_options", []),
                "intent_confidence": result.get("intent_confidence", 1.0),
                "refinement_options": result.get("refinement_options", []),
            }
            
        except Exception as e:
            logger.error(f"处理音乐推荐请求时发生错误: {str(e)}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
                "response": "抱歉，处理你的请求时遇到了问题。请稍后重试。",
                "recommendations": [],
                "search_results": [],
                "errors": [{"node": "main", "error": str(e)}],
                "timings": {
                    "agent_total_ms": round((time.perf_counter() - request_started) * 1000, 3)
                },
                "retrieval_meta": {},
            }
    
    async def stream_recommendations(
        self,
        query: str,
        chat_history: Optional[List[Dict[str, str]]] = None,
        user_preferences: Optional[Dict[str, Any]] = None,
        dialog_state: Optional[Dict[str, Any]] = None,
        user_id: str = "local_admin",
    ):
        """
        流式获取推荐结果（异步生成器）
        
        与 get_recommendations 不同，此方法在 LLM 生成推荐解释时
        逐 chunk 推送文本，而非等全部完成再返回。
        
        Yields:
            dict 事件: {"type": "thinking"|"response"|"songs"|"complete"|"error", ...}
        """
        import asyncio
        import time as _time
        import uuid as _uuid
        
        # 为本次请求生成唯一 ID，用于隔离并发请求的流式队列
        _request_id = str(_uuid.uuid4())
        
        try:
            logger.info(f"开始处理音乐推荐请求(流式): {query} [req={_request_id[:8]}]")
            _stream_start = _time.time()
            
            # 构建对话历史
            formatted_history: List[BaseMessage] = []
            if chat_history:
                for msg in chat_history:
                    role = msg.get("role", "")
                    content = msg.get("content", "")
                    if role == "user":
                        formatted_history.append(HumanMessage(content=content))
                    elif role == "assistant":
                        formatted_history.append(AIMessage(content=content))
            
            # 创建本次请求专属的队列，并注册到 graph 的队列表中
            # generate_explanation 节点通过 state.metadata.request_id 找到对应的队列
            explanation_queue = asyncio.Queue()
            self.graph._explanation_queues[_request_id] = explanation_queue
            
            initial_state: MusicAgentState = {
                "user_id": user_id,
                "input": query,
                "chat_history": formatted_history,
                "user_preferences": user_preferences or {},
                "favorite_songs": [],
                "intent_type": "",
                "intent_parameters": {},
                "intent_context": "",
                "search_results": [],
                "recommendations": [],
                "explanation": "",
                "final_response": "",
                "playlist": None,
                "step_count": 0,
                "error_log": [],
                "metadata": {"request_id": _request_id, "user_id": user_id},
                "timings": {},
                "retrieval_meta": {},
                "tool_plan": {},
                "tool_observations": [],
                "dialog_state": dialog_state or {},
            }
            
            config = {"recursion_limit": 50}
            # MemorySaver Checkpoint: 传入 thread_id 实现对话状态持久化
            if getattr(self.graph, 'checkpointer', None):
                thread_id = _request_id  # 复用 request_id 作为 thread_id
                config["configurable"] = {"thread_id": thread_id}
                logger.info(f"[Checkpoint] stream thread_id={thread_id[:8]}")
            
            # 后台任务运行 LangGraph
            result_holder = {}
            
            async def _run_graph():
                try:
                    result = await self.app.ainvoke(initial_state, config=config)
                    result_holder["result"] = result
                except Exception as e:
                    result_holder["error"] = str(e)
                    # 确保队列收到终止信号
                    try:
                        await explanation_queue.put(None)
                    except Exception:
                        pass
            
            graph_task = asyncio.create_task(_run_graph())
            self._current_graph_task = graph_task  # 暴露给 server.py 断连取消用
            
            # 发送思考状态
            yield {"type": "thinking", "message": "正在理解你的音乐偏好..."}
            
            # 从队列读取流式解释文本（歌曲数据也会通过队列提前到达）
            accumulated_text = ""
            songs_already_sent = False
            clarification_already_sent = False
            # Docker 环境冷启动可能需要较长时间（GraphZep + LLM + 检索 + 精排串行叠加）
            # 180s 给予充足的首次请求余量，热缓存时通常 20-40s 内即可收到首 chunk
            _STREAM_TIMEOUT = 180
            while True:
                try:
                    chunk = await asyncio.wait_for(explanation_queue.get(), timeout=_STREAM_TIMEOUT)
                except asyncio.TimeoutError:
                    _elapsed = _time.time() - _stream_start
                    logger.error(f"流式推荐超时: 已等待 {_elapsed:.1f}s (timeout={_STREAM_TIMEOUT}s) [req={_request_id[:8]}]")
                    yield {"type": "error", "error": f"推荐生成超时({_elapsed:.0f}s)，请重试"}
                    graph_task.cancel()
                    return
                
                if chunk is None:
                    # 流式结束
                    break
                
                # ★ 处理歌曲数据（在解释文本之前到达）
                if isinstance(chunk, dict) and "__songs__" in chunk:
                    songs_list = chunk["__songs__"]
                    yield {
                        "type": "recommendations_start",
                        "count": len(songs_list),
                        "exposure_id": _request_id,
                    }
                    for item in songs_list:
                        yield {
                            "type": "song",
                            "song": item["song"],
                            "index": item["index"],
                            "total": len(songs_list),
                            "exposure_id": _request_id,
                        }
                    yield {"type": "recommendations_complete"}
                    songs_already_sent = True
                    continue

                if isinstance(chunk, dict) and "__clarification__" in chunk:
                    clarification = chunk["__clarification__"]
                    accumulated_text = str(clarification.get("question") or "")
                    clarification_already_sent = True
                    yield {
                        "type": "clarification_required",
                        "text": accumulated_text,
                        "clarification_options": clarification.get("options") or [],
                        "clarification_reason": clarification.get("reason"),
                    }
                    continue
                
                accumulated_text += chunk
                yield {"type": "response", "text": accumulated_text, "is_complete": False}
            
            # 发送完整文本
            if accumulated_text and not clarification_already_sent:
                yield {"type": "response", "text": accumulated_text, "is_complete": True}
            
            # 等待图执行完毕
            await graph_task
            
            if "error" in result_holder:
                yield {"type": "error", "error": result_holder["error"]}
                return
            
            result = result_holder.get("result", {})
            raw_for_log = result.get("recommendations", [])
            recommendations_for_log = getattr(raw_for_log, "data", raw_for_log)
            if (
                isinstance(recommendations_for_log, list)
                and recommendations_for_log
                and not settings.eval_disable_side_effects
            ):
                try:
                    log_exposure(
                        query=query,
                        user_id=user_id,
                        request_id=_request_id,
                        recommendations=recommendations_for_log,
                        intent_type=result.get("intent_type", ""),
                        retrieval_meta=result.get("retrieval_meta", {}),
                        dialog_state=result.get("dialog_state", {}),
                        timings=result.get("timings", {}),
                    )
                except Exception as log_error:
                    logger.warning(f"[Feedback] 流式曝光日志写入失败: {log_error}")
            
            # 如果歌曲还没通过队列发送（兜底：非流式路径或队列推送失败）
            if not songs_already_sent:
                raw_recommendations = result.get("recommendations", [])
                recommendations = getattr(raw_recommendations, "data", raw_recommendations)
                if isinstance(recommendations, list) and recommendations:
                    yield {
                        "type": "recommendations_start",
                        "count": len(recommendations),
                        "exposure_id": _request_id,
                    }
                    for i, rec in enumerate(recommendations):
                        song = rec.get("song", rec) if isinstance(rec, dict) else rec
                        if isinstance(song, dict) and song.get("title"):
                            yield {
                                "type": "song",
                                "song": song,
                                "index": i,
                                "total": len(recommendations),
                                "exposure_id": _request_id,
                            }
                    yield {"type": "recommendations_complete"}
            
            yield {
                "type": "complete",
                "success": True,
                "exposure_id": _request_id,
                "retrieval_meta": result.get("retrieval_meta", {}),
                "dialog_state": result.get("dialog_state", {}),
                "dialog_delta": result.get("dialog_delta", {}),
                "clarification_options": result.get("clarification_options", []),
                "intent_confidence": result.get("intent_confidence", 1.0),
                "refinement_options": result.get("refinement_options", []),
            }
            logger.info(f"流式音乐推荐完成 [req={_request_id[:8]}]")

            # ── 微调方向 chips：歌曲与 complete 已送达后异步生成，fail-soft ──
            # 依据 LLM-first 原则：chips 由模型基于完整上下文 + 本轮 slate 产出，
            # 生成失败或超时只影响 chips，不影响歌曲与回复。
            if not clarification_already_sent and isinstance(recommendations_for_log, list) and recommendations_for_log:
                try:
                    from services.refinement_generator import get_refinement_generator

                    chips = await get_refinement_generator().generate(
                        user_id=user_id,
                        user_input=query,
                        chat_history=chat_history or [],
                        plan=result.get("retrieval_plan") or {},
                        dialog_state=result.get("dialog_state") or {},
                        memory_snapshot=result.get("graphzep_facts") or "",
                        recommendations=recommendations_for_log,
                        catalog_gap=(result.get("retrieval_meta") or {}).get("catalog_gap") or {},
                    )
                    yield {
                        "type": "refinement",
                        "exposure_id": _request_id,
                        "options": [option.model_dump() for option in chips],
                    }
                except Exception as chip_error:
                    logger.warning(f"[Refinement] chips 生成失败，已跳过: {chip_error} [req={_request_id[:8]}]")
            
        except asyncio.CancelledError:
            logger.info(f"🛑 流式推荐被取消 [req={_request_id[:8]}]")
            yield {"type": "error", "error": "推荐已被用户取消"}
        except Exception as e:
            logger.error(f"流式推荐失败: {str(e)} [req={_request_id[:8]}]", exc_info=True)
            yield {"type": "error", "error": str(e)}
        finally:
            # 清理本次请求的队列，防止内存泄漏
            self.graph._explanation_queues.pop(_request_id, None)
            self._current_graph_task = None  # 清理 task 引用
    
    def get_status(self) -> Dict[str, Any]:
        """获取智能体状态信息"""
        return {
            "status": "ready",
            "agent_type": "music_recommendation",
            "features": [
                "音乐搜索",
                "心情推荐",
                "场景推荐",
                "相似歌曲推荐",
                "艺术家推荐",
                "流派推荐",
                "智能对话"
            ],
            "supported_genres": [
                "流行", "摇滚", "民谣", "电子", 
                "说唱", "抒情", "古风", "爵士"
            ]
        }



