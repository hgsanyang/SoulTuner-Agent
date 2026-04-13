"""
音乐推荐Agent主入口
提供完整的音乐推荐功能
"""

import asyncio
import os
from typing import Dict, Any, Optional, List


from config.logging_config import get_logger
from agent.music_graph import MusicRecommendationGraph
from schemas.music_state import MusicAgentState
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
        user_preferences: Optional[Dict[str, Any]] = None
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
                "metadata": {}
            }
            
            # 执行工作流
            config = {
                "recursion_limit": 50
            }
            # MemorySaver Checkpoint: 传入 thread_id 实现对话状态持久化
            if getattr(self.graph, 'checkpointer', None):
                import uuid
                thread_id = config.get("configurable", {}).get("thread_id", str(uuid.uuid4()))
                config["configurable"] = {"thread_id": thread_id}
                logger.info(f"[Checkpoint] thread_id={thread_id}")
            result = await self.app.ainvoke(initial_state, config=config)
            
            logger.info("音乐推荐完成")
            
            return {
                "success": True,
                "response": result.get("final_response", ""),
                "recommendations": result.get("recommendations", []),
                "search_results": result.get("search_results", []),
                "intent_type": result.get("intent_type", ""),
                "explanation": result.get("explanation", ""),
                "playlist": result.get("playlist"),
                "errors": result.get("error_log", [])
            }
            
        except Exception as e:
            logger.error(f"处理音乐推荐请求时发生错误: {str(e)}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
                "response": "抱歉，处理你的请求时遇到了问题。请稍后重试。",
                "recommendations": [],
                "search_results": [],
                "errors": [{"node": "main", "error": str(e)}]
            }
    
    async def stream_recommendations(
        self,
        query: str,
        chat_history: Optional[List[Dict[str, str]]] = None,
        user_preferences: Optional[Dict[str, Any]] = None
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
                "metadata": {"request_id": _request_id},
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
                    yield {"type": "recommendations_start", "count": len(songs_list)}
                    for item in songs_list:
                        yield {"type": "song", "song": item["song"], "index": item["index"], "total": len(songs_list)}
                    yield {"type": "recommendations_complete"}
                    songs_already_sent = True
                    continue
                
                accumulated_text += chunk
                yield {"type": "response", "text": accumulated_text, "is_complete": False}
            
            # 发送完整文本
            if accumulated_text:
                yield {"type": "response", "text": accumulated_text, "is_complete": True}
            
            # 等待图执行完毕
            await graph_task
            
            if "error" in result_holder:
                yield {"type": "error", "error": result_holder["error"]}
                return
            
            result = result_holder.get("result", {})
            
            # 如果歌曲还没通过队列发送（兜底：非流式路径或队列推送失败）
            if not songs_already_sent:
                raw_recommendations = result.get("recommendations", [])
                recommendations = getattr(raw_recommendations, "data", raw_recommendations)
                if isinstance(recommendations, list) and recommendations:
                    yield {"type": "recommendations_start", "count": len(recommendations)}
                    for i, rec in enumerate(recommendations):
                        song = rec.get("song", rec) if isinstance(rec, dict) else rec
                        if isinstance(song, dict) and song.get("title"):
                            yield {"type": "song", "song": song, "index": i, "total": len(recommendations)}
                    yield {"type": "recommendations_complete"}
            
            yield {"type": "complete", "success": True}
            logger.info(f"流式音乐推荐完成 [req={_request_id[:8]}]")
            
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



