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
        
        try:
            logger.info(f"开始处理音乐推荐请求(流式): {query}")
            
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
            
            # 创建共享队列：generate_explanation 节点会往这里推 chunk
            explanation_queue = asyncio.Queue()
            
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
                "metadata": {},
                "_explanation_queue": explanation_queue,
            }
            
            config = {"recursion_limit": 50}
            
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
            
            # 发送思考状态
            yield {"type": "thinking", "message": "正在理解你的音乐偏好..."}
            
            # 从队列读取流式解释文本（歌曲数据也会通过队列提前到达）
            accumulated_text = ""
            songs_already_sent = False
            while True:
                try:
                    chunk = await asyncio.wait_for(explanation_queue.get(), timeout=90)
                except asyncio.TimeoutError:
                    yield {"type": "error", "error": "推荐生成超时，请重试"}
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
            logger.info("流式音乐推荐完成")
            
        except Exception as e:
            logger.error(f"流式推荐失败: {str(e)}", exc_info=True)
            yield {"type": "error", "error": str(e)}
    
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



