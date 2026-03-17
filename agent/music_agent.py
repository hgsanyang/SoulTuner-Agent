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



