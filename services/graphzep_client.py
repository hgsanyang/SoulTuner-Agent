"""
GraphZep 微服务 HTTP 客户端
负责与 GraphZep Server (Node.js) 通信
"""

import httpx
import asyncio
from datetime import datetime, timezone
from typing import Optional
from config.logging_config import get_logger

logger = get_logger(__name__)

from config.settings import settings

GRAPHZEP_BASE_URL = settings.graphzep_base_url
DEFAULT_GROUP_ID = "music-agent-memory"


class GraphZepClient:
    """GraphZep HTTP 客户端（单例）"""

    def __init__(self, base_url: str = GRAPHZEP_BASE_URL):
        self.base_url = base_url
        self._client = httpx.AsyncClient(base_url=base_url, timeout=10.0)

    # ---- 写入：将对话/事件送入 GraphZep ----

    async def add_messages(
        self,
        user_message: str,
        bot_response: str,
        group_id: str = DEFAULT_GROUP_ID,
        user_name: str = "用户",
    ) -> bool:
        """
        将一轮对话（用户 + Bot）送入 GraphZep。
        GraphZep 内部会异步执行 LLM 实体抽取，调用方无需等待。
        """
        now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
        payload = {
            "group_id": group_id,
            "messages": [
                {
                    "content": user_message,
                    "role_type": "user",
                    "role": user_name,
                    "timestamp": now,
                    "source_description": "MusicAgent对话",
                },
                {
                    "content": bot_response,
                    "role_type": "assistant",
                    "role": "MusicBot",
                    "timestamp": now,
                    "source_description": "MusicAgent对话",
                },
            ],
        }
        try:
            resp = await self._client.post("/messages", json=payload)
            resp.raise_for_status()
            logger.info(f"[GraphZep] 对话已送入处理队列: group={group_id}")
            return True
        except Exception as e:
            logger.warning(f"[GraphZep] 写入失败（不影响主流程）: {e}")
            return False

    async def add_user_event(
        self,
        event_description: str,
        group_id: str = DEFAULT_GROUP_ID,
        user_name: str = "用户",
    ) -> bool:
        """
        将用户行为事件（点赞/跳过/收藏等）以自然语言送入 GraphZep。
        GraphZep 会自动从中抽取 (用户)-[LIKES/DISLIKES]->(歌曲) 等关系。
        """
        now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
        payload = {
            "group_id": group_id,
            "messages": [
                {
                    "content": event_description,
                    "role_type": "system",
                    "role": "EventCollector",
                    "timestamp": now,
                    "source_description": "用户行为事件",
                }
            ],
        }
        try:
            resp = await self._client.post("/messages", json=payload)
            resp.raise_for_status()
            logger.info(f"[GraphZep] 行为事件已记录: {event_description[:50]}...")
            return True
        except Exception as e:
            logger.warning(f"[GraphZep] 事件写入失败: {e}")
            return False

    # ---- 读取：从 GraphZep 检索事实 ----

    async def search_facts(
        self,
        query: str,
        group_ids: Optional[list[str]] = None,
        max_facts: int = 8,
        search_type: str = "hybrid",
    ) -> str:
        """
        语义检索相关事实。返回格式化的文本字符串，可直接注入 Prompt。
        search_type: 'semantic' | 'keyword' | 'hybrid' | 'mmr'
        """
        payload = {
            "query": query,
            "max_facts": max_facts,
            "search_type": search_type,
        }
        if group_ids:
            payload["group_ids"] = group_ids

        try:
            resp = await self._client.post("/search", json=payload)
            resp.raise_for_status()
            data = resp.json()
            facts = data.get("facts", [])
            if not facts:
                return "暂无用户长期记忆"
            lines = []
            for f in facts:
                fact_text = f.get("fact", "")
                valid_at = f.get("valid_at", "")
                lines.append(f"- {fact_text}" + (f" (时间: {valid_at})" if valid_at else ""))
            return "\n".join(lines)
        except Exception as e:
            logger.warning(f"[GraphZep] 检索失败: {e}")
            return "暂无用户长期记忆（GraphZep 服务不可用）"

    async def get_memory(
        self,
        recent_messages: list[dict],
        group_id: str = DEFAULT_GROUP_ID,
        max_facts: int = 8,
    ) -> str:
        """
        基于最近几条对话获取相关记忆上下文。
        recent_messages 格式: [{"content": "...", "role_type": "user/assistant"}]
        """
        payload = {
            "group_id": group_id,
            "max_facts": max_facts,
            "messages": [
                {
                    "content": m["content"],
                    "role_type": m.get("role_type", "user"),
                    "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z",
                }
                for m in recent_messages[-3:]  # 只取最近 3 条
            ],
        }
        try:
            resp = await self._client.post("/get-memory", json=payload)
            resp.raise_for_status()
            data = resp.json()
            facts = data.get("facts", [])
            if not facts:
                return "暂无用户长期记忆"
            return "\n".join(f"- {f['fact']}" for f in facts)
        except Exception as e:
            logger.warning(f"[GraphZep] 记忆获取失败: {e}")
            return "暂无用户长期记忆（GraphZep 服务不可用）"

    # ---- 健康检查 ----

    async def healthcheck(self) -> bool:
        try:
            resp = await self._client.get("/healthcheck")
            return resp.status_code == 200
        except Exception:
            return False

    async def close(self):
        await self._client.aclose()


# 全局单例
_graphzep_client: Optional[GraphZepClient] = None

def get_graphzep_client() -> GraphZepClient:
    global _graphzep_client
    if _graphzep_client is None:
        _graphzep_client = GraphZepClient()
    return _graphzep_client
