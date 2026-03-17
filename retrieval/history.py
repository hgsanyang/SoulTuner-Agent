from typing import List, Sequence
import tiktoken
from langchain_core.messages import BaseMessage, trim_messages
from config.settings import settings
from config.logging_config import get_logger

logger = get_logger(__name__)

def count_tokens(messages: Sequence[BaseMessage]) -> int:
    """估算消息列表的 Token 数量"""
    try:
        encoding = tiktoken.get_encoding("cl100k_base")
        count = 0
        for m in messages:
            if hasattr(m, 'content') and isinstance(m.content, str):
                count += len(encoding.encode(m.content, disallowed_special=()))
        return count
    except Exception:
        # Fallback 估算
        count = 0
        for m in messages:
            if hasattr(m, 'content') and isinstance(m.content, str):
                count += len(m.content) // 2
        return count

class MusicContextManager:
    """
    负责 Agent 会话上下文的动态截断和管理。
    """
    def __init__(self):
        self.max_tokens = settings.max_context_tokens

    def format_chat_history(self, chat_history: Sequence[BaseMessage]) -> str:
        """
        根据配置动态截断并格式化历史记录。
        利用 LangChain 原生 trim_messages 函数。
        返回格式化后的字符串。
        """
        if not chat_history:
            return "无"

        try:
            trimmed_messages = trim_messages(
                chat_history,
                max_tokens=self.max_tokens,
                strategy="last",
                token_counter=count_tokens,
                include_system=True,
                allow_partial=True
            )
        except Exception as e:
            logger.warning(f"历史记录截断出错: {e}，使用简单截断")
            retain_count = settings.memory_retain_rounds * 2
            trimmed_messages = list(chat_history)[-retain_count:] if retain_count > 0 else []
            
        if not trimmed_messages:
            return "无"
            
        formatted_lines = []
        for msg in trimmed_messages:
            role = "user" if msg.type == "human" else ("assistant" if msg.type == "ai" else msg.type)
            content = msg.content if isinstance(msg.content, str) else str(msg.content)
            formatted_lines.append(f"{role}: {content}")
            
        return "\n".join(formatted_lines)


