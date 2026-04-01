"""
P3: GSSC 上下文构建器 (Gather / Select / Structure / Compress)
灵感来源：HelloAgents 的 ContextBuilder 流水线
压缩升级：Claude Code compact.ts 的 LLM Agent 压缩模式

作用：在注入 LLM Prompt 之前，将多个上下文源（GraphZep 记忆、对话历史、
检索结果）按优先级筛选并截断到 Token 预算内，避免 Prompt 超长导致的
截断、遗忘和高成本问题。

Stage 4 升级（V2）：
  当 chat_history 远超预算（> 1.5 倍分配量）时，使用 LLM 生成摘要替代硬截断，
  减少信息损失。短对话仍保留原有的按行截断逻辑作为兜底。

典型调用点：
  analyze_intent / generate_explanation 等节点在拼 Prompt 前调用
  await build_context(sources, budget=2000) → 截断后的文本
"""

import logging
from typing import Dict

logger = logging.getLogger(__name__)

# ---- Token 估算（不依赖 tiktoken，适用于中英混合文本） ----
# 中文 1 字 ≈ 1.5 token，英文 1 word ≈ 1.3 token，标点忽略
# 这里用简单的字符数 / 2 作为保守估算

def estimate_tokens(text: str) -> int:
    """估算文本的 Token 数（保守估算，适用于中英混合）"""
    if not text:
        return 0
    # 中文字符数 + 英文单词数 × 1.3
    chinese_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
    remaining = len(text) - chinese_chars
    return int(chinese_chars * 1.5 + remaining * 0.4)


# ---- 上下文源优先级 ----
# 数字越小优先级越高
PRIORITY_USER_INPUT = 0        # 用户当前输入（不可截断）
PRIORITY_GRAPHZEP_FACTS = 1    # 长期记忆（高优先级）
PRIORITY_CHAT_HISTORY = 2      # 最近对话历史
PRIORITY_RETRIEVAL = 3         # 检索结果

# ---- LLM 压缩触发阈值 ----
LLM_COMPRESS_RATIO = 1.5       # 超出分配预算的倍数阈值


class ContextSource:
    """一个上下文源"""
    def __init__(self, name: str, content: str, priority: int, min_tokens: int = 0):
        self.name = name
        self.content = content
        self.priority = priority
        self.min_tokens = min_tokens  # 最少保留的 token 数（0=可完全截断）
        self.estimated_tokens = estimate_tokens(content)
    
    def truncate_to(self, max_tokens: int) -> str:
        """按行截断内容到指定 Token 数（传统兜底方式）"""
        if self.estimated_tokens <= max_tokens:
            return self.content
        
        lines = self.content.split("\n")
        result = []
        used = 0
        for line in lines:
            line_tokens = estimate_tokens(line)
            if used + line_tokens > max_tokens:
                if result:
                    result.append("... (已截断)")
                break
            result.append(line)
            used += line_tokens
        
        return "\n".join(result)


async def _llm_compress_chat_history(chat_history: str) -> str:
    """
    使用 LLM 将冗长的对话历史压缩为摘要。
    
    借鉴 Claude Code compact.ts 的思路：用一个轻量 LLM 调用，
    将旧对话轮次生成结构化摘要，替代硬截断。
    
    使用意图分析专用的小模型（如 Qwen3-4B），成本低、速度快。
    """
    try:
        from llms.multi_llm import get_intent_chat_model
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.output_parsers import StrOutputParser
        from llms.prompts import CONTEXT_COMPRESSOR_PROMPT
        
        llm = get_intent_chat_model()
        chain = (
            ChatPromptTemplate.from_template(CONTEXT_COMPRESSOR_PROMPT)
            | llm
            | StrOutputParser()
        )
        summary = await chain.ainvoke({"chat_history": chat_history})
        summary = summary.strip()
        
        # 清理可能的 <think>...</think> 残留（本地模型常见）
        if "<think>" in summary:
            think_end = summary.find("</think>")
            if think_end > 0:
                summary = summary[think_end + 8:].strip()
        
        original_tokens = estimate_tokens(chat_history)
        compressed_tokens = estimate_tokens(summary)
        logger.info(
            f"[GSSC] LLM 压缩成功: {original_tokens} → {compressed_tokens} tokens "
            f"(压缩率: {compressed_tokens / max(original_tokens, 1):.1%})"
        )
        return summary
        
    except Exception as e:
        logger.warning(f"[GSSC] LLM 压缩失败，退回按行截断: {e}")
        return None  # 返回 None 表示失败，调用方会退回 truncate_to


async def build_context(
    graphzep_facts: str = "",
    chat_history: str = "",
    retrieval_context: str = "",
    user_input: str = "",
    total_budget: int = 3000,
) -> Dict[str, str]:
    """
    GSSC 四阶段上下文构建（V2 异步版）

    V2 升级：Stage 4 新增 LLM 智能压缩分支。
    当 chat_history 的 token 数远超分配预算（> 1.5 倍）时，
    调用 LLM 生成摘要来替代硬截断，减少信息损失。

    Args:
        graphzep_facts: GraphZep 长期记忆文本
        chat_history: 格式化的对话历史文本
        retrieval_context: 检索结果文本（可选）
        user_input: 用户当前输入（不截断）
        total_budget: 总 Token 预算（不含 system prompt 和 user_input）

    Returns:
        dict: {"graphzep_facts": ..., "chat_history": ..., "retrieval_context": ...}
              各字段已按优先级截断到预算内
    """
    # ---- Stage 1: Gather（收集所有上下文源） ----
    sources = []
    
    if graphzep_facts and graphzep_facts != "暂无用户长期记忆":
        sources.append(ContextSource(
            name="graphzep_facts",
            content=graphzep_facts,
            priority=PRIORITY_GRAPHZEP_FACTS,
            min_tokens=100,  # 至少保留 100 token 的记忆
        ))
    
    if chat_history:
        sources.append(ContextSource(
            name="chat_history",
            content=chat_history,
            priority=PRIORITY_CHAT_HISTORY,
            min_tokens=200,  # 至少保留最近 2-3 轮对话
        ))
    
    if retrieval_context:
        sources.append(ContextSource(
            name="retrieval_context",
            content=retrieval_context,
            priority=PRIORITY_RETRIEVAL,
            min_tokens=0,  # 可以完全省略
        ))
    
    if not sources:
        return {
            "graphzep_facts": graphzep_facts,
            "chat_history": chat_history,
            "retrieval_context": retrieval_context,
        }
    
    # ---- Stage 2: Select（按优先级排序） ----
    sources.sort(key=lambda s: s.priority)
    total_estimated = sum(s.estimated_tokens for s in sources)
    
    if total_estimated <= total_budget:
        # 总量在预算内，无需截断
        logger.info(f"[GSSC] 上下文总量 {total_estimated} tokens ≤ 预算 {total_budget}，无需截断")
        return {
            "graphzep_facts": graphzep_facts,
            "chat_history": chat_history,
            "retrieval_context": retrieval_context,
        }
    
    logger.info(f"[GSSC] 上下文总量 {total_estimated} tokens > 预算 {total_budget}，启动截断")
    
    # ---- Stage 3: Structure（分配预算） ----
    # 先保证每个源的 min_tokens，剩余按优先级分配
    min_total = sum(s.min_tokens for s in sources)
    remaining_budget = total_budget - min_total
    
    allocations: Dict[str, int] = {}
    for src in sources:
        # 基础配额 = min_tokens
        # 额外配额 = 剩余预算按优先级倒序分配（优先级高的先分配）
        extra_needed = max(0, src.estimated_tokens - src.min_tokens)
        extra_allocated = min(extra_needed, remaining_budget)
        allocations[src.name] = src.min_tokens + extra_allocated
        remaining_budget -= extra_allocated
    
    # ---- Stage 4: Compress（智能压缩 — V2 升级） ----
    result = {
        "graphzep_facts": graphzep_facts,
        "chat_history": chat_history,
        "retrieval_context": retrieval_context,
    }
    
    for src in sources:
        budget = allocations.get(src.name, src.estimated_tokens)
        if src.estimated_tokens <= budget:
            continue  # 不需要压缩
        
        # V2 升级：chat_history 远超预算时尝试 LLM 摘要压缩
        if (
            src.name == "chat_history"
            and src.estimated_tokens > budget * LLM_COMPRESS_RATIO
        ):
            logger.info(
                f"[GSSC] chat_history ({src.estimated_tokens} tokens) 远超预算 "
                f"({budget} tokens, ×{LLM_COMPRESS_RATIO})，尝试 LLM 压缩"
            )
            compressed = await _llm_compress_chat_history(src.content)
            if compressed is not None:
                result[src.name] = compressed
                logger.info(
                    f"[GSSC] {src.name}: LLM 压缩 {src.estimated_tokens} → "
                    f"{estimate_tokens(compressed)} tokens (预算: {budget})"
                )
                continue
            # LLM 压缩失败，fall through 到按行截断
        
        # 按行截断兜底（V1 原有逻辑）
        truncated = src.truncate_to(budget)
        result[src.name] = truncated
        logger.info(
            f"[GSSC] {src.name}: 按行截断 {src.estimated_tokens} → "
            f"{estimate_tokens(truncated)} tokens (预算: {budget})"
        )
    
    return result
