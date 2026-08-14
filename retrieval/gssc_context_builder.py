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

import hashlib
import logging
import time
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple

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


TokenCounter = Callable[[str], int]


def _count_tokens(text: str, token_counter: Optional[TokenCounter] = None) -> int:
    """Count tokens with an injected model tokenizer, falling back to estimation."""
    return max(0, int((token_counter or estimate_tokens)(text or "")))


# ---- 上下文源优先级 ----
# 数字越小优先级越高
PRIORITY_USER_INPUT = 0        # 用户当前输入（不可截断）
PRIORITY_CHAT_HISTORY = 1      # 当前会话状态、纠正与否定
PRIORITY_EXPLICIT_PROFILE = 2  # 用户主动设置的明确画像
PRIORITY_GRAPHZEP_FACTS = 3    # 当前场景相关的长期记忆（兼容旧字段名）
PRIORITY_RETRIEVAL = 4         # 检索结果

# ---- LLM 压缩触发阈值 ----
LLM_COMPRESS_RATIO = 1.5       # 超出分配预算的倍数阈值

# ============================================================
# ★ 预压缩缓存（AsyncPreCompress Cache）
# 每轮对话结束后，异步预压缩对话历史并缓存，
# 下次请求直接读取，消除 17s 阻塞等待。
#
# 缓存按 user_id + conversation_id + history_fingerprint 隔离。
# 命中时还会校验 covered_message_ids 是当前历史的精确前缀；摘要之后新增的
# 对话必须以原文拼回 Prompt，避免旧摘要吞掉本轮纠正或否定。
# ============================================================
COMPRESSION_CACHE_VERSION = "gssc-summary-v2"
COMPRESSION_CACHE_TTL_SECONDS = 60 * 60


@dataclass(frozen=True)
class CompressionCacheEntry:
    user_id: str
    conversation_id: str
    history_fingerprint: str
    summary: str
    covered_message_ids: Tuple[str, ...]
    last_message_id: str
    generated_at: float
    original_token_count: int
    schema_version: str = COMPRESSION_CACHE_VERSION


@dataclass(frozen=True)
class CompressionCacheHit:
    entry: CompressionCacheEntry
    uncovered_text: str


_compress_cache: Dict[Tuple[str, str, str], CompressionCacheEntry] = {}


def _history_lines(chat_history_text: str) -> Tuple[str, ...]:
    """将历史稳定拆分为可做前缀校验的消息行。"""
    return tuple(chat_history_text.splitlines())


def _message_ids(lines: Tuple[str, ...]) -> Tuple[str, ...]:
    return tuple(
        hashlib.sha256(f"{index}\0{line}".encode("utf-8")).hexdigest()[:20]
        for index, line in enumerate(lines)
    )


def _history_fingerprint(lines: Tuple[str, ...]) -> str:
    return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()


def clear_compression_cache() -> None:
    """清空进程内摘要缓存；主要用于测试和显式会话重置。"""
    _compress_cache.clear()


def get_cached_compression(
    user_id: str,
    conversation_id: str,
    chat_history_text: str,
    *,
    now: Optional[float] = None,
) -> Optional[CompressionCacheHit]:
    """
    读取预压缩缓存。

    只有同一用户、同一会话且摘要覆盖的消息 ID 是当前历史的精确前缀时命中。
    TTL、摘要版本、任一前缀内容变化都会使缓存失效。
    """
    if not user_id or not conversation_id:
        return None

    current_time = time.time() if now is None else now
    current_lines = _history_lines(chat_history_text)
    current_ids = _message_ids(current_lines)
    scoped_entries = [
        entry
        for entry in _compress_cache.values()
        if entry.user_id == user_id and entry.conversation_id == conversation_id
    ]
    scoped_entries.sort(key=lambda entry: len(entry.covered_message_ids), reverse=True)

    for entry in scoped_entries:
        if entry.schema_version != COMPRESSION_CACHE_VERSION:
            continue
        if current_time - entry.generated_at > COMPRESSION_CACHE_TTL_SECONDS:
            _compress_cache.pop(
                (entry.user_id, entry.conversation_id, entry.history_fingerprint), None
            )
            continue
        covered = entry.covered_message_ids
        if len(current_ids) < len(covered) or current_ids[: len(covered)] != covered:
            continue
        uncovered_text = "\n".join(current_lines[len(covered) :])
        logger.info(
            "[GSSC-Cache] 命中会话摘要缓存 (user=%s, conversation=%s, "
            "covered=%s, appended=%s)",
            user_id,
            conversation_id[:12],
            len(covered),
            len(current_ids) - len(covered),
        )
        return CompressionCacheHit(entry=entry, uncovered_text=uncovered_text)

    # 同一会话出现非前缀历史，说明会话被重置或改写，立即清除旧摘要。
    if scoped_entries:
        for entry in scoped_entries:
            _compress_cache.pop(
                (entry.user_id, entry.conversation_id, entry.history_fingerprint), None
            )
        logger.info(
            "[GSSC-Cache] 会话历史不是缓存前缀，已清除 (user=%s, conversation=%s)",
            user_id,
            conversation_id[:12],
        )
    return None


async def pre_compress_and_cache(
    user_id: str,
    chat_history_text: str,
    conversation_id: str = "",
) -> None:
    """
    ★ 在每轮对话结束后异步调用（asyncio.create_task），
    预压缩本轮对话历史，写入缓存供下次请求直接使用。

    调用时机：与 extract_preferences_node 并行执行，
    在推荐解释生成完毕、返回响应之后触发。
    """
    if not conversation_id:
        logger.info("[GSSC-Cache] 缺少 conversation_id，跳过摘要缓存")
        return

    original_tokens = estimate_tokens(chat_history_text)

    # 只有历史足够长（> 1500 tokens）时才预压缩，短历史直接截断即可
    if original_tokens <= 1500:
        logger.info(
            f"[GSSC-Cache] 历史较短 ({original_tokens} tokens)，无需预压缩 (user={user_id})"
        )
        return

    logger.info(
        f"[GSSC-Cache] 开始异步预压缩 (user={user_id}, {original_tokens} tokens) ..."
    )
    compressed = await _llm_compress_chat_history(chat_history_text)
    if compressed:
        lines = _history_lines(chat_history_text)
        message_ids = _message_ids(lines)
        fingerprint = _history_fingerprint(lines)
        entry = CompressionCacheEntry(
            user_id=user_id,
            conversation_id=conversation_id,
            history_fingerprint=fingerprint,
            summary=compressed,
            covered_message_ids=message_ids,
            last_message_id=message_ids[-1] if message_ids else "",
            generated_at=time.time(),
            original_token_count=original_tokens,
        )
        for key, cached_entry in list(_compress_cache.items()):
            if (
                cached_entry.user_id == user_id
                and cached_entry.conversation_id == conversation_id
            ):
                _compress_cache.pop(key, None)
        _compress_cache[(user_id, conversation_id, fingerprint)] = entry
        logger.info(
            f"[GSSC-Cache] 预压缩完成并写入缓存 (user={user_id}): "
            f"{original_tokens} → {estimate_tokens(compressed)} tokens"
        )
    else:
        logger.warning(f"[GSSC-Cache] 预压缩失败 (user={user_id})，下次请求将重新压缩")



def _truncate_text_to_budget(
    text: str,
    max_tokens: int,
    *,
    from_end: bool,
    token_counter: Optional[TokenCounter] = None,
) -> str:
    """Truncate one text block against either an exact or estimated token budget."""
    if max_tokens <= 0 or not text:
        return ""
    if _count_tokens(text, token_counter) <= max_tokens:
        return text
    low, high = 0, len(text)
    while low < high:
        mid = (low + high + 1) // 2
        candidate = text[-mid:] if from_end else text[:mid]
        if _count_tokens(candidate, token_counter) <= max_tokens:
            low = mid
        else:
            high = mid - 1
    return text[-low:] if from_end else text[:low]


class ContextSource:
    """一个上下文源"""
    def __init__(
        self,
        name: str,
        content: str,
        priority: int,
        min_tokens: int = 0,
        preserve: str = "head",
        token_counter: Optional[TokenCounter] = None,
    ):
        self.name = name
        self.content = content
        self.priority = priority
        self.min_tokens = min_tokens  # 最少保留的 token 数（0=可完全截断）
        self.preserve = preserve
        self.token_counter = token_counter
        self.estimated_tokens = _count_tokens(content, token_counter)

    def truncate_to(self, max_tokens: int) -> str:
        """按行截断内容；会话历史保留最新内容，其他源默认保留开头。"""
        if max_tokens <= 0:
            return ""
        if self.estimated_tokens <= max_tokens:
            return self.content

        marker = "... (较早内容已截断)" if self.preserve == "tail" else "... (已截断)"
        marker_tokens = _count_tokens(marker, self.token_counter)
        content_budget = max(0, max_tokens - marker_tokens)
        lines = self.content.split("\n")
        indexed_lines = list(reversed(lines)) if self.preserve == "tail" else lines
        result = []
        used = 0
        for line in indexed_lines:
            line_tokens = _count_tokens(line, self.token_counter)
            if used + line_tokens > content_budget:
                if not result:
                    result.append(
                        _truncate_text_to_budget(
                            line,
                            content_budget,
                            from_end=self.preserve == "tail",
                            token_counter=self.token_counter,
                        )
                    )
                break
            result.append(line)
            used += line_tokens
        if self.preserve == "tail":
            result.reverse()
            result.insert(0, marker)
        else:
            result.append(marker)
        rendered = "\n".join(part for part in result if part)
        # Line separators and markers are real tokens too. Enforce the final
        # boundary with the same counter used for allocation.
        if _count_tokens(rendered, self.token_counter) > max_tokens:
            rendered = _truncate_text_to_budget(
                rendered,
                max_tokens,
                from_end=self.preserve == "tail",
                token_counter=self.token_counter,
            )
        return rendered


def _render_cached_history(
    hit: CompressionCacheHit,
    budget: int,
    token_counter: Optional[TokenCounter] = None,
) -> str:
    """摘要只代表已覆盖前缀；摘要后的新增轮次优先以原文保留。"""
    if not hit.uncovered_text:
        return _truncate_text_to_budget(
            hit.entry.summary,
            budget,
            from_end=False,
            token_counter=token_counter,
        )

    recent_marker = "【摘要后新增对话】"
    summary_marker = "【较早对话摘要】"
    marker_budget = _count_tokens(recent_marker + summary_marker, token_counter)
    content_budget = max(0, budget - marker_budget)
    recent = ContextSource(
        "uncovered_history",
        hit.uncovered_text,
        PRIORITY_CHAT_HISTORY,
        preserve="tail",
        token_counter=token_counter,
    ).truncate_to(content_budget)
    remaining = max(0, content_budget - _count_tokens(recent, token_counter))
    summary = _truncate_text_to_budget(
        hit.entry.summary,
        remaining,
        from_end=False,
        token_counter=token_counter,
    )
    parts = []
    if summary:
        parts.append(f"{summary_marker}\n{summary}")
    if recent:
        parts.append(f"{recent_marker}\n{recent}")
    return "\n".join(parts)


async def _llm_compress_chat_history(chat_history: str) -> str:
    """
    使用 LLM 将冗长的对话历史压缩为摘要。

    借鉴 Claude Code compact.ts 的思路：用一个轻量 LLM 调用，
    将旧对话轮次生成结构化摘要，替代硬截断。

    使用意图分析专用的小模型（如 Qwen3-4B），成本低、速度快。
    """
    try:
        from llms.multi_llm import get_compress_chat_model
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.output_parsers import StrOutputParser
        from llms.prompts import CONTEXT_COMPRESSOR_PROMPT

        llm = get_compress_chat_model()
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
    explicit_profile: str = "",
    graphzep_facts: str = "",
    chat_history: str = "",
    retrieval_context: str = "",
    user_input: str = "",
    total_budget: int = 0,
    user_id: str = "local_admin",
    conversation_id: str = "",
    token_counter: Optional[TokenCounter] = None,
) -> Dict[str, str]:
    """
    GSSC 四阶段上下文构建（V2 异步版）

    V2 升级：Stage 4 新增 LLM 智能压缩分支。
    当 chat_history 的 token 数远超分配预算（> 1.5 倍）时，
    调用 LLM 生成摘要来替代硬截断，减少信息损失。

    Args:
        explicit_profile: 用户主动设置的明确画像，优先级高于长期记忆
        graphzep_facts: 自研 MemoryGateway 召回的长期记忆文本（兼容旧字段名）
        chat_history: 格式化的对话历史文本
        retrieval_context: 检索结果文本（可选）
        user_input: 用户当前输入（不截断）
        total_budget: 总 Token 预算（不含 system prompt 和 user_input）

    Returns:
        dict: {"explicit_profile": ..., "graphzep_facts": ...,
               "chat_history": ..., "retrieval_context": ...}
              各字段已按优先级截断到预算内
    """
    # ---- 读取预算配置 ----
    if total_budget <= 0:
        try:
            from config.settings import settings
            total_budget = settings.context_total_budget
        except Exception:
            total_budget = 8000

    # ---- Stage 1: Gather（收集所有上下文源） ----
    sources = []

    if explicit_profile:
        sources.append(ContextSource(
            name="explicit_profile",
            content=explicit_profile,
            priority=PRIORITY_EXPLICIT_PROFILE,
            min_tokens=100,
            token_counter=token_counter,
        ))

    if graphzep_facts and graphzep_facts != "暂无用户长期记忆":
        sources.append(ContextSource(
            name="graphzep_facts",
            content=graphzep_facts,
            priority=PRIORITY_GRAPHZEP_FACTS,
            min_tokens=100,
            token_counter=token_counter,
        ))

    if chat_history:
        sources.append(ContextSource(
            name="chat_history",
            content=chat_history,
            priority=PRIORITY_CHAT_HISTORY,
            min_tokens=200,
            preserve="tail",
            token_counter=token_counter,
        ))

    if retrieval_context:
        sources.append(ContextSource(
            name="retrieval_context",
            content=retrieval_context,
            priority=PRIORITY_RETRIEVAL,
            min_tokens=0,  # 可以完全省略
            token_counter=token_counter,
        ))

    if not sources:
        return {
            "explicit_profile": explicit_profile,
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
            "explicit_profile": explicit_profile,
            "graphzep_facts": graphzep_facts,
            "chat_history": chat_history,
            "retrieval_context": retrieval_context,
        }

    logger.info(f"[GSSC] 上下文总量 {total_estimated} tokens > 预算 {total_budget}，启动截断")

    # ---- Stage 3: Structure（分配预算） ----
    # 先保证每个源的 min_tokens，剩余按优先级分配
    remaining_budget = total_budget
    allocations: Dict[str, int] = {}
    # Guarantee minima in priority order without ever oversubscribing a tiny budget.
    for src in sources:
        guaranteed = min(src.min_tokens, src.estimated_tokens, remaining_budget)
        allocations[src.name] = guaranteed
        remaining_budget -= guaranteed
    # Then let higher-priority sources consume the remaining budget first.
    for src in sources:
        extra_needed = max(0, src.estimated_tokens - allocations[src.name])
        extra_allocated = min(extra_needed, remaining_budget)
        allocations[src.name] += extra_allocated
        remaining_budget -= extra_allocated

    # ---- Stage 4: Compress（智能压缩 — V2 升级） ----
    result = {
        "explicit_profile": explicit_profile,
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
            # ★ 先查预压缩缓存（上一轮结束后异步预计算的结果）
            cached = get_cached_compression(
                user_id,
                conversation_id,
                chat_history,
            )
            if cached is not None:
                result[src.name] = _render_cached_history(
                    cached,
                    budget,
                    token_counter=token_counter,
                )
                logger.info(
                    "[GSSC] chat_history: 使用预压缩缓存，跳过 LLM 调用 "
                    "(节省 ~15-20s)"
                )
                continue

            # ★★ 缓存未命中：不再同步调用 LLM 压缩（会阻塞意图识别 15-80s）
            # 直接 fall through 到按行截断兜底
            # 预压缩将在本轮结束后由 pre_compress_and_cache 异步执行，
            # 下次请求就能命中缓存
            logger.info(
                f"[GSSC] chat_history ({src.estimated_tokens} tokens) 远超预算 "
                f"({budget} tokens)，预压缩缓存未命中，使用按行截断兜底"
            )
            # fall through 到下面的 truncate_to

        # 按行截断兜底（V1 原有逻辑）
        truncated = src.truncate_to(budget)
        result[src.name] = truncated
        logger.info(
            f"[GSSC] {src.name}: 按行截断 {src.estimated_tokens} → "
            f"{_count_tokens(truncated, token_counter)} tokens (预算: {budget})"
        )

    # ---- Token Tracking Report（结构化追踪日志） ----
    # 用于性能分析和面试展示：压缩前 vs 压缩后的 Token 对比
    _track_token_savings(
        before={
            "explicit_profile": _count_tokens(explicit_profile, token_counter),
            "graphzep_facts": _count_tokens(graphzep_facts, token_counter),
            "chat_history": _count_tokens(chat_history, token_counter),
            "retrieval_context": _count_tokens(retrieval_context, token_counter),
        },
        after={
            "explicit_profile": _count_tokens(result.get("explicit_profile", ""), token_counter),
            "graphzep_facts": _count_tokens(result.get("graphzep_facts", ""), token_counter),
            "chat_history": _count_tokens(result.get("chat_history", ""), token_counter),
            "retrieval_context": _count_tokens(result.get("retrieval_context", ""), token_counter),
        },
        budget=total_budget,
    )

    return result


def _track_token_savings(
    before: Dict[str, int],
    after: Dict[str, int],
    budget: int,
) -> None:
    """
    结构化 Token 消耗追踪日志

    输出格式（每次 GSSC 调用记录一次）：
    ┌─────────────────────────────────────────────────┐
    │ [GSSC Token Report]                             │
    │ Source            Before    After    Saved       │
    │ graphzep_facts       120      120       0 (0%)  │
    │ chat_history        2400      350   2050 (85%)  │
    │ retrieval_context    800      600    200 (25%)  │
    │ ─────────────────────────────────────────       │
    │ TOTAL              3320     1070   2250 (68%)   │
    │ Budget: 3000                                    │
    └─────────────────────────────────────────────────┘
    """
    total_before = sum(before.values())
    total_after = sum(after.values())
    total_saved = total_before - total_after

    if total_saved <= 0:
        return  # 没有压缩发生，不输出报告

    lines = ["[GSSC Token Report]"]
    lines.append(f"  {'Source':<22s} {'Before':>7s} {'After':>7s} {'Saved':>10s}")
    lines.append(f"  {'─' * 50}")

    for key in ["graphzep_facts", "chat_history", "retrieval_context"]:
        b = before.get(key, 0)
        a = after.get(key, 0)
        saved = b - a
        pct = f"({saved * 100 // max(b, 1)}%)" if b > 0 else ""
        lines.append(f"  {key:<22s} {b:>7d} {a:>7d} {saved:>6d} {pct}")

    lines.append(f"  {'─' * 50}")
    pct_total = f"({total_saved * 100 // max(total_before, 1)}%)"
    lines.append(f"  {'TOTAL':<22s} {total_before:>7d} {total_after:>7d} {total_saved:>6d} {pct_total}")
    lines.append(f"  Budget: {budget}")

    logger.info("\n".join(lines))
