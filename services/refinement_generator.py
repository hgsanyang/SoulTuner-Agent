"""LLM-first refinement chip generation.

Refinement chips are generated AFTER the slate is known, from the full turn
context: the current user turn, a recent dialogue window, the structured plan,
session dialog state, the relevant memory snapshot, a summary of the current
slate, catalog gap findings, and the previous turn's chips.

Deterministic code in this module only validates the typed LLM output
(schema, count, dedup, conflicts with explicitly avoided directions). It must
never invent semantic directions from keywords or profile text; when the model
fails we fall back to at most two content-free neutral chips.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from typing import Any, Awaitable, Callable

from pydantic import BaseModel, Field

from schemas.refinement import RefinementOption

logger = logging.getLogger(__name__)

MAX_OPTIONS = 6
MIN_TARGET_OPTIONS = 4
MAX_LABEL_CHARS = 24
MAX_PROMPT_CHARS = 120
MAX_REASON_CHARS = 160
_MAX_TRACKED_USERS = 512

REFINEMENT_SYSTEM_PROMPT = """你是音乐推荐产品里的「下一步方向」设计师。
用户刚收到一批推荐歌曲。你要基于完整上下文，提出 4-6 个真正互补的微调方向，
让用户一键说出"接下来想怎么调"。

规则：
1. 方向必须来自当前上下文：本轮请求、对话历史、当前歌曲批次的构成、
   曲库缺口、相关记忆。不要输出与当前场景无关的通用方向。
2. 当前请求优先于长期画像；与本轮无关的记忆不能影响方向。
3. 绝不提出与用户已明确否定/要求避开的方向相同或相反强化的选项。
   例如用户说过"不要悲伤"，就不能出现"更悲伤"。
4. 各方向之间要互补（覆盖不同维度：节奏/能量、人声、年代/语言、
   小众程度、场景贴合、情绪走向等），不要 4 个都在调同一维度。
5. 与上一轮的方向组合相比要有变化，不要原样重复整组。
6. reason 必须引用真实依据（如"本轮候选电子节拍占比较高"），
   不得编造没有证据的歌曲事实。
7. label 是按钮短文案（<=12 个汉字），prompt 是用户视角的自然语言
   follow-up（保留当前场景，只调一个维度），reason 是一句话依据。
8. source 取值：current_slate / dialogue / memory / catalog_gap 之一。

输出 JSON：{"options": [{"label": "...", "prompt": "...", "reason": "...", "source": "..."}]}
"""

# Content-free fallbacks: no semantic direction is invented on model failure.
NEUTRAL_FALLBACK_OPTIONS = (
    RefinementOption(
        label="换一批",
        prompt="保持刚才的方向，换一批不一样的歌",
        reason="模型微调方向暂不可用时的中性操作",
        source="fallback",
    ),
    RefinementOption(
        label="换个方向",
        prompt="换一个不同的方向推荐试试",
        reason="模型微调方向暂不可用时的中性操作",
        source="fallback",
    ),
)


class RefinementProposal(BaseModel):
    """Typed output contract for the chip-generation model call."""

    options: list[RefinementOption] = Field(default_factory=list)


# Per-user previous chip sets, kept in-process so consecutive turns can be
# compared without widening the dialog-state schema.
_previous_options: dict[str, list[dict[str, str]]] = {}


def _norm(text: Any) -> str:
    return str(text or "").strip().casefold()


def remember_emitted_options(user_id: str, options: list[RefinementOption]) -> None:
    if len(_previous_options) >= _MAX_TRACKED_USERS and user_id not in _previous_options:
        _previous_options.pop(next(iter(_previous_options)), None)
    _previous_options[user_id] = [
        {"label": option.label, "prompt": option.prompt} for option in options
    ]


def previous_options_for(user_id: str) -> list[dict[str, str]]:
    return list(_previous_options.get(user_id, []))


def summarize_slate(recommendations: list[Any], limit: int = 12) -> list[dict[str, Any]]:
    """Compact, factual summary of the current slate for the model."""
    summary: list[dict[str, Any]] = []
    for item in recommendations[:limit]:
        song = item.get("song", item) if isinstance(item, dict) else item
        if not isinstance(song, dict):
            continue
        entry = {
            "title": str(song.get("title") or "")[:60],
            "artist": str(song.get("artist") or "")[:40],
        }
        if not entry["title"] and not entry["artist"]:
            continue
        for key in ("genre", "language", "mood", "moods", "scenarios", "year"):
            value = song.get(key)
            if value:
                entry[key] = value if not isinstance(value, list) else value[:3]
        summary.append(entry)
    return summary


def extract_avoid_texts(plan: dict[str, Any] | None, dialog_state: dict[str, Any] | None) -> list[str]:
    """Collect the user's explicitly avoided directions from structured state."""
    avoid: list[str] = []
    for source in (plan or {}, ((dialog_state or {}).get("soft_intent") or {})):
        if not isinstance(source, dict):
            continue
        soft = source.get("soft_intent") if "soft_intent" in source else source
        if isinstance(soft, dict):
            for item in soft.get("avoid") or []:
                text = str(item or "").strip()
                if text:
                    avoid.append(text)
    seen: set[str] = set()
    unique: list[str] = []
    for text in avoid:
        key = _norm(text)
        if key and key not in seen:
            seen.add(key)
            unique.append(text)
    return unique


def validate_options(
    raw_options: list[RefinementOption],
    *,
    avoid_texts: list[str] | None = None,
    previous: list[dict[str, str]] | None = None,
) -> list[RefinementOption]:
    """Deterministic guardrails: schema bounds, dedup, avoid-conflicts, caps.

    This function must stay purely mechanical — it filters what the model
    proposed and never adds semantic directions of its own.
    """
    avoid_normed = [_norm(text) for text in (avoid_texts or []) if _norm(text)]
    seen_labels: set[str] = set()
    seen_prompts: set[str] = set()
    valid: list[RefinementOption] = []
    for option in raw_options:
        label = str(option.label or "").strip()
        prompt = str(option.prompt or "").strip()
        reason = str(option.reason or "").strip()[:MAX_REASON_CHARS]
        source = str(option.source or "context").strip() or "context"
        if not label or not prompt:
            continue
        if len(label) > MAX_LABEL_CHARS or len(prompt) > MAX_PROMPT_CHARS:
            continue
        label_key, prompt_key = _norm(label), _norm(prompt)
        if label_key in seen_labels or prompt_key in seen_prompts:
            continue
        # Never re-offer a direction the user explicitly asked to avoid.
        if any(avoided in label_key or avoided in prompt_key for avoided in avoid_normed):
            continue
        seen_labels.add(label_key)
        seen_prompts.add(prompt_key)
        valid.append(RefinementOption(label=label, prompt=prompt, reason=reason, source=source))
        if len(valid) >= MAX_OPTIONS:
            break

    # Do not repeat the previous turn's entire set verbatim; dropping the last
    # option is enough to break exact repetition without discarding directions.
    if previous and len(valid) > 1:
        prev_pairs = {(_norm(p.get("label")), _norm(p.get("prompt"))) for p in previous}
        new_pairs = {(_norm(o.label), _norm(o.prompt)) for o in valid}
        if new_pairs == prev_pairs:
            valid = valid[:-1]
    return valid


class RefinementChipGenerator:
    """Generates context-grounded refinement chips via a typed LLM call.

    ``generator`` may be injected in tests: a callable taking the payload dict
    and returning a ``RefinementProposal`` (or dict), sync or async.
    """

    def __init__(
        self,
        *,
        generator: Callable[[dict[str, Any]], Any] | None = None,
        timeout_seconds: float = 12.0,
    ):
        self.generator = generator
        self.timeout_seconds = float(timeout_seconds)

    @staticmethod
    def prompt_hash() -> str:
        return hashlib.sha256(REFINEMENT_SYSTEM_PROMPT.encode("utf-8")).hexdigest()[:16]

    async def generate(
        self,
        *,
        user_id: str,
        user_input: str,
        chat_history: list[dict[str, str]] | None = None,
        plan: dict[str, Any] | None = None,
        dialog_state: dict[str, Any] | None = None,
        memory_snapshot: str = "",
        recommendations: list[Any] | None = None,
        catalog_gap: dict[str, Any] | None = None,
    ) -> list[RefinementOption]:
        previous = previous_options_for(user_id)
        avoid_texts = extract_avoid_texts(plan, dialog_state)
        payload = {
            "current_user_turn": str(user_input or "")[:500],
            "recent_dialogue_window": [
                {"role": str(m.get("role") or ""), "content": str(m.get("content") or "")[:300]}
                for m in (chat_history or [])[-6:]
            ],
            "current_structured_plan": _compact_plan(plan),
            "current_session_state": _compact_dialog_state(dialog_state),
            "top_relevant_memory_snapshot": str(memory_snapshot or "")[:800],
            "current_slate_summary": summarize_slate(list(recommendations or [])),
            "catalog_gap_summary": _compact_catalog_gap(catalog_gap),
            "previous_refinement_options": previous,
            "explicitly_avoided": avoid_texts,
        }

        try:
            proposal = await asyncio.wait_for(
                self._invoke(payload), timeout=self.timeout_seconds
            )
            options = validate_options(
                proposal.options, avoid_texts=avoid_texts, previous=previous
            )
        except Exception as exc:
            logger.warning("[Refinement] LLM 方向生成失败，使用中性 fallback: %s", exc)
            options = []

        if not options:
            options = list(NEUTRAL_FALLBACK_OPTIONS)
        remember_emitted_options(user_id, options)
        return options

    async def _invoke(self, payload: dict[str, Any]) -> RefinementProposal:
        if self.generator is not None:
            result = self.generator(payload)
            if asyncio.iscoroutine(result) or isinstance(result, Awaitable):
                result = await result
            if isinstance(result, RefinementProposal):
                return result
            return RefinementProposal.model_validate(result)

        from config.settings import settings
        from llms.chat_models import get_chat_model

        provider = settings.intent_llm_provider or settings.llm_default_provider
        model_name = settings.intent_llm_model or settings.llm_default_model
        llm = get_chat_model(
            provider=provider,
            model_name=model_name,
            temperature=0.0,
            max_tokens=900,
        )
        try:
            structured = llm.with_structured_output(
                RefinementProposal, include_raw=True, method="json_mode"
            )
        except (TypeError, ValueError):
            structured = llm.with_structured_output(RefinementProposal, include_raw=True)
        messages = [
            ("system", REFINEMENT_SYSTEM_PROMPT),
            ("human", json.dumps(payload, ensure_ascii=False, separators=(",", ":"))),
        ]
        result = await structured.ainvoke(messages)
        if isinstance(result, RefinementProposal):
            return result
        if isinstance(result, dict) and isinstance(result.get("parsed"), RefinementProposal):
            return result["parsed"]
        raw = result.get("raw") if isinstance(result, dict) else result
        content = getattr(raw, "content", raw)
        return RefinementProposal.model_validate(_decode_json_payload(content))


def _decode_json_payload(content: Any) -> dict[str, Any]:
    text = str(content or "").strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
    start, end = text.find("{"), text.rfind("}")
    if start >= 0 and end > start:
        text = text[start : end + 1]
    return json.loads(text)


def _compact_plan(plan: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(plan, dict):
        return {}
    compact: dict[str, Any] = {}
    for key in ("hard_constraints", "soft_intent", "hints", "metadata_constraints"):
        value = plan.get(key)
        if isinstance(value, dict):
            trimmed = {k: v for k, v in value.items() if v not in (None, "", [], False)}
            if trimmed:
                compact[key] = trimmed
    intent_type = plan.get("_intent_type") or plan.get("intent_type")
    if intent_type:
        compact["intent_type"] = intent_type
    return compact


def _compact_dialog_state(dialog_state: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(dialog_state, dict):
        return {}
    compact: dict[str, Any] = {}
    for key in ("hard_constraints", "soft_intent", "hints", "scene", "resolved_references"):
        value = dialog_state.get(key)
        if isinstance(value, dict):
            trimmed = {k: v for k, v in value.items() if v not in (None, "", [], False)}
            if trimmed:
                compact[key] = trimmed
        elif value not in (None, "", [], False):
            compact[key] = value
    return compact


def _compact_catalog_gap(catalog_gap: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(catalog_gap, dict):
        return {}
    compact: dict[str, Any] = {}
    for key in ("has_gap", "gap_reasons", "local_result_count", "suggestion"):
        value = catalog_gap.get(key)
        if value not in (None, "", []):
            compact[key] = value
    return compact


_default_generator: RefinementChipGenerator | None = None


def get_refinement_generator() -> RefinementChipGenerator:
    global _default_generator
    if _default_generator is None:
        _default_generator = RefinementChipGenerator()
    return _default_generator
