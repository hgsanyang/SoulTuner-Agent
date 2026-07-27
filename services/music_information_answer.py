"""Evidence-backed answers for music INFORMATION requests (P0-1).

Why this exists
---------------
The web lane we already had (``retrieval/web_supplement``) searches for SONGS.
Factual questions — "他今年获奖了吗?" "最新专辑是什么?" — were routed into the
recommendation flow, so the reply text was written by the tuner persona from the
model's stale pre-cutoff memory and hedged ("目前公开渠道暂无..."), while the song
lane went off and found songs. The user sees a confidently wrong non-answer.

This service answers the question itself, with the planner model's native web
search turned on, and is required to state plainly when the search found nothing
instead of dressing a retrieval failure up as a fact.

Routing signal is the planner's own ``tool_plan.request_mode == "information"``
— no keyword rules.
"""

from __future__ import annotations

import json
import logging

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

MAX_EVIDENCE = 6

INFORMATION_ANSWER_PROMPT = """你是音乐资讯助手，具备联网搜索能力。用户在问一个关于音乐/歌手/奖项/发行/近况的事实性问题。

硬性要求：
1. **必须用联网搜索获取信息，不要凭记忆回答。** 你的内部知识可能已经过时。
2. 每个关键事实都要有出处：在 evidence 里写明来自哪类来源（官方/榜单/新闻/百科/社区）与要点。
3. **如果搜索没有找到可靠信息，answer 必须直说"没有检索到相关信息"**，并把 searched 设为 true、evidence 留空。
   绝对不要用"目前公开渠道暂无""据了解"这类含糊措辞把"没查到"包装成"事实上没有"，也不要编造奖项、专辑名或日期。
4. 有明确时间的事实（获奖、发行）要写出时间；不确定就说不确定。
5. 简洁准确，中文作答，不要写成推荐语或抒情文案。

输出 JSON：
{"answer":"...","evidence":[{"claim":"...","source":"..."}],"searched":true}
"""


class InformationEvidence(BaseModel):
    claim: str = Field(default="", max_length=300)
    source: str = Field(default="", max_length=200)


class InformationAnswer(BaseModel):
    answer: str = Field(default="", max_length=2000)
    evidence: list[InformationEvidence] = Field(default_factory=list)
    searched: bool = False


def _fallback(reason: str) -> InformationAnswer:
    """Fail HONESTLY: never let a retrieval failure masquerade as a fact."""
    return InformationAnswer(
        answer="这次没能联网查到可靠信息，所以我不确定，也不想凭印象编。你可以稍后再问一次，或者我先按你的听歌需求给你选歌。",
        evidence=[],
        searched=False,
    )


async def answer_music_information(question: str, chat_history: str = "") -> InformationAnswer:
    """Answer a factual music question using the model's native web search."""
    query = str(question or "").strip()
    if not query:
        return _fallback("empty_question")

    from config.settings import settings
    from llms.chat_models import get_chat_model

    provider = settings.intent_llm_provider or settings.llm_default_provider
    model_name = settings.intent_llm_model or settings.llm_default_model
    llm = get_chat_model(
        provider=provider,
        model_name=model_name,
        temperature=0.0,
        max_tokens=1200,
        enable_web_search=True,
    )
    payload = {"question": query}
    if chat_history:
        payload["recent_dialog"] = chat_history[-1200:]

    try:
        try:
            structured = llm.with_structured_output(InformationAnswer, include_raw=True, method="json_mode")
        except (TypeError, ValueError):
            structured = llm.with_structured_output(InformationAnswer, include_raw=True)
        result = await structured.ainvoke([
            ("system", INFORMATION_ANSWER_PROMPT),
            ("human", json.dumps(payload, ensure_ascii=False, separators=(",", ":"))),
        ])
        if isinstance(result, InformationAnswer):
            answer = result
        elif isinstance(result, dict) and isinstance(result.get("parsed"), InformationAnswer):
            answer = result["parsed"]
        else:
            raw = result.get("raw") if isinstance(result, dict) else result
            content = str(getattr(raw, "content", raw) or "").strip()
            start, end = content.find("{"), content.rfind("}")
            if start < 0 or end <= start:
                return _fallback("unparseable")
            answer = InformationAnswer.model_validate(json.loads(content[start : end + 1]))
    except Exception as exc:  # network/provider/parse — never fabricate
        logger.warning("[InformationAnswer] failed: %s: %s", type(exc).__name__, exc)
        return _fallback(type(exc).__name__)

    answer.evidence = answer.evidence[:MAX_EVIDENCE]
    if not str(answer.answer or "").strip():
        return _fallback("empty_answer")
    logger.info(
        "[InformationAnswer] answered q=%r searched=%s evidence=%d",
        query[:40], answer.searched, len(answer.evidence),
    )
    return answer
