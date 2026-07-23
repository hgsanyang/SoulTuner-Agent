"""PlannerDecisionV3 — request kind and tool choice, finally separated.

Why V3
------
V2's single ``intent`` field mixed two different things:
  - what the USER wants (recommend / ask a fact / download / manage library / chat)
  - which TOOLS the system runs (graph / dense / web)
``graph_search`` vs ``vector_search`` vs ``hybrid_search`` are not user intents at
all — they are recall-lane combinations, and they duplicated ``tool_names``.
Distilling that as-is would bake the redundancy into the student.

V3 splits them:
  request_kind   recommendation | information | acquisition | library | conversation
  response_mode  answer | clarify        (clarify is an AGENT ACTION, not a request type)
  tool_names     graph | dense | web | library | ingest   (authoritative tool choice)

Deliberately NOT included: ``preference_source``. Real requests lean on the
current turn, session, profile, episodic memory and time habits *at once*;
forcing the planner to predict one source is noise. Context is assembled
deterministically upstream (see ``schemas/agent_context.py``) and recorded in the
trace, never guessed as a training label.

Compatibility: V3 compiles by mapping to the legacy intent and reusing the V2
compiler, so the existing pipeline consumes it unchanged. Migration from the
1491 collected V2 targets is a pure function — no teacher re-call.
"""

from __future__ import annotations

from typing import Literal

from pydantic import ConfigDict, Field, model_validator

from schemas.planner_decision import (
    DecisionHard,
    DecisionHints,
    DecisionMeta,
    DecisionSoft,
    IntentType,
    PlannerDecisionV2,
    _Strict,
    _normalized_tool_lanes,
)

PLANNER_DECISION_V3_VERSION = "planner_decision_v3"

RequestKind = Literal["recommendation", "information", "acquisition", "library", "conversation"]
ResponseMode = Literal["answer", "clarify"]
ToolLane = Literal["graph", "dense", "web", "library", "ingest"]

# Lanes V2 can represent; library/ingest have no V2 encoding and are dropped on
# the way back (documented asymmetry — V2 simply cannot say them).
_V2_LANES = ("graph", "dense", "web")


class PlannerDecisionV3(_Strict):
    """Minimal planner output with user-intent and tool-choice decoupled."""

    model_config = ConfigDict(extra="forbid")

    request_kind: RequestKind
    response_mode: ResponseMode = "answer"
    tool_names: list[ToolLane] = Field(default_factory=list)
    hard: DecisionHard = Field(default_factory=DecisionHard)
    soft: DecisionSoft = Field(default_factory=DecisionSoft)
    hints: DecisionHints = Field(default_factory=DecisionHints)
    metadata: DecisionMeta = Field(default_factory=DecisionMeta)
    acoustic_queries: list[str] = Field(default_factory=list, max_length=4)
    clarification: str | None = None
    decision_summary: str = Field(default="", max_length=200)

    @model_validator(mode="after")
    def _enforce_invariants(self) -> "PlannerDecisionV3":
        lanes = set(self.tool_names)
        has_text = bool((self.clarification or "").strip())
        # clarify is an agent action: locked to the text, and it runs no tools.
        if self.response_mode == "clarify":
            if not has_text:
                raise ValueError("response_mode=clarify requires clarification text")
            if lanes:
                raise ValueError("clarify must carry no tool lanes")
            return self
        if has_text:
            raise ValueError("clarification text only allowed when response_mode=clarify")
        if self.request_kind == "conversation":
            if lanes:
                raise ValueError("conversation must carry no tool lanes")
            return self
        # answer-mode request kinds must name a lane that can serve them.
        if not lanes:
            raise ValueError(f"request_kind={self.request_kind} requires at least one tool lane")
        required_any = {
            "recommendation": {"graph", "dense"},
            "information": {"web"},
            "acquisition": {"ingest"},
            "library": {"library"},
        }[self.request_kind]
        if not (lanes & required_any):
            raise ValueError(
                f"request_kind={self.request_kind} requires one of {sorted(required_any)}, got {sorted(lanes)}"
            )
        return self


def v3_to_legacy_intent(decision: PlannerDecisionV3) -> IntentType:
    """Deterministic V3 -> legacy intent, so the current pipeline still routes."""
    if decision.response_mode == "clarify":
        return "clarification"
    kind = decision.request_kind
    if kind == "conversation":
        return "general_chat"
    if kind == "acquisition":
        return "acquire_music"
    if kind == "library":
        return "recommend_by_favorites"
    if kind == "information":
        return "web_search"
    lanes = set(decision.tool_names)
    if "graph" in lanes and "dense" in lanes:
        return "hybrid_search"
    if "dense" in lanes:
        return "vector_search"
    return "graph_search"


def _legacy_intent_to_kind(intent: str) -> tuple[RequestKind, ResponseMode]:
    if intent == "clarification":
        # V2 does not record what the user was asking about when the agent chose
        # to clarify; recommendation is the dominant case in collected data.
        return "recommendation", "clarify"
    if intent == "general_chat":
        return "conversation", "answer"
    if intent == "acquire_music":
        return "acquisition", "answer"
    if intent == "recommend_by_favorites":
        # current code seeds recommendation from favorites (not a library view)
        return "recommendation", "answer"
    if intent == "web_search":
        return "information", "answer"
    return "recommendation", "answer"


def migrate_v2_to_v3(decision: PlannerDecisionV2) -> PlannerDecisionV3:
    """Pure, offline migration of a collected V2 target to V3 (no teacher call)."""
    kind, mode = _legacy_intent_to_kind(decision.intent)
    lanes = sorted(_normalized_tool_lanes(decision.tool_names))
    if mode == "clarify" or kind == "conversation":
        lanes = []
    elif not lanes:
        # V2 allowed empty lanes (compiler fallback); V3 requires an explicit
        # lane, so materialise the fallback the compiler would have used.
        lanes = {"graph_search": ["graph"], "vector_search": ["dense"],
                 "hybrid_search": ["graph", "dense"], "web_search": ["web"]}.get(decision.intent, ["graph"])
    if kind == "acquisition":
        lanes = ["ingest"]
    elif kind == "library":
        lanes = ["library"]
    return PlannerDecisionV3(
        request_kind=kind,
        response_mode=mode,
        tool_names=lanes,  # type: ignore[arg-type]
        hard=decision.hard,
        soft=decision.soft,
        hints=decision.hints,
        metadata=decision.metadata,
        acoustic_queries=list(decision.acoustic_queries),
        clarification=decision.clarification if mode == "clarify" else None,
        decision_summary=decision.decision_summary,
    )


def migrate_v3_to_v2(decision: PlannerDecisionV3) -> PlannerDecisionV2:
    """Back-migration for round-trip checks. library/ingest have no V2 encoding
    and are dropped — round-trip identity therefore holds for the graph/dense/web
    subset (which is all collected data), not for acquisition/library."""
    intent = v3_to_legacy_intent(decision)
    lanes = [lane for lane in decision.tool_names if lane in _V2_LANES]
    return PlannerDecisionV2(
        intent=intent,
        hard=decision.hard,
        soft=decision.soft,
        hints=decision.hints,
        metadata=decision.metadata,
        acoustic_queries=list(decision.acoustic_queries),
        clarification=decision.clarification if intent == "clarification" else None,
        tool_names=lanes,
        decision_summary=decision.decision_summary,
    )


def compile_v3_to_query_plan(decision: PlannerDecisionV3):
    """V3 -> MusicQueryPlan by reusing the V2 compiler (one compiler, no drift)."""
    from schemas.planner_decision import compile_to_query_plan

    return compile_to_query_plan(migrate_v3_to_v2(decision))
