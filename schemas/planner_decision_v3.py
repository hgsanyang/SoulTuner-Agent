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

from dataclasses import dataclass
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

# High-level lanes backed by registered ToolPlan tools. ``ingest`` remains a
# shadow preview: selecting the lane does not authorize a catalog mutation.
ToolLane = Literal["graph", "dense", "web", "library", "ingest"]
_V2_LANES = ("graph", "dense", "web")

# A kind may be served by any of these lanes. `information` is deliberately not
# web-only: "这首歌哪年发行" is often already in the graph/knowledge cards, and
# forcing a web call there would be both slower and wrong.
KIND_LANES: dict[str, set[str]] = {
    "recommendation": {"graph", "dense", "web"},
    "information": {"graph", "web"},
    "acquisition": {"web", "ingest"},
    "library": {"library"},
}
KIND_REQUIRED_ANY: dict[str, set[str]] = {
    # A catalog-gap recovery may intentionally use the web lane on its own
    # after both local anchors have proved insufficient. Requiring a local
    # lane here made that safe recovery impossible to represent.
    "recommendation": {"graph", "dense", "web"},
    "information": {"graph", "web"},
    "acquisition": {"ingest"},
    "library": {"library"},
}


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
        if ("dense" in lanes) != bool(self.acoustic_queries):
            raise ValueError("dense lane and acoustic_queries must be selected together")
        if self.request_kind == "conversation":
            if lanes:
                raise ValueError("conversation must carry no tool lanes")
            return self
        # answer-mode retrieval kinds must name a lane that can serve them.
        if not lanes:
            raise ValueError(f"request_kind={self.request_kind} requires at least one tool lane")
        allowed = KIND_LANES[self.request_kind]
        if not lanes <= allowed:
            raise ValueError(
                f"request_kind={self.request_kind} allows {sorted(allowed)}, got {sorted(lanes)}"
            )
        required_any = KIND_REQUIRED_ANY[self.request_kind]
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
    lanes = set(decision.tool_names)
    if kind == "information":
        # V2 conflates request kind with retrieval lane. Pick the legacy label
        # that can compile the selected lane, then restore the V3 request mode
        # on the resulting ToolPlan.
        return "web_search" if "web" in lanes else "graph_search"
    if "graph" in lanes and "dense" in lanes:
        return "hybrid_search"
    if "dense" in lanes:
        return "vector_search"
    if "web" in lanes:
        return "web_search"
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


@dataclass(frozen=True)
class MigrationNote:
    """Whether a migrated sample is semantically safe for training."""

    ambiguous: bool = False
    reason: str = ""


def migration_note(decision: PlannerDecisionV2) -> MigrationNote:
    """Flag V2 samples whose V3 meaning cannot be derived from the intent alone.

    ``web_search`` is the big one: the same legacy intent covers "Billboard 本周
    冠军是谁" (information) and "求推荐几首本周发行的 K-hiphop" (recommendation +
    web). Blanket-mapping it to ``information`` would mint wrong labels, so these
    are marked and must be re-judged from the original query before training.
    """
    if decision.intent == "web_search":
        return MigrationNote(True, "web_search covers information / recommendation+web; re-judge from query")
    if decision.intent == "recommend_by_favorites":
        return MigrationNote(True, "favorites = library view vs favorites-seeded recommendation; re-judge")
    if decision.intent == "clarification":
        return MigrationNote(True, "V2 does not record what the user was asking when the agent clarified")
    return MigrationNote(False, "")


def migrate_v2_to_v3(decision: PlannerDecisionV2) -> PlannerDecisionV3:
    """Pure, offline structural migration (no teacher call).

    Structure only — see ``migration_note`` for samples whose *meaning* needs a
    re-judgement before they may enter a training set.
    """
    kind, mode = _legacy_intent_to_kind(decision.intent)
    lanes = sorted(_normalized_tool_lanes(decision.tool_names))
    if mode == "clarify" or kind == "conversation":
        lanes = []
    elif kind == "acquisition":
        lanes = ["web", "ingest"]
    elif kind == "library":
        lanes = ["library"]
    elif not lanes:
        # V2 allowed empty lanes (compiler fallback); V3 requires an explicit
        # lane, so materialise the fallback the compiler would have used.
        lanes = {"graph_search": ["graph"], "vector_search": ["dense"],
                 "hybrid_search": ["graph", "dense"], "web_search": ["web"]}.get(decision.intent, ["graph"])
    if kind == "information":
        # information may be served by graph or web; drop lanes it cannot use.
        lanes = [x for x in lanes if x in KIND_LANES["information"]] or ["web"]
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


def compile_v3_to_tool_plan(decision: PlannerDecisionV3):
    """Compile every V3 lane into the executable ToolPlan 1.1 protocol.

    Graph, dense and web still reuse the mature legacy compiler.  ``library``
    and ``ingest`` cannot take that route because V2 has no representation for
    either lane; dropping them made valid V3 targets look executable while the
    requested tool never ran.
    """
    from schemas.planner_decision import compile_to_query_plan
    from schemas.tool_plan import ToolCall, ToolName, ToolPlan

    legacy_plan = compile_to_query_plan(migrate_v3_to_v2(decision))
    base = legacy_plan.tool_plan
    if decision.response_mode == "clarify" or decision.request_kind == "conversation":
        return base
    if decision.request_kind == "library":
        return ToolPlan(
            version="1.1",
            origin="legacy_compiler",
            request_mode="library",
            tool_calls=[
                ToolCall(
                    id="library_read",
                    name=ToolName.READ_LIBRARY,
                    arguments={"collection": "liked", "query": decision.decision_summary},
                    reason="read the user's library without mutating it",
                )
            ],
            decision_summary=decision.decision_summary,
            max_replans=0,
        )
    if decision.request_kind == "acquisition":
        calls = list(base.tool_calls if base else [])
        dependencies = [call.id for call in calls]
        calls.append(
            ToolCall(
                id="ingest_preview",
                name=ToolName.STAGE_INGEST,
                arguments={
                    "candidate_source_ids": [],
                    "preserve_audio": False,
                    "reason": decision.decision_summary,
                    "mode": "preview",
                },
                depends_on=dependencies,
                reason="stage a reversible ingest preview after discovery",
            )
        )
        return ToolPlan(
            version="1.1",
            origin="legacy_compiler",
            request_mode="acquisition",
            tool_calls=calls,
            decision_summary=decision.decision_summary,
            max_replans=1,
        )
    if base is None:
        raise ValueError("compiled V3 decision has no ToolPlan")
    return base.model_copy(update={"request_mode": decision.request_kind})


def compile_v3_to_query_plan(decision: PlannerDecisionV3):
    """V3 -> MusicQueryPlan with a lossless ToolPlan 1.1 compilation."""
    from schemas.planner_decision import compile_to_query_plan

    plan = compile_to_query_plan(migrate_v3_to_v2(decision))
    plan.tool_plan = compile_v3_to_tool_plan(decision)
    return plan
