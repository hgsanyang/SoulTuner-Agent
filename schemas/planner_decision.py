"""PlannerDecisionV2 — compact planner output contract for distillation.

The production `MusicQueryPlan` carries canonical layered intent AND legacy flat
fields AND full ToolPlan arguments AND reasoning. That is convenient for the
existing pipeline but verbose to generate, so distilling it inflates training
tokens and inference latency and makes "planner p50 < 2s" unrealistic.

`PlannerDecisionV2` is the minimal set a model must actually GENERATE — everything
else (legacy RetrievalPlan fields, ToolPlan arguments/dependencies, limits,
timeouts, audit trace) is filled by a deterministic compiler. The student learns
this small target; the compiler makes the rest of the pipeline consume it
unchanged.

Two directions, both must stay faithful (round-trip tested):
- ``compile_to_query_plan``  V2 -> MusicQueryPlan   (inference / execution)
- ``from_query_plan``        MusicQueryPlan -> V2    (build training targets from
                                                     the strong planner's output)
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

from schemas.query_plan import (
    HardConstraints,
    IntentHints,
    MetadataConstraints,
    MusicQueryPlan,
    RetrievalPlan,
    SoftIntent,
)
from schemas.tool_plan import ToolName

PLANNER_DECISION_VERSION = "planner_decision_v2"

IntentType = Literal[
    "graph_search",
    "hybrid_search",
    "vector_search",
    "clarification",
    "general_chat",
    "web_search",
    "acquire_music",
    "recommend_by_favorites",
]

# Compact tool names the model emits -> whether they turn on each recall lane.
# Deterministic compilation derives the actual ToolPlan arguments from the
# canonical fields, so the model never has to spell out tool arguments.
_TOOL_NAME_ALIASES: dict[str, str] = {
    "graph": "graph", "graph_recall": "graph", "search_graph": "graph",
    "dense": "dense", "dense_recall": "dense", "audio_recall": "dense",
    "search_audio": "dense", "vector": "dense",
    "web": "web", "search_external_music": "web", "external": "web",
}


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class DecisionHard(_Strict):
    artist: list[str] = Field(default_factory=list)
    song: list[str] = Field(default_factory=list)
    language: Optional[str] = None
    region: Optional[str] = None
    instrumental: bool = False


class DecisionSoft(_Strict):
    goal: str = ""
    trajectory: str = ""
    vibe: list[str] = Field(default_factory=list)
    avoid: list[str] = Field(default_factory=list)


class DecisionHints(_Strict):
    mood: list[str] = Field(default_factory=list)
    scenario: list[str] = Field(default_factory=list)
    genre: list[str] = Field(default_factory=list)


class DecisionMeta(_Strict):
    era: Optional[str] = None
    release_year_from: Optional[int] = Field(default=None, ge=1800, le=2200)
    release_year_to: Optional[int] = Field(default=None, ge=1800, le=2200)
    recency_required: bool = False
    external_knowledge_required: bool = False


class PlannerDecisionV2(_Strict):
    """The minimal, semantically non-derivable planner output for distillation."""

    intent: IntentType
    hard: DecisionHard = Field(default_factory=DecisionHard)
    soft: DecisionSoft = Field(default_factory=DecisionSoft)
    hints: DecisionHints = Field(default_factory=DecisionHints)
    metadata: DecisionMeta = Field(default_factory=DecisionMeta)
    acoustic_queries: list[str] = Field(default_factory=list, max_length=4)
    clarification: Optional[str] = None
    tool_names: list[str] = Field(default_factory=list)
    # short human audit note only; execution never depends on it, and hidden
    # chain-of-thought is never stored.
    decision_summary: str = Field(default="", max_length=200)


def _normalized_tool_lanes(tool_names: list[str]) -> set[str]:
    lanes: set[str] = set()
    for name in tool_names or []:
        lane = _TOOL_NAME_ALIASES.get(str(name or "").strip().casefold())
        if lane:
            lanes.add(lane)
    return lanes


def compile_to_query_plan(decision: PlannerDecisionV2) -> MusicQueryPlan:
    """V2 -> MusicQueryPlan. The existing pipeline consumes the result unchanged."""
    if decision.clarification:
        return MusicQueryPlan(
            intent_type="clarification",
            parameters={"question": decision.clarification},
            context=decision.decision_summary,
            reasoning=decision.decision_summary,
        )

    lanes = _normalized_tool_lanes(decision.tool_names)
    # If the model named no lanes, fall back to the intent's natural lanes so a
    # sparse decision still executes rather than silently doing nothing.
    if not lanes:
        if decision.intent in {"graph_search", "web_search"}:
            lanes.add("graph")
        if decision.intent in {"vector_search", "hybrid_search"}:
            lanes.add("dense")
        if decision.intent == "hybrid_search":
            lanes.add("graph")
        if decision.intent == "web_search":
            lanes.add("web")

    retrieval_plan = RetrievalPlan(
        hard_constraints=HardConstraints(
            artist_entities=list(decision.hard.artist),
            song_entities=list(decision.hard.song),
            language=decision.hard.language,
            region=decision.hard.region,
            instrumental=decision.hard.instrumental,
        ),
        soft_intent=SoftIntent(
            goal=decision.soft.goal,
            trajectory=decision.soft.trajectory,
            vibe=" ".join(v for v in decision.soft.vibe if v).strip(),
            avoid=list(decision.soft.avoid),
        ),
        hints=IntentHints(
            genres=list(decision.hints.genre),
            mood=decision.hints.mood[0] if decision.hints.mood else None,
            scenario=decision.hints.scenario[0] if decision.hints.scenario else None,
        ),
        metadata_constraints=MetadataConstraints(
            era=decision.metadata.era,
            release_year_from=decision.metadata.release_year_from,
            release_year_to=decision.metadata.release_year_to,
            recency_required=decision.metadata.recency_required,
            external_knowledge_required=decision.metadata.external_knowledge_required,
        ),
        vector_acoustic_queries=list(decision.acoustic_queries),
        use_graph="graph" in lanes,
        use_vector="dense" in lanes,
        use_web_search="web" in lanes,
    )
    return MusicQueryPlan(
        intent_type=decision.intent,
        parameters={},
        context=decision.decision_summary,
        retrieval_plan=retrieval_plan,
        reasoning=decision.decision_summary,
    )


def from_query_plan(plan: MusicQueryPlan) -> PlannerDecisionV2:
    """MusicQueryPlan -> V2. Used to build compact training targets from the
    strong planner's (verbose) output, and to convert legacy seed answers."""
    rp = plan.retrieval_plan
    if plan.intent_type == "clarification":
        question = str((plan.parameters or {}).get("question") or plan.context or "请再具体说明一下你的音乐需求。")
        return PlannerDecisionV2(
            intent="clarification",
            clarification=question,
            decision_summary=str(plan.reasoning or "")[:200],
        )

    tool_names: list[str] = []
    for call in (plan.tool_plan.tool_calls if plan.tool_plan else []):
        if call.name == ToolName.SEARCH_GRAPH:
            tool_names.append("graph")
        elif call.name == ToolName.SEARCH_AUDIO:
            tool_names.append("dense")
        elif call.name == ToolName.SEARCH_EXTERNAL_MUSIC:
            tool_names.append("web")

    acoustic = list(rp.vector_acoustic_queries or [])
    if rp.vector_acoustic_query and rp.vector_acoustic_query not in acoustic:
        acoustic.insert(0, rp.vector_acoustic_query)

    return PlannerDecisionV2(
        intent=plan.intent_type,
        hard=DecisionHard(
            artist=list(rp.hard_constraints.artist_entities),
            song=list(rp.hard_constraints.song_entities),
            language=rp.hard_constraints.language,
            region=rp.hard_constraints.region,
            instrumental=rp.hard_constraints.instrumental,
        ),
        soft=DecisionSoft(
            goal=rp.soft_intent.goal,
            trajectory=rp.soft_intent.trajectory,
            vibe=[rp.soft_intent.vibe] if rp.soft_intent.vibe else [],
            avoid=list(rp.soft_intent.avoid),
        ),
        hints=DecisionHints(
            mood=[rp.hints.mood] if rp.hints.mood else [],
            scenario=[rp.hints.scenario] if rp.hints.scenario else [],
            genre=list(rp.hints.genres),
        ),
        metadata=DecisionMeta(
            era=rp.metadata_constraints.era,
            release_year_from=rp.metadata_constraints.release_year_from,
            release_year_to=rp.metadata_constraints.release_year_to,
            recency_required=rp.metadata_constraints.recency_required,
            external_knowledge_required=rp.metadata_constraints.external_knowledge_required,
        ),
        acoustic_queries=acoustic[:4],
        tool_names=list(dict.fromkeys(tool_names)),
        decision_summary=str(plan.reasoning or "")[:200],
    )


def decision_token_estimate(decision: PlannerDecisionV2) -> int:
    """Rough token estimate of the compact JSON target (chars/4 heuristic)."""
    import json

    text = json.dumps(decision.model_dump(exclude_none=True), ensure_ascii=False, separators=(",", ":"))
    return max(0, len(text) // 4)
