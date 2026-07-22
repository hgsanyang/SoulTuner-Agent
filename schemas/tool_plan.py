"""Versioned schema for SoulTuner's bounded tool-calling policy."""

from __future__ import annotations

from enum import Enum
import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


TOOL_PLAN_VERSION = "1.0"


class StrictToolModel(BaseModel):
    """Reject undeclared fields at the LLM/tool security boundary."""

    model_config = ConfigDict(extra="forbid")


class ToolName(str, Enum):
    RETRIEVE_MEMORY = "retrieve_memory"
    SEARCH_GRAPH = "search_graph"
    SEARCH_AUDIO = "search_audio"
    INSPECT_CATALOG_GAP = "inspect_catalog_gap"
    SEARCH_EXTERNAL_MUSIC = "search_external_music"
    RESOLVE_PLAYABLE_TRACKS = "resolve_playable_tracks"
    COMMIT_MEMORY_DELTA = "commit_memory_delta"


class RetrieveMemoryArguments(StrictToolModel):
    query: str = ""
    scope: Literal["session", "preferences", "episodic", "all"] = "all"
    limit: int = Field(default=8, ge=1, le=30)


class SearchGraphArguments(StrictToolModel):
    artist_entities: list[str] = Field(default_factory=list)
    song_entities: list[str] = Field(default_factory=list)
    language: str | None = None
    region: str | None = None
    instrumental: bool = False
    genres: list[str] = Field(default_factory=list)
    moods: list[str] = Field(default_factory=list)
    scenarios: list[str] = Field(default_factory=list)
    release_year_from: int | None = Field(default=None, ge=1800, le=2200)
    release_year_to: int | None = Field(default=None, ge=1800, le=2200)
    era: str | None = None
    limit: int = Field(default=30, ge=1, le=200)

    @model_validator(mode="after")
    def validate_years(self) -> "SearchGraphArguments":
        if (
            self.release_year_from is not None
            and self.release_year_to is not None
            and self.release_year_from > self.release_year_to
        ):
            raise ValueError("release_year_from must not exceed release_year_to")
        return self


class SearchAudioArguments(StrictToolModel):
    acoustic_queries: list[str] = Field(min_length=1, max_length=4)
    negative_targets: list[str] = Field(default_factory=list, max_length=12)
    limit: int = Field(default=30, ge=1, le=200)

    @field_validator("acoustic_queries")
    @classmethod
    def clean_queries(cls, values: list[str]) -> list[str]:
        cleaned = list(dict.fromkeys(str(value or "").strip() for value in values))
        cleaned = [value for value in cleaned if value]
        if not cleaned:
            raise ValueError("search_audio requires at least one acoustic query")
        return cleaned[:4]


class CatalogGapArguments(StrictToolModel):
    requirements: dict[str, Any] = Field(default_factory=dict)
    candidate_source_ids: list[str] = Field(default_factory=list)


class ExternalMusicArguments(StrictToolModel):
    requirements: str = Field(min_length=1, max_length=1200)
    entities: list[str] = Field(default_factory=list, max_length=30)
    limit: int = Field(default=10, ge=1, le=30)

    @field_validator("requirements")
    @classmethod
    def reject_arbitrary_urls(cls, value: str) -> str:
        text = value.strip()
        if re.search(r"https?://", text, flags=re.IGNORECASE):
            raise ValueError("external discovery accepts requirements, not arbitrary URLs")
        return text


class ResolvePlayableArguments(StrictToolModel):
    candidate_source_ids: list[str] = Field(min_length=1, max_length=100)
    limit: int = Field(default=10, ge=1, le=30)


class CommitMemoryArguments(StrictToolModel):
    memory_type: Literal["explicit_preference", "inferred_preference", "episodic"]
    values: dict[str, Any] = Field(default_factory=dict)
    evidence_id: str = Field(min_length=1, max_length=200)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)


TOOL_ARGUMENT_MODELS: dict[ToolName, type[BaseModel]] = {
    ToolName.RETRIEVE_MEMORY: RetrieveMemoryArguments,
    ToolName.SEARCH_GRAPH: SearchGraphArguments,
    ToolName.SEARCH_AUDIO: SearchAudioArguments,
    ToolName.INSPECT_CATALOG_GAP: CatalogGapArguments,
    ToolName.SEARCH_EXTERNAL_MUSIC: ExternalMusicArguments,
    ToolName.RESOLVE_PLAYABLE_TRACKS: ResolvePlayableArguments,
    ToolName.COMMIT_MEMORY_DELTA: CommitMemoryArguments,
}


class ToolCall(StrictToolModel):
    id: str = Field(pattern=r"^[a-z][a-z0-9_]{0,47}$")
    name: ToolName
    arguments: dict[str, Any] = Field(default_factory=dict)
    depends_on: list[str] = Field(default_factory=list, max_length=8)
    reason: str = Field(default="", max_length=300)

    @model_validator(mode="after")
    def validate_arguments(self) -> "ToolCall":
        argument_model = TOOL_ARGUMENT_MODELS[self.name]
        validated = argument_model.model_validate(self.arguments)
        self.arguments = validated.model_dump(exclude_none=True)
        self.depends_on = list(dict.fromkeys(self.depends_on))
        if self.id in self.depends_on:
            raise ValueError("tool call cannot depend on itself")
        return self


class ToolPlan(StrictToolModel):
    version: Literal["1.0"] = TOOL_PLAN_VERSION
    origin: Literal["planner", "legacy_compiler", "replanner"] = "planner"
    request_mode: Literal["recommendation", "information", "conversation", "acquisition"]
    tool_calls: list[ToolCall] = Field(default_factory=list, max_length=8)
    needs_clarification: bool = False
    clarification_question: str = Field(default="", max_length=500)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    decision_summary: str = Field(default="", max_length=500)
    max_replans: int = Field(default=1, ge=0, le=1)

    @model_validator(mode="after")
    def validate_graph(self) -> "ToolPlan":
        if self.needs_clarification:
            if not self.clarification_question.strip():
                raise ValueError("clarification requires a question")
            if self.tool_calls:
                raise ValueError("clarification plan must not execute tools")

        ids = [call.id for call in self.tool_calls]
        if len(ids) != len(set(ids)):
            raise ValueError("tool call ids must be unique")
        known = set(ids)
        for call in self.tool_calls:
            unknown = set(call.depends_on) - known
            if unknown:
                raise ValueError(f"unknown tool dependencies: {sorted(unknown)}")

        dependencies = {call.id: set(call.depends_on) for call in self.tool_calls}
        visiting: set[str] = set()
        visited: set[str] = set()

        def visit(call_id: str) -> None:
            if call_id in visiting:
                raise ValueError("tool dependency cycle detected")
            if call_id in visited:
                return
            visiting.add(call_id)
            for dependency in dependencies.get(call_id, set()):
                visit(dependency)
            visiting.remove(call_id)
            visited.add(call_id)

        for call_id in ids:
            visit(call_id)
        return self


class ToolObservation(StrictToolModel):
    call_id: str
    tool_name: ToolName
    success: bool
    status: Literal["success", "empty", "error", "timeout", "skipped"]
    data: Any = None
    error: str = ""
    duration_ms: float = Field(default=0.0, ge=0.0)
    metadata: dict[str, Any] = Field(default_factory=dict)


def compile_legacy_tool_plan(plan: Any) -> ToolPlan:
    """Compile the existing MusicQueryPlan into ToolPlan v1 as a migration fallback.

    The LLM should emit ToolPlan directly after rollout.  This compiler keeps
    old local models and recorded plans executable while the new protocol is
    evaluated in shadow mode.
    """

    intent_type = str(getattr(plan, "intent_type", "") or "")
    retrieval = getattr(plan, "retrieval_plan", None)
    reasoning = str(getattr(plan, "reasoning", "") or "")
    if intent_type == "clarification":
        parameters = getattr(plan, "parameters", {}) or {}
        return ToolPlan(
            origin="legacy_compiler",
            request_mode="conversation",
            needs_clarification=True,
            clarification_question=str(parameters.get("question") or "请再具体说明一下你的音乐需求。"),
            confidence=0.5,
            decision_summary=reasoning,
            max_replans=0,
        )
    if intent_type == "general_chat":
        return ToolPlan(
            origin="legacy_compiler",
            request_mode="conversation",
            decision_summary=reasoning,
            max_replans=0,
        )

    hard = getattr(retrieval, "hard_constraints", None)
    hints = getattr(retrieval, "hints", None)
    metadata = getattr(retrieval, "metadata_constraints", None)
    soft = getattr(retrieval, "soft_intent", None)
    calls: list[ToolCall] = []

    graph_args = SearchGraphArguments(
        artist_entities=list(getattr(hard, "artist_entities", []) or []),
        song_entities=list(getattr(hard, "song_entities", []) or []),
        language=getattr(hard, "language", None),
        region=getattr(hard, "region", None),
        instrumental=bool(getattr(hard, "instrumental", False)),
        genres=list(getattr(hints, "genres", []) or []),
        moods=[getattr(hints, "mood", None)] if getattr(hints, "mood", None) else [],
        scenarios=[getattr(hints, "scenario", None)] if getattr(hints, "scenario", None) else [],
        release_year_from=getattr(metadata, "release_year_from", None),
        release_year_to=getattr(metadata, "release_year_to", None),
        era=getattr(metadata, "era", None),
    )
    graph_signal = any(
        value
        for key, value in graph_args.model_dump(exclude_none=True).items()
        if key != "limit"
    )
    if bool(getattr(retrieval, "use_graph", False)) or graph_signal:
        calls.append(
            ToolCall(
                id="graph_recall",
                name=ToolName.SEARCH_GRAPH,
                arguments=graph_args.model_dump(exclude_none=True),
                reason="structured catalog and entity constraints",
            )
        )

    acoustic_queries = list(getattr(retrieval, "vector_acoustic_queries", []) or [])
    primary = str(getattr(retrieval, "vector_acoustic_query", "") or "").strip()
    if primary and primary not in acoustic_queries:
        acoustic_queries.insert(0, primary)
    if not acoustic_queries:
        fallback_vibe = " ".join(
            str(value or "").strip()
            for value in (
                getattr(soft, "goal", ""),
                getattr(soft, "trajectory", ""),
                getattr(soft, "vibe", ""),
            )
            if str(value or "").strip()
        )
        if fallback_vibe:
            acoustic_queries = [fallback_vibe]
    if bool(getattr(retrieval, "use_vector", False)) and acoustic_queries:
        calls.append(
            ToolCall(
                id="audio_recall",
                name=ToolName.SEARCH_AUDIO,
                arguments={
                    "acoustic_queries": acoustic_queries[:4],
                    "negative_targets": list(getattr(soft, "avoid", []) or []),
                    "limit": 30,
                },
                reason="continuous text-to-audio intent",
            )
        )

    local_ids = [call.id for call in calls]
    needs_gap = bool(
        getattr(retrieval, "use_web_search", False)
        or getattr(metadata, "release_year_from", None)
        or getattr(metadata, "release_year_to", None)
        or getattr(metadata, "era", None)
        or getattr(metadata, "recency_required", False)
        or getattr(metadata, "external_knowledge_required", False)
    )
    if needs_gap:
        calls.append(
            ToolCall(
                id="catalog_gap",
                name=ToolName.INSPECT_CATALOG_GAP,
                arguments={"requirements": metadata.model_dump(exclude_none=True) if metadata else {}},
                depends_on=local_ids,
                reason="verify whether the local catalog can satisfy metadata requirements",
            )
        )

    if intent_type == "web_search" or bool(getattr(retrieval, "use_web_search", False)):
        requirements = str(getattr(retrieval, "web_search_keywords", "") or getattr(plan, "context", "") or "music discovery")
        dependencies = ["catalog_gap"] if needs_gap else []
        calls.append(
            ToolCall(
                id="external_discovery",
                name=ToolName.SEARCH_EXTERNAL_MUSIC,
                arguments={"requirements": requirements, "limit": 10},
                depends_on=dependencies,
                reason="local catalog or external knowledge is insufficient",
            )
        )

    request_mode = "information" if intent_type == "web_search" else "recommendation"
    if intent_type == "acquire_music":
        request_mode = "acquisition"
    return ToolPlan(
        origin="legacy_compiler",
        request_mode=request_mode,
        tool_calls=calls,
        confidence=1.0,
        decision_summary=reasoning,
        max_replans=1,
    )


def tool_plan_alignment_issues(plan: Any) -> list[str]:
    """Check that ToolPlan and RetrievalPlan express the same LLM decision.

    This validates protocol consistency only.  It deliberately does not parse
    user text or override the planner's semantic judgment.
    """

    tool_plan = getattr(plan, "tool_plan", None)
    retrieval = getattr(plan, "retrieval_plan", None)
    if tool_plan is None or retrieval is None:
        return ["missing_plan"]

    names = {call.name for call in tool_plan.tool_calls}
    issues: list[str] = []
    intent_type = str(getattr(plan, "intent_type", "") or "")
    direct_external_information = bool(
        ToolName.SEARCH_EXTERNAL_MUSIC in names
        and (tool_plan.request_mode == "information" or intent_type == "web_search")
    )
    if intent_type in {"clarification", "general_chat"} and names:
        issues.append("conversation_has_tools")

    hard = getattr(retrieval, "hard_constraints", None)
    metadata = getattr(retrieval, "metadata_constraints", None)
    graph_signal = bool(
        getattr(retrieval, "use_graph", False)
        or getattr(hard, "artist_entities", [])
        or getattr(hard, "song_entities", [])
        or getattr(hard, "language", None)
        or getattr(hard, "region", None)
        or getattr(metadata, "release_year_from", None)
        or getattr(metadata, "release_year_to", None)
        or getattr(metadata, "era", None)
    )
    if graph_signal and not direct_external_information and ToolName.SEARCH_GRAPH not in names:
        issues.append("graph_signal_without_search_graph")

    audio_signal = bool(
        getattr(retrieval, "use_vector", False)
        or getattr(retrieval, "vector_acoustic_query", None)
        or getattr(retrieval, "vector_acoustic_queries", [])
        or getattr(hard, "instrumental", False)
    )
    if audio_signal and ToolName.SEARCH_AUDIO not in names:
        issues.append("audio_signal_without_search_audio")

    web_signal = bool(intent_type == "web_search" or getattr(retrieval, "use_web_search", False))
    if web_signal and ToolName.SEARCH_EXTERNAL_MUSIC not in names:
        issues.append("web_signal_without_external_search")

    gap_signal = bool(
        not direct_external_information
        and (
        getattr(metadata, "release_year_from", None)
        or getattr(metadata, "release_year_to", None)
        or getattr(metadata, "era", None)
        or getattr(metadata, "recency_required", False)
        or getattr(metadata, "external_knowledge_required", False)
        )
    )
    if gap_signal and ToolName.INSPECT_CATALOG_GAP not in names:
        issues.append("catalog_requirement_without_gap_check")
    return issues
