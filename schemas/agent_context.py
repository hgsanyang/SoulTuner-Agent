"""Assembled planner context + temporal habit cards (input protocol).

Why this exists
---------------
A real request leans on the current turn, the running session, the explicit
profile, episodic memory, time-of-day habits and library state *simultaneously*.
Asking the planner to predict a single ``preference_source`` would be noise, so
V3 has no such field: the system assembles context deterministically here, hands
it to the planner as INPUT, and records it in the trace. What the planner
outputs stays a decision; what fed that decision stays evidence.

This module freezes the SCHEMA only. The statistical aggregator
(TemporalHabitAnalyzer) runs asynchronously off the hot path — habit cards are
produced from counted evidence, never from an LLM "noticing a pattern".

Precedence — a habit may only tilt ranking, never override what the user just
said:

    1. this turn's explicit request
    2. this turn's correction / negation
    3. preferences the user set deliberately
    4. episodic memory matching the current scene
    5. temporal habit priors
    6. global behaviour statistics

"今天我就想听热闹一点" must beat "工作日深夜通常听低动态".

Time unit: every timestamp is UTC epoch **milliseconds** (``*_at_ms``), matching
``MemoryGateway`` (``int(time.time() * 1000)``). Mixing seconds in here would
make TTL and ordering wrong by 1000x.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

CONTEXT_SCHEMA_VERSION = "agent_context_v1"

DayType = Literal["weekday", "weekend"]
DayPart = Literal["late_night", "early_morning", "morning", "afternoon", "evening", "night"]
HabitTargetKind = Literal["genre", "mood", "scenario", "acoustic", "language", "artist"]
HabitDirection = Literal["prefer", "avoid"]
# corrected = the user edited the statement; it is MORE trustworthy than a raw
# statistical card, not less. Only `disabled` is withheld from the planner.
HabitStatus = Literal["active", "disabled", "corrected"]
HabitSource = Literal["statistical", "user_confirmed"]

# Evidence gates. Below these we keep raw candidates and stay silent.
HABIT_MIN_SESSIONS = 6
HABIT_MIN_DISTINCT_WEEKS = 3
HABIT_MIN_LIFT = 1.2


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class CurrentContext(_Strict):
    """Deterministic facts about this turn — measured, never inferred."""

    local_time: str = ""            # "22:40" in the USER's timezone
    timezone: str = ""              # IANA, e.g. "Asia/Shanghai"
    local_hour: Optional[int] = Field(default=None, ge=0, le=23)
    day_of_week: Optional[int] = Field(default=None, ge=0, le=6)  # 0=Mon
    day_type: Optional[DayType] = None
    day_part: Optional[DayPart] = None
    explicit_scene: str = ""        # scene the user stated this turn, free text
    session_id: str = ""
    turn_at_ms: Optional[int] = None


class RecommendedAnchor(_Strict):
    """A song actually shown last turn — what "第二首""更像第三首" refer to.

    Without these anchors the planner can only inherit plan-level tags and will
    keep failing on song-level reference resolution.
    """

    music_id: str = ""
    title: str = ""
    artist: str = ""
    rank: Optional[int] = Field(default=None, ge=1)
    feedback: str = ""              # like/save/skip/dislike/"" (unknown)


class SessionContext(_Strict):
    """What already happened in this conversation (highest priority after the ask)."""

    refinements: list[str] = Field(default_factory=list)   # "安静一点" / "不要太悲伤"
    rejected: list[str] = Field(default_factory=list)      # directions user pushed away
    last_plan_summary: str = ""
    last_recommendations: list[RecommendedAnchor] = Field(default_factory=list, max_length=10)


class HabitEvidence(_Strict):
    """Counts behind a habit — so "为什么得出这个习惯" is answerable."""

    support_sessions: int = Field(default=0, ge=0)
    distinct_weeks: int = Field(default=0, ge=0)
    counter_examples: int = Field(default=0, ge=0)
    context_rate: float = Field(default=0.0, ge=0.0, le=1.0)   # rate in this context
    baseline_rate: float = Field(default=0.0, ge=0.0, le=1.0)  # user's own baseline
    lift_vs_baseline: float = Field(default=1.0, ge=0.0)
    window_start_ms: Optional[int] = None
    window_end_ms: Optional[int] = None
    evidence_ids: list[str] = Field(default_factory=list, max_length=50)


class HabitCandidate(_Strict):
    """A raw statistical finding. NOT planner input until it clears the gate."""

    habit_id: str
    statement: str
    target_kind: Optional[HabitTargetKind] = None
    target_value: str = ""
    direction: HabitDirection = "prefer"
    day_type: Optional[DayType] = None
    day_part: Optional[DayPart] = None
    scene: str = ""
    evidence: HabitEvidence = Field(default_factory=HabitEvidence)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    status: HabitStatus = "active"
    source: HabitSource = "statistical"
    computed_at_ms: Optional[int] = None
    valid_until_ms: Optional[int] = None
    policy_version: str = ""

    def passes_gate(self) -> bool:
        """User-confirmed cards bypass the statistical gate (a person said so)."""
        if self.status == "disabled":
            return False
        if self.source == "user_confirmed" or self.status == "corrected":
            return True
        e = self.evidence
        return (
            e.support_sessions >= HABIT_MIN_SESSIONS
            and e.distinct_weeks >= HABIT_MIN_DISTINCT_WEEKS
            and e.lift_vs_baseline >= HABIT_MIN_LIFT
        )


class HabitCard(HabitCandidate):
    """A candidate that HAS cleared the gate — the only kind the planner sees."""

    def model_post_init(self, __context) -> None:  # noqa: D105
        if not self.passes_gate():
            raise ValueError(
                f"HabitCard {self.habit_id} does not pass the evidence gate; keep it a HabitCandidate"
            )


class RetrievedMemory(_Strict):
    memory_id: str
    statement: str
    layer: str = ""                          # explicit / inferred / episodic
    occurred_at_ms: Optional[int] = None
    relevance: float = Field(default=0.0, ge=0.0, le=1.0)


class LibrarySummary(_Strict):
    favorites_available: bool = False
    catalog_size: int = Field(default=0, ge=0)
    recent_ingest_count: int = Field(default=0, ge=0)


class AssembledContext(_Strict):
    """Everything the planner is ALLOWED to condition on, assembled by code.

    ``temporal_habits`` is typed as ``HabitCard``, so an ungated candidate cannot
    be attached at all — the gate is enforced by the schema, not by remembering
    to call a filter.
    """

    schema_version: str = CONTEXT_SCHEMA_VERSION
    current: CurrentContext = Field(default_factory=CurrentContext)
    session: SessionContext = Field(default_factory=SessionContext)
    explicit_profile: str = ""
    temporal_habits: list[HabitCard] = Field(default_factory=list, max_length=4)
    retrieved_memories: list[RetrievedMemory] = Field(default_factory=list, max_length=8)
    library: LibrarySummary = Field(default_factory=LibrarySummary)


def promote_candidates(candidates: list[HabitCandidate], limit: int = 4) -> list[HabitCard]:
    """Gate raw candidates into planner-visible cards, user-confirmed first."""
    passing = [c for c in candidates if c.passes_gate()]
    passing.sort(
        key=lambda c: (
            0 if (c.source == "user_confirmed" or c.status == "corrected") else 1,
            -c.evidence.lift_vs_baseline,
        )
    )
    return [HabitCard.model_validate(c.model_dump()) for c in passing[:limit]]
