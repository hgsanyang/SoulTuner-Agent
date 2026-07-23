"""Assembled planner context + temporal habit cards (input protocol freeze).

Why this exists
---------------
A real request leans on the current turn, the running session, the explicit
profile, episodic memory, time-of-day habits and library state *simultaneously*.
Asking the planner to predict a single ``preference_source`` would be noise, so
V3 has no such field: the system assembles context deterministically here, hands
it to the planner as INPUT, and records it in the trace. What the planner
outputs stays a decision; what fed that decision stays evidence.

This module freezes the SCHEMA only. The statistical aggregator
(TemporalHabitAnalyzer) runs asynchronously off the hot path and is implemented
separately — habit cards are produced from counted evidence, never from an LLM
"noticing a pattern".

Precedence — a habit may only tilt ranking, never override what the user just
said:

    1. this turn's explicit request
    2. this turn's correction / negation
    3. preferences the user set deliberately
    4. episodic memory matching the current scene
    5. temporal habit priors
    6. global behaviour statistics

"今天我就想听热闹一点" must beat "工作日深夜通常听低动态".
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

CONTEXT_SCHEMA_VERSION = "agent_context_v1"

DayType = Literal["weekday", "weekend"]
DayPart = Literal["late_night", "early_morning", "morning", "afternoon", "evening", "night"]

# A habit card is only emitted when the counted evidence clears every gate;
# below the gate we keep raw statistics and stay silent.
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


class SessionContext(_Strict):
    """What already happened in this conversation (highest-priority after the ask)."""

    refinements: list[str] = Field(default_factory=list)   # "安静一点" / "不要太悲伤"
    rejected: list[str] = Field(default_factory=list)      # directions user pushed away
    last_plan_summary: str = ""


class HabitCard(_Strict):
    """A counted, gated periodic listening habit — a PRIOR, never a constraint."""

    habit_id: str
    statement: str                          # natural-language summary of the counts
    day_type: Optional[DayType] = None
    day_part: Optional[DayPart] = None
    scene: str = ""
    support_sessions: int = Field(default=0, ge=0)
    distinct_weeks: int = Field(default=0, ge=0)
    lift_vs_baseline: float = Field(default=1.0, ge=0.0)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    window_days: int = Field(default=28, ge=1)
    computed_at: Optional[int] = None        # epoch seconds
    valid_until: Optional[int] = None
    user_edited: bool = False                # user may correct/disable a habit

    def passes_gate(self) -> bool:
        return (
            self.support_sessions >= HABIT_MIN_SESSIONS
            and self.distinct_weeks >= HABIT_MIN_DISTINCT_WEEKS
            and self.lift_vs_baseline >= HABIT_MIN_LIFT
        )


class RetrievedMemory(_Strict):
    memory_id: str
    statement: str
    layer: str = ""                          # explicit / inferred / episodic
    occurred_at: Optional[int] = None
    relevance: float = Field(default=0.0, ge=0.0, le=1.0)


class LibrarySummary(_Strict):
    favorites_available: bool = False
    catalog_size: int = Field(default=0, ge=0)
    recent_ingest_count: int = Field(default=0, ge=0)


class AssembledContext(_Strict):
    """Everything the planner is ALLOWED to condition on, assembled by code.

    Carried in the trace so an eval can attribute a decision to its inputs; the
    planner never has to guess which source it used.
    """

    schema_version: str = CONTEXT_SCHEMA_VERSION
    current: CurrentContext = Field(default_factory=CurrentContext)
    session: SessionContext = Field(default_factory=SessionContext)
    explicit_profile: str = ""
    temporal_habits: list[HabitCard] = Field(default_factory=list, max_length=4)
    retrieved_memories: list[RetrievedMemory] = Field(default_factory=list, max_length=8)
    library: LibrarySummary = Field(default_factory=LibrarySummary)

    def gated_habits(self) -> list[HabitCard]:
        """Only habits that cleared the evidence gate may reach the planner."""
        return [h for h in self.temporal_habits if h.passes_gate() and not h.user_edited]
