"""Feedback & exposure event contracts (P1 data protocol).

Three DIFFERENT purposes must never be collapsed into one training set:

  1. personal adaptation   -> update session state / long-term memory   (no LLM training)
  2. ranking learning      -> train a reranker, exposure/novelty weights (not the planner)
  3. agent behaviour       -> find wrong ToolPlans -> correction SFT / preference pairs

So per-song feedback is captured on TWO SEPARATE CHANNELS:

  taste channel    like / save / dislike / block      -> long-term preference (purpose 1)
  context channel  fits / partial / off + reason      -> THIS slate in THIS context (purpose 2)

A song can be loved and still wrong for tonight; conflating them poisons both.

Weak signals (play duration, skip, repeat) stay weak, and **not clicked is
UNKNOWN, never negative** — the user simply may not have reached it.

Every event carries the listening context (timezone/local hour/day type/scene/
session) plus exposure bookkeeping (rank, propensity, policy_version) so ranking
work can debias later. Missing these at write time is unrecoverable: you cannot
backfill what time it was for the user.

Timestamps are UTC epoch MILLISECONDS, matching MemoryGateway and agent_context.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

FEEDBACK_SCHEMA_VERSION = "feedback_events_v1"

TasteSignal = Literal["like", "save", "dislike", "block"]
ContextFit = Literal["fits", "partial", "off"]
# Why a track did not fit *this* context. Free text is always allowed too.
OffReason = Literal[
    "mood_mismatch",     # 氛围不对
    "too_loud",          # 太吵
    "too_flat",          # 太平
    "wrong_language",    # 语言不对
    "wrong_era",         # 年代不对
    "overplayed",        # 已经听腻
    "want_unfamiliar",   # 想要更陌生
    "bad_audio",         # 版本或音源有问题
]
# Why a whole SLATE missed. Deliberately a separate vocabulary from OffReason:
# these judge the LIST, not a track. "太重复" and "太冷门" cannot be said about a
# single song, and "this one is too loud" is not the same complaint as "the whole
# set is too mainstream". Slugs (not display labels) are stored so renaming a
# button never splits the history into two categories.
SlateReason = Literal[
    "too_loud",               # 太吵
    "too_sad",                # 太悲伤
    "too_mainstream",         # 太热门
    "too_obscure",            # 太冷门
    "too_repetitive",         # 重复太多
    "scene_mismatch",         # 场景不合
    "wrong_language_or_era",  # 语言/年代不准
    "other",                  # 其他（配合 note 自由文本）
]
# Where a track came from — needed to split evaluation cohorts, otherwise a
# local library full of the user's own favourites flatters every metric.
CatalogOrigin = Literal["prior_favorite", "local_unheard", "online_new", "neutral_open"]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ListeningContext(_Strict):
    """When/where this happened, in the USER's frame of reference."""

    ts_ms: int = 0
    timezone: str = ""                                  # IANA
    local_hour: Optional[int] = Field(default=None, ge=0, le=23)
    day_of_week: Optional[int] = Field(default=None, ge=0, le=6)   # 0=Mon
    day_type: Optional[Literal["weekday", "weekend"]] = None
    day_part: Optional[Literal["late_night", "early_morning", "morning",
                               "afternoon", "evening", "night"]] = None
    session_id: str = ""
    scene: str = ""                                     # scene the user stated
    device: str = ""


class ExposureBookkeeping(_Strict):
    """What the policy did, so ranking can be debiased later."""

    rank: Optional[int] = Field(default=None, ge=1)
    propensity: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    policy_version: str = ""
    is_exploration: bool = False
    catalog_origin: Optional[CatalogOrigin] = None
    known_to_user: Optional[bool] = None                # None = unknown, not False
    # Two DIFFERENT quantities that used to be crushed into one int:
    #   historical_exposure_count — how many times we have shown this before (a count)
    #   effective_exposure         — the time-decayed float the ranker penalised on
    # None = unknown (we did not record it), which is not the same as 0.
    historical_exposure_count: Optional[int] = Field(default=None, ge=0)
    effective_exposure: Optional[float] = Field(default=None, ge=0.0)


class SongFeedback(_Strict):
    """Per-song feedback on two independent channels."""

    schema_version: str = FEEDBACK_SCHEMA_VERSION
    exposure_id: str = ""
    music_id: str = ""
    title: str = ""
    artist: str = ""
    user_id: str = "local_admin"

    taste: Optional[TasteSignal] = None       # long-term preference (purpose 1)
    context_fit: Optional[ContextFit] = None  # this slate, this context (purpose 2)
    off_reasons: list[OffReason] = Field(default_factory=list)
    note: str = Field(default="", max_length=500)   # free-text, always allowed

    context: ListeningContext = Field(default_factory=ListeningContext)
    exposure: ExposureBookkeeping = Field(default_factory=ExposureBookkeeping)

    def is_empty(self) -> bool:
        return not (self.taste or self.context_fit or self.off_reasons or self.note.strip())


class SlateFeedback(_Strict):
    """Slate-level judgement. Best/worst picks give attribution without needing
    the user to rate (or even hear) every track."""

    schema_version: str = FEEDBACK_SCHEMA_VERSION
    exposure_id: str = ""
    user_id: str = "local_admin"
    overall: Optional[ContextFit] = None
    best_music_ids: list[str] = Field(default_factory=list, max_length=3)
    worst_music_ids: list[str] = Field(default_factory=list, max_length=3)
    reasons: list[SlateReason] = Field(default_factory=list)
    note: str = Field(default="", max_length=500)
    context: ListeningContext = Field(default_factory=ListeningContext)


# The UI rates a slate 整体合适/部分合适/不太合适; the stored judgement uses the
# same three-way scale as per-song context_fit so the two channels are directly
# comparable offline.
SLATE_RATING_TO_OVERALL: dict[str, ContextFit] = {
    "great": "fits",
    "partial": "partial",
    "off": "off",
}


def derive_context(ts_ms: int, timezone: str, *, session_id: str = "", scene: str = "",
                   device: str = "") -> ListeningContext:
    """Fill the derivable time fields from a timestamp + IANA timezone.

    Derived deterministically (never guessed); if the timezone is unknown we
    leave the local fields None rather than silently assuming the server's.
    """
    ctx = ListeningContext(ts_ms=ts_ms, timezone=timezone, session_id=session_id,
                           scene=scene, device=device)
    if not timezone:
        return ctx
    try:
        from datetime import datetime, timezone as _tz
        from zoneinfo import ZoneInfo

        local = datetime.fromtimestamp(ts_ms / 1000, tz=_tz.utc).astimezone(ZoneInfo(timezone))
    except Exception:
        return ctx
    hour = local.hour
    ctx.local_hour = hour
    ctx.day_of_week = local.weekday()
    ctx.day_type = "weekend" if local.weekday() >= 5 else "weekday"
    if hour < 5:
        ctx.day_part = "late_night"
    elif hour < 8:
        ctx.day_part = "early_morning"
    elif hour < 12:
        ctx.day_part = "morning"
    elif hour < 18:
        ctx.day_part = "afternoon"
    elif hour < 22:
        ctx.day_part = "evening"
    else:
        ctx.day_part = "night"
    return ctx
