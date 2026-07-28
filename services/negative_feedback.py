"""Use per-song "doesn't fit right now" feedback to change what we recommend.

Two mechanisms, deliberately at very different confidence levels:

1. ``recent_context_rejections`` — exact-song suppression. You said a track does
   not fit; for a while, we stop showing you that exact track. This is the ONE
   thing a single negative rating tells us with high confidence, and it cannot
   over-generalise because it applies to nothing else.

2. ``negative_anchor_penalty`` — a bounded score reduction for songs that are
   acoustically close to something you rejected. Much lower confidence, so it is
   **off by default** and every term in it is capped.

What this module deliberately does NOT do
-----------------------------------------
It never touches the query. An earlier design extracted a rejected song's
acoustic traits into ``soft_intent.avoid``; that reads as "reject piano, slow
tempo and Chinese" from one data point about one song, and it propagates into
the planner, HyDE and hard filtering where it is invisible and hard to undo.
Negative feedback should lower confidence, not invert a preference.

Scoping is by TIME, not by session id: the retrieval layer has a user id but no
session id, and inventing a fake session boundary would be worse than saying
plainly that this is a recency window.
"""

from __future__ import annotations

import os
import time
from typing import Any, Iterable

from config.logging_config import get_logger

logger = get_logger(__name__)

# How long a "doesn't fit" keeps a song out of your results. Roughly one
# listening session — long enough that asking again in five minutes does not
# hand you back the song you just rejected, short enough that a mood from last
# week does not follow you around.
DEFAULT_SUPPRESSION_WINDOW_MS = 2 * 60 * 60 * 1000      # 2 hours

# Hard cap on how much the (default-off) similarity penalty may subtract,
# expressed as a fraction of the song's own score. Four data points must not be
# able to reorder a slate.
MAX_PENALTY_RATIO = 0.20


def _now_ms() -> int:
    return int(time.time() * 1000)


def _truthy(value: object) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def similarity_penalty_enabled() -> bool:
    """The bounded similarity penalty is OFF until a targeted eval says otherwise.

    Exact-song suppression needs no such gate — it changes nothing except not
    repeating a song you explicitly rejected.
    """
    return _truthy(os.getenv("MUSIC_NEGATIVE_ANCHOR_PENALTY"))


def recent_context_rejections(
    user_id: str,
    *,
    now_ms: int | None = None,
    window_ms: int = DEFAULT_SUPPRESSION_WINDOW_MS,
    feedback_rows: list[dict[str, Any]] | None = None,
) -> set[str]:
    """music_ids this user marked ``context_fit=off`` inside the window.

    Only an explicit "off" counts. ``partial`` is genuinely mixed and unrated is
    UNKNOWN — treating either as a rejection would punish songs the user never
    complained about.
    """
    now = int(now_ms if now_ms is not None else _now_ms())
    cutoff = now - max(0, int(window_ms))
    rows = feedback_rows
    if rows is None:
        try:
            from services.feedback_logger import load_song_feedback_canonical

            rows = load_song_feedback_canonical()
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("[NegativeFeedback] 读取逐首反馈失败，跳过抑制: %s: %s",
                         type(exc).__name__, exc)
            return set()

    target_user = str(user_id or "").strip()
    rejected: set[str] = set()
    for row in rows:
        if str(row.get("context_fit") or "").strip() != "off":
            continue
        if target_user and str(row.get("user_id") or "").strip() != target_user:
            continue
        ts = int(row.get("ts") or 0)
        if ts and ts < cutoff:
            continue
        music_id = str(row.get("music_id") or "").strip()
        if music_id:
            rejected.add(music_id)
    return rejected


def suppress_rejected(
    items: list[dict[str, Any]],
    rejected_music_ids: Iterable[str],
    *,
    keep_at_least: int = 5,
) -> tuple[list[dict[str, Any]], int]:
    """Drop candidates the user just rejected, without emptying the slate.

    ``keep_at_least`` is a floor, not a preference: if suppression would leave
    almost nothing, returning a thin useless slate is worse than showing a
    rejected song again. Returns (kept, dropped_count).
    """
    rejected = {str(mid).strip() for mid in rejected_music_ids if str(mid).strip()}
    if not rejected:
        return items, 0

    kept, dropped = [], []
    for item in items:
        song = item.get("song") or item
        music_id = str(song.get("music_id") or "").strip()
        (dropped if music_id and music_id in rejected else kept).append(item)

    if not dropped:
        return items, 0
    if len(kept) < max(0, int(keep_at_least)):
        logger.info("[NegativeFeedback] 抑制会让结果只剩 %d 首，放弃抑制（保底 %d）",
                    len(kept), keep_at_least)
        return items, 0
    logger.info("[NegativeFeedback] 按最近的「不符合」抑制了 %d 首", len(dropped))
    return kept, len(dropped)


def negative_anchor_penalty(
    similarity_to_nearest_rejection: float,
    *,
    context_similarity: float = 1.0,
    age_ms: int = 0,
    half_life_ms: int = 14 * 24 * 60 * 60 * 1000,
    max_ratio: float = MAX_PENALTY_RATIO,
) -> float:
    """Bounded penalty ratio in [0, max_ratio] for one candidate.

    penalty = neg_similarity x context_similarity x max_ratio x time_decay

    Every factor exists to limit blast radius:
      * neg_similarity  — far-away songs are barely touched
      * context_similarity — a rejection in one kind of request must not follow
        you into an unrelated one
      * max_ratio       — the hard cap; nothing can exceed it
      * time_decay      — an old "not tonight" should fade

    Returns a RATIO to subtract, so the caller keeps ownership of the score.
    This function does not know what a song is and cannot touch the query.
    """
    sim = max(0.0, min(float(similarity_to_nearest_rejection), 1.0))
    ctx = max(0.0, min(float(context_similarity), 1.0))
    cap = max(0.0, min(float(max_ratio), MAX_PENALTY_RATIO))
    if sim <= 0.0 or ctx <= 0.0 or cap <= 0.0:
        return 0.0
    decay = 0.5 ** (max(0, int(age_ms)) / max(1, int(half_life_ms)))
    return round(sim * ctx * cap * decay, 6)
