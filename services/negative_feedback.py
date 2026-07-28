"""Use per-song "doesn't fit right now" feedback to change what we recommend.

Two mechanisms, deliberately at very different confidence levels:

1. ``recent_context_rejections`` — exact-song suppression. You said a track does
   not fit; for a while, we stop showing you that exact track. This is the ONE
   thing a single negative rating tells us with high confidence, and it cannot
   over-generalise because it applies to nothing else.

2. ``negative_anchor_penalty`` — a bounded score reduction for songs that are
   acoustically close to something you rejected. Much lower confidence, so it is
   **off by default** and every term in it is capped.

Scoping
-------
A rejection applies within its SESSION when one is known, and otherwise inside a
recency window. Session is the honest boundary: rejecting a track while asking
for "quiet, before sleep" should not keep it out of a road-trip request an hour
later. The window is the fallback for callers that have no session id, not the
primary rule.

What this module deliberately does NOT do
-----------------------------------------
It never touches the query. An earlier design extracted a rejected song's
acoustic traits into ``soft_intent.avoid``; that reads as "reject piano, slow
tempo and Chinese" from one data point about one song, and it propagates into
the planner, HyDE and hard filtering where it is invisible and hard to undo.
Negative feedback should lower confidence, not invert a preference.
"""

from __future__ import annotations

import os
import re
import time
from typing import Any, Iterable

from config.logging_config import get_logger

logger = get_logger(__name__)

# Fallback scope when the caller has no session id. Roughly one sitting.
DEFAULT_SUPPRESSION_WINDOW_MS = 2 * 60 * 60 * 1000      # 2 hours

# Hard cap on how much the (default-off) similarity penalty may subtract,
# expressed as a fraction of the song's own score.
MAX_PENALTY_RATIO = 0.20

_NORMALISE = re.compile(r"[\s\-_—·・,，.。!！?？'\"()（）\[\]【】]+")


def _now_ms() -> int:
    return int(time.time() * 1000)


def _truthy(value: object) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def song_key(title: Any, artist: Any) -> str:
    """Normalised title+artist, the fallback identity when music_id is absent.

    Web-supplement candidates often arrive without a music_id, so an id-only
    match let a track you just rejected come back through the online lane.
    """
    text = "%s|%s" % (str(title or ""), str(artist or ""))
    return _NORMALISE.sub("", text).casefold()


def similarity_penalty_enabled() -> bool:
    """The bounded similarity penalty is OFF until a targeted eval says otherwise."""
    return _truthy(os.getenv("MUSIC_NEGATIVE_ANCHOR_PENALTY"))


def recent_context_rejection_rows(
    user_id: str,
    *,
    session_id: str = "",
    now_ms: int | None = None,
    window_ms: int = DEFAULT_SUPPRESSION_WINDOW_MS,
    feedback_rows: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Return the canonical rejection snapshot used by suppression and ranking."""
    now = int(now_ms if now_ms is not None else _now_ms())
    cutoff = now - max(0, int(window_ms))
    rows = feedback_rows
    if rows is None:
        try:
            from services.feedback_logger import load_song_feedback_canonical

            rows = load_song_feedback_canonical()
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug(
                "[NegativeFeedback] 读取逐首反馈失败，跳过抑制: %s: %s",
                type(exc).__name__,
                exc,
            )
            return []

    target_user = str(user_id or "").strip()
    target_session = str(session_id or "").strip()

    # Collapse to the latest judgement per (exposure, song).
    latest: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        if target_user and str(row.get("user_id") or "").strip() != target_user:
            continue
        identity = (
            str(row.get("exposure_id") or ""),
            str(row.get("music_id") or "")
            or song_key(row.get("title"), row.get("artist")),
        )
        current = latest.get(identity)
        if current is None or int(row.get("ts") or 0) >= int(current.get("ts") or 0):
            latest[identity] = row

    selected: list[dict[str, Any]] = []
    for row in latest.values():
        if str(row.get("context_fit") or "").strip() != "off":
            continue
        if target_session:
            row_session = str(
                (row.get("context") or {}).get("session_id")
                or row.get("session_id")
                or ""
            ).strip()
            if row_session and row_session != target_session:
                continue
            if not row_session and int(row.get("ts") or 0) < cutoff:
                continue
        elif int(row.get("ts") or 0) and int(row.get("ts") or 0) < cutoff:
            continue
        selected.append(dict(row))
    return selected


def recent_context_rejections(
    user_id: str,
    *,
    session_id: str = "",
    now_ms: int | None = None,
    window_ms: int = DEFAULT_SUPPRESSION_WINDOW_MS,
    feedback_rows: list[dict[str, Any]] | None = None,
    with_names: bool = False,
):
    """What this user rejected, scoped to the session when one is known.

    Only an explicit ``context_fit == "off"`` counts. ``partial`` is genuinely
    mixed and unrated is UNKNOWN — treating either as a rejection would punish
    songs the user never complained about.

    Returns a set of music_ids, or ``(music_ids, song_keys)`` when
    ``with_names`` — the second set covers candidates that carry no music_id.

    LATEST WINS: if you rate the same song in the same exposure twice, only the
    most recent judgement counts. Rating "off" then correcting to "fits" must
    lift the suppression, not leave both records fighting.
    """
    ids: set[str] = set()
    names: set[str] = set()
    for row in recent_context_rejection_rows(
        user_id,
        session_id=session_id,
        now_ms=now_ms,
        window_ms=window_ms,
        feedback_rows=feedback_rows,
    ):
        music_id = str(row.get("music_id") or "").strip()
        if music_id:
            ids.add(music_id)
        key = song_key(row.get("title"), row.get("artist"))
        if key:
            names.add(key)
    return (ids, names) if with_names else ids


def suppress_rejected(
    items: list[dict[str, Any]],
    rejected_music_ids: Iterable[str],
    *,
    rejected_names: Iterable[str] | None = None,
    keep_at_least: int = 5,
) -> tuple[list[dict[str, Any]], int]:
    """Drop candidates the user just rejected, without emptying the slate.

    Matches on music_id first and on normalised title+artist second, because web
    candidates frequently have no music_id — matching only on id let a rejected
    song return through the online lane.

    ``keep_at_least`` is a floor, not a preference: if suppression would leave
    almost nothing, a thin useless slate is worse than a repeat.
    """
    rejected = {str(mid).strip() for mid in rejected_music_ids if str(mid).strip()}
    names = {str(n).strip() for n in (rejected_names or ()) if str(n).strip()}
    if not rejected and not names:
        return items, 0

    kept, dropped = [], []
    for item in items:
        song = item.get("song") or item
        music_id = str(song.get("music_id") or "").strip()
        key = song_key(song.get("title"), song.get("artist"))
        hit = (music_id and music_id in rejected) or (key and key in names)
        (dropped if hit else kept).append(item)

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
    """
    sim = max(0.0, min(float(similarity_to_nearest_rejection), 1.0))
    ctx = max(0.0, min(float(context_similarity), 1.0))
    cap = max(0.0, min(float(max_ratio), MAX_PENALTY_RATIO))
    if sim <= 0.0 or ctx <= 0.0 or cap <= 0.0:
        return 0.0
    decay = 0.5 ** (max(0, int(age_ms)) / max(1, int(half_life_ms)))
    return round(sim * ctx * cap * decay, 6)


def apply_negative_anchor_penalty(
    items: list[dict[str, Any]],
    rejected_rows: list[dict[str, Any]],
    *,
    now_ms: int | None = None,
    score_field: str = "similarity_score",
    candidate_vectors: dict[str, dict[str, Any]] | None = None,
) -> int:
    """Apply the bounded penalty to a candidate list, in place. Returns count.

    THE production entry point. Without this the penalty was a tested function
    nobody called — setting the flag changed nothing, which made "implemented,
    off by default" an overstatement.

    Similarity is cosine over whichever audio embedding both the candidate and
    the rejected song carry. A candidate with no comparable embedding is left
    alone rather than guessed at.
    """
    if not similarity_penalty_enabled() or not items or not rejected_rows:
        return 0

    now = int(now_ms if now_ms is not None else _now_ms())
    touched = 0
    for item in items:
        song = item.get("song") or item
        candidate = song
        if candidate_vectors:
            music_id = str(song.get("music_id") or "").strip()
            candidate = (
                candidate_vectors.get(f"id:{music_id}") if music_id else None
            ) or candidate_vectors.get(
                f"song:{song_key(song.get('title'), song.get('artist'))}"
            ) or song
        best_sim, best_age = 0.0, 0
        for rejected in rejected_rows:
            sim = _embedding_similarity(candidate, rejected)
            if sim > best_sim:
                best_sim = sim
                best_age = max(0, now - int(rejected.get("ts") or now))
        if best_sim <= 0.0:
            continue
        ratio = negative_anchor_penalty(best_sim, age_ms=best_age)
        if ratio <= 0.0:
            continue
        base = float(item.get(score_field) or 0.0)
        item[score_field] = base * (1.0 - ratio)
        item["_negative_anchor_penalty"] = round(ratio, 4)
        touched += 1
    if touched:
        logger.info("[NegativeFeedback] 负例相似度惩罚已应用于 %d 首（上限 %.0f%%）",
                    touched, MAX_PENALTY_RATIO * 100)
    return touched


def load_rejection_anchors(
    rejections: Iterable[Any],
    *,
    rows_by_id: dict[str, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Fetch embeddings for the rejected songs.

    Feedback records store no vectors -- only ids, titles and the judgement --
    so the anchors have to be read from the catalogue. Without this the penalty
    compared nothing against nothing and silently scored 0, which is worse than
    being off: it looks enabled.
    """
    ids: list[str] = []
    ts_by_id: dict[str, int] = {}
    for rejection in rejections:
        if isinstance(rejection, dict):
            music_id = str(rejection.get("music_id") or "").strip()
            ts = int(rejection.get("ts") or 0)
        else:
            music_id = str(rejection or "").strip()
            ts = 0
        if not music_id:
            continue
        if music_id not in ids:
            ids.append(music_id)
        ts_by_id[music_id] = max(ts_by_id.get(music_id, 0), ts)
    if not ids:
        return []
    if rows_by_id is not None:              # injected in tests
        result = []
        for music_id in ids:
            if music_id not in rows_by_id:
                continue
            row = dict(rows_by_id[music_id])
            if ts_by_id.get(music_id):
                row["ts"] = ts_by_id[music_id]
            result.append(row)
        return result
    try:
        from retrieval.neo4j_client import get_neo4j_client

        rows = get_neo4j_client().execute_query(
            """
            MATCH (s:Song) WHERE s.music_id IN $ids
            RETURN s.music_id AS music_id,
                   s.muq_embedding AS muq_embedding,
                   s.m2d2_embedding AS m2d2_embedding,
                   s.omar_embedding AS omar_embedding
            """,
            {"ids": ids},
        )
        result = []
        for raw in rows or []:
            row = dict(raw)
            music_id = str(row.get("music_id") or "").strip()
            if ts_by_id.get(music_id):
                row["ts"] = ts_by_id[music_id]
            result.append(row)
        return result
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("[NegativeFeedback] 取负例向量失败，跳过惩罚: %s: %s",
                       type(exc).__name__, exc)
        return []


def load_candidate_vectors(
    items: list[dict[str, Any]],
    *,
    rows: list[dict[str, Any]] | None = None,
) -> dict[str, dict[str, Any]]:
    """Load candidate embeddings once and index them by id and title+artist."""
    candidates = []
    for item in items:
        song = item.get("song") or item
        candidates.append(
            {
                "music_id": str(song.get("music_id") or "").strip(),
                "title": str(song.get("title") or "").strip(),
                "artist": str(song.get("artist") or "").strip(),
            }
        )
    if not candidates:
        return {}

    result_rows = rows
    if result_rows is None:
        try:
            from retrieval.neo4j_client import get_neo4j_client

            result_rows = get_neo4j_client().execute_query(
                """
                UNWIND $candidates AS candidate
                MATCH (s:Song)
                WHERE (
                    candidate.music_id <> ''
                    AND s.music_id = candidate.music_id
                ) OR (
                    candidate.title <> ''
                    AND toLower(trim(coalesce(s.title, ''))) =
                        toLower(trim(candidate.title))
                    AND (
                        candidate.artist = ''
                        OR toLower(trim(coalesce(s.artist, ''))) =
                            toLower(trim(candidate.artist))
                    )
                )
                RETURN DISTINCT s.music_id AS music_id,
                       s.title AS title,
                       s.artist AS artist,
                       s.muq_embedding AS muq_embedding,
                       s.m2d2_embedding AS m2d2_embedding,
                       s.omar_embedding AS omar_embedding
                """,
                {"candidates": candidates},
            )
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning(
                "[NegativeFeedback] 取候选向量失败，跳过惩罚: %s: %s",
                type(exc).__name__,
                exc,
            )
            return {}

    indexed: dict[str, dict[str, Any]] = {}
    for raw in result_rows or []:
        row = dict(raw)
        music_id = str(row.get("music_id") or "").strip()
        if music_id:
            indexed[f"id:{music_id}"] = row
        key = song_key(row.get("title"), row.get("artist"))
        if key:
            indexed[f"song:{key}"] = row
    return indexed


def _embedding_similarity(song: dict[str, Any], rejected: dict[str, Any]) -> float:
    """Cosine over a shared embedding field, or 0.0 when there is none."""
    for field in ("muq_embedding", "m2d2_embedding", "omar_embedding"):
        left, right = song.get(field), rejected.get(field)
        if not left or not right or len(left) != len(right):
            continue
        dot = sum(a * b for a, b in zip(left, right))
        na = sum(a * a for a in left) ** 0.5
        nb = sum(b * b for b in right) ** 0.5
        if na <= 0 or nb <= 0:
            continue
        return max(0.0, min(dot / (na * nb), 1.0))
    return 0.0
