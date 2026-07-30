"""JSONL feedback logs for exposure replay and lightweight rank learning."""

from __future__ import annotations

import json
import hashlib
import logging
import math
import os
import time
import uuid
from pathlib import Path
from typing import Any

from services.runtime_mode import side_effects_disabled

logger = logging.getLogger(__name__)


POSITIVE_EVENTS = {"like", "save", "full_play", "repeat"}
NEGATIVE_EVENTS = {"skip", "dislike"}
WEIGHTS_FILE = "ranking_weights.json"
SLATE_FEEDBACK_FILE = "slate_feedback.jsonl"
FEATURE_FIELDS = {
    "semantic": "semantic_score",
    "acoustic": "acoustic_score",
    "personal": "personal_score",
}


def _first_present(item: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in item and item.get(key) is not None:
            return item.get(key)
    return None


def _feedback_dir() -> Path:
    root = os.getenv("MUSIC_FEEDBACK_DIR")
    path = Path(root) if root else Path("data") / "feedback"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _jsonl_path(name: str) -> Path:
    return _feedback_dir() / name


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    if side_effects_disabled():
        return
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n")


def _song_identity(song: dict[str, Any]) -> dict[str, str]:
    return {
        "title": str(song.get("title") or "").strip(),
        "artist": str(song.get("artist") or "").strip(),
    }


def _catalog_origin(song: dict[str, Any], item: dict[str, Any]) -> str | None:
    """Where this candidate came from — or None when we genuinely do not know.

    Routed through the shared source taxonomy so every online/web/acquired label
    is recognised, not just the literal ``web``. See services/source_taxonomy.py
    for why a local-only hit stays unknown rather than being guessed.
    """
    from services.source_taxonomy import catalog_origin_from_sources

    sources = list(song.get("recall_sources") or song.get("_recall_sources")
                   or item.get("_recall_sources") or [])
    for candidate in (song.get("source"), item.get("source"),
                      song.get("recall_source"), item.get("recall_source")):
        if candidate:
            sources.append(str(candidate))
    return catalog_origin_from_sources(sources)


def _feature_snapshot(item: dict[str, Any], rank: int) -> dict[str, Any]:
    song = item.get("song") if isinstance(item.get("song"), dict) else item
    identity = _song_identity(song)
    source_ranks = (
        item.get("_source_ranks")
        or song.get("_source_ranks")
        or {}
    )
    return {
        # Exposure bookkeeping for later debiasing. `propensity` stays absent on
        # purpose: the ranker is deterministic apart from the Thompson-sampled
        # exploration slots and does not yet report the probability it showed an
        # item with, so writing 1.0 would be a fabricated number.
        "catalog_origin": _catalog_origin(song, item),
        # `effective_exposure` is the time-DECAYED float the ranker penalised on;
        # it is NOT a count. Keeping the two apart stops downstream code from
        # int()-truncating a 2.5 decay score and calling it "shown 2 times".
        "effective_exposure": _first_present(item, "_post_effective_exposure"),
        "historical_exposure_count": _first_present(item, "_exposure_count", "exposure_count"),
        **identity,
        "rank": rank,
        "music_id": song.get("music_id") or song.get("id"),
        "source": song.get("source") or song.get("recall_source") or item.get("source") or item.get("recall_source"),
        "recall_sources": song.get("recall_sources") or song.get("_recall_sources") or item.get("_recall_sources") or [],
        "score": _first_present(item, "similarity_score", "_post_final_score"),
        "rrf_score": _first_present(item, "_rrf_score"),
        "source_ranks": source_ranks,
        "semantic_score": _first_present(item, "_semantic_score"),
        "acoustic_score": _first_present(item, "_acoustic_score"),
        "personal_score": _first_present(item, "_post_personal_score", "_personal_score"),
        "freshness_score": _first_present(item, "_post_freshness_score"),
        "longtail_score": _first_present(item, "_post_longtail_score"),
        "exposure_penalty": _first_present(item, "_post_exposure_penalty"),
        "post_recall_delta": _first_present(item, "_post_recall_delta"),
        "is_exploration": bool(item.get("_is_exploration")),
        "language": song.get("language"),
        "genres": song.get("genres") or song.get("genre"),
        "moods": song.get("moods"),
        "scenarios": song.get("scenarios"),
    }


def log_exposure(
    *,
    query: str,
    recommendations: list[dict[str, Any]],
    user_id: str = "local_admin",
    request_id: str | None = None,
    intent_type: str = "",
    retrieval_meta: dict[str, Any] | None = None,
    dialog_state: dict[str, Any] | None = None,
    timings: dict[str, Any] | None = None,
    context: dict[str, Any] | None = None,
    policy_version: str = "",
    provisional: bool = False,
) -> str:
    """Persist one recommendation slate for later offline replay.

    ``context`` carries the listening context (timezone / local hour / day type /
    session / scene). It cannot be backfilled later — you cannot reconstruct what
    time it was for the user — so it is written at exposure time even when empty.
    """
    exposure_id = request_id or str(uuid.uuid4())
    rows = [
        _feature_snapshot(song if isinstance(song, dict) else {}, rank=i + 1)
        for i, song in enumerate(recommendations or [])
    ]
    now_ms = int(time.time() * 1000)
    from services.runtime_context import provenance_fields

    payload = {
        "type": "exposure",
        "schema_version": "feedback_events_v2",
        "exposure_id": exposure_id,
        # `ts` is kept for backward compatibility but is now the time of THIS
        # write. Attribution must not use it: the provisional and final records
        # share an id and the final one is written later, so a user who rates the
        # moment the card appears has feedback.ts < final.ts and would be judged
        # "outside the attribution window". Use `shown_at_ms` instead — the
        # immutable instant the slate first reached the user, carried forward
        # from the provisional write. `completed_at_ms` is when the graph finished.
        "ts": now_ms,
        "shown_at_ms": now_ms,          # provisional write stamps it; final carries it forward
        "completed_at_ms": None if provisional else now_ms,
        "user_id": user_id,
        "query_hash": hashlib.sha256(str(query or "").encode("utf-8")).hexdigest(),
        "intent_type": intent_type,
        "policy_version": policy_version,
        # A provisional record is written BEFORE the songs are streamed, so
        # feedback always has something to attribute to; the final record is
        # written with the same exposure_id once the graph finishes and
        # supersedes it (lookup_exposure keeps the last write).
        "provisional": bool(provisional),
        "context": context or {},
        "count": len(rows),
        "items": rows,
        "retrieval_meta": retrieval_meta or {},
        "dialog_state": dialog_state or {},
        "timings": timings or {},
        **provenance_fields("ranking"),
    }
    if os.getenv("FEEDBACK_LOG_RAW_QUERY", "0").lower() in {"1", "true", "yes"}:
        payload["query"] = query
    if not provisional:
        _carry_forward_from_provisional(payload)
    _append_jsonl(_jsonl_path("exposures.jsonl"), payload)  # export snapshot
    _store("upsert_exposure", payload)  # canonical
    return exposure_id


# Fields the provisional write captures that the final write cannot reconstruct.
# The listening context is measured on the client at the moment the cards are
# shown; by the time the graph finishes, "what time was it for the user" is gone.
_CARRY_FORWARD_FIELDS = ("context", "policy_version")


def _carry_forward_from_provisional(payload: dict[str, Any]) -> None:
    """Let the final record REFINE the provisional one instead of replacing it.

    Provisional and final writes share an exposure_id and the last one wins, so a
    caller that simply forgets an argument silently destroys what the first write
    captured. That is exactly what happened to the listening context on the
    streaming path. Rather than trusting every call site to restate every field,
    carry the unreconstructable ones forward whenever the final write left them
    empty — the caller can still override by passing a value.

    `shown_at_ms` is handled separately: the final write always stamps its own
    now, but attribution must anchor on when the user first saw the slate, so we
    always pin it back to the provisional value (never the final's later clock).
    """
    needs_fill = any(not payload.get(field) for field in _CARRY_FORWARD_FIELDS)
    try:
        prior = lookup_exposure(str(payload.get("exposure_id") or ""))
    except Exception as exc:  # pragma: no cover - telemetry must not fail a request
        logger.debug("[feedback] carry-forward lookup skipped: %s", exc)
        return
    if not prior:
        return
    prior_shown = prior.get("shown_at_ms") or prior.get("ts")
    if prior_shown:
        payload["shown_at_ms"] = prior_shown   # earliest write wins, unconditionally
    if not needs_fill:
        return
    for field in _CARRY_FORWARD_FIELDS:
        if not payload.get(field) and prior.get(field):
            payload[field] = prior[field]


class FeedbackStoreError(RuntimeError):
    """The canonical SQLite write failed for a record that must not be lost."""


def _store(fn_name: str, payload: dict[str, Any], *, required: bool = False) -> None:
    """Write an event into the canonical SQLite store.

    SQLite is the system of record; JSONL is an audit/export copy. For USER-
    submitted feedback (song/slate) the store is authoritative: if the write
    fails we raise, so the API returns an error instead of telling the user
    "recorded" for something that was not. For passive telemetry (exposure,
    behaviour events) the write is best-effort — it must never fail a request,
    and the JSONL copy still retains it.
    """
    try:
        from services import feedback_store

        getattr(feedback_store, fn_name)(payload)
    except Exception as exc:
        if required:
            raise FeedbackStoreError(f"{fn_name} failed: {exc}") from exc
        logger.debug("[feedback] store write skipped (%s): %s", fn_name, exc)


def lookup_exposure(exposure_id: str) -> dict[str, Any] | None:
    """Return the persisted exposure record, or None if it does not exist.

    The exposure is the server's own record of what the policy showed. Feedback
    must be validated and enriched against it — never against numbers the
    browser reports about our own ranking.
    """
    exposure_id = str(exposure_id or "").strip()
    if not exposure_id:
        return None
    # SQLite is canonical: an upsert gives us last-write-wins for the
    # provisional/final pair without scanning the whole log.
    try:
        from services import feedback_store

        found = feedback_store.get_exposure(exposure_id)
        if found is not None:
            return found
    except Exception as exc:  # pragma: no cover - fall back to the JSONL scan
        logger.debug("[feedback] store lookup skipped: %s", exc)
    path = _jsonl_path("exposures.jsonl")
    if not path.exists():
        return None
    found = None
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or exposure_id not in line:
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            if row.get("exposure_id") == exposure_id:
                found = row  # keep the last write for this id
    return found


def find_exposure_item(exposure: dict[str, Any], *, music_id: str = "",
                       title: str = "", artist: str = "") -> dict[str, Any] | None:
    """Locate the exposed item this feedback is about (identity, not position)."""
    items = exposure.get("items") or []
    mid = str(music_id or "").strip()
    if mid:
        for item in items:
            if str(item.get("music_id") or "") == mid:
                return item
    t, a = str(title or "").strip().casefold(), str(artist or "").strip().casefold()
    if t:
        for item in items:
            if str(item.get("title") or "").strip().casefold() == t and (
                not a or str(item.get("artist") or "").strip().casefold() == a
            ):
                return item
    return None


def log_song_feedback(feedback: Any) -> str:
    """Persist per-song feedback on the two independent channels.

    ``feedback`` is a ``schemas.feedback_events.SongFeedback``. The taste channel
    (like/save/dislike/block) and the context channel (fits/partial/off + reason)
    are stored side by side and NEVER merged: a track can be a long-term
    favourite and still wrong for tonight.
    """
    payload = feedback.model_dump(mode="json") if hasattr(feedback, "model_dump") else dict(feedback)
    feedback_id = str(uuid.uuid4())
    from services.runtime_context import provenance_fields

    payload.update({
        "type": "song_feedback",
        "song_feedback_id": feedback_id,
        "ts": int(time.time() * 1000),
        **provenance_fields("ranking"),
    })
    _store("insert_song_feedback", payload, required=True)  # authoritative
    _append_jsonl(_jsonl_path("song_feedback.jsonl"), payload)  # audit copy
    return feedback_id


def log_user_event(
    *,
    event_type: str,
    song_title: str,
    artist: str,
    user_id: str = "local_admin",
    exposure_id: str | None = None,
    extra: Any = None,
) -> str:
    extra_payload = extra if isinstance(extra, dict) else {"value": extra} if extra is not None else {}
    event_id = str(uuid.uuid4())
    from services.runtime_context import provenance_fields

    payload = {
        "type": "event",
        "event_id": event_id,
        "ts": int(time.time() * 1000),
        "user_id": user_id,
        "event_type": event_type,
        "title": str(song_title or "").strip(),
        "artist": str(artist or "").strip(),
        "exposure_id": exposure_id,
        "extra": extra_payload,
        "position": extra_payload.get("position"),
        "play_duration_ms": extra_payload.get("play_duration_ms"),
        "progress_ratio": extra_payload.get("progress_ratio"),
        "session_id": extra_payload.get("session_id"),
        **provenance_fields("preference_and_ranking"),
    }
    _append_jsonl(_jsonl_path("events.jsonl"), payload)  # export snapshot
    _store("insert_user_event", payload)  # canonical
    return event_id


def log_slate_feedback(
    *,
    exposure_id: str,
    rating: str,
    reasons: list[str] | None = None,
    note: str = "",
    user_id: str = "local_admin",
    extra: dict[str, Any] | None = None,
) -> str:
    """Persist feedback for an entire recommendation slate.

    Song-level feedback tells us which item worked.  Slate-level feedback tells
    us whether the whole ranked list satisfied the current intent, which is the
    signal needed for offline replay and future ranking-policy learning.

    The stored record is a SINGLE canonical object: `overall`, `reasons`,
    `best_music_ids`, `worst_music_ids` and `context` live at the top level, not
    buried in `extra`, so training and diagnostics read one place. `rating` and
    `feedback_id` remain as legacy read-only aliases; nothing new should write or
    depend on them.
    """
    feedback_id = str(uuid.uuid4())
    extra_payload = dict(extra or {})
    # Lift the canonical fields the API computed (from the strict SlateFeedback)
    # out of `extra` and onto the record itself. Pop them so they are not stored
    # twice and cannot drift between the two copies.
    overall = extra_payload.pop("overall", None)
    best_music_ids = extra_payload.pop("best_music_ids", [])
    worst_music_ids = extra_payload.pop("worst_music_ids", [])
    context = extra_payload.pop("context", {})
    from services.runtime_context import provenance_fields

    payload = {
        "type": "slate_feedback",
        "schema_version": extra_payload.pop("schema_version", "feedback_events_v2"),
        # `slate_feedback_id` is the canonical key (it is the store's primary key
        # and matches song_feedback_id / event_id). `feedback_id` is kept as a
        # legacy alias because existing JSONL logs and replay code read it.
        "slate_feedback_id": feedback_id,
        "feedback_id": feedback_id,
        "ts": int(time.time() * 1000),
        "user_id": user_id,
        "exposure_id": str(exposure_id or "").strip(),
        # canonical judgement (fits/partial/off) — same scale as per-song context_fit
        "overall": overall,
        # legacy alias (great/partial/off); read-only, do not depend on it
        "rating": str(rating or "").strip(),
        "reasons": [str(item).strip() for item in (reasons or []) if str(item).strip()],
        "best_music_ids": list(best_music_ids or []),
        "worst_music_ids": list(worst_music_ids or []),
        "note": str(note or "").strip()[:1000],
        "context": context or {},
        "extra": extra_payload,   # only what is not canonical (e.g. reasons_raw, song_count)
        **provenance_fields("ranking"),
    }
    _store("insert_slate_feedback", payload, required=True)  # authoritative
    _append_jsonl(_jsonl_path(SLATE_FEEDBACK_FILE), payload)  # audit copy
    return feedback_id


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rows.append(json.loads(line))
    return rows


# The identity of each record type, most-specific key first. Slate feedback is
# the load-bearing case: records written before the primary-key fix carry only
# `feedback_id`, so keying on `slate_feedback_id` alone silently drops all of
# them (135 of 175 in the real log). The effective id is the first key present.
ID_KEYS: dict[str, tuple[str, ...]] = {
    "exposures": ("exposure_id",),
    "events": ("event_id",),
    "slate_feedback": ("slate_feedback_id", "feedback_id"),
    "song_feedback": ("song_feedback_id",),
}


def effective_id(row: dict[str, Any], id_keys: tuple[str, ...]) -> str:
    """First present id among id_keys — the record's stable identity."""
    for key in id_keys:
        value = row.get(key)
        if value:
            return str(value)
    return ""


def _dedupe_rows(rows: list[dict[str, Any]], id_keys: tuple[str, ...]) -> list[dict[str, Any]]:
    """Keep the last row per effective id (provisional/final collapse to one).

    Rows with no id at all are kept as-is rather than dropped — losing an
    un-keyed record is exactly the failure this function exists to prevent.
    """
    latest: dict[str, dict[str, Any]] = {}
    unkeyed: list[dict[str, Any]] = []
    for row in rows:
        k = effective_id(row, id_keys)
        if k:
            latest[k] = row
        else:
            unkeyed.append(row)
    return list(latest.values()) + unkeyed


def _merge_canonical(table: str, sqlite_rows: list[dict[str, Any]],
                     jsonl_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Union of SQLite and JSONL, deduped by effective id, SQLite winning.

    Not "SQLite if non-empty else JSONL": before the one-shot migration runs, the
    store holds only NEW events while the JSONL holds the whole history, so an
    either-or read makes years of feedback vanish the moment one new event lands.
    Merging keeps every record visible and lets SQLite override once migrated.
    """
    id_keys = ID_KEYS[table]
    merged: dict[str, dict[str, Any]] = {}
    extras: list[dict[str, Any]] = []
    # JSONL first (deduped), then SQLite overlays — SQLite wins on conflict.
    for row in _dedupe_rows(jsonl_rows, id_keys) + _dedupe_rows(sqlite_rows, id_keys):
        k = effective_id(row, id_keys)
        if k:
            merged[k] = row
        else:
            extras.append(row)
    # SQLite rows come last, so re-applying them guarantees precedence.
    for row in sqlite_rows:
        k = effective_id(row, id_keys)
        if k:
            merged[k] = row
    return list(merged.values()) + extras


def _load_store(fn_name: str) -> list[dict[str, Any]]:
    try:
        from services import feedback_store

        return getattr(feedback_store, fn_name)()
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("[feedback] store %s skipped: %s", fn_name, exc)
        return []


def load_exposures_canonical() -> list[dict[str, Any]]:
    """Exposures deduped by id, merged from SQLite + JSONL.

    Every offline consumer — replay, ranking, catalog diagnostics — must go
    through here so one exposure is counted once, regardless of how many times it
    was written or whether it has been migrated into the store yet.
    """
    from services.runtime_context import normalize_provenance

    return [
        normalize_provenance(row)
        for row in _merge_canonical(
            "exposures",
            _load_store("load_exposures"),
            load_jsonl(_jsonl_path("exposures.jsonl")),
        )
    ]


def load_events_canonical() -> list[dict[str, Any]]:
    from services.runtime_context import normalize_provenance

    return [
        normalize_provenance(row)
        for row in _merge_canonical(
            "events",
            _load_store("load_events"),
            load_jsonl(_jsonl_path("events.jsonl")),
        )
    ]


def load_slate_feedback_canonical() -> list[dict[str, Any]]:
    from services.runtime_context import normalize_provenance

    return [
        normalize_provenance(row)
        for row in _merge_canonical(
            "slate_feedback",
            _load_store("load_slate_feedback"),
            load_jsonl(_jsonl_path(SLATE_FEEDBACK_FILE)),
        )
    ]


def load_song_feedback_canonical() -> list[dict[str, Any]]:
    """Per-song feedback, merged the same way as every other canonical read.

    Added for the history view: this table had a write path and no read path, so
    a rating you gave was invisible the moment the panel closed.
    """
    from services.runtime_context import normalize_provenance

    return [
        normalize_provenance(row)
        for row in _merge_canonical(
            "song_feedback",
            _load_store("load_song_feedback"),
            load_jsonl(_jsonl_path("song_feedback.jsonl")),
        )
    ]


def load_song_feedback_canonical_strict() -> list[dict[str, Any]]:
    """Merge SQLite and legacy JSONL without hiding a read failure.

    Normal read paths may tolerate one unavailable source. Destructive catalog
    cleanup may not: treating an unreadable ledger as an empty ledger can delete
    the Song node that a stored rating still references.
    """
    from services import feedback_store
    from services.runtime_context import normalize_provenance

    sqlite_rows = feedback_store.load_song_feedback()
    jsonl_rows = load_jsonl(_jsonl_path("song_feedback.jsonl"))
    return [
        normalize_provenance(row)
        for row in _merge_canonical("song_feedback", sqlite_rows, jsonl_rows)
    ]


def _eligible(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    from services.runtime_context import is_training_eligible

    return [row for row in rows if is_training_eligible(row)]


def load_training_exposures() -> list[dict[str, Any]]:
    return _eligible(load_exposures_canonical())


def load_training_events() -> list[dict[str, Any]]:
    return _eligible(load_events_canonical())


def load_training_slate_feedback() -> list[dict[str, Any]]:
    return _eligible(load_slate_feedback_canonical())


def load_training_song_feedback() -> list[dict[str, Any]]:
    return _eligible(load_song_feedback_canonical())


def learned_weights_path() -> Path:
    return _feedback_dir() / WEIGHTS_FILE


def load_learned_tri_anchor_weights() -> dict[str, float] | None:
    path = learned_weights_path()
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        weights = payload.get("weights", payload)
        values = {
            "semantic": float(weights.get("semantic", 0.0)),
            "acoustic": float(weights.get("acoustic", 0.0)),
            "personal": float(weights.get("personal", 0.0)),
        }
        total = sum(max(0.0, v) for v in values.values())
        if total <= 0:
            return None
        return {key: max(0.0, value) / total for key, value in values.items()}
    except Exception:
        return None


def _event_label(event_type: str) -> int | None:
    if event_type in POSITIVE_EVENTS:
        return 1
    if event_type in NEGATIVE_EVENTS:
        return 0
    return None


def _safe_feature(value: Any) -> float:
    if value is None:
        return 0.5
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return 0.5


def build_feedback_training_rows(
    exposures: list[dict[str, Any]],
    events: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Join user feedback events to exposed items and return audit-friendly rows."""
    by_exposure: dict[str, list[dict[str, Any]]] = {
        str(row.get("exposure_id")): list(row.get("items") or [])
        for row in exposures
        if row.get("exposure_id")
    }
    all_items: list[dict[str, Any]] = [item for row in exposures for item in (row.get("items") or [])]
    rows: list[dict[str, Any]] = []

    for event in events:
        label = _event_label(str(event.get("event_type") or ""))
        if label is None:
            continue
        title = str(event.get("title") or "").casefold()
        artist = str(event.get("artist") or "").casefold()
        items = by_exposure.get(str(event.get("exposure_id"))) or all_items
        match = next(
            (
                item
                for item in items
                if str(item.get("title") or "").casefold() == title
                and str(item.get("artist") or "").casefold() == artist
            ),
            None,
        )
        if not match:
            continue
        features = {
            key: _safe_feature(match.get(field))
            for key, field in FEATURE_FIELDS.items()
        }
        rows.append({
            "label": label,
            "event_type": event.get("event_type"),
            "exposure_id": event.get("exposure_id"),
            "title": match.get("title"),
            "artist": match.get("artist"),
            "rank": match.get("rank"),
            "features": features,
            "feature_present": {
                key: match.get(field) is not None
                for key, field in FEATURE_FIELDS.items()
            },
        })
    return rows


def _sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def _log_loss(rows: list[dict[str, Any]], coefficients: dict[str, float], bias: float) -> float:
    if not rows:
        return 0.0
    total = 0.0
    for row in rows:
        score = bias + sum(coefficients[key] * row["features"][key] for key in FEATURE_FIELDS)
        pred = max(1e-6, min(1.0 - 1e-6, _sigmoid(score)))
        label = float(row["label"])
        total += -(label * math.log(pred) + (1.0 - label) * math.log(1.0 - pred))
    return total / len(rows)


def _coefficients_to_weights(coefficients: dict[str, float]) -> dict[str, float]:
    # Softmax keeps every anchor non-negative and makes the learned file safe to load.
    largest = max(coefficients.values()) if coefficients else 0.0
    exp_values = {
        key: math.exp(float(value) - largest)
        for key, value in coefficients.items()
    }
    total = sum(exp_values.values()) or 1.0
    return {key: round(value / total, 4) for key, value in exp_values.items()}


def learn_tri_anchor_weights(
    exposures: list[dict[str, Any]],
    events: list[dict[str, Any]],
    *,
    min_events: int = 8,
    learning_rate: float = 0.15,
    epochs: int = 240,
    l2: float = 0.02,
) -> dict[str, Any]:
    """Learn tri-anchor weights from explicit feedback with an audit trail."""
    rows = build_feedback_training_rows(exposures, events)
    positives = sum(1 for row in rows if row["label"] == 1)
    negatives = sum(1 for row in rows if row["label"] == 0)
    feature_coverage = {
        key: round(
            sum(1 for row in rows if row["feature_present"][key]) / len(rows),
            4,
        ) if rows else 0.0
        for key in FEATURE_FIELDS
    }

    audit: dict[str, Any] = {
        "method": "logistic_tri_anchor_v1",
        "status": "ok",
        "matched_events": len(rows),
        "positive_events": positives,
        "negative_events": negatives,
        "feature_coverage": feature_coverage,
        "min_events": min_events,
    }
    if len(rows) < min_events or positives == 0 or negatives == 0:
        audit.update({
            "status": "insufficient_data",
            "reason": "needs at least min_events with both positive and negative labels",
        })
        return audit

    coefficients = {key: 0.0 for key in FEATURE_FIELDS}
    bias = 0.0
    baseline_loss = _log_loss(rows, coefficients, bias)
    for _ in range(max(1, int(epochs))):
        grad = {key: 0.0 for key in FEATURE_FIELDS}
        grad_bias = 0.0
        for row in rows:
            score = bias + sum(coefficients[key] * row["features"][key] for key in FEATURE_FIELDS)
            error = _sigmoid(score) - float(row["label"])
            grad_bias += error
            for key in FEATURE_FIELDS:
                grad[key] += error * row["features"][key] + l2 * coefficients[key]
        scale = 1.0 / len(rows)
        bias -= learning_rate * grad_bias * scale
        for key in FEATURE_FIELDS:
            coefficients[key] -= learning_rate * grad[key] * scale

    learned_loss = _log_loss(rows, coefficients, bias)
    audit.update({
        "weights": _coefficients_to_weights(coefficients),
        "coefficients": {key: round(value, 6) for key, value in coefficients.items()},
        "bias": round(bias, 6),
        "baseline_log_loss": round(baseline_loss, 6),
        "learned_log_loss": round(learned_loss, 6),
        "loss_delta": round(baseline_loss - learned_loss, 6),
    })
    return audit


def estimate_tri_anchor_weights(
    exposures: list[dict[str, Any]],
    events: list[dict[str, Any]],
) -> dict[str, Any]:
    """Estimate simple tri-anchor weights from logged events.

    This is intentionally transparent rather than clever: it joins events to
    exposed items by exposure_id when available, otherwise by title+artist, then
    rewards feature dimensions that were high on positive events and low on
    negative events.
    """
    by_exposure: dict[str, list[dict[str, Any]]] = {
        str(row.get("exposure_id")): list(row.get("items") or [])
        for row in exposures
        if row.get("exposure_id")
    }
    all_items: list[dict[str, Any]] = [item for row in exposures for item in (row.get("items") or [])]

    accum = {"semantic": 1.0, "acoustic": 1.0, "personal": 1.0}
    matched = 0
    positives = 0
    negatives = 0

    for event in events:
        title = str(event.get("title") or "").casefold()
        artist = str(event.get("artist") or "").casefold()
        items = by_exposure.get(str(event.get("exposure_id"))) or all_items
        match = next(
            (
                item
                for item in items
                if str(item.get("title") or "").casefold() == title
                and str(item.get("artist") or "").casefold() == artist
            ),
            None,
        )
        if not match:
            continue
        matched += 1
        sign = 1.0 if event.get("event_type") in POSITIVE_EVENTS else -0.6 if event.get("event_type") in NEGATIVE_EVENTS else 0.0
        if sign > 0:
            positives += 1
        elif sign < 0:
            negatives += 1
        for key, field in (
            ("semantic", "semantic_score"),
            ("acoustic", "acoustic_score"),
            ("personal", "personal_score"),
        ):
            value = match.get(field)
            if value is None:
                continue
            score = max(0.0, min(1.0, float(value)))
            accum[key] += sign * (score - 0.5)

    cleaned = {key: max(0.05, value) for key, value in accum.items()}
    total = sum(cleaned.values())
    weights = {key: round(value / total, 4) for key, value in cleaned.items()}
    return {
        "weights": weights,
        "matched_events": matched,
        "positive_events": positives,
        "negative_events": negatives,
        "method": "transparent_event_correlation_v1",
    }
