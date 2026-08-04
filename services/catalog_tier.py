"""Separate "my library" from "the cache the recommender filled behind my back".

Every online candidate that merely *appears* in a recommendation is downloaded
and written to Neo4j as a permanent ``:Song`` node by the acquisition flywheel.
The 24h cleanup then deletes only the MP3 — the node stays forever. So a song the
user never asked for, never played to the end, never saved, still shows up in
我的曲库 with a 25% quality score and five missing fields.

That is a shelf/cache confusion, not a bug in any one function. This module is
the vocabulary that separates the two:

``library``    the user's own catalogue — imported locally, or explicitly saved.
``candidate``  a temporary online fetch kept only so the track is playable.

Reads must stay correct on databases written before ``catalog_tier`` existed,
so :func:`candidate_predicate` also infers the tier from ``source`` +
``audio_retention`` when the property is absent. Backfilling is an optimisation,
never a precondition.
"""

from __future__ import annotations

from typing import Any, Iterable

LIBRARY = "library"
CANDIDATE = "candidate"
TIERS = (LIBRARY, CANDIDATE)

# Raw ``s.source`` values the acquisition path writes for an online fetch.
_ONLINE_SOURCES = ("online", "online_search", "web", "online_acquired")


def tier_for_ingest(song: dict[str, Any]) -> str:
    """Decide the tier for a song about to be written by the acquirer.

    Only an explicit save earns a place in the library. ``requested_by`` carries
    the intent when retention has not been set yet: ``user_like``/``user_save``
    and ``explicit_acquire`` are the user asking for the track,
    ``auto_recommendation`` is the recommender helping itself.
    """
    retention = str(song.get("audio_retention") or "").strip().lower()
    if retention == "saved":
        return LIBRARY
    requested_by = str(song.get("requested_by") or "").strip().lower()
    if requested_by.startswith("user_") or requested_by.startswith("explicit"):
        return LIBRARY
    return CANDIDATE


def resolve_ingest_tier(
    song: dict[str, Any],
    *,
    existing_tier: str | None = None,
    existing_source: str | None = None,
    node_exists: bool = False,
) -> str:
    """Tier to write, given what is already on the node. Never demotes.

    The acquirer overwrites ``s.source = 'online'`` whenever a track is fetched,
    so a locally-seeded song that also happens to exist on a streaming platform
    would otherwise be relabelled a cache entry the first time it is recommended
    — the user would watch their own library drain into the candidate shelf.
    """
    desired = tier_for_ingest(song)
    if desired == LIBRARY or not node_exists:
        return desired
    if str(existing_tier or "").strip().lower() == LIBRARY:
        return LIBRARY
    if not str(existing_tier or "").strip():
        # Predates this property. Only an online-sourced node can be a cache
        # entry; anything else came from the seeded local catalogue.
        if str(existing_source or "").strip().lower() not in _ONLINE_SOURCES:
            return LIBRARY
    return CANDIDATE


def candidate_predicate(var: str = "s") -> str:
    """Cypher boolean that is true for temporary online candidates.

    Legacy-safe: nodes written before this property existed carry no
    ``catalog_tier``, and the 1800+ locally-seeded songs carry no ``source``
    either. Treating "unknown" as library is the conservative direction — it can
    only ever leave a cache entry visible, never hide something the user owns.
    """
    online = ", ".join(f"'{value}'" for value in _ONLINE_SOURCES)
    return (
        f"(coalesce({var}.catalog_tier, '') = '{CANDIDATE}' "
        f"OR (coalesce({var}.catalog_tier, '') = '' "
        f"AND toLower(coalesce({var}.source, '')) IN [{online}] "
        f"AND coalesce({var}.audio_retention, 'temporary') <> 'saved'))"
    )


def tier_filter_clause(tier: str, var: str = "s") -> str:
    """WHERE fragment for ``tier``; empty string means "no filter"."""
    normalized = str(tier or LIBRARY).strip().lower()
    if normalized == "all":
        return ""
    if normalized == CANDIDATE:
        return candidate_predicate(var)
    return f"NOT {candidate_predicate(var)}"


# Being shown a track is not an interaction with it — EXPOSED is written for
# every recommended song, so counting it would make every candidate look
# "touched" and nothing would ever be purgeable.
USER_INTERACTION_RELS = ("LIKES", "SAVES", "DISLIKES", "SKIPPED", "LISTENED_TO")


def purgeable_predicate(var: str = "s") -> str:
    """Cypher boolean for candidates that are safe to delete.

    Three conditions, all required: it is a cache entry, its audio has already
    expired (so nothing is playing it), and the user never acted on it. A skip or
    a dislike counts as acting on it — those rows are what the negative-feedback
    ranker learns from, so deleting them would erase the evidence.

    This covers only what the *graph* knows. Per-song ratings live in SQLite and
    write no relationship at all, so :func:`rated_identities` has to filter the
    result of this predicate before anything is deleted — see
    :func:`unprotected`. The graph half alone is not a safe gate.
    """
    rels = ", ".join(f"'{rel}'" for rel in USER_INTERACTION_RELS)
    return (
        f"{candidate_predicate(var)} "
        f"AND coalesce({var}.audio_status, '') = 'released' "
        f"AND NOT EXISTS {{ MATCH (:User)-[r]->({var}) WHERE type(r) IN [{rels}] }}"
    )


def _rating_is_empty(row: dict[str, Any]) -> bool:
    """Mirror of ``SongFeedback.is_empty`` for a raw stored row.

    Any of the four channels counts. Protecting only ``context_fit='off'`` would
    drop a track the user took the trouble to write a note about.
    """
    if row.get("taste") or row.get("context_fit"):
        return False
    if row.get("off_reasons"):
        return False
    return not str(row.get("note") or "").strip()


def rated_identities(feedback_rows: Iterable[dict[str, Any]]) -> tuple[set[str], set[str]]:
    """``(music_ids, song_keys)`` for songs carrying any per-song rating.

    Two identities because the graph and the ledger disagree about which one
    exists: web-lane candidates are frequently ingested without a music_id, so an
    id-only check would leave exactly the rated-but-unidentified tracks
    unprotected — the subset most likely to be a cache entry in the first place.
    """
    from services.negative_feedback import song_key

    ids: set[str] = set()
    keys: set[str] = set()
    for row in feedback_rows or []:
        if not isinstance(row, dict) or _rating_is_empty(row):
            continue
        music_id = str(row.get("music_id") or "").strip()
        if music_id:
            ids.add(music_id)
        key = song_key(row.get("title"), row.get("artist"))
        if key.strip("|"):
            keys.add(key)
    return ids, keys


class RatingLedgerUnavailable(RuntimeError):
    """The per-song rating ledger could not be read, so nothing may be deleted."""


def load_rated_identities(
    feedback_rows: Iterable[dict[str, Any]] | None = None,
) -> tuple[set[str], set[str]]:
    """Read the rating ledger and return the identities that must not be deleted.

    Raises rather than returning empty sets on failure. An unreadable ledger and
    an empty ledger produce the same value, and only one of those two readings
    is recoverable — treating "I couldn't check" as "nothing to protect" is the
    fail-open shape that turns a missing file into silent data loss.
    """
    if feedback_rows is None:
        try:
            from services.feedback_store import load_song_feedback

            feedback_rows = load_song_feedback()
        except Exception as exc:
            raise RatingLedgerUnavailable(str(exc)) from exc
    return rated_identities(feedback_rows)


def unprotected(
    rows: Iterable[dict[str, Any]],
    protected_ids: set[str],
    protected_keys: set[str],
) -> list[dict[str, Any]]:
    """Drop rows whose song carries a per-song rating.

    Done in Python on purpose: the fallback identity is a normalised
    title+artist, and reimplementing that normalisation in Cypher would give two
    definitions of "the same song" that drift apart silently.
    """
    from services.negative_feedback import song_key

    kept: list[dict[str, Any]] = []
    for row in rows or []:
        music_id = str(row.get("music_id") or "").strip()
        if music_id and music_id in protected_ids:
            continue
        if song_key(row.get("title"), row.get("artist")) in protected_keys:
            continue
        kept.append(row)
    return kept
