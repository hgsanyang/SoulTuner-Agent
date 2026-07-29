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

from typing import Any

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
    """
    rels = ", ".join(f"'{rel}'" for rel in USER_INTERACTION_RELS)
    return (
        f"{candidate_predicate(var)} "
        f"AND coalesce({var}.audio_status, '') = 'released' "
        f"AND NOT EXISTS {{ MATCH (:User)-[r]->({var}) WHERE type(r) IN [{rels}] }}"
    )
