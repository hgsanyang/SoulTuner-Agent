"""One place that says where a track came from.

Recall paths, the acquisition flywheel and the UI have each grown their own
source strings over time — ``web``, ``online``, ``online_search``,
``online_acquired``, ``online_auto_flywheel_candidates``, ``local``, ``graph``,
``dense`` … The feedback ledger was reading only the literal ``web``, so every
other online label was silently filed as "origin unknown", which quietly biases
any cohort split built on it.

This module is the single normalization point. Adopt it incrementally: the
feedback ledger uses it now; recall and the UI should route their labels through
``normalize_source`` too so the vocabulary can never fork again.
"""

from __future__ import annotations

from typing import Iterable, Optional

# The normalized vocabulary. Everything else is an alias for one of these.
LOCAL_CATALOG = "local_catalog"          # already in the user's library
WEB_CANDIDATE = "web_candidate"          # surfaced from the web, not yet ingested
TEMPORARY_INGEST = "temporary_ingest"    # downloaded to the staging area
RETAINED_AUDIO = "retained_audio"        # kept from a positive-feedback save

# Substring → normalized. Checked in order; first hit wins. Substring (not exact)
# so future ``online_*`` variants are covered without another edit here.
_RULES: list[tuple[str, str]] = [
    ("retention", RETAINED_AUDIO),
    ("retain", RETAINED_AUDIO),
    ("pending", TEMPORARY_INGEST),
    ("staging", TEMPORARY_INGEST),
    ("temporary", TEMPORARY_INGEST),
    ("acquire", TEMPORARY_INGEST),
    ("acquired", TEMPORARY_INGEST),
    ("web", WEB_CANDIDATE),
    ("online", WEB_CANDIDATE),
    ("search", WEB_CANDIDATE),
    ("local", LOCAL_CATALOG),
    ("graph", LOCAL_CATALOG),
    ("dense", LOCAL_CATALOG),
    ("vector", LOCAL_CATALOG),
    ("neo4j", LOCAL_CATALOG),
]

# Which normalized origins mean "this was not already in the local library".
_NON_LOCAL = {WEB_CANDIDATE, TEMPORARY_INGEST, RETAINED_AUDIO}


def normalize_source(raw: Optional[str]) -> Optional[str]:
    """Map one raw source string to the normalized vocabulary, or None."""
    text = str(raw or "").strip().casefold()
    if not text:
        return None
    for needle, normalized in _RULES:
        if needle in text:
            return normalized
    return None


def catalog_origin_from_sources(sources: Iterable[str]) -> Optional[str]:
    """Decide the feedback-ledger ``catalog_origin`` from a set of raw sources.

    Only the "came from the web / not previously local" case is unambiguous
    without the user's favourites set, so that is all we assert. Distinguishing a
    fresh local hit from the user's own prior favourite needs data we do not have
    here, and guessing would manufacture the very bias this field exists to
    expose — so a local-only hit stays None (unknown), never a fabricated label.
    """
    normalized = {normalize_source(s) for s in sources}
    normalized.discard(None)
    if normalized & _NON_LOCAL:
        return "online_new"
    return None
