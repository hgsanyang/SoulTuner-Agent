"""A song the user never asked for must not count as part of their library.

The acquisition flywheel downloads and ingests every online candidate that merely
*appears* in a recommendation, and the 24h cleanup deletes only the MP3 — so 54
tracks nobody requested were sitting in 我的曲库 with 25% quality scores. These
tests pin the shelf/cache boundary and, more importantly, the direction it is
allowed to move in: never demote something the user owns.
"""

from __future__ import annotations

import pytest

from services.catalog_tier import (
    CANDIDATE,
    LIBRARY,
    candidate_predicate,
    purgeable_predicate,
    resolve_ingest_tier,
    tier_filter_clause,
    tier_for_ingest,
)


def test_auto_recommendation_fetch_is_only_a_candidate():
    song = {"audio_retention": "temporary", "requested_by": "auto_recommendation"}
    assert tier_for_ingest(song) == CANDIDATE


@pytest.mark.parametrize("requested_by", ["user_like", "user_save", "explicit_acquire"])
def test_the_user_asking_for_it_earns_a_library_slot(requested_by):
    assert tier_for_ingest({"requested_by": requested_by}) == LIBRARY


def test_saved_retention_is_a_library_slot_regardless_of_who_asked():
    song = {"audio_retention": "saved", "requested_by": "auto_recommendation"}
    assert tier_for_ingest(song) == LIBRARY


def test_a_local_song_found_online_is_not_demoted():
    """The acquirer overwrites s.source = 'online' on every fetch. Without this
    the user's own catalogue would drain into the cache shelf one recommendation
    at a time — the failure would look like the library slowly shrinking."""
    fetched = {"audio_retention": "temporary", "requested_by": "auto_recommendation"}
    tier = resolve_ingest_tier(fetched, existing_tier="", existing_source="", node_exists=True)
    assert tier == LIBRARY


def test_an_explicitly_saved_node_is_never_demoted():
    fetched = {"audio_retention": "temporary", "requested_by": "auto_recommendation"}
    tier = resolve_ingest_tier(fetched, existing_tier=LIBRARY, existing_source="online", node_exists=True)
    assert tier == LIBRARY


def test_refetching_a_candidate_keeps_it_a_candidate():
    fetched = {"audio_retention": "temporary", "requested_by": "auto_recommendation"}
    tier = resolve_ingest_tier(fetched, existing_tier=CANDIDATE, existing_source="online", node_exists=True)
    assert tier == CANDIDATE


def test_a_brand_new_auto_fetch_is_a_candidate():
    fetched = {"audio_retention": "temporary", "requested_by": "auto_recommendation"}
    assert resolve_ingest_tier(fetched, node_exists=False) == CANDIDATE


def test_a_user_save_promotes_an_existing_candidate():
    saved = {"audio_retention": "saved", "requested_by": "user_save"}
    tier = resolve_ingest_tier(saved, existing_tier=CANDIDATE, existing_source="online", node_exists=True)
    assert tier == LIBRARY


# ---- the read path has to survive databases written before this property ----

def test_predicate_reads_legacy_rows_without_the_property():
    """Backfilling is an optimisation, not a precondition: a fresh clone of an
    older volume must still separate the two shelves correctly."""
    predicate = candidate_predicate()
    assert "catalog_tier" in predicate
    assert "audio_retention" in predicate and "source" in predicate


def test_unknown_source_is_treated_as_library_not_cache():
    """1872 locally-seeded songs carry no source at all. Guessing "cache" there
    would hide the user's entire library — the conservative direction is the
    only safe one, and it is the one encoded in the predicate."""
    predicate = candidate_predicate()
    # The online-source test is an AND-condition of the inferred branch, so a row
    # with no source can never satisfy it.
    assert "toLower(coalesce(s.source, '')) IN [" in predicate


def test_filter_clauses_are_complementary():
    assert tier_filter_clause("library") == f"NOT {candidate_predicate()}"
    assert tier_filter_clause("candidate") == candidate_predicate()
    assert tier_filter_clause("all") == ""


def test_unknown_tier_falls_back_to_library():
    assert tier_filter_clause("nonsense") == f"NOT {candidate_predicate()}"
    assert tier_filter_clause("") == f"NOT {candidate_predicate()}"


# ---- deletion ----

def test_being_recommended_does_not_protect_a_candidate_from_purge():
    """EXPOSED is written for every recommended song. Counting it as an
    interaction would make every candidate look touched and nothing would ever
    be purgeable — the button would silently always report zero."""
    assert "EXPOSED" not in purgeable_predicate()


@pytest.mark.parametrize("rel", ["LIKES", "SAVES", "DISLIKES", "SKIPPED", "LISTENED_TO"])
def test_any_real_interaction_protects_a_candidate(rel):
    """A skip or a dislike is evidence the negative-feedback ranker learns from;
    deleting those nodes would erase it."""
    assert f"'{rel}'" in purgeable_predicate()


def test_purge_only_touches_expired_audio():
    """Still-cached candidates may be playing right now."""
    assert "'released'" in purgeable_predicate()


def test_purge_never_reaches_the_library_shelf():
    predicate = purgeable_predicate()
    assert predicate.startswith(candidate_predicate())
