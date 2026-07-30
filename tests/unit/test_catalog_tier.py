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
    RatingLedgerUnavailable,
    candidate_predicate,
    load_rated_identities,
    purgeable_predicate,
    rated_identities,
    resolve_ingest_tier,
    tier_filter_clause,
    tier_for_ingest,
    unprotected,
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


# ---- cross-store protection: ratings live in SQLite and write no relation ----
#
# The graph-only gate above was not enough. /api/song-feedback stores per-song
# ratings in SQLite and creates no Neo4j relationship at all, so a track the user
# explicitly rated looked untouched to the purge — and deleting it takes the
# vector the negative-feedback anchor needs, leaving the rating in place but
# pointing at nothing.

def test_a_song_rated_only_in_sqlite_is_protected():
    rows = [{"music_id": "m-1", "title": "Summer", "artist": "Calvin Harris",
             "context_fit": "off"}]
    ids, keys = rated_identities(rows)
    candidates = [{"eid": "1", "music_id": "m-1", "title": "Summer", "artist": "Calvin Harris"}]
    assert unprotected(candidates, ids, keys) == []


def test_a_rated_song_without_music_id_is_still_protected():
    """Web-lane candidates are routinely ingested with no music_id — an id-only
    check would leave exactly the rated-but-unidentified tracks unprotected."""
    rows = [{"music_id": "", "title": "Summer", "artist": "Calvin Harris", "note": "太吵"}]
    ids, keys = rated_identities(rows)
    assert ids == set()
    candidates = [{"eid": "1", "music_id": "", "title": "Summer", "artist": "Calvin Harris"}]
    assert unprotected(candidates, ids, keys) == []


@pytest.mark.parametrize("rating", [
    {"context_fit": "off"},
    {"context_fit": "fits"},
    {"taste": "love"},
    {"off_reasons": ["too_fast"]},
    {"note": "  下次别再放了  "},
])
def test_any_non_empty_rating_protects_not_just_a_rejection(rating):
    """Protecting only context_fit='off' would drop a track the user took the
    trouble to write a note about."""
    rows = [{"music_id": "m-1", "title": "A", "artist": "B", **rating}]
    ids, keys = rated_identities(rows)
    assert ids == {"m-1"}


@pytest.mark.parametrize("rating", [
    {}, {"note": "   "}, {"off_reasons": []}, {"context_fit": None, "taste": None},
])
def test_an_empty_rating_row_protects_nothing(rating):
    rows = [{"music_id": "m-1", "title": "A", "artist": "B", **rating}]
    assert rated_identities(rows) == (set(), set())


def test_an_unrated_candidate_is_still_deletable():
    """The protection must not swallow the whole feature."""
    rows = [{"music_id": "m-rated", "title": "A", "artist": "B", "context_fit": "off"}]
    ids, keys = rated_identities(rows)
    candidates = [
        {"eid": "1", "music_id": "m-rated", "title": "A", "artist": "B"},
        {"eid": "2", "music_id": "m-other", "title": "C", "artist": "D"},
    ]
    assert [row["eid"] for row in unprotected(candidates, ids, keys)] == ["2"]


def test_an_unreadable_rating_ledger_refuses_rather_than_deletes(monkeypatch):
    """"I couldn't check" must not collapse into "nothing to protect" — that is
    the fail-open shape that turns a missing file into silent data loss."""
    import services.feedback_store as store

    def boom():
        raise OSError("database is locked")

    monkeypatch.setattr(store, "load_song_feedback", boom)
    with pytest.raises(RatingLedgerUnavailable):
        load_rated_identities()


def test_an_unmigrated_jsonl_rating_is_also_protected(tmp_path, monkeypatch):
    """Cleanup must protect feedback written before the SQLite migration."""
    import json

    import services.feedback_store as store

    monkeypatch.setenv("MUSIC_FEEDBACK_DIR", str(tmp_path))
    monkeypatch.setattr(store, "load_song_feedback", lambda: [])
    row = {
        "song_feedback_id": "legacy-1",
        "music_id": "legacy-song",
        "title": "Old Song",
        "artist": "Old Artist",
        "context_fit": "off",
    }
    (tmp_path / "song_feedback.jsonl").write_text(
        json.dumps(row, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    ids, keys = load_rated_identities()

    assert ids == {"legacy-song"}
    assert keys


def test_an_unreadable_legacy_jsonl_ledger_also_refuses_cleanup(tmp_path, monkeypatch):
    import services.feedback_store as store

    monkeypatch.setenv("MUSIC_FEEDBACK_DIR", str(tmp_path))
    monkeypatch.setattr(store, "load_song_feedback", lambda: [])
    (tmp_path / "song_feedback.jsonl").write_text("{broken-json\n", encoding="utf-8")

    with pytest.raises(RatingLedgerUnavailable):
        load_rated_identities()


def test_a_readable_empty_ledger_protects_nothing():
    """The fail-closed path must not fire on a legitimately empty ledger."""
    assert load_rated_identities([]) == (set(), set())


def test_the_purge_endpoint_consults_the_rating_ledger():
    """A pure-function fix nobody calls is the same bug in a new place."""
    from pathlib import Path

    src = Path(__file__).resolve().parents[2] / "api" / "server.py"
    code = src.read_text(encoding="utf-8")
    start = code.index("async def purge_catalog_candidates")
    body = code[start : start + 4000]
    assert "load_rated_identities()" in body
    assert "unprotected(" in body
    assert "RatingLedgerUnavailable" in body
    # protection must apply to the dry run too, not only the delete
    assert body.index("unprotected(") < body.index("if dry_run:")
