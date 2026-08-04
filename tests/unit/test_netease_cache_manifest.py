"""The manifest must never claim two recordings are the same on weak evidence.

Title+artist agreeing is not identity: a live take, a remaster and a cover all
share those two fields. The handoff is explicit that such conflicts must not be
auto-merged, so the strongest verdict this classifier may reach from them is
"suspected", which a human resolves.

Also pinned: the inventory scanner reads file names and sizes; audio validation
belongs to the separate staging command.
"""

from __future__ import annotations

import pytest

from scripts.import_netease_cache_manifest import (
    DUPLICATE_EXACT,
    DUPLICATE_SUSPECTED,
    METADATA_MISSING,
    METADATA_READY,
    CacheEntry,
    classify,
    fetch_lyrics,
    load_library_index,
    scan_cache,
    summarise,
)


def _entry(song_id="1054603", quality="320", size=10 * 1024 * 1024):
    return CacheEntry(song_id=song_id, quality=quality, bytes=size, path=f"/c/{song_id}.uc")


def _meta(title="海阔天空", artist="Beyond", duration=326000):
    return {"title": title, "artist": artist, "album": "乐与怒", "duration_ms": duration}


# ---- scanning: names and sizes only -----------------------------------------

def test_scan_parses_id_quality_and_size(tmp_path):
    (tmp_path / "1054603-320-1964b09d14bbce4ed9956b680068bad3.uc").write_bytes(b"x" * 2048)
    (tmp_path / "1090226-999-d685396808f76b98bfc32e78f71909b6.uc").write_bytes(b"y" * 4096)
    entries = {e.song_id: e for e in scan_cache(tmp_path)}
    assert entries["1054603"].quality == "320"
    assert entries["1090226"].quality == "999"
    assert entries["1054603"].bytes == 2048


def test_scan_ignores_files_that_are_not_cache_entries(tmp_path):
    (tmp_path / "notes.txt").write_text("x", encoding="utf-8")
    (tmp_path / "1054603-320-abc.mp3").write_bytes(b"x")
    (tmp_path / "garbage.uc").write_bytes(b"x")
    assert scan_cache(tmp_path) == []


def test_scan_accepts_the_partial_download_suffix(tmp_path):
    (tmp_path / "1054603-320-1964b09d14bbce4ed9956b680068bad3.uc!").write_bytes(b"x")
    assert len(scan_cache(tmp_path)) == 1


def test_a_missing_cache_directory_is_not_an_error(tmp_path):
    assert scan_cache(tmp_path / "nope") == []


# ---- duplicate levels -------------------------------------------------------

def test_a_matching_song_id_is_an_exact_duplicate():
    entries = classify(
        [_entry("1054603")],
        {"1054603": _meta()},
        {"1054603": {"music_id": "1054603", "title": "海阔天空", "duration": 326000}},
        {},
    )
    assert entries[0].state == DUPLICATE_EXACT
    assert entries[0].matched_music_id == "1054603"


def test_same_title_and_artist_with_matching_duration_is_only_suspected():
    """Not exact. A remaster shares the title, artist and very nearly the
    duration, and merging it would silently discard one of the two."""
    from services.negative_feedback import song_key

    entries = classify(
        [_entry("999")],
        {"999": _meta(duration=326000)},
        {},
        {song_key("海阔天空", "Beyond"): [{"music_id": "m-1", "title": "海阔天空",
                                            "artist": "Beyond", "duration": 327000}]},
    )
    assert entries[0].state == DUPLICATE_SUSPECTED
    assert "人工确认" in entries[0].reasons[0]


def test_same_title_but_a_very_different_duration_is_flagged_as_a_version_difference():
    from services.negative_feedback import song_key

    entries = classify(
        [_entry("999")],
        {"999": _meta(duration=326000)},
        {},
        {song_key("海阔天空", "Beyond"): [{"music_id": "m-1", "title": "海阔天空",
                                            "artist": "Beyond", "duration": 500000}]},
    )
    assert entries[0].state == DUPLICATE_SUSPECTED
    assert "Live/重制/翻唱" in entries[0].reasons[0]


def test_a_library_duration_stored_in_seconds_is_still_compared_correctly():
    """Some rows hold seconds rather than milliseconds; treating 326 as 326ms
    would call every one of them a version mismatch."""
    from services.negative_feedback import song_key

    entries = classify(
        [_entry("999")],
        {"999": _meta(duration=326000)},
        {},
        {song_key("海阔天空", "Beyond"): [{"music_id": "m-1", "title": "海阔天空",
                                            "artist": "Beyond", "duration": 326}]},
    )
    assert "人工确认" in entries[0].reasons[0]


def test_an_unknown_track_stays_metadata_ready():
    entries = classify([_entry("777")], {"777": _meta("新歌", "新歌手")}, {}, {})
    assert entries[0].state == METADATA_READY
    assert entries[0].matched_music_id == ""


def test_a_track_the_proxy_cannot_resolve_is_not_given_invented_metadata():
    entries = classify([_entry("404")], {}, {}, {})
    assert entries[0].state == METADATA_MISSING
    assert entries[0].title == "" and entries[0].artist == ""


def test_exact_wins_over_suspected():
    """An id match must not be downgraded by a title collision elsewhere."""
    from services.negative_feedback import song_key

    entries = classify(
        [_entry("1054603")],
        {"1054603": _meta()},
        {"1054603": {"music_id": "1054603", "title": "海阔天空", "duration": 326000}},
        {song_key("海阔天空", "Beyond"): [{"music_id": "other", "title": "海阔天空(Live)"}]},
    )
    assert entries[0].state == DUPLICATE_EXACT


# ---- index construction -----------------------------------------------------

def test_the_id_index_ignores_non_numeric_local_ids():
    """Local rows carry ids like 'local_0421' that must never collide with a
    NetEase song id."""
    by_id, by_key = load_library_index(
        rows=[{"music_id": "local_0421", "source_id": "", "title": "A", "artist": "B"}]
    )
    assert by_id == {}
    assert len(by_key) == 1


def test_source_id_also_indexes():
    by_id, _ = load_library_index(
        rows=[{"music_id": "local_1", "source_id": "1054603", "title": "A", "artist": "B"}]
    )
    assert "1054603" in by_id


# ---- summary ----------------------------------------------------------------

def test_summary_counts_what_we_already_have():
    entries = [
        CacheEntry("1", "320", 1024 ** 3, "/c/1.uc", state=DUPLICATE_EXACT),
        CacheEntry("2", "999", 1024 ** 3, "/c/2.uc", state=DUPLICATE_SUSPECTED),
        CacheEntry("3", "320", 2 * 1024 ** 3, "/c/3.uc", state=METADATA_READY),
    ]
    summary = summarise(entries)
    assert summary["cache_files"] == 3
    assert summary["already_have"] == 2
    assert summary["already_have_gib"] == pytest.approx(2.0, abs=0.01)
    assert summary["quality_mix"] == {"320": 2, "999": 1}


# ---- lyrics -----------------------------------------------------------------

def test_classify_attaches_lyrics_when_available():
    entries = classify(
        [_entry("777")],
        {"777": _meta("新歌", "新歌手")},
        {}, {},
        lyrics={"777": "[00:00.00]第一行歌词\n[00:05.00]第二行歌词"},
    )
    assert entries[0].lyrics == "[00:00.00]第一行歌词\n[00:05.00]第二行歌词"
    assert entries[0].state == METADATA_READY


def test_classify_leaves_lyrics_empty_when_not_fetched():
    entries = classify([_entry("777")], {"777": _meta("新歌", "新歌手")}, {}, {})
    assert entries[0].lyrics == ""


def test_classify_lyrics_not_attached_for_metadata_missing():
    """Tracks that the proxy can't resolve should not get lyrics either."""
    entries = classify(
        [_entry("404")], {}, {}, {},
        lyrics={"404": "[00:00.00]this should not appear"},
    )
    assert entries[0].state == METADATA_MISSING
    assert entries[0].lyrics == ""


def test_summarise_counts_lyrics():
    entries = [
        CacheEntry("1", "320", 1024, "/c/1.uc", state=METADATA_READY, lyrics="[00:00.00]有歌词"),
        CacheEntry("2", "320", 1024, "/c/2.uc", state=METADATA_READY, lyrics=""),
        CacheEntry("3", "320", 1024, "/c/3.uc", state=METADATA_READY, lyrics="   "),
    ]
    summary = summarise(entries)
    assert summary["lyrics_available"] == 1
    assert summary["lyrics_missing"] == 2
