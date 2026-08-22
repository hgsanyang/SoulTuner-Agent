from __future__ import annotations

from tools.data import fma_balanced_pipeline as fma


def test_balanced_selection_excludes_unrequested_buckets_and_prefers_vocal_metadata(
    monkeypatch,
    tmp_path,
) -> None:
    rows = [
        {
            "track_id": 1,
            "album_id": 1,
            "genre": "Rock",
            "language": "",
            "lyricist": "",
            "listens": 999,
        },
        {
            "track_id": 2,
            "album_id": 1,
            "genre": "Rock",
            "language": "en",
            "lyricist": "",
            "listens": 10,
        },
        {
            "track_id": 3,
            "album_id": 1,
            "genre": "Experimental",
            "language": "en",
            "lyricist": "Writer",
            "listens": 9999,
        },
    ]
    monkeypatch.setattr(fma, "_stream_small_tracks", lambda _path: rows)
    monkeypatch.setattr(fma, "_album_metadata", lambda _path: {1: {}})

    selected = fma.select_tracks(tmp_path, {"Rock": 1})

    assert [row["track_id"] for row in selected] == [2]
    assert selected[0]["vocal_priority"] == 1


def test_license_links_cover_common_fma_labels() -> None:
    assert fma._licence_url("CC0 1.0 Universal").endswith("/zero/1.0/")
    assert fma._licence_url("Attribution-NonCommercial-NoDerivatives 4.0 International").endswith("/by-nc-nd/4.0/")
    assert fma._licence_url("Attribution-ShareAlike 3.0 International").endswith("/by-sa/3.0/")


def test_fma_archive_paths_are_stable_and_namespaced() -> None:
    assert fma._archive_name(2) == "fma_small/000/000002.mp3"
    assert fma._archive_name(123456) == "fma_small/123/123456.mp3"
