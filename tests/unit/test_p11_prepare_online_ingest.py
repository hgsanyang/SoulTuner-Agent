import json

from scripts.p11_prepare_online_ingest import build_song_from_meta, filter_ingestable, load_online_songs


def test_build_song_from_meta_carries_provenance_and_asset_flags(tmp_path):
    root = tmp_path / "online_acquired"
    (root / "audio").mkdir(parents=True)
    (root / "covers").mkdir()
    (root / "lyrics").mkdir()

    meta = {
        "musicId": 42,
        "musicName": "Song A",
        "artist": [["Artist A", 7]],
        "album": "Album A",
        "format": "mp3",
        "source_platform": "netease",
        "release_year": 1999,
    }
    (root / "audio" / "Song A - Artist A.mp3").write_bytes(b"fake")
    (root / "covers" / "Song A - Artist A_cover.jpg").write_bytes(b"fake")

    song = build_song_from_meta(meta, root)

    assert song["song_id"] == "42"
    assert song["title"] == "Song A"
    assert song["artist"] == "Artist A"
    assert song["release_year"] == 1999
    assert song["has_audio"] is True
    assert song["has_cover"] is True
    assert song["has_lyrics"] is False


def test_load_online_songs_and_filter_rejects_missing_audio(tmp_path):
    root = tmp_path / "online_acquired"
    (root / "metadata").mkdir(parents=True)
    (root / "audio").mkdir()
    (root / "covers").mkdir()
    (root / "lyrics").mkdir()
    (root / "metadata" / "Song A - Artist A_meta.json").write_text(
        json.dumps({"musicId": 1, "musicName": "Song A", "artist": [["Artist A", 1]], "format": "mp3"}),
        encoding="utf-8",
    )

    songs = load_online_songs(root)
    ok, rejected = filter_ingestable(songs)

    assert ok == []
    assert rejected[0]["reject_reason"] == "missing audio file"
