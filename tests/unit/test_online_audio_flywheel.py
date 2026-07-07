import json

from services.online_audio_flywheel import (
    cleanup_expired_temporary_audio,
    collect_online_candidates,
    is_online_candidate,
    should_auto_acquire_feedback,
)


def test_collect_online_candidates_dedupes_and_skips_local_catalog_rows():
    rows = [
        {
            "song": {
                "title": "Online A",
                "artist": "Artist",
                "song_id": "100",
                "source": "online_search",
                "preview_url": "https://example.test/100.mp3",
            }
        },
        {
            "song": {
                "title": "Online A duplicate",
                "artist": "Artist",
                "song_id": "100",
                "source": "online_search",
            }
        },
        {
            "song": {
                "title": "Already acquired",
                "artist": "Artist",
                "song_id": "200",
                "source": "online_search",
                "audio_url": "/static/online_audio/Already acquired - Artist.mp3",
            }
        },
        {"song": {"title": "Local", "artist": "Artist", "music_id": "300", "source": "local"}},
    ]

    candidates = collect_online_candidates(rows, limit=10)

    assert [song["song_id"] for song in candidates] == ["100"]


def test_positive_feedback_only_auto_acquires_online_song_with_identity():
    assert should_auto_acquire_feedback(
        "like",
        {"source": "online_search", "song_id": "100"},
    )
    assert should_auto_acquire_feedback(
        "save",
        {"source": "online_search", "song_id": "100"},
    )
    assert not should_auto_acquire_feedback(
        "dislike",
        {"source": "online_search", "song_id": "100"},
    )
    assert not should_auto_acquire_feedback(
        "like",
        {"source": "local", "music_id": "local-1"},
    )


def test_static_online_audio_is_already_acquired_not_live_candidate():
    assert not is_online_candidate(
        {
            "source": "online_search",
            "audio_url": "/static/online_audio/A - B.mp3",
        }
    )


def test_cleanup_releases_only_expired_temporary_audio(tmp_path, monkeypatch):
    metadata_dir = tmp_path / "metadata"
    audio_dir = tmp_path / "audio"
    metadata_dir.mkdir()
    audio_dir.mkdir()
    monkeypatch.setattr("services.ingest_queue.list_jobs", lambda limit=5000: [])

    temporary = {
        "musicId": 100,
        "musicName": "Temporary",
        "artist": [["Artist", 0]],
        "format": "mp3",
        "acquired_at": "2020-01-01T00:00:00",
        "acquire_status": "ready",
        "audio_retention": "temporary",
    }
    saved = {
        **temporary,
        "musicId": 200,
        "musicName": "Saved",
        "audio_retention": "saved",
    }
    (metadata_dir / "Temporary - Artist_meta.json").write_text(json.dumps(temporary), encoding="utf-8")
    (metadata_dir / "Saved - Artist_meta.json").write_text(json.dumps(saved), encoding="utf-8")
    (audio_dir / "Temporary - Artist.mp3").write_bytes(b"temporary")
    (audio_dir / "Saved - Artist.mp3").write_bytes(b"saved")

    result = cleanup_expired_temporary_audio(
        ttl_hours=1,
        now=1_800_000_000,
        metadata_dir=metadata_dir,
        audio_dir=audio_dir,
        update_neo4j=False,
    )

    assert result["released"] == 1
    assert not (audio_dir / "Temporary - Artist.mp3").exists()
    assert (audio_dir / "Saved - Artist.mp3").exists()
    updated = json.loads((metadata_dir / "Temporary - Artist_meta.json").read_text(encoding="utf-8"))
    assert updated["audio_status"] == "released"
