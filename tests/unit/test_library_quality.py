from services.library_quality import (
    duplicate_key,
    is_playable_song,
    missing_fields_for_song,
    pending_asset_status,
    quality_score,
    vector_coverage_from_dims,
)


def test_vector_coverage_and_missing_fields_drive_quality_score():
    coverage = vector_coverage_from_dims(muq_dim=512, m2d_dim=0, omar_dim=1024)
    missing = missing_fields_for_song(
        {
            "title": "A",
            "artist": "",
            "audio_url": "/a.mp3",
            "cover_url": "",
            "lrc_url": "/a.lrc",
            "language": "Chinese",
            "release_year": None,
        },
        coverage,
    )

    assert "artist" in missing
    assert "cover" in missing
    assert "release_year" in missing
    assert "m2d_embedding" in missing
    assert "audio" not in missing
    assert 0.0 < quality_score(missing) < 1.0


def test_duplicate_key_removes_common_version_noise():
    assert duplicate_key("Song (Live版)", "Artist") == duplicate_key("song", "artist")


def test_pending_asset_status_marks_missing_audio_invalid():
    status = pending_asset_status(has_audio=False, has_cover=True, has_lyrics=False)

    assert status["valid"] is False
    assert status["status"] == "invalid"
    assert status["missing_assets"] == ["audio", "lyrics"]


def test_is_playable_song_rejects_stubs_and_missing_audio():
    assert is_playable_song({"audio_url": "/static/audio/a.mp3"})
    assert is_playable_song({"preview_url": "/static/audio/a.mp3"})
    assert not is_playable_song({"audio_url": ""})
    assert not is_playable_song({"audio_url": "/static/audio/a.mp3", "unplayable_stub": True})
