import json

import pytest

from demos.public_gradio_app import (
    DemoTrack,
    format_recommendations,
    is_public_cc_track,
    load_demo_tracks,
    sanitize_query,
    score_track,
    validate_demo_root,
)


def test_load_demo_tracks_reads_mtg_metadata(tmp_path):
    root = tmp_path / "mtg"
    (root / "metadata").mkdir(parents=True)
    (root / "audio").mkdir(parents=True)
    (root / "audio" / "1.low.mp3").write_bytes(b"demo")
    (root / "metadata" / "1.low_meta.json").write_text(
        json.dumps({
            "musicName": "Rain Study",
            "artist": [["CC Artist", 0]],
            "moods": ["Relaxing"],
            "themes": ["Background"],
            "scenarios": ["Study"],
            "source": "mtg",
            "license": "CC-BY-NC",
        }),
        encoding="utf-8",
    )

    tracks = load_demo_tracks(root)

    assert len(tracks) == 1
    assert tracks[0].title == "Rain Study"
    assert tracks[0].artist == "CC Artist"
    assert tracks[0].license == "CC-BY-NC"


def test_score_track_matches_query_tags(tmp_path):
    track = DemoTrack(
        title="Soft Rain",
        artist="CC Artist",
        audio_path=tmp_path / "x.mp3",
        moods=("Relaxing", "Soft"),
        themes=("Background",),
        scenarios=("Study",),
        source="mtg",
    )

    score, reasons = score_track(track, "雨天学习，柔软一点")

    assert score > 0
    assert reasons


def test_format_recommendations_mentions_cc_only(tmp_path):
    path = tmp_path / "x.mp3"
    path.write_bytes(b"demo")
    track = DemoTrack(
        title="Soft Rain",
        artist="CC Artist",
        audio_path=path,
        moods=("Relaxing", "Soft"),
        themes=("Background",),
        scenarios=("Study",),
        source="mtg",
    )

    text, audio = format_recommendations("rain study", [track])

    assert "CC-only" in text
    assert "Soft Rain" in text
    assert "license=" in text
    assert str(tmp_path) not in text
    assert audio == [str(path)]


def test_load_demo_tracks_filters_non_cc_metadata(tmp_path):
    root = tmp_path / "demo"
    (root / "metadata").mkdir(parents=True)
    (root / "audio").mkdir(parents=True)
    (root / "audio" / "private.mp3").write_bytes(b"demo")
    (root / "metadata" / "private_meta.json").write_text(
        json.dumps({
            "musicName": "Private Track",
            "artist": "Private Artist",
            "moods": ["Happy"],
            "source": "private_catalog",
            "license": "all rights reserved",
        }),
        encoding="utf-8",
    )

    assert load_demo_tracks(root) == []


def test_public_demo_rejects_private_catalog_paths(tmp_path):
    private_root = tmp_path / "processed_audio"
    private_root.mkdir()

    with pytest.raises(ValueError):
        validate_demo_root(private_root)


def test_public_cc_detection_accepts_jamendo_or_cc_license():
    assert is_public_cc_track({"source": "mtg"})
    assert is_public_cc_track({"license": "CC-BY-SA 4.0"})
    assert not is_public_cc_track({"source": "private", "license": "all rights reserved"})


def test_sanitize_query_strips_controls_and_truncates():
    query = "雨天\x00学习" + "x" * 300

    cleaned = sanitize_query(query)

    assert "\x00" not in cleaned
    assert len(cleaned) == 240
