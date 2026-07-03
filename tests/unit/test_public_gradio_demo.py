import json

from demos.public_gradio_app import DemoTrack, format_recommendations, load_demo_tracks, score_track


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
        }),
        encoding="utf-8",
    )

    tracks = load_demo_tracks(root)

    assert len(tracks) == 1
    assert tracks[0].title == "Rain Study"
    assert tracks[0].artist == "CC Artist"


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
    assert audio == [str(path)]
