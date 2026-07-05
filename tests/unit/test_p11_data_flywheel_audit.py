import json

from scripts.p11_data_flywheel_audit import expected_basename, summarize_online_acquired


def test_expected_basename_matches_download_policy():
    basename = expected_basename(
        {
            "musicName": "A/B:Song",
            "artist": [["Artist?", 1]],
        }
    )
    assert basename == "ABSong - Artist"


def test_summarize_online_acquired_reports_missing_assets(tmp_path):
    root = tmp_path / "online_acquired"
    (root / "metadata").mkdir(parents=True)
    (root / "audio").mkdir()
    (root / "covers").mkdir()
    (root / "lyrics").mkdir()

    meta = {
        "musicId": 1,
        "musicName": "Song A",
        "artist": [["Artist A", 1]],
        "format": "mp3",
    }
    (root / "metadata" / "Song A - Artist A_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False),
        encoding="utf-8",
    )
    (root / "audio" / "Song A - Artist A.mp3").write_bytes(b"fake")

    summary = summarize_online_acquired(root)

    assert summary["totals"]["metadata_files"] == 1
    assert summary["totals"]["has_audio"] == 1
    assert summary["totals"]["missing_cover"] == 1
    assert summary["totals"]["missing_lyrics"] == 1
    assert summary["totals"]["missing_release_year"] == 1
    assert summary["problem_rows"][0]["title"] == "Song A"
