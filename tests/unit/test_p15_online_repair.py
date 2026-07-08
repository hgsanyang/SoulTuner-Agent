import json
import asyncio

from scripts.p15_repair_online_acquired_metadata import repair_online_acquired


def test_repair_online_acquired_marks_missing_audio_and_updates_year(tmp_path, monkeypatch):
    root = tmp_path / "online_acquired"
    (root / "metadata").mkdir(parents=True)
    (root / "audio").mkdir()
    (root / "covers").mkdir()
    (root / "lyrics").mkdir()
    meta_path = root / "metadata" / "Song A - Artist A_meta.json"
    meta_path.write_text(
        json.dumps({"musicId": 1, "musicName": "Song A", "artist": [["Artist A", 1]], "format": "mp3"}),
        encoding="utf-8",
    )

    async def fake_batch(songs, **kwargs):
        assert songs == [{"title": "Song A", "artist": "Artist A"}]
        return [{"title": "Song A", "artist": "Artist A", "release_year": 1999, "source_url": "https://example.com"}]

    monkeypatch.setattr("scripts.p15_repair_online_acquired_metadata.enrich_song_cards_batch", fake_batch)

    result = asyncio.run(repair_online_acquired(root, batch_size=3))
    updated = json.loads(meta_path.read_text(encoding="utf-8"))

    assert result["release_year_updated"] == 1
    assert result["missing_audio_marked_failed"] == 1
    assert updated["release_year"] == 1999
    assert updated["acquire_status"] == "failed"
