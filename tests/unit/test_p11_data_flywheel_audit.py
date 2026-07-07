import json

from scripts.p11_data_flywheel_audit import (
    expected_basename,
    normalize_catalog_key,
    summarize_knowledge_store,
    summarize_online_acquired,
    summarize_qdrant_health,
)
from services.music_knowledge_store import MusicKnowledgeStore


def test_expected_basename_matches_download_policy():
    basename = expected_basename(
        {
            "musicName": "A/B:Song",
            "artist": [["Artist?", 1]],
        }
    )
    assert basename == "ABSong - Artist"


def test_normalize_catalog_key_removes_common_version_noise():
    assert normalize_catalog_key("Song (Live版)", "Artist") == normalize_catalog_key("song", "artist")


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
    assert summary["readiness"]["ready_for_ingest_minimal"] == 1
    assert summary["readiness"]["ready_for_enrichment"] == 0
    assert summary["problem_rows"][0]["title"] == "Song A"


def test_summarize_online_acquired_reports_duplicate_groups(tmp_path):
    root = tmp_path / "online_acquired"
    (root / "metadata").mkdir(parents=True)
    (root / "audio").mkdir()
    (root / "covers").mkdir()
    (root / "lyrics").mkdir()

    for index, title in enumerate(["Song A", "Song A (Live版)"]):
        meta = {
            "musicId": index,
            "musicName": title,
            "artist": [["Artist A", 1]],
            "format": "mp3",
        }
        (root / "metadata" / f"{title} - Artist A_meta.json").write_text(
            json.dumps(meta, ensure_ascii=False),
            encoding="utf-8",
        )

    summary = summarize_online_acquired(root)

    assert summary["readiness"]["duplicate_groups"] == 1
    assert summary["duplicate_groups"][0]["count"] == 2


def test_summarize_knowledge_store_reports_card_coverage(tmp_path):
    store = MusicKnowledgeStore(tmp_path / "knowledge.sqlite")
    store.upsert_artist_card(
        artist="The Cure",
        summary="English post-punk band.",
        style_tags=["Post-Punk"],
        source_url="https://example.com/the-cure",
        confidence=0.9,
    )
    store.upsert_song_card(
        title="Friday I'm in Love",
        artist="The Cure",
        summary="A jangly pop rock single.",
        release_year=1992,
        source_url="https://example.com/friday",
        confidence=0.85,
    )

    summary = summarize_knowledge_store(store.path)

    assert summary["available"] is True
    assert summary["artist_cards"] == 1
    assert summary["song_cards"] == 1
    assert summary["song_cards_with_release_year"] == 1
    assert summary["trusted_artist_cards"] == 1
    assert summary["trusted_song_cards"] == 1


def test_summarize_qdrant_health_is_best_effort_for_missing_service():
    summary = summarize_qdrant_health(
        base_url="http://127.0.0.1:1",
        collection="missing",
        timeout=0.01,
    )

    assert summary["available"] is False
    assert summary["collection"] == "missing"
