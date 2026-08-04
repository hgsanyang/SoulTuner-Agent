import json

import pytest

from services.offline_tag_review import (
    OfflineTagReviewError,
    build_review_prompt,
    make_task,
    read_jsonl,
    validate_result,
)
from scripts.export_catalog_tag_review import export_runs


def _record(tmp_path):
    lyric = tmp_path / "song.lrc"
    lyric.write_text("[00:01.00]first line\nsecond line", encoding="utf-8")
    return {
        "song_id": "42",
        "title": "Track",
        "artist": "Artist",
        "album": "Album",
        "release_year": 2001,
        "lrc_path": str(lyric),
    }


def test_task_is_stable_and_prompt_requests_jsonl(tmp_path):
    first = make_task(_record(tmp_path))
    second = make_task(_record(tmp_path))

    assert first == second
    assert first["lyrics_excerpt"] == "first line\nsecond line"
    prompt = build_review_prompt([first])
    assert "只返回 JSONL" in prompt
    assert first["task_id"] in prompt


def test_validate_result_caps_tags_and_ignores_model_confidence(tmp_path):
    task = make_task(_record(tmp_path))
    result = validate_result({
        "task_id": task["task_id"],
        "music_id": "42",
        "title": "Track",
        "artist": "Artist",
        "genres": ["Pop", "Pop", "Indie", "Rock", "Folk", "Soul", "Jazz"],
        "moods": ["Warm"],
        "themes": [],
        "scenarios": ["Late Night"],
        "language": "English",
        "region": "Western",
        "vibe": "Intimate",
        "confidence": 0.999,
        "evidence_urls": ["javascript:bad", "https://example.com/song"],
        "evidence_basis": "mixed",
        "taxonomy_feedback": {"suggested_additions": ["Intimate Folk"]},
    }, {task["task_id"]: task})

    assert result["genres"] == ["Pop", "Indie", "Rock", "Folk", "Soul"]
    assert result["evidence_urls"] == ["https://example.com/song"]
    confidence = json.loads(result["tag_confidence_json"])
    assert confidence["genres"]["Pop"] == 0.7
    assert result["taxonomy_feedback"]["suggested_additions"] == ["Intimate Folk"]


def test_validate_result_rejects_identity_drift(tmp_path):
    task = make_task(_record(tmp_path))
    with pytest.raises(OfflineTagReviewError, match="artist does not match"):
        validate_result({
            "task_id": task["task_id"],
            "music_id": "42",
            "title": "Track",
            "artist": "Wrong Artist",
        }, {task["task_id"]: task})


def test_read_jsonl_reports_bad_line(tmp_path):
    path = tmp_path / "results.jsonl"
    path.write_text('{"ok": true}\nnot-json\n', encoding="utf-8")
    with pytest.raises(OfflineTagReviewError, match="line 2"):
        read_jsonl(path)


def test_export_runs_combines_reports_and_deduplicates_tasks(tmp_path):
    record = _record(tmp_path)
    reports = []
    for index in range(2):
        path = tmp_path / f"run-{index}.json"
        path.write_text(json.dumps({
            "run_id": f"run-{index}",
            "published": [{"record": record}],
        }), encoding="utf-8")
        reports.append(path)

    summary = export_runs(reports, tmp_path / "bundle", batch_size=10)

    assert summary["run_ids"] == ["run-0", "run-1"]
    assert summary["tasks"] == 1
    assert summary["prompt_files"] == 1
