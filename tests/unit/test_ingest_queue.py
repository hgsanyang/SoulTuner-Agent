import json

import pytest

from services import ingest_queue


def _song(title: str = "Song A", artist: str = "Artist A") -> dict:
    return {
        "title": title,
        "artist": artist,
        "audio_url": "/static/online_audio/Song A - Artist A.mp3",
        "file_basename": "Song A - Artist A",
    }


def test_ingest_queue_lifecycle(tmp_path, monkeypatch):
    monkeypatch.setattr(ingest_queue, "QUEUE_ROOT", tmp_path)
    monkeypatch.setattr(ingest_queue, "PENDING_DIR", tmp_path / "pending")
    monkeypatch.setattr(ingest_queue, "PROCESSING_DIR", tmp_path / "processing")
    monkeypatch.setattr(ingest_queue, "DONE_DIR", tmp_path / "done")
    monkeypatch.setattr(ingest_queue, "FAILED_DIR", tmp_path / "failed")

    job_id = ingest_queue.enqueue_songs([_song()])
    claimed = ingest_queue.claim_next_job()

    assert claimed is not None
    job_path, payload = claimed
    assert payload["job_id"] == job_id
    assert payload["songs"][0]["title"] == "Song A"

    ingest_queue.complete_job(job_path)
    done_payload = json.loads((tmp_path / "done" / job_path.name).read_text(encoding="utf-8"))
    assert done_payload["job_id"] == job_id


def test_ingest_queue_lists_and_retries_failed_jobs(tmp_path, monkeypatch):
    monkeypatch.setattr(ingest_queue, "QUEUE_ROOT", tmp_path)
    monkeypatch.setattr(ingest_queue, "PENDING_DIR", tmp_path / "pending")
    monkeypatch.setattr(ingest_queue, "PROCESSING_DIR", tmp_path / "processing")
    monkeypatch.setattr(ingest_queue, "DONE_DIR", tmp_path / "done")
    monkeypatch.setattr(ingest_queue, "FAILED_DIR", tmp_path / "failed")

    job_id = ingest_queue.enqueue_songs([_song("Song B", "Artist B")])
    claimed = ingest_queue.claim_next_job()
    assert claimed is not None
    job_path, _payload = claimed
    ingest_queue.fail_job(job_path, "gpu unavailable")

    jobs = ingest_queue.list_jobs()
    assert jobs[0]["job_id"] == job_id
    assert jobs[0]["status"] == "failed"
    assert jobs[0]["song_count"] == 1
    assert jobs[0]["error"] == "gpu unavailable"

    assert ingest_queue.retry_failed_job(job_id) is True
    assert (tmp_path / "pending" / f"{job_id}.json").exists()
    assert not (tmp_path / "failed" / f"{job_id}.json").exists()


def test_ingest_queue_rejects_placeholder_or_audio_less_songs(tmp_path, monkeypatch):
    monkeypatch.setattr(ingest_queue, "QUEUE_ROOT", tmp_path)
    monkeypatch.setattr(ingest_queue, "PENDING_DIR", tmp_path / "pending")
    monkeypatch.setattr(ingest_queue, "PROCESSING_DIR", tmp_path / "processing")
    monkeypatch.setattr(ingest_queue, "DONE_DIR", tmp_path / "done")
    monkeypatch.setattr(ingest_queue, "FAILED_DIR", tmp_path / "failed")

    with pytest.raises(ingest_queue.IngestQueueValidationError, match="placeholder title"):
        ingest_queue.enqueue_songs([{"title": "New Song", "artist": "A", "audio_url": "/x.mp3"}])

    with pytest.raises(ingest_queue.IngestQueueValidationError, match="missing audio_url"):
        ingest_queue.enqueue_songs([{"title": "Real Song", "artist": "Real Artist"}])


def test_claim_skips_invalid_pending_jobs_and_marks_failed(tmp_path, monkeypatch):
    monkeypatch.setattr(ingest_queue, "QUEUE_ROOT", tmp_path)
    monkeypatch.setattr(ingest_queue, "PENDING_DIR", tmp_path / "pending")
    monkeypatch.setattr(ingest_queue, "PROCESSING_DIR", tmp_path / "processing")
    monkeypatch.setattr(ingest_queue, "DONE_DIR", tmp_path / "done")
    monkeypatch.setattr(ingest_queue, "FAILED_DIR", tmp_path / "failed")

    (tmp_path / "pending").mkdir(parents=True)
    invalid_id = "100-invalid"
    (tmp_path / "pending" / f"{invalid_id}.json").write_text(
        json.dumps({"job_id": invalid_id, "songs": [{"title": "New Song", "artist": "A"}]}),
        encoding="utf-8",
    )
    valid_id = ingest_queue.enqueue_songs([_song("Real Song", "Real Artist")])

    claimed = ingest_queue.claim_next_job()

    assert claimed is not None
    _job_path, payload = claimed
    assert payload["job_id"] == valid_id
    failed_payload = json.loads((tmp_path / "failed" / f"{invalid_id}.json").read_text(encoding="utf-8"))
    assert "placeholder title" in failed_payload["error"]
