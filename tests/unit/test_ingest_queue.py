import json

import pytest

from services import ingest_queue


def test_non_object_job_is_visible_and_quarantined(tmp_path, monkeypatch):
    for name in ("PENDING_DIR", "PROCESSING_DIR", "DONE_DIR", "FAILED_DIR"):
        monkeypatch.setattr(ingest_queue, name, tmp_path / name)
    ingest_queue._ensure_dirs()
    (ingest_queue.PENDING_DIR / "malformed.json").write_text("[]", encoding="utf-8")
    assert ingest_queue.list_jobs()[0]["valid"] is False
    assert ingest_queue.claim_next_job() is None
    assert ingest_queue.list_jobs()[0]["status"] == "failed"


def _song(title: str = "Song A", artist: str = "Artist A") -> dict:
    return {
        "title": title,
        "artist": artist,
        "audio_url": "/static/online_audio/Song A - Artist A.mp3",
        "file_basename": "Song A - Artist A",
    }


def test_duplicate_submission_returns_same_active_job(tmp_path, monkeypatch):
    for name in ("PENDING_DIR", "PROCESSING_DIR", "DONE_DIR", "FAILED_DIR"):
        monkeypatch.setattr(ingest_queue, name, tmp_path / name)
    job = ingest_queue.enqueue_songs([_song()])
    assert ingest_queue.enqueue_songs([_song()]) == job
    path, _ = ingest_queue.claim_next_job()
    assert ingest_queue.enqueue_songs([_song()]) == job
    ingest_queue.complete_job(path)
    assert ingest_queue.enqueue_songs([_song()]) != job  # Explicit re-enrichment remains possible.


def test_lease_recovery_fences_old_worker_and_bounds_retries(tmp_path, monkeypatch):
    for name in ("PENDING_DIR", "PROCESSING_DIR", "DONE_DIR", "FAILED_DIR"):
        monkeypatch.setattr(ingest_queue, name, tmp_path / name)
    now = [1000.]
    monkeypatch.setattr(ingest_queue.time, "time", lambda: now[0])
    job_id = ingest_queue.enqueue_songs([_song()])
    first, payload = ingest_queue.claim_next_job()
    assert payload["attempt"] == 1
    ingest_queue.checkpoint_song(first, 0, {"song_count": 1})
    now[0] += 60
    assert ingest_queue.heartbeat(first)
    now[0] += 61
    assert ingest_queue.recover_expired_jobs() == 0
    now[0] += 61
    second, payload = ingest_queue.claim_next_job()
    assert payload["attempt"] == 2 and second != first
    assert payload["completed_songs"] == {"0": {"song_count": 1}}
    with pytest.raises(FileNotFoundError):
        ingest_queue.checkpoint_song(first, 0, {})
    assert not ingest_queue.heartbeat(first)
    with pytest.raises(FileNotFoundError):
        ingest_queue.complete_job(first)
    now[0] += 121
    third, payload = ingest_queue.claim_next_job()
    assert payload["attempt"] == 3
    now[0] += 121
    assert ingest_queue.claim_next_job() is None
    assert (ingest_queue.FAILED_DIR / f"{job_id}.json").exists()


@pytest.mark.parametrize("job_id", ["../outside", "..\\outside", "C:\\outside", "/outside", "x/y", "x:y", "", "a" * 81])
def test_retry_rejects_unsafe_job_ids(tmp_path, monkeypatch, job_id):
    for name in ("PENDING_DIR", "PROCESSING_DIR", "DONE_DIR", "FAILED_DIR"):
        monkeypatch.setattr(ingest_queue, name, tmp_path / name)
    assert ingest_queue.retry_failed_job(job_id) is False


@pytest.mark.parametrize("bad", ["{broken", "[]", "null"])
def test_bad_job_is_quarantined_and_next_job_claimed(tmp_path, monkeypatch, bad):
    for name in ("PENDING_DIR", "PROCESSING_DIR", "DONE_DIR", "FAILED_DIR"):
        monkeypatch.setattr(ingest_queue, name, tmp_path / name)
    good = ingest_queue.enqueue_songs([_song()])
    (ingest_queue.PENDING_DIR / "000-bad.json").write_text(bad, encoding="utf-8")
    path, payload = ingest_queue.claim_next_job()
    assert payload["job_id"] == good
    assert path.parent == ingest_queue.PROCESSING_DIR
    assert (ingest_queue.FAILED_DIR / "000-bad.json").exists()


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

    ingest_queue.complete_job(job_path, result={"song_count": 1, "warnings": []})
    done_payload = json.loads((tmp_path / "done" / job_path.name).read_text(encoding="utf-8"))
    assert done_payload["job_id"] == job_id
    assert done_payload["result"]["song_count"] == 1
    assert done_payload["completed_at"] > 0
    assert ingest_queue.list_jobs()[0]["result"]["song_count"] == 1


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
