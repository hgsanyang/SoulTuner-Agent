import json

from services import ingest_queue


def test_ingest_queue_lifecycle(tmp_path, monkeypatch):
    monkeypatch.setattr(ingest_queue, "QUEUE_ROOT", tmp_path)
    monkeypatch.setattr(ingest_queue, "PENDING_DIR", tmp_path / "pending")
    monkeypatch.setattr(ingest_queue, "PROCESSING_DIR", tmp_path / "processing")
    monkeypatch.setattr(ingest_queue, "DONE_DIR", tmp_path / "done")
    monkeypatch.setattr(ingest_queue, "FAILED_DIR", tmp_path / "failed")

    job_id = ingest_queue.enqueue_songs([{"title": "Song A"}])
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

    job_id = ingest_queue.enqueue_songs([{"title": "Song B"}])
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
