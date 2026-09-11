import asyncio

from scripts import ingest_worker


def test_worker_resumes_after_last_completed_song(tmp_path, monkeypatch):
    from services import ingest_queue
    for name in ("PENDING_DIR", "PROCESSING_DIR", "DONE_DIR", "FAILED_DIR"):
        monkeypatch.setattr(ingest_queue, name, tmp_path / name)
    songs = [{"title": title, "artist": "Artist", "audio_url": "/audio/test.mp3"}
             for title in ("First Track", "Second Track")]
    job_id = ingest_queue.enqueue_songs(songs)
    calls = []
    fail_once = [True]

    async def enrich(batch, **kwargs):
        title = batch[0]["title"]
        calls.append(title)
        if title == "Second Track" and fail_once[0]:
            fail_once[0] = False
            raise RuntimeError("encoder unavailable")
        return {"song_count": 1, "songs": [{"title": title}], "warnings": []}

    monkeypatch.setattr(ingest_worker, "_background_flywheel", enrich)
    assert asyncio.run(ingest_worker.process_one())
    assert ingest_queue.list_jobs()[0]["status"] == "failed"
    assert ingest_queue.retry_failed_job(job_id)
    assert asyncio.run(ingest_worker.process_one())
    row = ingest_queue.list_jobs()[0]
    assert row["status"] == "done"
    assert row["result"]["song_count"] == 2
    assert calls == ["First Track", "Second Track", "Second Track"]


def test_worker_persists_success_result(tmp_path, monkeypatch):
    job_path = tmp_path / "job.json"
    payload = {"job_id": "job", "songs": [{"title": "Track"}]}
    completed = []

    monkeypatch.setattr(ingest_worker, "claim_next_job", lambda: (job_path, payload))

    async def enrich(_songs):
        return {"song_count": 1, "warnings": ["omar optional"]}

    monkeypatch.setattr(ingest_worker, "_background_flywheel", enrich)
    monkeypatch.setattr(
        ingest_worker,
        "complete_job",
        lambda path, result=None: completed.append((path, result)),
    )

    assert asyncio.run(ingest_worker.process_one()) is True
    assert completed == [(job_path, {"song_count": 1, "warnings": ["omar optional"]})]


def test_worker_marks_incomplete_enrichment_failed(tmp_path, monkeypatch):
    job_path = tmp_path / "job.json"
    payload = {"job_id": "job", "songs": [{"title": "Track"}]}
    failures = []

    monkeypatch.setattr(ingest_worker, "claim_next_job", lambda: (job_path, payload))

    async def enrich(_songs):
        raise RuntimeError("missing muq_embedding")

    monkeypatch.setattr(ingest_worker, "_background_flywheel", enrich)
    monkeypatch.setattr(
        ingest_worker,
        "fail_job",
        lambda path, error: failures.append((path, error)),
    )

    assert asyncio.run(ingest_worker.process_one()) is True
    assert failures == [(job_path, "missing muq_embedding")]
