"""Process queued music enrichment jobs outside the online API process."""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
import threading
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.logging_config import get_logger
from services.ingest_queue import claim_next_job, complete_job, fail_job, heartbeat, checkpoint_song, checkpoint_stage
from tools.acquire_music import _background_flywheel

logger = get_logger(__name__)


async def process_one() -> bool:
    claimed = claim_next_job()
    if claimed is None:
        return False

    job_path, payload = claimed
    songs = payload.get("songs", [])
    job_id = payload.get("job_id", job_path.stem)
    stop = threading.Event()
    lost = threading.Event()
    def renew():
        while not stop.wait(20):
            try:
                if not heartbeat(job_path):
                    lost.set()
                    return
            except Exception:
                lost.set()
                return
    keeper = threading.Thread(target=renew, daemon=True)
    if payload.get("lease_until"):
        keeper.start()
    try:
        logger.info("[IngestWorker] processing job=%s songs=%s", job_id, len(songs))
        if payload.get("lease_until"):
            finished = dict(payload.get("completed_songs") or {})
            for index, song in enumerate(songs):
                if lost.is_set():
                    raise RuntimeError("ingest lease lost")
                if str(index) in finished:
                    continue
                async def save_stage(key, value):
                    if lost.is_set():
                        raise RuntimeError("ingest lease lost")
                    await asyncio.to_thread(checkpoint_stage, job_path, index, key, value)
                song_result = await _background_flywheel(
                    [song], stage_cache=(payload.get("song_stages") or {}).get(str(index), {}),
                    save_stage=save_stage,
                )
                checkpoint_song(job_path, index, song_result)
                finished[str(index)] = song_result
            result = {
                "song_count": len(songs),
                "songs": [song for value in finished.values() for song in value.get("songs", [])],
                "warnings": [warning for value in finished.values() for warning in value.get("warnings", [])],
                "failure_count": 0,
            }
        else:
            result = await _background_flywheel(songs)
        if lost.is_set():
            raise RuntimeError("ingest lease lost; refusing late completion")
        complete_job(job_path, result=result)
        logger.info("[IngestWorker] completed job=%s", job_id)
    except Exception as exc:
        try:
            fail_job(job_path, str(exc))
        except FileNotFoundError:
            logger.warning("[IngestWorker] claim no longer owned job=%s", job_id)
        logger.exception("[IngestWorker] failed job=%s", job_id)
    finally:
        stop.set()
        if keeper.is_alive():
            keeper.join(timeout=1)
    return True


async def run(watch: bool, interval: float) -> None:
    while True:
        processed = await process_one()
        if not watch:
            if not processed:
                return
            continue
        if not processed:
            await asyncio.sleep(interval)


def main() -> None:
    parser = argparse.ArgumentParser(description="SoulTuner offline music ingestion worker")
    parser.add_argument("--watch", action="store_true", help="Keep polling for new jobs")
    parser.add_argument("--interval", type=float, default=3.0, help="Polling interval in seconds")
    args = parser.parse_args()
    started = time.time()
    asyncio.run(run(args.watch, args.interval))
    logger.info("[IngestWorker] stopped after %.1fs", time.time() - started)


if __name__ == "__main__":
    main()
