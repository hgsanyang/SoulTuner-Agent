"""Explicit positive feedback (like/save) must ingest the online track and KEEP it.

Chain under test:
    online candidate -> like/save -> retention="saved" -> ingest job -> MP3 kept

`retention="saved"` is the load-bearing bit: auto-recommendation ingests with
retention="temporary" and those files are reclaimed by cleanup_expired_temporary_audio,
so a liked song landing in "temporary" would silently disappear.
"""

from __future__ import annotations

import asyncio

import pytest

import services.online_audio_flywheel as flywheel


@pytest.fixture
def captured(monkeypatch):
    """Capture what would be ingested without touching the network/disk."""
    calls: list[dict] = []

    async def _fake_acquire(candidates, retention="temporary", requested_by=""):
        calls.append({"candidates": candidates, "retention": retention, "requested_by": requested_by})
        return len(candidates)

    monkeypatch.setattr(flywheel, "acquire_and_ingest_online_candidates", _fake_acquire)
    monkeypatch.setattr(flywheel.settings, "eval_disable_side_effects", False, raising=False)
    monkeypatch.setattr(flywheel.settings, "online_auto_flywheel_enabled", True, raising=False)
    return calls


def _run(coro_fn, *args, **kwargs):
    """schedule_* uses asyncio.create_task, so it needs a running loop."""
    async def _main():
        result = coro_fn(*args, **kwargs)
        await asyncio.sleep(0)  # let the scheduled task start
        return result

    return asyncio.run(_main())


ONLINE_EXTRA = {"source": "online_search", "song_id": "1901371647", "platform": "netease",
                "album": "跟着感觉走", "duration": 245}


@pytest.mark.parametrize("event", ["like", "save"])
def test_positive_feedback_ingests_with_saved_retention(captured, event):
    scheduled = _run(flywheel.schedule_online_feedback_flywheel,
                     event_type=event, title="再见", artist="张震岳", extra=dict(ONLINE_EXTRA))
    assert scheduled is True
    assert len(captured) == 1
    assert captured[0]["retention"] == "saved", "liked songs must be kept, not reclaimed as temporary"
    assert captured[0]["requested_by"] == f"user_{event}"
    song = captured[0]["candidates"][0]
    assert song["title"] == "再见" and song["song_id"] == "1901371647"


@pytest.mark.parametrize("event", ["skip", "dislike", "play"])
def test_non_positive_events_do_not_ingest(captured, event):
    scheduled = _run(flywheel.schedule_online_feedback_flywheel,
                     event_type=event, title="再见", artist="张震岳", extra=dict(ONLINE_EXTRA))
    assert scheduled is False
    assert captured == []


def test_like_without_source_id_is_not_ingested(captured):
    extra = {k: v for k, v in ONLINE_EXTRA.items() if k != "song_id"}
    assert _run(flywheel.schedule_online_feedback_flywheel,
                event_type="like", title="再见", artist="张震岳", extra=extra) is False
    assert captured == []


def test_like_on_local_track_is_not_re_ingested(captured):
    extra = {**ONLINE_EXTRA, "source": "local_catalog"}
    assert _run(flywheel.schedule_online_feedback_flywheel,
                event_type="like", title="再见", artist="张震岳", extra=extra) is False
    assert captured == []


def test_side_effect_guard_blocks_ingest(captured, monkeypatch):
    monkeypatch.setattr(flywheel.settings, "eval_disable_side_effects", True, raising=False)
    assert _run(flywheel.schedule_online_feedback_flywheel,
                event_type="like", title="再见", artist="张震岳", extra=dict(ONLINE_EXTRA)) is False
    assert captured == []
