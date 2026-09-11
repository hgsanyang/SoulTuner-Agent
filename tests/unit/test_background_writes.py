import contextvars
import threading

from services.background_writes import BackgroundWrites


def test_capacity_is_held_until_real_write_finishes_and_context_is_preserved():
    pool = BackgroundWrites(workers=1, capacity=1)
    started, release = threading.Event(), threading.Event()
    owner = contextvars.ContextVar("test_owner", default="none")
    seen = []

    def write():
        seen.append(owner.get())
        started.set()
        assert release.wait(3)

    token = owner.set("alice")
    try:
        assert pool.submit(write)
        assert started.wait(3)
        assert not pool.submit(lambda: None)
    finally:
        owner.reset(token)
        release.set()
        pool.close()
    assert seen == ["alice"]


def test_failure_releases_capacity(caplog):
    pool = BackgroundWrites(workers=1, capacity=1)
    def broken():
        raise RuntimeError("test failure")
    try:
        assert pool.submit(broken)
    finally:
        pool.close()
    assert pool._slots.acquire(blocking=False)
    assert "background write failed" in caplog.text
