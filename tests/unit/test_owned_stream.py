"""Task ownership, backpressure, context and disconnect regression tests."""

import asyncio
from contextlib import aclosing

import pytest

from services.owned_stream import owned_stream
from services.recommendation_execution import (
    RecommendationExecution, RecommendationModels, current_execution,
)


async def collect(stream):
    return [value async for value in stream]


def test_natural_completion_and_error_close_source():
    async def run():
        closed = []

        async def source(fail):
            try:
                yield 1
                if fail:
                    raise ValueError("tool failed")
                yield 2
            finally:
                closed.append(fail)

        assert await collect(owned_stream(source(False))) == [1, 2]
        with pytest.raises(ValueError, match="tool failed"):
            await collect(owned_stream(source(True)))
        assert closed == [False, True]

    asyncio.run(run())


def test_disconnect_during_silent_work_cancels_only_its_owner():
    async def run():
        entered, closed, disconnect, release_other = [asyncio.Event() for _ in range(4)]

        async def source():
            try:
                entered.set()
                await asyncio.Event().wait()
                yield "never"
            finally:
                closed.set()

        async def other():
            await release_other.wait()
            yield "other result"

        async def probe():
            return disconnect.is_set()

        first = asyncio.create_task(collect(owned_stream(source(), is_disconnected=probe, poll_seconds=.01)))
        second = asyncio.create_task(collect(owned_stream(other(), poll_seconds=.01)))
        await asyncio.wait_for(entered.wait(), 1)
        disconnect.set()
        assert await asyncio.wait_for(first, 1) == []
        assert closed.is_set()
        assert not second.done()
        release_other.set()
        assert await second == ["other result"]

    asyncio.run(run())


def test_consumer_cancel_propagates_and_awaits_cleanup():
    async def run():
        entered, closed = asyncio.Event(), asyncio.Event()

        async def source():
            try:
                entered.set()
                await asyncio.Event().wait()
                yield 1
            finally:
                await asyncio.sleep(0)
                closed.set()

        consumer = asyncio.create_task(collect(owned_stream(source(), poll_seconds=.01)))
        await entered.wait()
        consumer.cancel()
        with pytest.raises(asyncio.CancelledError):
            await consumer
        assert closed.is_set()

    asyncio.run(run())


def test_early_close_on_different_task_restores_producer_context():
    async def run():
        closed = asyncio.Event()
        value = RecommendationExecution(False, RecommendationModels(1, 2, 3, 4))

        async def source():
            try:
                assert current_execution() is value
                yield 1
                await asyncio.Event().wait()
            finally:
                assert current_execution() is value
                closed.set()

        stream = owned_stream(source(), execution=value)
        assert await asyncio.create_task(anext(stream)) == 1
        assert current_execution() is None
        await asyncio.create_task(stream.aclose())
        assert closed.is_set()
        assert current_execution() is None

    asyncio.run(run())


def test_slow_consumer_does_not_build_an_unbounded_prefetch_queue():
    async def run():
        produced = []

        async def source():
            for number in range(10000):
                produced.append(number)
                yield number

        async with aclosing(owned_stream(source())) as stream:
            assert await anext(stream) == 0
            await asyncio.sleep(.02)
            assert len(produced) <= 3

    asyncio.run(run())


def test_failed_probe_does_not_drop_data():
    async def run():
        async def probe():
            raise OSError("probe unavailable")

        async def source():
            for value in range(5):
                yield value

        assert await collect(owned_stream(source(), is_disconnected=probe)) == list(range(5))

    asyncio.run(run())


def test_cancelled_source_does_not_leave_consumer_waiting_forever():
    async def run():
        async def source():
            raise asyncio.CancelledError
            yield 1

        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(collect(owned_stream(source(), poll_seconds=.01)), 1)

    asyncio.run(run())
