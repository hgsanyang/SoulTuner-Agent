"""Consume and close a stream in its owning task, including during silent work."""

from __future__ import annotations

import asyncio
from contextlib import aclosing, nullcontext, suppress
from typing import AsyncGenerator, Awaitable, Callable, TypeVar

from services.recommendation_execution import RecommendationExecution, execution_scope


T = TypeVar("T")


async def owned_stream(
    source: AsyncGenerator[T, None],
    *,
    is_disconnected: Callable[[], Awaitable[bool]] | None = None,
    execution: RecommendationExecution | None = None,
    poll_seconds: float = 0.25,
) -> AsyncGenerator[T, None]:
    """Bounded prefetch; cancellation closes only this source and awaits cleanup.

    The producer owns all source iterations and its aclose, so ContextVar tokens
    never cross tasks even if the HTTP consumer is closed by a different task.
    This cancels async work, not arbitrary blocking threads or remote GPU jobs.
    """
    interval = max(0.01, poll_seconds)
    queue: asyncio.Queue[tuple[bool, object]] = asyncio.Queue(maxsize=1)

    async def produce() -> None:
        with execution_scope(execution) if execution is not None else nullcontext():
            try:
                async with aclosing(source):
                    async for item in source:
                        await queue.put((True, item))
            except Exception as exc:
                await queue.put((False, exc))
            else:
                await queue.put((False, None))

    producer = asyncio.create_task(produce(), name="soultuner-stream-owner")
    pending = None
    try:
        while True:
            if is_disconnected is not None:
                try:
                    if await asyncio.wait_for(is_disconnected(), timeout=interval):
                        return
                except Exception:
                    # A failed liveness probe is not proof that the client left.
                    pass
            if pending is None:
                pending = asyncio.create_task(queue.get())
            done, _ = await asyncio.wait({pending}, timeout=interval)
            if not done:
                if producer.cancelled():
                    raise asyncio.CancelledError
                continue
            is_value, value = pending.result()
            pending = None
            if not is_value:
                if isinstance(value, Exception):
                    raise value
                return
            yield value
    finally:
        if pending is not None:
            pending.cancel()
            with suppress(asyncio.CancelledError):
                await pending
        if not producer.done():
            producer.cancel()
        with suppress(asyncio.CancelledError):
            await producer
