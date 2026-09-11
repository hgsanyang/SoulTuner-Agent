"""Shared graph admission and total wall-clock budget for API and stream paths."""
import asyncio
import os
import threading

_lock = threading.Lock()
_active: set[tuple[str, str]] = set()


class ConversationBusyError(RuntimeError):
    pass


async def invoke_graph_turn(app, state, *, config, owner: str, conversation: str, timeout_seconds=None):
    budget = float(timeout_seconds if timeout_seconds is not None else os.getenv("SOULTUNER_TURN_TIMEOUT_SECONDS", "180"))
    if not 0 < budget <= 600:
        raise ValueError("turn timeout must be within (0, 600] seconds")
    key = (owner, conversation)
    with _lock:
        if key in _active:
            raise ConversationBusyError("当前会话仍在处理上一轮请求，请稍后再试")
        if len(_active) >= 128:
            raise ConversationBusyError("推荐服务当前繁忙，请稍后再试")
        _active.add(key)
    try:
        async with asyncio.timeout(budget):
            return await app.ainvoke(state, config=config)
    finally:
        with _lock:
            _active.discard(key)
