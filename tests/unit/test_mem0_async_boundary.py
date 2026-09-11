import asyncio
import threading
from types import SimpleNamespace

from services.memory_gateway import Mem0Adapter


def test_sync_mem0_calls_do_not_run_on_event_loop_thread():
    loop_thread = threading.get_ident()
    seen = []
    def call(*args, **kwargs):
        assert threading.get_ident() != loop_thread
        seen.append(kwargs["user_id"])
        return {"results": [{"memory": "Quiet music"}]}
    adapter = Mem0Adapter()
    adapter._client = SimpleNamespace(add=call, search=call, delete_all=call)

    async def run():
        assert await adapter.remember_text("quiet", user_id="alice")
        assert await adapter.retrieve_context("music", user_id="alice") == "Quiet music"
        assert await adapter.clear_user(user_id="alice")
    asyncio.run(run())
    assert seen == ["alice"] * 3


def test_async_mem0_client_result_is_still_awaited():
    adapter = Mem0Adapter()
    async def search(*args, **kwargs):
        return {"results": [{"memory": "Warm music"}]}
    adapter._client = SimpleNamespace(search=search)
    assert asyncio.run(adapter.retrieve_context("warm", user_id="alice")) == "Warm music"
