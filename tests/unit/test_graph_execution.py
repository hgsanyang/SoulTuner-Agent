import asyncio
from types import SimpleNamespace
import pytest
from services.graph_execution import invoke_graph_turn, ConversationBusyError


def test_overlapping_same_conversation_is_rejected_but_other_users_run():
    async def run():
        entered, release = asyncio.Event(), asyncio.Event()
        async def graph(state, config):
            entered.set()
            await release.wait()
            return state
        app = SimpleNamespace(ainvoke=graph)
        first = asyncio.create_task(invoke_graph_turn(app, {}, config={}, owner="a", conversation="s"))
        await entered.wait()
        with pytest.raises(ConversationBusyError):
            await invoke_graph_turn(app, {}, config={}, owner="a", conversation="s")
        other = asyncio.create_task(invoke_graph_turn(app, {}, config={}, owner="b", conversation="s"))
        release.set()
        assert await first == await other == {}
        assert await invoke_graph_turn(app, {}, config={}, owner="a", conversation="s") == {}
    asyncio.run(run())


def test_total_timeout_cancels_graph_and_releases_conversation():
    async def run():
        stopped = asyncio.Event()
        async def graph(state, config):
            try:
                await asyncio.Event().wait()
            finally:
                stopped.set()
        app = SimpleNamespace(ainvoke=graph)
        for _ in range(2):
            with pytest.raises(TimeoutError):
                await invoke_graph_turn(app, {}, config={}, owner="a", conversation="s", timeout_seconds=.01)
        assert stopped.is_set()
    asyncio.run(run())
