"""会话标识与 LangGraph checkpoint 隔离契约。"""

from agent.session_identity import checkpoint_thread_id, resolve_conversation_id


def test_supplied_conversation_id_is_stable():
    assert resolve_conversation_id({"session_id": "session-42"}) == "session-42"


def test_missing_conversation_id_is_request_scoped():
    first = resolve_conversation_id({})
    second = resolve_conversation_id(None)

    assert first.startswith("ephemeral-")
    assert second.startswith("ephemeral-")
    assert first != second


def test_checkpoint_id_is_stable_and_user_scoped():
    first = checkpoint_thread_id("user-a", "conversation-a")
    repeated = checkpoint_thread_id("user-a", "conversation-a")
    other_user = checkpoint_thread_id("user-b", "conversation-a")
    other_conversation = checkpoint_thread_id("user-a", "conversation-b")

    assert first == repeated
    assert first != other_user
    assert first != other_conversation
    assert "user-a" not in first
    assert "conversation-a" not in first
