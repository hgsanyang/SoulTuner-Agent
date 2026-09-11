import pytest

from retrieval.user_memory import MemoryProfileUnavailable, UserMemoryManager


class ReadOnlyClient:
    def __init__(self, fail_at=0):
        self.reads = 0
        self.fail_at = fail_at

    def execute_query(self, *args, **kwargs):
        raise AssertionError("profile read must use read-only reconnect path")

    def execute_read_query(self, query, parameters=None):
        self.reads += 1
        assert "CREATE " not in query and "SET " not in query
        if self.reads == self.fail_at:
            raise ConnectionError("internal endpoint")
        return []


@pytest.mark.parametrize("fail_at", [1, 2, 3])
def test_each_profile_read_failure_propagates(fail_at):
    manager = UserMemoryManager(neo4j_client=ReadOnlyClient(fail_at))
    with pytest.raises(MemoryProfileUnavailable, match="^memory_profile_unavailable$"):
        manager.get_user_preferences("alice")


def test_empty_profile_is_success_and_read_has_no_maintenance_writes():
    client = ReadOnlyClient()
    manager = UserMemoryManager(neo4j_client=client)
    assert manager.get_user_preferences("alice") == {"inferred_preferences": []}
    assert client.reads == 3


def test_missing_client_is_unavailable_not_empty():
    manager = UserMemoryManager(neo4j_client=ReadOnlyClient())
    manager.neo4j_client = None
    with pytest.raises(MemoryProfileUnavailable):
        manager.get_user_preferences("alice")
