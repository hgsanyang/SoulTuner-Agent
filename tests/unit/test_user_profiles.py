from __future__ import annotations

from datetime import datetime, timezone
from uuid import UUID

import pytest
from pydantic import ValidationError

from schemas.user_profile import (
    CreateUserProfile,
    UpdateUserProfile,
    UserProfile,
    developer_sandbox_id,
)
from services.user_profiles import (
    UserProfileConflict,
    UserProfileNotFound,
    UserProfileService,
    UserProfileStoreUnavailable,
)


NOW = datetime(2026, 7, 28, tzinfo=timezone.utc).isoformat()


def _profile(
    profile_id: str,
    *,
    display_name: str = "Listener",
    profile_type: str = "personal",
    status: str = "active",
) -> dict:
    return {
        "profile_id": profile_id,
        "display_name": display_name,
        "profile_type": profile_type,
        "status": status,
        "created_at": NOW,
        "updated_at": NOW,
        "deleted_at": NOW if status == "deleted" else None,
    }


class FakeNeo4j:
    def __init__(self):
        self.profiles = {
            "local_admin": _profile("local_admin", display_name="Local Admin"),
        }
        self.sandboxes: set[str] = set()
        self.calls: list[tuple[str, dict]] = []

    def execute_query(self, query, parameters=None):
        params = parameters or {}
        self.calls.append((query, params))
        compact = " ".join(query.split())

        if "profile_registry_health" in compact:
            return [{"profile_registry_health": "ok"}]
        if "RETURN collect({" in compact:
            profiles = list(self.profiles.values())
            if not params["include_deleted"]:
                profiles = [row for row in profiles if row["status"] != "deleted"]
            profiles.sort(key=lambda row: (row["profile_id"] != "local_admin", row["profile_id"]))
            return [{"profiles": profiles}]
        if "MERGE (u:User {id: $profile_id})" in compact:
            row = self.profiles.setdefault(
                params["profile_id"],
                _profile(params["profile_id"], display_name=params["display_name"]),
            )
            return [{"profile": row}]
        if "CREATE (u:User" in compact:
            row = _profile(
                params["profile_id"],
                display_name=params["display_name"],
                profile_type=params["profile_type"],
            )
            self.profiles[params["profile_id"]] = row
            return [{"profile": row}]
        if "u.profile_display_name = $display_name" in compact:
            row = self.profiles[params["profile_id"]]
            row.update(
                display_name=params["display_name"],
                profile_type=params["profile_type"],
                status="active",
                updated_at=NOW,
            )
            return [{"profile": row}]
        if "u.profile_status = 'deleted'" in compact:
            row = self.profiles[params["profile_id"]]
            row.update(status="deleted", updated_at=NOW, deleted_at=NOW)
            return [{"profile": row}]
        if "sandbox IS NOT NULL AS exists" in compact:
            return [{"exists": params["sandbox_user_id"] in self.sandboxes}]
        if "DETACH DELETE sandbox" in compact:
            self.sandboxes.remove(params["sandbox_user_id"])
            return [{"reset": True}]
        if "OPTIONAL MATCH (u:User {id: $profile_id})" in compact:
            return [{"profile": self.profiles.get(params["profile_id"])}]
        raise AssertionError(f"unexpected query: {compact}")


class OfflineNeo4j:
    driver = None

    def execute_query(self, query, parameters=None):
        raise AssertionError("unavailable clients must not be queried")


class RaisingNeo4j:
    def execute_query(self, query, parameters=None):
        raise RuntimeError("connection lost")


def test_profile_id_accepts_uuid_and_legacy_local_admin_only():
    generated = str(UUID("8d3936f3-617b-4b70-bc21-6da9de734b50"))
    assert UserProfile(**_profile(generated)).profile_id == generated
    assert UserProfile(**_profile("local_admin")).profile_id == "local_admin"

    with pytest.raises(ValidationError, match="UUID or 'local_admin'"):
        UserProfile(**_profile("someone-else"))


def test_create_profile_generates_uuid_and_never_reuses_default_user():
    client = FakeNeo4j()
    service = UserProfileService(client)

    created = service.create_profile(CreateUserProfile(display_name="  New Listener  ", profile_type="test"))

    assert UUID(created.profile_id)
    assert created.profile_id != "local_admin"
    assert created.display_name == "New Listener"
    assert created.profile_type == "test"
    create_params = next(params for query, params in client.calls if "CREATE (u:User" in query)
    assert create_params["profile_id"] == created.profile_id


def test_list_profiles_preserves_local_admin_and_hides_soft_deleted_by_default():
    client = FakeNeo4j()
    deleted_id = "a69c145b-50ed-433b-882c-bcb2e0c69811"
    client.profiles[deleted_id] = _profile(deleted_id, status="deleted")
    service = UserProfileService(client)

    assert [p.profile_id for p in service.list_profiles()] == ["local_admin"]
    assert {p.profile_id for p in service.list_profiles(include_deleted=True)} == {
        "local_admin",
        deleted_id,
    }


def test_reads_do_not_silently_create_missing_profile():
    client = FakeNeo4j()
    service = UserProfileService(client)
    missing_id = "49704855-c6be-44d0-a71b-9baa3fac1aed"

    with pytest.raises(UserProfileNotFound, match=missing_id):
        service.get_profile(missing_id)

    assert missing_id not in client.profiles
    assert not any("MERGE" in query for query, _ in client.calls)


def test_default_profile_registration_is_explicit_and_idempotent():
    client = FakeNeo4j()
    client.profiles.clear()
    service = UserProfileService(client)

    assert service.list_profiles() == []
    registered = service.register_default_profile()
    registered_again = service.register_default_profile(display_name="Ignored New Name")

    assert registered.profile_id == "local_admin"
    assert registered.display_name == "Local Admin"
    assert registered_again.display_name == "Local Admin"
    assert list(client.profiles) == ["local_admin"]


def test_update_profile_changes_only_explicit_fields():
    client = FakeNeo4j()
    service = UserProfileService(client)

    updated = service.update_profile("local_admin", UpdateUserProfile(display_name="Daily Listener"))

    assert updated.display_name == "Daily Listener"
    assert updated.profile_type == "personal"


def test_soft_delete_is_idempotent_and_protects_default_profile():
    client = FakeNeo4j()
    profile_id = "39b7cdcb-deb8-49c1-926b-a9939e9dd0e3"
    client.profiles[profile_id] = _profile(profile_id)
    service = UserProfileService(client)

    deleted = service.soft_delete_profile(profile_id)
    assert deleted.status == "deleted"
    delete_call_count = sum("u.profile_status = 'deleted'" in query for query, _ in client.calls)
    assert service.soft_delete_profile(profile_id).status == "deleted"
    assert sum("u.profile_status = 'deleted'" in query for query, _ in client.calls) == delete_call_count

    with pytest.raises(UserProfileConflict, match="cannot be deleted"):
        service.soft_delete_profile("local_admin")


def test_reset_developer_sandbox_deletes_only_derived_user():
    client = FakeNeo4j()
    profile_id = "5a70ffec-f3dc-4b05-a2eb-d13956a5d04d"
    client.profiles[profile_id] = _profile(profile_id)
    sandbox_id = developer_sandbox_id(profile_id)
    client.sandboxes.add(sandbox_id)
    service = UserProfileService(client)

    result = service.reset_developer_sandbox(profile_id)

    assert result.reset is True
    assert result.sandbox_user_id == sandbox_id
    assert sandbox_id not in client.sandboxes
    assert profile_id in client.profiles
    delete_params = next(params for query, params in client.calls if "DETACH DELETE sandbox" in query)
    assert delete_params == {"sandbox_user_id": sandbox_id}


def test_reset_missing_sandbox_is_successful_noop():
    client = FakeNeo4j()
    service = UserProfileService(client)

    result = service.reset_developer_sandbox("local_admin")

    assert result.reset is False
    assert not any("DETACH DELETE sandbox" in query for query, _ in client.calls)


@pytest.mark.parametrize("client", [OfflineNeo4j(), RaisingNeo4j()])
def test_store_unavailable_raises_clear_error_without_fallback(client):
    service = UserProfileService(client)

    with pytest.raises(UserProfileStoreUnavailable, match="unavailable|could not"):
        service.list_profiles()
