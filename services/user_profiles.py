"""Neo4j-backed registry for local application profiles.

This service deliberately does not implement authentication. It provides
explicit profile lifecycle operations and never creates a profile as a side
effect of a read. The Neo4j client is injectable so all behaviour can be tested
without a running database.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol
from uuid import uuid4

from schemas.user_profile import (
    DEFAULT_PROFILE_ID,
    CreateUserProfile,
    DeveloperSandboxReset,
    UpdateUserProfile,
    UserProfile,
    developer_sandbox_id,
    validate_profile_id,
)


class Neo4jQueryClient(Protocol):
    def execute_query(
        self,
        query: str,
        parameters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]: ...


class UserProfileServiceError(RuntimeError):
    """Base error for local profile operations."""


class UserProfileStoreUnavailable(UserProfileServiceError):
    """The profile registry cannot safely read from or write to Neo4j."""


class UserProfileNotFound(UserProfileServiceError):
    """The requested registered profile does not exist."""


class UserProfileConflict(UserProfileServiceError):
    """The requested profile operation would violate a lifecycle invariant."""


class UserProfileService:
    """Manage registered profiles on existing Neo4j ``User`` nodes."""

    def __init__(
        self,
        client: Neo4jQueryClient | None = None,
        *,
        client_factory: Callable[[], Neo4jQueryClient] | None = None,
    ):
        self._client = client
        self._client_factory = client_factory

    def list_profiles(self, *, include_deleted: bool = False) -> list[UserProfile]:
        client = self._available_client()
        rows = self._run(
            client,
            """
            MATCH (u:User)
            WHERE u.profile_registry = true OR u.id = $default_profile_id
            WITH u
            WHERE $include_deleted OR coalesce(u.profile_status, 'active') <> 'deleted'
            ORDER BY
                CASE WHEN u.id = $default_profile_id THEN 0 ELSE 1 END,
                coalesce(u.profile_created_at, datetime({epochMillis: 0})),
                u.id
            RETURN collect({
                profile_id: u.id,
                display_name: coalesce(u.profile_display_name, 'Local Admin'),
                profile_type: coalesce(u.profile_type, 'personal'),
                status: coalesce(u.profile_status, 'active'),
                created_at: CASE
                    WHEN u.profile_created_at IS NULL THEN null
                    ELSE toString(u.profile_created_at)
                END,
                updated_at: CASE
                    WHEN u.profile_updated_at IS NULL THEN null
                    ELSE toString(u.profile_updated_at)
                END,
                deleted_at: CASE
                    WHEN u.profile_deleted_at IS NULL THEN null
                    ELSE toString(u.profile_deleted_at)
                END
            }) AS profiles
            """,
            {
                "default_profile_id": DEFAULT_PROFILE_ID,
                "include_deleted": include_deleted,
            },
            operation="list profiles",
        )
        if len(rows) != 1 or not isinstance(rows[0].get("profiles"), list):
            raise UserProfileStoreUnavailable("Neo4j returned an invalid profile registry response")
        return [self._profile_from_row(row) for row in rows[0]["profiles"]]

    def get_profile(self, profile_id: str) -> UserProfile:
        canonical_id = validate_profile_id(profile_id)
        client = self._available_client()
        return self._get_profile(client, canonical_id)

    def register_default_profile(self, *, display_name: str = "Local Admin") -> UserProfile:
        """Explicitly register the legacy default profile on a fresh install."""

        cleaned_name = display_name.strip()
        if not cleaned_name:
            raise ValueError("display_name must not be blank")

        client = self._available_client()
        rows = self._run(
            client,
            """
            MERGE (u:User {id: $profile_id})
            ON CREATE SET
                u.profile_created_at = datetime(),
                u.profile_status = 'active'
            SET u.profile_registry = true,
                u.profile_display_name = coalesce(u.profile_display_name, $display_name),
                u.profile_type = coalesce(u.profile_type, 'personal'),
                u.profile_status = coalesce(u.profile_status, 'active'),
                u.profile_updated_at = datetime()
            RETURN {
                profile_id: u.id,
                display_name: u.profile_display_name,
                profile_type: u.profile_type,
                status: u.profile_status,
                created_at: CASE
                    WHEN u.profile_created_at IS NULL THEN null
                    ELSE toString(u.profile_created_at)
                END,
                updated_at: toString(u.profile_updated_at),
                deleted_at: CASE
                    WHEN u.profile_deleted_at IS NULL THEN null
                    ELSE toString(u.profile_deleted_at)
                END
            } AS profile
            """,
            {
                "profile_id": DEFAULT_PROFILE_ID,
                "display_name": cleaned_name,
            },
            operation="register default profile",
        )
        return self._single_profile(rows, operation="register default profile")

    def create_profile(self, request: CreateUserProfile) -> UserProfile:
        client = self._available_client()
        profile_id = str(uuid4())
        rows = self._run(
            client,
            """
            CREATE (u:User {
                id: $profile_id,
                profile_registry: true,
                profile_display_name: $display_name,
                profile_type: $profile_type,
                profile_status: 'active',
                profile_created_at: datetime(),
                profile_updated_at: datetime()
            })
            RETURN {
                profile_id: u.id,
                display_name: u.profile_display_name,
                profile_type: u.profile_type,
                status: u.profile_status,
                created_at: toString(u.profile_created_at),
                updated_at: toString(u.profile_updated_at),
                deleted_at: null
            } AS profile
            """,
            {
                "profile_id": profile_id,
                "display_name": request.display_name,
                "profile_type": request.profile_type,
            },
            operation="create profile",
        )
        return self._single_profile(rows, operation="create profile")

    def update_profile(self, profile_id: str, request: UpdateUserProfile) -> UserProfile:
        canonical_id = validate_profile_id(profile_id)
        client = self._available_client()
        current = self._get_profile(client, canonical_id)
        if current.status == "deleted":
            raise UserProfileConflict(f"profile '{canonical_id}' is deleted")

        rows = self._run(
            client,
            """
            MATCH (u:User {id: $profile_id})
            WHERE u.profile_registry = true OR u.id = $default_profile_id
            SET u.profile_registry = true,
                u.profile_display_name = $display_name,
                u.profile_type = $profile_type,
                u.profile_status = 'active',
                u.profile_updated_at = datetime()
            RETURN {
                profile_id: u.id,
                display_name: u.profile_display_name,
                profile_type: u.profile_type,
                status: u.profile_status,
                created_at: CASE
                    WHEN u.profile_created_at IS NULL THEN null
                    ELSE toString(u.profile_created_at)
                END,
                updated_at: toString(u.profile_updated_at),
                deleted_at: CASE
                    WHEN u.profile_deleted_at IS NULL THEN null
                    ELSE toString(u.profile_deleted_at)
                END
            } AS profile
            """,
            {
                "profile_id": canonical_id,
                "default_profile_id": DEFAULT_PROFILE_ID,
                "display_name": request.display_name or current.display_name,
                "profile_type": request.profile_type or current.profile_type,
            },
            operation="update profile",
        )
        return self._single_profile(rows, operation="update profile")

    def soft_delete_profile(self, profile_id: str) -> UserProfile:
        canonical_id = validate_profile_id(profile_id)
        if canonical_id == DEFAULT_PROFILE_ID:
            raise UserProfileConflict("the default local_admin profile cannot be deleted")

        client = self._available_client()
        current = self._get_profile(client, canonical_id)
        if current.status == "deleted":
            return current

        rows = self._run(
            client,
            """
            MATCH (u:User {id: $profile_id, profile_registry: true})
            SET u.profile_status = 'deleted',
                u.profile_deleted_at = datetime(),
                u.profile_updated_at = datetime()
            RETURN {
                profile_id: u.id,
                display_name: u.profile_display_name,
                profile_type: u.profile_type,
                status: u.profile_status,
                created_at: CASE
                    WHEN u.profile_created_at IS NULL THEN null
                    ELSE toString(u.profile_created_at)
                END,
                updated_at: toString(u.profile_updated_at),
                deleted_at: toString(u.profile_deleted_at)
            } AS profile
            """,
            {"profile_id": canonical_id},
            operation="soft-delete profile",
        )
        return self._single_profile(rows, operation="soft-delete profile")

    def reset_developer_sandbox(self, profile_id: str) -> DeveloperSandboxReset:
        canonical_id = validate_profile_id(profile_id)
        client = self._available_client()
        profile = self._get_profile(client, canonical_id)
        if profile.status == "deleted":
            raise UserProfileConflict(f"profile '{canonical_id}' is deleted")

        sandbox_id = developer_sandbox_id(canonical_id)
        existence_rows = self._run(
            client,
            """
            OPTIONAL MATCH (sandbox:User {id: $sandbox_user_id})
            RETURN sandbox IS NOT NULL AS exists
            """,
            {"sandbox_user_id": sandbox_id},
            operation="inspect developer sandbox",
        )
        if len(existence_rows) != 1 or "exists" not in existence_rows[0]:
            raise UserProfileStoreUnavailable("Neo4j returned an invalid developer sandbox response")

        existed = bool(existence_rows[0]["exists"])
        if existed:
            rows = self._run(
                client,
                """
                MATCH (sandbox:User {id: $sandbox_user_id})
                DETACH DELETE sandbox
                RETURN true AS reset
                """,
                {"sandbox_user_id": sandbox_id},
                operation="reset developer sandbox",
            )
            if len(rows) != 1 or rows[0].get("reset") is not True:
                raise UserProfileStoreUnavailable("Neo4j did not confirm developer sandbox reset")

        return DeveloperSandboxReset(
            profile_id=canonical_id,
            sandbox_user_id=sandbox_id,
            reset=existed,
        )

    def _available_client(self) -> Neo4jQueryClient:
        client = self._client
        if client is None:
            try:
                if self._client_factory is not None:
                    client = self._client_factory()
                else:
                    from retrieval.neo4j_client import get_neo4j_client

                    client = get_neo4j_client()
            except Exception as exc:
                raise UserProfileStoreUnavailable(
                    "Neo4j profile registry is unavailable; no profile data was changed"
                ) from exc
            self._client = client

        if client is None or getattr(client, "driver", object()) is None:
            raise UserProfileStoreUnavailable(
                "Neo4j profile registry is unavailable; no profile data was changed"
            )

        rows = self._run(
            client,
            "RETURN 'ok' AS profile_registry_health",
            {},
            operation="check profile registry",
        )
        if len(rows) != 1 or rows[0].get("profile_registry_health") != "ok":
            raise UserProfileStoreUnavailable(
                "Neo4j profile registry health check failed; no profile data was changed"
            )
        return client

    def _get_profile(self, client: Neo4jQueryClient, profile_id: str) -> UserProfile:
        rows = self._run(
            client,
            """
            OPTIONAL MATCH (u:User {id: $profile_id})
            WHERE u.profile_registry = true OR u.id = $default_profile_id
            RETURN CASE
                WHEN u IS NULL THEN null
                ELSE {
                    profile_id: u.id,
                    display_name: coalesce(u.profile_display_name, 'Local Admin'),
                    profile_type: coalesce(u.profile_type, 'personal'),
                    status: coalesce(u.profile_status, 'active'),
                    created_at: CASE
                        WHEN u.profile_created_at IS NULL THEN null
                        ELSE toString(u.profile_created_at)
                    END,
                    updated_at: CASE
                        WHEN u.profile_updated_at IS NULL THEN null
                        ELSE toString(u.profile_updated_at)
                    END,
                    deleted_at: CASE
                        WHEN u.profile_deleted_at IS NULL THEN null
                        ELSE toString(u.profile_deleted_at)
                    END
                }
            END AS profile
            """,
            {
                "profile_id": profile_id,
                "default_profile_id": DEFAULT_PROFILE_ID,
            },
            operation="get profile",
        )
        if len(rows) != 1 or "profile" not in rows[0]:
            raise UserProfileStoreUnavailable("Neo4j returned an invalid profile lookup response")
        if rows[0]["profile"] is None:
            raise UserProfileNotFound(f"profile '{profile_id}' was not found")
        return self._profile_from_row(rows[0]["profile"])

    @staticmethod
    def _run(
        client: Neo4jQueryClient,
        query: str,
        parameters: dict[str, Any],
        *,
        operation: str,
    ) -> list[dict[str, Any]]:
        try:
            rows = client.execute_query(query, parameters)
        except Exception as exc:
            raise UserProfileStoreUnavailable(
                f"Neo4j profile registry could not {operation}; no cross-user fallback was used"
            ) from exc
        if not isinstance(rows, list):
            raise UserProfileStoreUnavailable(
                f"Neo4j profile registry returned an invalid result while trying to {operation}"
            )
        return rows

    @staticmethod
    def _profile_from_row(row: dict[str, Any]) -> UserProfile:
        try:
            return UserProfile.model_validate(row)
        except Exception as exc:
            raise UserProfileStoreUnavailable("Neo4j contains an invalid registered profile") from exc

    def _single_profile(self, rows: list[dict[str, Any]], *, operation: str) -> UserProfile:
        if len(rows) != 1 or not isinstance(rows[0].get("profile"), dict):
            raise UserProfileStoreUnavailable(
                f"Neo4j did not confirm the requested {operation}; no fallback profile was created"
            )
        return self._profile_from_row(rows[0]["profile"])
