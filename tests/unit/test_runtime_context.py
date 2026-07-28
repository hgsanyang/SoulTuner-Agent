from schemas.runtime_context import build_runtime_context
from services.runtime_context import (
    current_runtime_context,
    is_training_eligible,
    normalize_provenance,
    provenance_fields,
    runtime_context_scope,
    shared_catalog_side_effects_allowed,
)


def test_personal_context_is_training_eligible():
    ctx = build_runtime_context(profile_id="alice", interaction_mode="personal", session_id="s1")
    assert ctx.effective_user_id == "alice"
    assert ctx.training_eligible is True
    with runtime_context_scope(ctx):
        assert current_runtime_context().profile_id == "alice"
        assert provenance_fields("ranking")["training_eligible"] is True


def test_developer_context_uses_isolated_user_and_is_not_trainable():
    ctx = build_runtime_context(profile_id="alice", interaction_mode="developer")
    assert ctx.effective_user_id == "__dev__:alice"
    assert ctx.training_eligible is False
    with runtime_context_scope(ctx):
        fields = provenance_fields("preference_and_ranking")
    assert fields["interaction_mode"] == "developer"
    assert fields["training_eligible"] is False


def test_test_profile_is_not_trainable_in_personal_mode():
    ctx = build_runtime_context(
        profile_id="11111111-1111-4111-8111-111111111111",
        profile_type="test",
        interaction_mode="personal",
    )
    assert ctx.profile_type == "test"
    assert ctx.effective_user_id == "11111111-1111-4111-8111-111111111111"
    assert ctx.training_eligible is False
    assert ctx.teacher_log_eligible is False


def test_legacy_payload_is_quarantined_by_default():
    row = normalize_provenance({"user_id": "local_admin", "event_id": "old"})
    assert row["interaction_mode"] == "legacy"
    assert row["data_purpose"] == "legacy_unclassified"
    assert row["training_eligible"] is False
    assert is_training_eligible(row) is False


def test_client_cannot_make_developer_record_trainable():
    row = normalize_provenance(
        {
            "interaction_mode": "developer",
            "training_eligible": True,
            "data_purpose": "ranking",
        }
    )
    assert row["training_eligible"] is False


def test_unscoped_and_invalid_mode_contexts_fail_closed():
    assert current_runtime_context().interaction_mode == "legacy"
    assert current_runtime_context().training_eligible is False
    invalid = build_runtime_context(
        profile_id="alice",
        interaction_mode="persnoal",
    )
    assert invalid.interaction_mode == "legacy"
    assert invalid.training_eligible is False


def test_shared_catalog_side_effects_only_allow_personal_profile_normal_use():
    personal = build_runtime_context(
        profile_id="local_admin",
        profile_type="personal",
        interaction_mode="personal",
    )
    developer = build_runtime_context(
        profile_id="local_admin",
        profile_type="personal",
        interaction_mode="developer",
    )
    test_profile = build_runtime_context(
        profile_id="11111111-1111-4111-8111-111111111111",
        profile_type="test",
        interaction_mode="personal",
    )
    assert shared_catalog_side_effects_allowed(context=personal) is True
    assert shared_catalog_side_effects_allowed(context=developer) is False
    assert shared_catalog_side_effects_allowed(context=test_profile) is False


def test_developer_mode_cannot_schedule_shared_knowledge_backfill(monkeypatch):
    from config.settings import settings
    from services.recommendation_knowledge_backfill import (
        schedule_recommendation_knowledge_backfill,
    )

    monkeypatch.setattr(settings, "eval_disable_side_effects", False)
    monkeypatch.setattr(settings, "knowledge_background_enrichment_enabled", True)
    developer = build_runtime_context(
        profile_id="local_admin",
        profile_type="personal",
        interaction_mode="developer",
    )
    with runtime_context_scope(developer):
        result = schedule_recommendation_knowledge_backfill(
            [{"title": "A", "artist": "B"}]
        )
    assert result == {"scheduled": 0, "reason": "runtime_context"}
