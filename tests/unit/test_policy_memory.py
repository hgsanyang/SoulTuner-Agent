from services.policy_memory import (
    MAX_MULTIPLIER,
    MIN_MULTIPLIER,
    build_user_policy_profile,
    policy_runtime_payload,
)


def test_discovery_memory_increases_longtail_without_unbounded_weights():
    profile = build_user_policy_profile(
        {
            "activity_contexts": ["discovery", "longtail", "less_familiar"],
            "avoid_moods": ["Aggressive"],
        }
    )
    multipliers = profile.multipliers()

    assert multipliers["longtail"] > 1.0
    assert multipliers["freshness"] > 1.0
    assert multipliers["personal"] < 1.0
    assert multipliers["semantic_conflict"] > 1.0
    assert all(MIN_MULTIPLIER <= value <= MAX_MULTIPLIER for value in multipliers.values())


def test_closer_to_seed_memory_reduces_exploration_bias():
    payload = policy_runtime_payload(
        {
            "activity_contexts": ["closer_to_seed_song"],
            "favorite_genres": ["Folk", "Indie", "Dream Pop"],
            "favorite_moods": ["Healing", "Warm", "Calm"],
        }
    )
    multipliers = payload["post_recall_multipliers"]

    assert multipliers["personal"] > 1.0
    assert multipliers["semantic_preference"] > 1.0
    assert multipliers["longtail"] < 1.0
    assert payload["rationale"]


def test_empty_memory_is_neutral_policy():
    payload = policy_runtime_payload({})

    assert set(payload["post_recall_multipliers"].values()) == {1.0}
    assert payload["rationale"] == []
