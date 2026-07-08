"""Bounded ranking-policy knobs derived from editable user memory.

This layer is intentionally small: it does not recall songs and does not let
memory override graph/MuQ content relevance.  It only converts durable,
explicit preference signals into safe post-recall multipliers.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
import time
from typing import Any, Mapping


MIN_MULTIPLIER = 0.75
MAX_MULTIPLIER = 1.35
POLICY_CACHE_TTL_SECONDS = 30.0
_POLICY_CACHE: dict[str, tuple[float, dict[str, Any] | None]] = {}


@dataclass(frozen=True)
class UserPolicyProfile:
    personal: float = 1.0
    freshness: float = 1.0
    longtail: float = 1.0
    exposure_penalty: float = 1.0
    semantic_preference: float = 1.0
    semantic_conflict: float = 1.0
    rationale: tuple[str, ...] = ()

    def multipliers(self) -> dict[str, float]:
        payload = asdict(self)
        payload.pop("rationale", None)
        return {key: _clamp_multiplier(value) for key, value in payload.items()}


def _clamp_multiplier(value: Any) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        numeric = 1.0
    return round(max(MIN_MULTIPLIER, min(MAX_MULTIPLIER, numeric)), 3)


def _terms(value: Any) -> set[str]:
    if isinstance(value, (list, tuple, set)):
        return {" ".join(str(item or "").casefold().split()) for item in value if str(item or "").strip()}
    text = " ".join(str(value or "").casefold().split())
    return {text} if text else set()


def build_user_policy_profile(memory_profile: Mapping[str, Any] | None) -> UserPolicyProfile:
    """Build safe post-recall multipliers from durable memory fields.

    The inputs are already user-feedback or LLM-extracted preference fields.
    This function does not infer intent from the current query text.
    """

    profile = memory_profile or {}
    activity = _terms(profile.get("activity_contexts"))
    avoid_genres = _terms(profile.get("avoid_genres"))
    avoid_moods = _terms(profile.get("avoid_moods"))
    avoid_scenarios = _terms(profile.get("avoid_scenarios"))
    positive_count = sum(
        len(_terms(profile.get(key)))
        for key in (
            "favorite_genres",
            "favorite_artists",
            "favorite_moods",
            "favorite_scenarios",
            "preferred_genres",
            "preferred_moods",
            "preferred_scenarios",
            "preferred_languages",
        )
    )

    personal = 1.0
    freshness = 1.0
    longtail = 1.0
    exposure_penalty = 1.0
    semantic_preference = 1.0
    semantic_conflict = 1.0
    rationale: list[str] = []

    if {"discovery", "longtail", "less_familiar", "more_distinctive"} & activity:
        longtail += 0.22
        freshness += 0.12
        exposure_penalty += 0.15
        personal -= 0.08
        rationale.append("memory asks for discovery/less familiar music")

    if "avoid_overexposed" in activity or "too_familiar" in activity:
        exposure_penalty += 0.18
        personal -= 0.06
        rationale.append("memory asks to reduce overexposed familiar songs")

    if "closer_to_seed_song" in activity:
        personal += 0.10
        semantic_preference += 0.08
        longtail -= 0.08
        rationale.append("memory asks to stay closer to the current seed")

    if "needs_context_refinement" in activity:
        semantic_conflict += 0.12
        semantic_preference += 0.06
        rationale.append("memory says context matching needs stricter handling")

    if avoid_genres or avoid_moods or avoid_scenarios:
        semantic_conflict += 0.10
        exposure_penalty += 0.04
        rationale.append("memory contains explicit avoid preferences")

    if positive_count >= 6 and not ({"discovery", "less_familiar"} & activity):
        personal += 0.06
        semantic_preference += 0.04
        rationale.append("memory has enough positive preference signal")

    return UserPolicyProfile(
        personal=_clamp_multiplier(personal),
        freshness=_clamp_multiplier(freshness),
        longtail=_clamp_multiplier(longtail),
        exposure_penalty=_clamp_multiplier(exposure_penalty),
        semantic_preference=_clamp_multiplier(semantic_preference),
        semantic_conflict=_clamp_multiplier(semantic_conflict),
        rationale=tuple(rationale),
    )


def policy_runtime_payload(memory_profile: Mapping[str, Any] | None) -> dict[str, Any]:
    profile = build_user_policy_profile(memory_profile)
    return {
        "post_recall_multipliers": profile.multipliers(),
        "rationale": list(profile.rationale),
        "bounds": {"min": MIN_MULTIPLIER, "max": MAX_MULTIPLIER},
    }


def policy_runtime_payload_for_user(user_id: str = "local_admin") -> dict[str, Any] | None:
    now = time.monotonic()
    cached = _POLICY_CACHE.get(user_id)
    if cached and now - cached[0] < POLICY_CACHE_TTL_SECONDS:
        return cached[1]
    try:
        from services.memory_gateway import get_memory_gateway

        profile = get_memory_gateway().get_user_profile(user_id)
    except Exception:
        return None
    payload = policy_runtime_payload(profile)
    _POLICY_CACHE[user_id] = (now, payload)
    return payload


def invalidate_policy_memory_cache(user_id: str | None = None) -> None:
    if user_id:
        _POLICY_CACHE.pop(user_id, None)
    else:
        _POLICY_CACHE.clear()
