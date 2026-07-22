"""A-MEM link validation + episodic temporal helpers (memory v3 MV3-1)."""

from services.memory_links import (
    EPISODE_OCCURRED_AT,
    EPISODE_VALID_UNTIL,
    MAX_LINKS_PER_MEMORY,
    CONTRADICTING_RELATIONS,
    MemoryLink,
    MemoryRelation,
    RETRIEVAL_EXPAND_RELATIONS,
    SUPERSEDING_RELATIONS,
    episode_temporal,
    validate_links,
)


def _link(target: str, relation: str = "same_scene") -> dict:
    return {"target_memory_id": target, "relation": relation, "reason": "r"}


def test_validate_links_keeps_only_known_active_targets():
    known = {"m1", "m2", "m3"}
    links = validate_links(
        [_link("m1"), _link("ghost"), _link("m2", "refines")],
        known_memory_ids=known,
        self_id="m9",
    )
    assert [(link.target_memory_id, link.relation.value) for link in links] == [
        ("m1", "same_scene"),
        ("m2", "refines"),
    ]


def test_validate_links_drops_self_link_and_dedupes():
    links = validate_links(
        [_link("m1"), _link("m1"), _link("self")],
        known_memory_ids={"m1", "self"},
        self_id="self",
    )
    assert [link.target_memory_id for link in links] == ["m1"]


def test_validate_links_caps_count():
    known = {f"m{i}" for i in range(10)}
    raw = [_link(f"m{i}") for i in range(10)]
    links = validate_links(raw, known_memory_ids=known, self_id="x")
    assert len(links) == MAX_LINKS_PER_MEMORY


def test_validate_links_rejects_unknown_relation_and_bad_shape():
    links = validate_links(
        [_link("m1", "not_a_relation"), {"garbage": True}, "nonsense"],
        known_memory_ids={"m1"},
    )
    assert links == []


def test_validate_links_handles_non_list():
    assert validate_links(None, known_memory_ids={"m1"}) == []
    assert validate_links("m1", known_memory_ids={"m1"}) == []


def test_relation_side_effect_sets_are_disjoint_and_complete():
    # supersede vs contradict vs retrieval-only must not overlap dangerously
    assert MemoryRelation.EVOLVES_FROM in SUPERSEDING_RELATIONS
    assert MemoryRelation.CONTRADICTS in CONTRADICTING_RELATIONS
    assert MemoryRelation.SAME_SCENE in RETRIEVAL_EXPAND_RELATIONS
    # contradiction never silently supersedes
    assert not (SUPERSEDING_RELATIONS & CONTRADICTING_RELATIONS)


def test_memory_link_model_strips_and_validates():
    link = MemoryLink.model_validate({"target_memory_id": "  m1 ", "relation": "co_occurs"})
    assert link.target_memory_id == "m1"
    assert link.relation is MemoryRelation.CO_OCCURS


def test_episode_temporal_uses_fields_then_falls_back():
    occurred, valid = episode_temporal(
        {EPISODE_OCCURRED_AT: 1000, EPISODE_VALID_UNTIL: 2000},
        now_ms=5000,
        created_at=9999,
    )
    assert occurred == 1000
    assert valid == 2000


def test_episode_temporal_fallback_to_created_at_and_none_window():
    occurred, valid = episode_temporal({}, now_ms=5000, created_at=777)
    assert occurred == 777
    assert valid is None
