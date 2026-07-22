"""A-MEM style memory linking + episodic temporal fields (memory v3).

Memories are cards in a network: the consolidation LLM proposes directed
links between a new memory and the user's existing ones, and episodic (L3)
records carry structured temporal fields. All of this lives in the record
``payload`` dict, so nothing here changes the frozen MemoryRecord dataclass
or the SQLite schema — it is an additive, append-only extension.

Deterministic code in this module only validates what the LLM proposed
(known targets, relation enum, dedup, self-link ban, bounded count). Links
are retrieval enrichment and supersede/contradiction signals; they never
bypass a relevance or safety gate.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator

# ---- Episodic temporal payload keys (L3) ----
EPISODE_OCCURRED_AT = "occurred_at"   # int ms: when the event actually happened
EPISODE_VALID_UNTIL = "valid_until"   # int ms: hard relevance cutoff (also mirrored to expires_at)
EPISODE_MOOD = "mood"                 # free text: emotional context
EPISODE_SCENE = "scene"               # free text: situational scope

MAX_LINKS_PER_MEMORY = 4


class MemoryRelation(str, Enum):
    REFINES = "refines"            # new memory narrows/updates the target
    CONTRADICTS = "contradicts"    # new memory conflicts with the target
    SAME_SCENE = "same_scene"      # both apply to the same life scene
    EVOLVES_FROM = "evolves_from"  # new memory is a later version of the target
    CO_OCCURS = "co_occurs"        # observed together, no dominance


# Relations whose deterministic side effect is to supersede the target
# (the newer record wins; the ledger keeps history).
SUPERSEDING_RELATIONS = frozenset({MemoryRelation.EVOLVES_FROM, MemoryRelation.REFINES})
# Relations that flag the target for confidence review, never silent deletion.
CONTRADICTING_RELATIONS = frozenset({MemoryRelation.CONTRADICTS})
# Relations safe to follow during bounded retrieval expansion.
RETRIEVAL_EXPAND_RELATIONS = frozenset({MemoryRelation.SAME_SCENE, MemoryRelation.REFINES})


class MemoryLink(BaseModel):
    target_memory_id: str = Field(min_length=1, max_length=80)
    relation: MemoryRelation
    reason: str = Field(default="", max_length=200)

    @field_validator("target_memory_id", "reason")
    @classmethod
    def _strip(cls, value: str) -> str:
        return str(value or "").strip()


def validate_links(
    raw_links: Any,
    *,
    known_memory_ids: set[str],
    self_id: str = "",
    max_links: int = MAX_LINKS_PER_MEMORY,
) -> list[MemoryLink]:
    """Deterministic guardrail for LLM-proposed links.

    Drops links whose target is not a real, active memory id for this user,
    self-links, unknown relations, and duplicates; caps the count. ``known_memory_ids``
    must already exclude tombstoned/superseded memories, so a link can never
    resurrect an invalidated record.
    """
    if not isinstance(raw_links, (list, tuple)):
        return []
    self_key = str(self_id or "").strip()
    seen: set[tuple[str, str]] = set()
    valid: list[MemoryLink] = []
    for raw in raw_links:
        try:
            link = raw if isinstance(raw, MemoryLink) else MemoryLink.model_validate(raw)
        except Exception:
            continue
        target = link.target_memory_id
        if not target or target == self_key:
            continue
        if target not in known_memory_ids:
            continue
        identity = (target, link.relation.value)
        if identity in seen:
            continue
        seen.add(identity)
        valid.append(link)
        if len(valid) >= max(1, int(max_links)):
            break
    return valid


def episode_temporal(payload: dict[str, Any], *, now_ms: int, created_at: int) -> tuple[int, int | None]:
    """Return (occurred_at, valid_until) for an episodic record, with fallbacks.

    occurred_at falls back to the record's created_at when the event time was
    not captured; valid_until stays None (no hard cutoff) unless set.
    """
    occurred_raw = payload.get(EPISODE_OCCURRED_AT)
    try:
        occurred_at = int(occurred_raw)
    except (TypeError, ValueError):
        occurred_at = int(created_at)
    valid_raw = payload.get(EPISODE_VALID_UNTIL)
    try:
        valid_until: int | None = int(valid_raw)
    except (TypeError, ValueError):
        valid_until = None
    del now_ms  # reserved for future relative-window resolution
    return occurred_at, valid_until
