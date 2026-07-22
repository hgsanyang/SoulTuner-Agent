"""Versioned memory records shared by storage, API, and evaluation code."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class MemoryLayer(str, Enum):
    RAW_EVENT = "L0"
    EXPLICIT = "L1"
    INFERRED = "L2"
    EPISODIC = "L3"


class MemoryStatus(str, Enum):
    ACTIVE = "active"
    DELETED = "deleted"
    EXPIRED = "expired"
    SUPERSEDED = "superseded"


@dataclass(frozen=True)
class MemoryRecord:
    record_id: str
    user_id: str
    layer: MemoryLayer
    kind: str
    source: str
    evidence_id: str
    confidence: float
    created_at: int
    valid_from: int
    expires_at: int | None = None
    status: MemoryStatus = MemoryStatus.ACTIVE
    memory_key: str = ""
    payload: dict[str, Any] = field(default_factory=dict)
    why_used: str = ""
    target_record_id: str | None = None

    def model_dump(self) -> dict[str, Any]:
        data = asdict(self)
        data["layer"] = self.layer.value
        data["status"] = self.status.value
        return data
