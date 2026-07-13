"""LLM-assisted, evidence-bound consolidation of long-term music preferences."""

from __future__ import annotations

import inspect
import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Literal

from pydantic import BaseModel, Field, field_validator

from retrieval.user_memory import SEMANTIC_CONFLICT_FIELDS, SEMANTIC_LIST_FIELDS
from services.memory_event_store import MemoryEventStore
from services.memory_models import MemoryLayer, MemoryRecord

logger = logging.getLogger(__name__)

ALLOWED_MEMORY_FIELDS = frozenset(
    {*SEMANTIC_LIST_FIELDS, "mood_tendency", "language_preference"}
)


class InferredPreferenceCandidate(BaseModel):
    field: str = Field(description="One allowed structured preference field")
    value: str = Field(min_length=1, max_length=160)
    scope: Literal["global", "contextual", "temporary"] = "contextual"
    confidence: float = Field(ge=0.0, le=1.0)
    evidence_ids: list[str] = Field(default_factory=list, max_length=20)
    counter_evidence_ids: list[str] = Field(default_factory=list, max_length=20)
    ttl_days: int | None = Field(default=None, ge=1, le=365)
    retrieval_cues: list[str] = Field(default_factory=list, max_length=8)
    decision_summary: str = Field(default="", max_length=240)

    @field_validator("field", "value", "decision_summary")
    @classmethod
    def strip_text(cls, value: str) -> str:
        return str(value or "").strip()

    @field_validator("retrieval_cues")
    @classmethod
    def clean_cues(cls, values: list[str]) -> list[str]:
        cleaned: list[str] = []
        seen: set[str] = set()
        for value in values:
            text = str(value or "").strip()
            key = text.casefold()
            if text and key not in seen:
                seen.add(key)
                cleaned.append(text[:120])
        return cleaned[:8]


class MemoryConsolidationProposal(BaseModel):
    candidates: list[InferredPreferenceCandidate] = Field(default_factory=list, max_length=20)
    abstained: bool = False
    summary: str = Field(default="", max_length=300)


@dataclass
class RejectedMemoryCandidate:
    field: str
    value: str
    reason: str


@dataclass
class MemoryConsolidationReport:
    user_id: str
    evidence_count: int
    accepted: list[InferredPreferenceCandidate] = field(default_factory=list)
    rejected: list[RejectedMemoryCandidate] = field(default_factory=list)
    abstained: bool = False
    summary: str = ""
    model: str = ""

    def model_dump(self) -> dict[str, Any]:
        return {
            "user_id": self.user_id,
            "evidence_count": self.evidence_count,
            "accepted": [item.model_dump() for item in self.accepted],
            "rejected": [item.__dict__ for item in self.rejected],
            "abstained": self.abstained,
            "summary": self.summary,
            "model": self.model,
        }


ProposalGenerator = Callable[
    [str, list[dict[str, Any]]],
    MemoryConsolidationProposal | dict[str, Any] | Awaitable[MemoryConsolidationProposal | dict[str, Any]],
]


CONSOLIDATION_SYSTEM_PROMPT = """You consolidate durable music preferences from bounded user evidence.

Rules:
- Infer only from the supplied evidence. Never treat assistant text as user evidence.
- Prefer abstaining over inventing a preference.
- A durable inference normally needs at least two independent evidence records.
- When two or more records consistently support the same music preference and no
  supplied record contradicts it, emit a candidate instead of merely summarizing it.
- Keep explicit likes/dislikes, inferred tendencies, and temporary contexts distinct.
- Use only these fields: {allowed_fields}.
- add_* means preference; avoid_* means aversion. Do not output contradictory pairs.
- Include every supporting record_id in evidence_ids and any contradiction in counter_evidence_ids.
- retrieval_cues should be short natural-language paraphrases in the user's likely language(s).
- decision_summary is a short audit explanation, not hidden chain-of-thought.
- Do not emit song facts, recommendations, or personally identifying information.
- Set abstained=true exactly when no candidate is emitted.

Output one candidate per field/value. Never group add_moods/add_genres/etc in one object.
Exact shape example:
{{"candidates":[{{"field":"add_moods","value":"Warm","scope":"contextual",
"confidence":0.86,"evidence_ids":["record-1","record-2"],
"counter_evidence_ids":[],"ttl_days":45,
"retrieval_cues":["warm healing music"],
"decision_summary":"Repeated positive evidence supports this preference."}}],
"abstained":false,"summary":"One validated tendency."}}
"""


class MemoryConsolidator:
    def __init__(
        self,
        event_store: MemoryEventStore,
        *,
        generator: ProposalGenerator | None = None,
        min_confidence: float | None = None,
        min_evidence: int | None = None,
        default_ttl_days: int | None = None,
        max_evidence: int | None = None,
    ):
        self.event_store = event_store
        self.generator = generator
        self.min_confidence = float(
            min_confidence if min_confidence is not None else os.getenv("MEMORY_CONSOLIDATION_MIN_CONFIDENCE", "0.72")
        )
        self.min_evidence = int(
            min_evidence if min_evidence is not None else os.getenv("MEMORY_CONSOLIDATION_MIN_EVIDENCE", "2")
        )
        self.default_ttl_days = int(
            default_ttl_days if default_ttl_days is not None else os.getenv("MEMORY_INFERRED_TTL_DAYS", "45")
        )
        self.max_evidence = int(
            max_evidence if max_evidence is not None else os.getenv("MEMORY_CONSOLIDATION_MAX_EVIDENCE", "40")
        )

    async def consolidate(self, *, user_id: str) -> MemoryConsolidationReport:
        evidence = list(reversed(self.event_store.recent_evidence(user_id=user_id, limit=self.max_evidence)))
        report = MemoryConsolidationReport(user_id=user_id, evidence_count=len(evidence))
        if len(evidence) < self.min_evidence:
            report.abstained = True
            report.summary = "Insufficient independent evidence"
            return report

        proposal, model_name = await self._generate(user_id, evidence)
        report.model = model_name
        report.abstained = proposal.abstained or not proposal.candidates
        report.summary = proposal.summary
        accepted, rejected = self._validate(user_id=user_id, evidence=evidence, proposal=proposal)
        report.accepted = accepted
        report.rejected = rejected
        return report

    async def _generate(
        self,
        user_id: str,
        evidence: list[MemoryRecord],
    ) -> tuple[MemoryConsolidationProposal, str]:
        payload = [self._evidence_payload(record) for record in evidence]
        if self.generator is not None:
            result = self.generator(user_id, payload)
            if inspect.isawaitable(result):
                result = await result
            proposal = result if isinstance(result, MemoryConsolidationProposal) else MemoryConsolidationProposal.model_validate(result)
            return proposal, "injected-test-generator"

        from config.settings import settings
        from llms.chat_models import get_chat_model

        provider = settings.intent_llm_provider or settings.llm_default_provider
        model_name = settings.intent_llm_model or settings.llm_default_model
        llm = get_chat_model(
            provider=provider,
            model_name=model_name,
            temperature=0.0,
            max_tokens=1800,
        )
        try:
            structured = llm.with_structured_output(
                MemoryConsolidationProposal,
                include_raw=True,
                method="json_mode",
            )
        except (TypeError, ValueError):
            structured = llm.with_structured_output(MemoryConsolidationProposal, include_raw=True)
        system = CONSOLIDATION_SYSTEM_PROMPT.format(
            allowed_fields=", ".join(sorted(ALLOWED_MEMORY_FIELDS))
        )
        messages = [
            ("system", system),
            (
                "human",
                "user_id is already trusted by the application. Consolidate only this JSON evidence:\n"
                + json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
            ),
        ]
        result = await structured.ainvoke(messages)
        if isinstance(result, MemoryConsolidationProposal):
            proposal = result
        elif isinstance(result, dict) and isinstance(result.get("parsed"), MemoryConsolidationProposal):
            proposal = result["parsed"]
        else:
            raw = result.get("raw") if isinstance(result, dict) else result
            content = getattr(raw, "content", raw)
            payload = self._decode_json_payload(content)
            proposal = MemoryConsolidationProposal.model_validate(
                self._normalize_llm_payload(payload)
            )
        return proposal, f"{provider}:{model_name}"

    def _validate(
        self,
        *,
        user_id: str,
        evidence: list[MemoryRecord],
        proposal: MemoryConsolidationProposal,
    ) -> tuple[list[InferredPreferenceCandidate], list[RejectedMemoryCandidate]]:
        evidence_by_id = {record.record_id: record for record in evidence if record.user_id == user_id}
        explicit = self.event_store.effective_records(user_id=user_id, limit=1000)
        explicit_keys = {
            record.memory_key
            for record in explicit
            if record.layer == MemoryLayer.EXPLICIT and record.memory_key
        }
        accepted: list[InferredPreferenceCandidate] = []
        rejected: list[RejectedMemoryCandidate] = []
        seen_keys: set[str] = set()

        for raw in proposal.candidates:
            candidate = raw.model_copy(deep=True)
            candidate.field = candidate.field.strip()
            candidate.value = candidate.value.strip()
            key = self.memory_key(candidate.field, candidate.value)
            reason = ""
            cited = {value for value in candidate.evidence_ids if value in evidence_by_id}
            unknown = set(candidate.evidence_ids) - cited
            counter = {value for value in candidate.counter_evidence_ids if value in evidence_by_id}

            if candidate.field not in ALLOWED_MEMORY_FIELDS:
                reason = "unsupported_field"
            elif not candidate.value:
                reason = "empty_value"
            elif candidate.confidence < self.min_confidence:
                reason = "low_confidence"
            elif unknown:
                reason = "unknown_or_cross_user_evidence"
            elif len(cited) < self.min_evidence:
                reason = "insufficient_evidence"
            elif counter and candidate.confidence < 0.9:
                reason = "unresolved_counter_evidence"
            elif key in explicit_keys or self._inverse_key(candidate.field, candidate.value) in explicit_keys:
                reason = "explicit_memory_takes_precedence"
            elif key in seen_keys:
                reason = "duplicate_candidate"

            if reason:
                rejected.append(RejectedMemoryCandidate(candidate.field, candidate.value, reason))
                continue

            candidate.evidence_ids = sorted(cited)
            candidate.counter_evidence_ids = sorted(counter)
            candidate.ttl_days = self._bounded_ttl(candidate)
            if not candidate.retrieval_cues:
                candidate.retrieval_cues = [candidate.value]
            seen_keys.add(key)
            accepted.append(candidate)
        return accepted, rejected

    def _bounded_ttl(self, candidate: InferredPreferenceCandidate) -> int:
        defaults = {
            "global": max(self.default_ttl_days, 90),
            "contextual": self.default_ttl_days,
            "temporary": min(self.default_ttl_days, 14),
        }
        requested = int(candidate.ttl_days or defaults[candidate.scope])
        return max(7, min(requested, 180))

    @staticmethod
    def memory_key(field: str, value: str) -> str:
        return f"preference:{field}:{value.casefold()}"

    @staticmethod
    def _inverse_key(field: str, value: str) -> str:
        inverse = SEMANTIC_CONFLICT_FIELDS.get(field, "")
        return MemoryConsolidator.memory_key(inverse, value) if inverse else ""

    @staticmethod
    def _evidence_payload(record: MemoryRecord) -> dict[str, Any]:
        return {
            "record_id": record.record_id,
            "kind": record.kind,
            "source": record.source,
            "created_at": record.created_at,
            "payload": record.payload,
        }

    @staticmethod
    def _decode_json_payload(content: Any) -> dict[str, Any]:
        if isinstance(content, dict):
            return content
        if isinstance(content, list):
            content = "".join(
                str(item.get("text") or item.get("content") or "") if isinstance(item, dict) else str(item)
                for item in content
            )
        text = str(content or "").strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end < start:
            raise ValueError("memory consolidator did not return a JSON object")
        return json.loads(text[start:end + 1])

    @staticmethod
    def _normalize_llm_payload(payload: dict[str, Any]) -> dict[str, Any]:
        """Normalize provider-specific grouped fields without inventing confidence."""
        normalized: list[dict[str, Any]] = []
        for item in payload.get("candidates") or []:
            if not isinstance(item, dict):
                continue
            if item.get("field") and item.get("value"):
                normalized.append(item)
                continue
            common = {
                key: item.get(key)
                for key in (
                    "scope", "confidence", "evidence_ids", "counter_evidence_ids",
                    "ttl_days", "retrieval_cues", "decision_summary",
                )
                if item.get(key) is not None
            }
            for field_name in sorted(ALLOWED_MEMORY_FIELDS):
                raw_values = item.get(field_name)
                if raw_values is None or raw_values == "" or raw_values == []:
                    continue
                values = raw_values if isinstance(raw_values, list) else [raw_values]
                for value in values:
                    normalized.append(
                        {
                            **common,
                            "field": field_name,
                            "value": value,
                            # Missing confidence must fail the deterministic gate.
                            "confidence": common.get("confidence", 0.0),
                        }
                    )
        return {
            "candidates": normalized,
            "abstained": bool(payload.get("abstained", not normalized)),
            "summary": str(payload.get("summary") or "")[:300],
        }
