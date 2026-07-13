"""Isolated JSONL worker for the sealed Memory blind-v2 benchmark.

The worker sees benchmark stimuli but never the scoring oracle. Each arm runs in
its own process and each bundle gets a fresh SQLite ledger. Synthetic candidates
are hard-filtered before memory can apply a bounded soft ordering adjustment.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import re
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from services.memory_consolidator import MemoryConsolidator  # noqa: E402
from services.memory_event_store import MemoryEventStore  # noqa: E402
from services.memory_gateway import Mem0Adapter  # noqa: E402
from services.memory_models import MemoryLayer, MemoryRecord, MemoryStatus  # noqa: E402
from services.memory_retriever import (  # noqa: E402
    DEFAULT_LAYER_THRESHOLDS,
    MemoryRelevanceRetriever,
    RetrievedMemory,
)
from services.memory_semantic_scorer import BgeMemorySemanticScorer  # noqa: E402

PROTOCOL_VERSION = "2.1.0"
# D_graphzep retired by product decision: GraphZep is no longer run as a comparison arm.
VALID_ARMS = {"A", "B", "C", "D_mem0"}
VALID_MODES = {"off", "structured", "semantic", "sidecar"}
OPTIONAL_CANDIDATE_FIELDS = {
    "artist", "title", "language", "year", "genre", "mood", "scenario",
    "instrumental", "instrument", "style", "era", "version", "playable",
    "request_mode", "purpose",
}
FIELD_TO_CANDIDATE = {
    "add_genres": "genre",
    "avoid_genres": "genre",
    "add_moods": "mood",
    "avoid_moods": "mood",
    "add_scenarios": "scenario",
    "avoid_scenarios": "scenario",
    "add_artists": "artist",
    "avoid_artists": "artist",
    "language_preference": "language",
    "mood_tendency": "mood",
}


def _stable_error(exc: BaseException, code: str = "worker_action_failed") -> dict[str, str]:
    return {"code": code, "class": type(exc).__name__}


def _norm(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").casefold()).strip()


def _candidate_satisfies(candidate: dict[str, Any], constraints: dict[str, Any]) -> bool:
    for key, expected in constraints.items():
        if key == "year_min":
            year = candidate.get("year")
            if year is None or int(year) < int(expected):
                return False
        elif key == "year_max":
            year = candidate.get("year")
            if year is None or int(year) > int(expected):
                return False
        elif isinstance(expected, str):
            if _norm(candidate.get(key)) != _norm(expected):
                return False
        elif candidate.get(key) != expected:
            return False
    return True


def _memory_matches_candidate(memory: RetrievedMemory, candidate: dict[str, Any]) -> float:
    field = str(memory.record.payload.get("field") or "")
    candidate_field = FIELD_TO_CANDIDATE.get(field)
    if not candidate_field:
        return 0.0
    expected = _norm(memory.record.payload.get("value"))
    actual = _norm(candidate.get(candidate_field))
    if not expected or not actual:
        return 0.0
    matched = expected == actual or expected in actual or actual in expected
    if not matched:
        return 0.0
    direction = -1.0 if field.startswith("avoid_") else 1.0
    return direction * memory.record.confidence * max(0.1, memory.relevance)


def _memory_can_discriminate(record: MemoryRecord, candidates: list[dict[str, Any]]) -> bool:
    field = str(record.payload.get("field") or "")
    candidate_field = FIELD_TO_CANDIDATE.get(field)
    if not candidate_field or len(candidates) < 2:
        return False
    expected = _norm(record.payload.get("value"))
    matches = []
    for candidate in candidates:
        actual = _norm(candidate.get(candidate_field))
        matches.append(bool(expected and actual and (expected == actual or expected in actual or actual in expected)))
    return any(matches) and not all(matches)


class _DeterministicIds:
    def __init__(self, namespace: str):
        self.namespace = namespace
        self.counter = 0
        self.forced: deque[str] = deque()

    def force(self, value: str) -> None:
        self.forced.append(str(value))

    def __call__(self) -> str:
        if self.forced:
            return self.forced.popleft()
        self.counter += 1
        raw = f"{self.namespace}:{self.counter}".encode("utf-8")
        return "bench-" + hashlib.sha256(raw).hexdigest()[:24]


class MemoryBenchmarkWorker:
    def __init__(self) -> None:
        self.arm = ""
        self.mode = "off"
        self.sidecar: Mem0Adapter | None = None
        self.capabilities: list[str] = []
        self.blocked = False
        self.bundle_id = ""
        self.namespace = ""
        self.user_ids: set[str] = set()
        self.primary_user_id = ""
        self.clock_ms = 0
        self.ids = _DeterministicIds("uninitialized")
        self.store: MemoryEventStore | None = None
        self.semantic_scorer = BgeMemorySemanticScorer()
        self.retriever = MemoryRelevanceRetriever(
            min_relevance=0.08,
            semantic_scorer=self.semantic_scorer,
            layer_thresholds=DEFAULT_LAYER_THRESHOLDS,
            max_per_layer=1,
        )
        self.sidecar_dirty = False
        self.bundle_model_calls = 0
        self.bundle_latency_ms = 0.0
        self.bundle_input_tokens = 0
        self.bundle_output_tokens = 0

    async def dispatch(self, action: str, payload: dict[str, Any]) -> dict[str, Any]:
        handler = getattr(self, f"action_{action}", None)
        if not callable(handler):
            raise ValueError("unsupported_action")
        if self.blocked and action not in {"initialize", "shutdown"}:
            raise RuntimeError("worker_unavailable")
        return await handler(payload)

    async def action_initialize(self, payload: dict[str, Any]) -> dict[str, Any]:
        if os.getenv("MEMORY_BENCHMARK_ISOLATED") != "1":
            raise RuntimeError("isolation_guard_missing")
        arm = str(payload.get("arm") or "")
        mode = str(payload.get("memory_mode") or "")
        if arm not in VALID_ARMS or mode not in VALID_MODES or payload.get("evaluation_mode") is not True:
            raise ValueError("invalid_initialization")
        self.arm = arm
        self.mode = mode
        self.capabilities = ["isolated_sqlite", "virtual_clock", "hard_constraints_first"]
        if arm != "A":
            self.capabilities.extend(["l1_explicit", "l2_expiring", "relevance_retrieval"])
            scorer = self.semantic_scorer.preflight()
            self.capabilities.extend([
                f"relevance_backend:{scorer['backend']}",
                f"relevance_revision:{scorer['revision']}",
                f"relevance_device:{scorer['device']}",
                "relevance_policy:open-calibration-v1",
            ])
        if arm in {"C", "D_mem0"}:
            self.capabilities.append("local_l3")
        if arm == "D_mem0":
            self.sidecar = Mem0Adapter()
        if self.sidecar is not None:
            healthy = await self.sidecar.healthcheck()
            if not healthy:
                self.blocked = True
                raise RuntimeError("sidecar_unavailable")
            self.capabilities.append(f"sidecar:{self.sidecar.name}")
        return {"arm": arm, "capabilities": self.capabilities, "isolated": True}

    async def action_begin_bundle(self, payload: dict[str, Any]) -> dict[str, Any]:
        bundle_id = str(payload.get("bundle_id") or "")
        namespace = str(payload.get("isolated_namespace") or "")
        users = {str(value) for value in payload.get("user_ids") or [] if str(value)}
        primary = str(payload.get("primary_user_id") or "")
        if not bundle_id or not namespace or primary not in users:
            raise ValueError("invalid_bundle")
        self.bundle_id = bundle_id
        self.namespace = namespace
        self.user_ids = users
        self.primary_user_id = primary
        self.clock_ms = 0
        self.ids = _DeterministicIds(namespace)
        base = Path(os.environ["MEMORY_EVENT_DB"])
        digest = hashlib.sha256(namespace.encode("utf-8")).hexdigest()[:16]
        path = base.with_name(f"{base.stem}-{digest}{base.suffix}")
        if path.exists():
            path.unlink()
        self.store = None if self.arm == "A" else MemoryEventStore(
            path,
            clock_ms=lambda: self.clock_ms,
            id_factory=self.ids,
        )
        self.sidecar_dirty = self.sidecar is not None
        self.bundle_model_calls = 0
        self.bundle_latency_ms = 0.0
        self.bundle_input_tokens = 0
        self.bundle_output_tokens = 0
        if self.sidecar is not None:
            for user_id in users:
                await self.sidecar.clear_user(user_id=self._sidecar_user(user_id))
        return {"bundle_id": bundle_id, "ready": True}

    async def action_seed_memory(self, payload: dict[str, Any]) -> dict[str, Any]:
        self._require_bundle(payload)
        memory = dict(payload.get("memory") or {})
        user_id = self._require_user(memory.get("user_id"))
        memory_id = str(memory.get("memory_id") or "")
        if self.arm == "A":
            return {"accepted": False, "memory_id": memory_id, "user_id": user_id, "status": memory.get("status", "active")}
        if not memory_id or self.store is None:
            raise ValueError("invalid_seed_memory")
        self.clock_ms = max(self.clock_ms, int(memory.get("created_at_ms") or 0))
        layer = MemoryLayer(str(memory.get("layer")))
        status = MemoryStatus(str(memory.get("status") or "active"))
        field = str(memory.get("field") or "")
        value = str(memory.get("value") or "")
        self.ids.force(memory_id)
        record = self.store.append(
            user_id=user_id,
            layer=layer,
            kind="episode" if layer == MemoryLayer.EPISODIC else "preference",
            source=str(memory.get("source") or "sealed_fixture"),
            evidence_id=memory_id,
            payload={
                "field": field,
                "value": value,
                "scope": str(memory.get("scope") or "global"),
                "retrieval_cues": list(memory.get("retrieval_cues") or []),
                "evidence_ids": [memory_id],
                "description": value if layer == MemoryLayer.EPISODIC else "",
            },
            confidence=float(memory.get("confidence") or 0.0),
            expires_at=memory.get("expires_at_ms"),
            memory_key=f"preference:{field}:{value.casefold()}" if field else f"episode:{memory_id}",
            status=status,
            target_record_id=memory.get("target_memory_id"),
            now_ms=int(memory.get("created_at_ms") or self.clock_ms),
            why_used="Sealed benchmark fixture",
        )
        if layer == MemoryLayer.EPISODIC and self.sidecar is not None:
            self.sidecar_dirty = True
        return {"accepted": record is not None, "memory_id": memory_id, "user_id": user_id, "status": status.value}

    async def action_append_event(self, payload: dict[str, Any]) -> dict[str, Any]:
        self._require_bundle(payload)
        event = dict(payload.get("event") or {})
        user_id = self._require_user(event.get("user_id"))
        evidence_id = str(event.get("evidence_id") or "")
        actor = str(event.get("actor") or "")
        event_type = str(event.get("type") or "")
        attributes = dict(event.get("attributes") or {})
        self.clock_ms = max(self.clock_ms, int(event.get("timestamp_ms") or 0))
        if self.arm == "A":
            return {"accepted": False, "evidence_id": evidence_id, "user_id": user_id, "record_id": None}
        if not evidence_id or self.store is None:
            raise ValueError("invalid_event")

        record: MemoryRecord | None
        target_id = str(attributes.get("target_memory_id") or "")
        if actor == "user" and event_type == "delete_request" and target_id:
            self.ids.force(evidence_id)
            record = self.store.tombstone(
                user_id=user_id,
                target_record_id=self._resolve_record_id(user_id, target_id),
                source="user_action",
                evidence_id=evidence_id,
            )
            self.sidecar_dirty = self.sidecar is not None
        elif actor == "user" and event_type == "explicit_preference" and attributes.get("field"):
            superseded = str(attributes.get("supersedes_memory_id") or "")
            if superseded:
                superseded_record_id = self._resolve_record_id(user_id, superseded)
                self.store.append(
                    user_id=user_id,
                    layer=MemoryLayer.EXPLICIT,
                    kind="supersession",
                    source="user_action",
                    evidence_id=evidence_id,
                    payload={"superseded_record_id": superseded_record_id},
                    status=MemoryStatus.SUPERSEDED,
                    target_record_id=superseded_record_id,
                    now_ms=self.clock_ms,
                )
            field = str(attributes.get("field") or "")
            value = str(attributes.get("value") or "")
            self.ids.force(evidence_id)
            record = self.store.append(
                user_id=user_id,
                layer=MemoryLayer.EXPLICIT,
                kind="preference",
                source="user_action",
                evidence_id=evidence_id,
                payload={
                    "field": field,
                    "value": value,
                    "scope": str(attributes.get("scope") or "global"),
                    "retrieval_cues": [value, str(event.get("content") or "")],
                    "evidence_ids": [evidence_id],
                },
                confidence=1.0,
                memory_key=f"preference:{field}:{value.casefold()}",
                now_ms=self.clock_ms,
                why_used="Explicit user preference",
            )
        else:
            source = "user_statement" if actor == "user" else f"{actor}_message"
            self.ids.force(evidence_id)
            record = self.store.append(
                user_id=user_id,
                layer=MemoryLayer.RAW_EVENT,
                kind=event_type,
                source=source,
                evidence_id=evidence_id,
                payload={
                    "user_text": str(attributes.get("user_text") or event.get("content") or "") if actor == "user" else "",
                    "actor": actor,
                    "attributes": attributes,
                },
                confidence=1.0,
                now_ms=self.clock_ms,
                why_used="User evidence" if actor == "user" else "Non-user event excluded from consolidation",
            )
        return {
            "accepted": record is not None,
            "evidence_id": evidence_id,
            "user_id": user_id,
            "record_id": record.record_id if record is not None else None,
        }

    async def action_consolidate(self, payload: dict[str, Any]) -> dict[str, Any]:
        self._require_bundle(payload)
        user_id = self._require_user(payload.get("user_id"))
        max_calls = int(payload.get("max_model_calls") or 0)
        empty = {
            "invoked": False,
            "model_calls": 0,
            "accepted": [],
            "rejected": [],
            "latency_ms": 0.0,
            "input_tokens": 0,
            "output_tokens": 0,
            "abstained": True,
            "model_id": None,
            "model_version": None,
            "prompt_hash": None,
        }
        if self.arm == "A" or max_calls == 0 or self.store is None:
            return empty
        started = time.perf_counter()
        report = await MemoryConsolidator(self.store).consolidate(user_id=user_id)
        latency = (time.perf_counter() - started) * 1000.0
        model_calls = 1 if report.model else 0
        if model_calls > max_calls:
            raise RuntimeError("model_call_budget_exceeded")
        accepted: list[dict[str, Any]] = []
        for candidate in report.accepted:
            expires_at = self.clock_ms + int(candidate.ttl_days or 45) * 86_400_000
            record = self.store.append(
                user_id=user_id,
                layer=MemoryLayer.INFERRED,
                kind="preference",
                source="memory_consolidator",
                evidence_id=candidate.evidence_ids[0],
                payload={
                    "field": candidate.field,
                    "value": candidate.value,
                    "scope": candidate.scope,
                    "evidence_ids": list(candidate.evidence_ids),
                    "counter_evidence_ids": list(candidate.counter_evidence_ids),
                    "retrieval_cues": list(candidate.retrieval_cues),
                    "decision_summary": candidate.decision_summary,
                },
                confidence=float(candidate.confidence),
                expires_at=expires_at,
                memory_key=MemoryConsolidator.memory_key(candidate.field, candidate.value),
                now_ms=self.clock_ms,
                why_used="Evidence-bound inferred preference",
            )
            accepted.append({
                "record_id": record.record_id,
                "user_id": user_id,
                "field": candidate.field,
                "value": candidate.value,
                "layer": "L2",
                "scope": candidate.scope,
                "confidence": candidate.confidence,
                "evidence_ids": list(candidate.evidence_ids),
                "counter_evidence_ids": list(candidate.counter_evidence_ids),
                "retrieval_cues": list(candidate.retrieval_cues),
                "decision_summary": candidate.decision_summary,
                "created_at_ms": self.clock_ms,
                "expires_at_ms": expires_at,
                "source": "memory_consolidator",
            })
        rejected = [
            {"field": item.field, "value": item.value, "reason": item.reason}
            for item in report.rejected
        ]
        self.bundle_model_calls += model_calls
        self.bundle_latency_ms += latency
        self.bundle_input_tokens += report.input_tokens
        self.bundle_output_tokens += report.output_tokens
        return {
            "invoked": bool(model_calls),
            "model_calls": model_calls,
            "accepted": accepted,
            "rejected": rejected,
            "latency_ms": round(latency, 3),
            "input_tokens": report.input_tokens,
            "output_tokens": report.output_tokens,
            "abstained": report.abstained,
            "model_id": report.model or None,
            "model_version": None,
            "prompt_hash": report.prompt_hash or None,
        }

    async def action_advance_clock(self, payload: dict[str, Any]) -> dict[str, Any]:
        self._require_bundle(payload)
        timestamp = int(payload.get("timestamp_ms") or 0)
        advanced = timestamp >= self.clock_ms
        self.clock_ms = timestamp
        return {"timestamp_ms": timestamp, "advanced": advanced}

    async def action_evaluate_query(self, payload: dict[str, Any]) -> dict[str, Any]:
        self._require_bundle(payload)
        query = dict(payload.get("query") or {})
        user_id = self._require_user(query.get("user_id"))
        self.clock_ms = int(query.get("timestamp_ms") or self.clock_ms)
        started = time.perf_counter()
        selected: list[RetrievedMemory] = []
        effective: list[MemoryRecord] = []
        backends: list[str] = []
        constraints = dict(query.get("hard_constraints") or {})
        candidates = [
            dict(item)
            for item in query.get("synthetic_candidates") or []
            if _candidate_satisfies(dict(item), constraints)
        ]
        if self.arm != "A" and self.store is not None:
            effective = self.store.effective_records(user_id=user_id, now_ms=self.clock_ms, limit=1000)
            include_l3 = self.arm in {"C", "D_mem0"}
            selected = self.retriever.retrieve(
                query=str(query.get("text") or ""),
                records=effective,
                max_facts=8,
                include_episodic=include_l3,
                now_ms=self.clock_ms,
            )
            backends.extend(["local_ledger", self.retriever.backend_name])
            if include_l3:
                backends.append("local_l3")
        if self.sidecar is not None:
            await self._sync_sidecar()
            text = await self.sidecar.retrieve_context(
                str(query.get("text") or ""),
                user_id=self._sidecar_user(user_id),
                max_facts=8,
            )
            if "服务不可用" in text or "暂时不可用" in text:
                raise RuntimeError("sidecar_retrieval_unavailable")
            selected = self._merge_sidecar_matches(selected, effective, text)
            backends.append(self.sidecar.name)

        scored = [
            (sum(_memory_matches_candidate(memory, item) for memory in selected), index, item)
            for index, item in enumerate(candidates)
        ]
        scored.sort(key=lambda row: (-row[0], row[1]))
        recommendations: list[dict[str, Any]] = []
        for rank, (_, _, item) in enumerate(scored[:10], start=1):
            recommendations.append({
                "candidate_id": str(item.get("candidate_id") or ""),
                **{key: item[key] for key in OPTIONAL_CANDIDATE_FIELDS if key in item},
                "rank": rank,
            })
        memory_items = [self._memory_observation(item) for item in selected]
        effective_items = [
            self._record_observation(record, relevance=None)
            for record in effective
            if record.layer in self._enabled_layers()
        ]
        latency = (time.perf_counter() - started) * 1000.0
        trace_raw = f"{self.arm}:{self.bundle_id}:{payload.get('case_id')}:{payload.get('phase')}"
        return {
            "retrieved_memories": memory_items,
            "effective_memories": effective_items,
            "recommendations": recommendations,
            "memory_latency_ms": round(latency, 3),
            "memory_input_tokens": 0,
            "memory_output_tokens": 0,
            "memory_overrode_hard_constraints": False,
            "retrieval_backend": backends,
            "memory_trace_id": hashlib.sha256(trace_raw.encode("utf-8")).hexdigest()[:24],
        }

    async def action_end_bundle(self, payload: dict[str, Any]) -> dict[str, Any]:
        self._require_bundle(payload)
        closed = True
        if self.sidecar is not None:
            if self.sidecar_dirty:
                await self._sync_sidecar()
            closed = await self.sidecar.await_idle(timeout_seconds=90.0)
            for user_id in self.user_ids:
                closed = await self.sidecar.clear_user(user_id=self._sidecar_user(user_id)) and closed
        observation = {
            "bundle_id": self.bundle_id,
            "closed": bool(closed),
            "model_calls": self.bundle_model_calls,
            "latency_ms": round(self.bundle_latency_ms, 3),
            "input_tokens": self.bundle_input_tokens,
            "output_tokens": self.bundle_output_tokens,
        }
        self.bundle_id = ""
        self.store = None
        return observation

    async def action_shutdown(self, payload: dict[str, Any]) -> dict[str, Any]:
        del payload
        return {"closed": True}

    async def _sync_sidecar(self) -> None:
        if self.sidecar is None or not self.sidecar_dirty or self.store is None:
            return
        for user_id in self.user_ids:
            sidecar_user = self._sidecar_user(user_id)
            if not await self.sidecar.clear_user(user_id=sidecar_user):
                raise RuntimeError("sidecar_cleanup_failed")
            records = self.store.effective_records(user_id=user_id, now_ms=self.clock_ms, limit=1000)
            for record in records:
                if record.layer != MemoryLayer.EPISODIC:
                    continue
                value = str(record.payload.get("value") or record.payload.get("description") or "")
                marker = f"memory_id={record.record_id}; field={record.payload.get('field')}; value={value}"
                if not await self.sidecar.remember_text(marker, user_id=sidecar_user, extra={"source": "sealed_benchmark"}):
                    raise RuntimeError("sidecar_write_failed")
        if not await self.sidecar.await_idle(timeout_seconds=90.0):
            raise RuntimeError("sidecar_not_idle")
        self.sidecar_dirty = False

    def _merge_sidecar_matches(
        self,
        selected: list[RetrievedMemory],
        effective: list[MemoryRecord],
        sidecar_text: str,
    ) -> list[RetrievedMemory]:
        existing = {item.record.record_id for item in selected}
        normalized = _norm(sidecar_text)
        merged = list(selected)
        for record in effective:
            if record.layer != MemoryLayer.EPISODIC or record.record_id in existing:
                continue
            value = _norm(record.payload.get("value") or record.payload.get("description"))
            if record.record_id.casefold() in normalized or (value and value in normalized):
                merged.append(RetrievedMemory(record, 0.7, 0.7, "sidecar matched sealed memory marker"))
        merged.sort(key=lambda item: (item.score, item.record.created_at), reverse=True)
        return merged[:8]

    def _record_observation(self, record: MemoryRecord, relevance: float | None) -> dict[str, Any]:
        evidence = record.payload.get("evidence_ids") or []
        if not isinstance(evidence, list):
            evidence = []
        if record.evidence_id and record.evidence_id not in evidence:
            evidence.append(record.evidence_id)
        item = {
            "memory_id": str(record.payload.get("canonical_memory_id") or record.record_id),
            "record_id": record.record_id,
            "user_id": record.user_id,
            "layer": record.layer.value,
            "field": str(record.payload.get("field") or ""),
            "value": str(record.payload.get("value") or record.payload.get("description") or ""),
            "status": record.status.value,
            "source": record.source,
            "evidence_ids": [str(value) for value in evidence],
            "scope": str(record.payload.get("scope") or "global"),
            "confidence": record.confidence,
            "created_at_ms": record.created_at,
            "expires_at_ms": record.expires_at,
            "backend": "local_ledger",
        }
        if relevance is not None:
            item["relevance"] = max(0.0, min(1.0, relevance))
        return item

    def _memory_observation(self, item: RetrievedMemory) -> dict[str, Any]:
        result = self._record_observation(item.record, item.relevance)
        result["why_used"] = item.why_used
        return result

    def _enabled_layers(self) -> set[MemoryLayer]:
        layers = {MemoryLayer.EXPLICIT, MemoryLayer.INFERRED}
        if self.arm in {"C", "D_mem0"}:
            layers.add(MemoryLayer.EPISODIC)
        return layers

    def _sidecar_user(self, user_id: str) -> str:
        return f"{self.namespace}:{user_id}"

    def _resolve_record_id(self, user_id: str, memory_id: str) -> str:
        if self.store is None:
            return memory_id
        for record in self.store.list_records(user_id=user_id, limit=1000):
            canonical = str(record.payload.get("canonical_memory_id") or record.record_id)
            if record.record_id == memory_id or canonical == memory_id:
                return record.record_id
        return memory_id

    def _require_bundle(self, payload: dict[str, Any]) -> None:
        if not self.bundle_id or str(payload.get("bundle_id") or "") != self.bundle_id:
            raise ValueError("bundle_mismatch")

    def _require_user(self, value: Any) -> str:
        user_id = str(value or "")
        if user_id not in self.user_ids:
            raise ValueError("unknown_user")
        return user_id


async def _run() -> int:
    if os.getenv("MEMORY_BENCHMARK_ISOLATED") == "1":
        os.environ["EVAL_DISABLE_SIDE_EFFECTS"] = "0"
    logging.basicConfig(stream=sys.stderr, level=logging.WARNING)
    worker = MemoryBenchmarkWorker()
    for raw_line in sys.stdin:
        line = raw_line.strip()
        if not line:
            continue
        request_id = "unknown"
        try:
            request = json.loads(line)
            request_id = str(request.get("request_id") or "unknown")
            if request.get("protocol_version") != PROTOCOL_VERSION:
                raise ValueError("protocol_version_mismatch")
            observation = await worker.dispatch(
                str(request.get("action") or ""),
                dict(request.get("payload") or {}),
            )
            response = {
                "protocol_version": PROTOCOL_VERSION,
                "request_id": request_id,
                "ok": True,
                "observation": observation,
            }
        except Exception as exc:
            response = {
                "protocol_version": PROTOCOL_VERSION,
                "request_id": request_id,
                "ok": False,
                "observation": {},
                "error": _stable_error(exc),
            }
        sys.stdout.write(json.dumps(response, ensure_ascii=False, separators=(",", ":")) + "\n")
        sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_run()))
