"""Request-scoped, application-issued memory capability; never issued by a model.

The application must approve the exact delta (e.g. user confirmation or its
validated consolidation policy) and inject its commit callback. This module
does not turn an existing evidence ID or model confidence into authorization.
"""
from __future__ import annotations

import asyncio
import copy
import inspect
import time

from schemas.tool_plan import CommitMemoryArguments
from services.memory_models import MemoryLayer
from services.runtime_mode import side_effects_disabled


class AuthorizedMemoryWriter:
    def __init__(self, *, user_id, approved_delta, evidence_ids, event_store, commit, lifetime_seconds=60):
        if not 0 < lifetime_seconds <= 300:
            raise ValueError("memory authorization lifetime must be within 300 seconds")
        self._expires_at = time.monotonic() + lifetime_seconds
        self._user_id = user_id
        self._approved = CommitMemoryArguments.model_validate(copy.deepcopy(approved_delta)).model_dump(mode="json")
        self._evidence_ids = tuple(dict.fromkeys(evidence_ids))
        if not self._evidence_ids or len(self._evidence_ids) > 20:
            raise ValueError("authorization requires bounded server-selected evidence")
        if self._approved["evidence_id"] not in self._evidence_ids:
            raise ValueError("primary evidence is not in the approved evidence set")
        self._store = event_store
        self._commit = commit
        self._lock = asyncio.Lock()
        self._state = "ready"
        self._receipt = None

    def _evidence_valid(self):
        for record_id in self._evidence_ids:
            record = self._store.get(user_id=self._user_id, record_id=record_id)
            if record is None or record.layer != MemoryLayer.RAW_EVENT:
                return False
            if record.source not in {"user_statement", "user_action", "slate_feedback", "user_confirmation"}:
                return False
            if not self._store.is_effective_record(user_id=self._user_id, record_id=record_id):
                return False
        return True

    async def __call__(self, user_id, arguments):
        denied = {"success": False, "error": "memory write not authorized"}
        if side_effects_disabled() or user_id != self._user_id or time.monotonic() >= self._expires_at:
            return denied
        try:
            delta = CommitMemoryArguments.model_validate(arguments).model_dump(mode="json")
        except (TypeError, ValueError):
            return denied
        if delta != self._approved:
            return denied
        async with self._lock:
            if time.monotonic() >= self._expires_at:
                return denied
            if self._state == "completed":
                return copy.deepcopy(self._receipt)
            if self._state != "ready":
                return {"success": False, "error": "memory write outcome requires reconciliation"}
            if not await asyncio.to_thread(self._evidence_valid):
                return denied
            # A timeout/cancellation may follow a committed external write.
            # Never blindly invoke this capability again in that uncertain state.
            self._state = "started"
            try:
                if inspect.iscoroutinefunction(self._commit):
                    result = await self._commit(user_id, copy.deepcopy(delta))
                else:
                    result = await asyncio.to_thread(self._commit, user_id, copy.deepcopy(delta))
                    if inspect.isawaitable(result):
                        result = await result
                if not isinstance(result, dict) or result.get("success") is not True:
                    raise RuntimeError("write not confirmed")
            except asyncio.CancelledError:
                raise
            except Exception:
                return {"success": False, "error": "memory write outcome requires reconciliation"}
            self._state = "completed"
            self._receipt = copy.deepcopy(result)
            return copy.deepcopy(self._receipt)
