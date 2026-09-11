"""Owner-scoped anonymous memory actions, separate from local admin routes."""
import asyncio

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field

router = APIRouter()


class ForgetRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")
    record_id: str = Field(min_length=1, max_length=200)


class ClearLearned(BaseModel):
    model_config = ConfigDict(extra="forbid")


@router.post("/api/visitor/memory/retry-writes")
async def retry_writes(payload: ClearLearned, request: Request):
    subject = getattr(request.state, "anonymous_subject", None)
    if not subject:
        raise HTTPException(401, "Anonymous session required")
    from services.memory_gateway import get_memory_gateway
    try:
        result = await asyncio.to_thread(get_memory_gateway().retry_pending_writes, user_id=subject)
        if not result["success"]:
            raise RuntimeError("write_not_confirmed")
    except Exception:
        raise HTTPException(503, "Memory write recovery unavailable") from None
    return {"success": True}


@router.post("/api/visitor/memory/retry-deletions")
async def retry_deletions(payload: ClearLearned, request: Request):
    subject = getattr(request.state, "anonymous_subject", None)
    if not subject:
        raise HTTPException(401, "Anonymous session required")
    from services.memory_gateway import get_memory_gateway
    gateway = get_memory_gateway()
    try:
        result = await asyncio.to_thread(gateway.retry_pending_deletions, user_id=subject, limit=20)
        remaining = await asyncio.to_thread(gateway.pending_deletion_count, user_id=subject)
    except Exception:
        raise HTTPException(503, "Memory deletion recovery unavailable") from None
    return {"success": remaining == 0, **result, "remaining": remaining}


@router.post("/api/visitor/memory/clear-learned")
async def clear_learned(payload: ClearLearned, request: Request):
    subject = getattr(request.state, "anonymous_subject", None)
    if not subject:
        raise HTTPException(401, "Anonymous session required")
    from services.memory_gateway import get_memory_gateway
    try:
        ok = await asyncio.to_thread(get_memory_gateway().clear_learned_preferences, user_id=subject)
    except Exception:
        raise HTTPException(503, "Memory deletion unavailable") from None
    if not ok:
        raise HTTPException(503, "Memory deletion unavailable")
    return {"success": True, "deletion": "inferred_preferences_tombstone", "audit_history_retained": True}


@router.post("/api/visitor/memory/forget")
async def forget_record(payload: ForgetRecord, request: Request):
    subject = getattr(request.state, "anonymous_subject", None)
    if not subject:
        raise HTTPException(401, "Anonymous session required")
    from services.memory_gateway import get_memory_gateway

    try:
        ok = await asyncio.to_thread(
            get_memory_gateway().delete_memory_record,
            user_id=subject, record_id=payload.record_id,
        )
    except Exception:
        raise HTTPException(503, "Memory deletion unavailable") from None
    if not ok:
        raise HTTPException(404, "Memory record not found")
    return {"success": True, "deletion": "tombstone", "audit_history_retained": True}
