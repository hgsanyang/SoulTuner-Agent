"""Stable conversation identity without importing the full agent runtime."""

import hashlib
import uuid
from typing import Any, Dict, Optional


def resolve_conversation_id(client_context: Optional[Dict[str, Any]]) -> str:
    """Return the stable client session id or a request-scoped fallback."""
    supplied = str((client_context or {}).get("session_id") or "").strip()
    return supplied or f"ephemeral-{uuid.uuid4()}"


def checkpoint_thread_id(user_id: str, conversation_id: str) -> str:
    """Derive a stable, non-reversible LangGraph checkpoint key."""
    material = f"{user_id.strip()}\0{conversation_id.strip()}".encode("utf-8")
    return "conversation-" + hashlib.sha256(material).hexdigest()[:32]
