"""Local presentation adapter; the API owns planning, memory and retrieval.

No model factory, catalog download or retry belongs in this client. Remote
multi-user transports need their own authenticated adapter, not arbitrary IDs.
"""
from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from urllib.parse import urlsplit

import httpx


class RecommendationClientError(RuntimeError):
    pass


def local_api_url(base_url: str) -> str:
    parsed = urlsplit(base_url)
    if (parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "::1"}
            or parsed.username or parsed.password or parsed.query or parsed.fragment
            or parsed.path not in {"", "/"}):
        raise ValueError("Only a literal loopback HTTP API address is supported")
    return base_url.rstrip("/") + "/api/recommendations/stream"


async def _bounded_lines(response: httpx.Response) -> AsyncIterator[str]:
    buffer = ""
    # Do not set httpx chunk_size: it coalesces small SSE events and delays songs.
    async for chunk in response.aiter_text():
        for offset in range(0, len(chunk), 8192):
            buffer += chunk[offset:offset + 8192]
            while "\n" in buffer:
                line, buffer = buffer.split("\n", 1)
                if len(line) > 1048576:
                    raise RecommendationClientError("Oversized recommendation event")
                yield line.rstrip("\r")
            if len(buffer) > 1048576:
                raise RecommendationClientError("Oversized recommendation event")
    if buffer:
        yield buffer.rstrip("\r")


async def recommend_events(
    query: str, *, base_url: str = "http://127.0.0.1:8000", session_id: str,
    profile_id: str = "local_admin", personal: bool = False, web_search_enabled: bool = False,
    device: str = "local-cli",
    chat_history: list[dict[str, str]] | None = None, dialog_state: dict | None = None,
    timeout_seconds: float = 180, transport: httpx.AsyncBaseTransport | None = None,
) -> AsyncIterator[dict]:
    """Preserve the server stream, including early songs and conversation state."""
    endpoint = local_api_url(base_url)
    if not query.strip() or len(query) > 8000:
        raise ValueError("Query must contain 1 to 8000 characters")
    if not session_id or len(session_id) > 200 or not profile_id or len(profile_id) > 200:
        raise ValueError("Invalid local session or profile")
    if not 0 < timeout_seconds <= 600:
        raise ValueError("Timeout must be between 0 and 600 seconds")
    if device not in {"local-cli", "local-mcp", "local-gradio"}:
        raise ValueError("Invalid local presentation device")
    payload = {
        "query": query, "session_id": session_id, "profile_id": profile_id,
        "interaction_mode": "personal" if personal else "developer",
        "web_search_enabled": web_search_enabled, "chat_history": chat_history or [],
        "dialog_state": dialog_state or {}, "device": device,
    }
    encoded = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    if len(encoded) > 65536:
        raise ValueError("Request exceeds 64 KiB")
    completed = False
    try:
        async with asyncio.timeout(timeout_seconds):
            async with httpx.AsyncClient(transport=transport, trust_env=False, follow_redirects=False,
                                         timeout=httpx.Timeout(30, connect=5, read=timeout_seconds)) as client:
                async with client.stream("POST", endpoint, content=encoded,
                                         headers={"Content-Type": "application/json", "Accept": "text/event-stream"}) as response:
                    if response.status_code != 200:
                        raise RecommendationClientError(f"Recommendation API returned HTTP {response.status_code}")
                    if "text/event-stream" not in response.headers.get("content-type", ""):
                        raise RecommendationClientError("Recommendation API did not return an event stream")
                    parts: list[str] = []
                    size = 0
                    async for line in _bounded_lines(response):
                        if len(line) > 1048576:
                            raise RecommendationClientError("Oversized recommendation event")
                        if line.startswith("data:"):
                            value = line[5:].lstrip(" ")
                            size += len(value)
                            if size > 1048576:
                                raise RecommendationClientError("Oversized recommendation event")
                            parts.append(value)
                        elif not line and parts:
                            event = json.loads("\n".join(parts))
                            parts, size = [], 0
                            if not isinstance(event, dict) or not isinstance(event.get("type"), str):
                                raise RecommendationClientError("Invalid recommendation event")
                            if event["type"] == "error":
                                raise RecommendationClientError("Recommendation did not complete; earlier songs are partial results")
                            if event["type"] == "complete":
                                if event.get("success") is not True:
                                    raise RecommendationClientError("Recommendation was not confirmed successful")
                                completed = True
                            yield event
        if not completed:
            raise RecommendationClientError("Recommendation stream ended before completion")
    except (httpx.HTTPError, TimeoutError, json.JSONDecodeError) as exc:
        raise RecommendationClientError("Recommendation connection, timeout or event decoding failed") from exc
