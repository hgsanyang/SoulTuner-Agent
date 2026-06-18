"""Helpers for validating structured planner responses."""

from __future__ import annotations

from schemas.query_plan import MusicQueryPlan


def clean_json_response(content: str) -> str:
    """Remove common model wrappers while preserving the JSON object."""
    cleaned = (content or "").strip()
    if "<think>" in cleaned:
        think_end = cleaned.find("</think>")
        if think_end >= 0:
            cleaned = cleaned[think_end + len("</think>"):].strip()

    if "```json" in cleaned:
        cleaned = cleaned.split("```json", 1)[1].split("```", 1)[0].strip()
    elif cleaned.startswith("```"):
        cleaned = cleaned.split("```", 1)[1].split("```", 1)[0].strip()
    return cleaned


def parse_music_query_plan(content: str) -> MusicQueryPlan:
    return MusicQueryPlan.model_validate_json(clean_json_response(content))
