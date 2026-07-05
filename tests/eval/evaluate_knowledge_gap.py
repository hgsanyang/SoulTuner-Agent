"""Deterministic Catalog Gap + knowledge-store targeted eval.

This ruler is intentionally small and cheap.  It does not call Neo4j, web
search, Netease, LLMs, or vector models.  It validates the decision boundary:
local knowledge cards can satisfy era/style/background evidence, while true
inventory gaps still ask for online fallback.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent.catalog_gap import analyze_catalog_gap  # noqa: E402
from config.settings import settings  # noqa: E402
from services.music_knowledge_store import MusicKnowledgeStore  # noqa: E402


def _song(title: str, artist: str = "A", **extra: Any) -> dict[str, Any]:
    song = {"title": title, "artist": artist, "preview_url": "u"}
    song.update(extra)
    return {"song": song}


def _populate_store(path: Path) -> MusicKnowledgeStore:
    store = MusicKnowledgeStore(path)
    store.upsert_artist_card(
        artist="The Cure",
        summary="The Cure are an English post-punk and gothic rock band.",
        style_tags=["Post-Punk", "Gothic Rock"],
        facts=["Known for atmospheric guitar textures."],
        source_url="https://example.com/the-cure",
        confidence=0.9,
    )
    for index in range(6):
        store.upsert_song_card(
            title=f"Old Song {index}",
            artist="A",
            summary="A classic 1980s pop song.",
            release_year=1985,
            style_tags=["Classic Pop"],
            facts=["Originally released in the 1980s."],
            source_url=f"https://example.com/old-song-{index}",
            confidence=0.86,
        )
    store.upsert_song_card(
        title="恋曲1980",
        artist="罗大佑",
        summary="A Chinese classic pop/folk-rock song associated with the early 1980s.",
        release_year=1982,
        style_tags=["Chinese Pop", "Folk Rock"],
        facts=["Often discussed as a Chinese-language classic."],
        source_url="https://example.com/love-song-1980",
        confidence=0.88,
    )
    return store


def _case_result(case_id: str, passed: bool, detail: dict[str, Any]) -> dict[str, Any]:
    return {"id": case_id, "status": "pass" if passed else "fail", "detail": detail}


def run_eval() -> dict[str, Any]:
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        store = _populate_store(Path(tmp) / "knowledge.sqlite")
        settings.knowledge_store_path = str(store.path)
        settings.knowledge_gap_enabled = True
        settings.knowledge_gap_min_confidence = 0.55

        local_oldies = [_song(f"Old Song {index}") for index in range(12)]
        local_cure = [_song(f"Song {index}", artist="The Cure") for index in range(12)]

        rows: list[dict[str, Any]] = []

        decision = analyze_catalog_gap(local_oldies, {}, "推荐80年代的老歌", web_enabled=True)
        rows.append(
            _case_result(
                "era_request_uses_local_year_cards",
                "metadata_release_year_missing" not in decision.reasons,
                decision.model_dump(),
            )
        )

        decision = analyze_catalog_gap(local_cure, {}, "讲讲 The Cure 的歌手背景和风格", web_enabled=True)
        rows.append(
            _case_result(
                "artist_style_uses_local_knowledge",
                "external_knowledge_required" not in decision.reasons
                and decision.details["knowledge_evidence"]["query_hits"] > 0,
                decision.model_dump(),
            )
        )

        same_era = [_song("恋曲1980", artist="罗大佑"), *_song_list("Old Song", 11)]
        decision = analyze_catalog_gap(same_era, {}, "找几首和恋曲1980同年代的歌", web_enabled=True)
        rows.append(
            _case_result(
                "same_era_similarity_uses_song_knowledge",
                "metadata_release_year_missing" not in decision.reasons,
                decision.model_dump(),
            )
        )

        decision = analyze_catalog_gap([], {}, "找几首80年代粤语老歌", web_enabled=True)
        rows.append(
            _case_result(
                "inventory_gap_triggers_online_fallback",
                decision.action == "fallback" and "local_inventory_low" in decision.reasons,
                decision.model_dump(),
            )
        )

        hits = store.search("The Cure gothic rock", kind="artist", min_confidence=0.55)
        rows.append(
            _case_result(
                "knowledge_result_has_source_url",
                bool(hits and hits[0].get("source_url")),
                {"hits": hits[:1]},
            )
        )

    passed = sum(1 for row in rows if row["status"] == "pass")
    return {"total": len(rows), "passed": passed, "failed": len(rows) - passed, "cases": rows}


def _song_list(prefix: str, count: int) -> list[dict[str, Any]]:
    return [_song(f"{prefix} {index}") for index in range(count)]


def main() -> None:
    report = run_eval()
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report["failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
