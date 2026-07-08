"""Repair small online-acquired metadata gaps.

This is intentionally targeted: it only touches files in data/online_acquired
and does not bulk-backfill the whole catalog.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.p11_prepare_online_ingest import build_song_from_meta, expected_basename  # noqa: E402
from services.music_knowledge_enrichment import enrich_song_cards_batch  # noqa: E402
from services.music_knowledge_store import MusicKnowledgeStore  # noqa: E402

DEFAULT_ONLINE_ROOT = PROJECT_ROOT.parent / "data" / "online_acquired"


def _load_meta(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _write_meta(path: Path, meta: dict[str, Any]) -> None:
    path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


def _has_audio(meta: dict[str, Any], root: Path) -> bool:
    song = build_song_from_meta(meta, root)
    return bool(song.get("has_audio"))


async def repair_online_acquired(root: Path, *, batch_size: int = 3, dry_run: bool = False) -> dict[str, Any]:
    meta_dir = root / "metadata"
    paths = sorted(meta_dir.glob("*_meta.json")) if meta_dir.exists() else []
    missing_year: list[tuple[Path, dict[str, Any]]] = []
    missing_audio: list[tuple[Path, dict[str, Any]]] = []
    for path in paths:
        meta = _load_meta(path)
        if not meta:
            continue
        if not meta.get("release_year"):
            missing_year.append((path, meta))
        if not _has_audio(meta, root):
            missing_audio.append((path, meta))

    updated_year = 0
    cards_total = 0
    store = MusicKnowledgeStore()
    for idx in range(0, len(missing_year), max(1, batch_size)):
        chunk = missing_year[idx : idx + max(1, batch_size)]
        songs = []
        for _, meta in chunk:
            song = build_song_from_meta(meta, root)
            songs.append({"title": song.get("title", ""), "artist": song.get("artist", "")})
        cards = await enrich_song_cards_batch(songs, store=store, dry_run=dry_run, use_llm_summary=True)
        cards_total += len(cards)
        by_key = {
            (str(card.get("title") or "").casefold(), str(card.get("artist") or "").casefold()): card
            for card in cards
            if card.get("release_year")
        }
        for path, meta in chunk:
            song = build_song_from_meta(meta, root)
            card = by_key.get((str(song.get("title") or "").casefold(), str(song.get("artist") or "").casefold()))
            if not card:
                continue
            try:
                year = int(card.get("release_year") or 0)
            except (TypeError, ValueError):
                continue
            if not 1900 <= year <= 2100:
                continue
            meta["release_year"] = year
            meta["release_year_source"] = "qwen_knowledge_card"
            meta["release_year_source_url"] = card.get("source_url", "")
            if not dry_run:
                _write_meta(path, meta)
            updated_year += 1

    marked_missing_audio = 0
    for path, meta in missing_audio:
        meta["acquire_status"] = "failed"
        meta["acquire_error"] = "missing audio file; keep metadata only until a playable source is acquired"
        meta["missing_audio"] = True
        meta["file_basename"] = expected_basename(meta)
        if not dry_run:
            _write_meta(path, meta)
        marked_missing_audio += 1

    return {
        "root": str(root),
        "metadata_files": len(paths),
        "missing_year_before": len(missing_year),
        "knowledge_cards_returned": cards_total,
        "release_year_updated": updated_year,
        "missing_audio_marked_failed": marked_missing_audio,
        "dry_run": dry_run,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Repair targeted online-acquired metadata gaps.")
    parser.add_argument("--online-root", default=str(DEFAULT_ONLINE_ROOT))
    parser.add_argument("--batch-size", type=int, default=3)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    print(json.dumps(
        asyncio.run(repair_online_acquired(Path(args.online_root), batch_size=args.batch_size, dry_run=args.dry_run)),
        ensure_ascii=False,
        indent=2,
    ))


if __name__ == "__main__":
    main()
