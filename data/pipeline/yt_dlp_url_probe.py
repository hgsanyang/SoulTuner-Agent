"""Probe yt-dlp candidate URLs for wishlist rows without downloading media.

This script is for manual review only. It uses yt-dlp search metadata to find
candidate URLs, scores title/artist/duration fit, and writes a JSON report.
It does not download audio and does not decide licensing. A candidate URL is not
treated as ingestable until the user confirms the source permits download/use.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.pipeline.netease_wishlist_acquire import _parse_indexes, parse_wishlist
from data.pipeline.local_download_flywheel import TAG_DIR

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


def _norm(text: str) -> str:
    text = str(text or "").lower()
    text = re.sub(r"\([^)]*\)|（[^）]*）", "", text)
    return re.sub(r"[\W_]+", "", text, flags=re.UNICODE)


def _score(row: dict[str, Any], candidate: dict[str, Any]) -> float:
    title = candidate.get("title") or ""
    uploader = candidate.get("uploader") or candidate.get("channel") or ""
    title_score = SequenceMatcher(None, _norm(row["title"]), _norm(title)).ratio()
    artist_scores = []
    for artist in row.get("artists") or []:
        an = _norm(artist)
        haystack = _norm(f"{title} {uploader}")
        if an and an in haystack:
            artist_scores.append(1.0)
        elif an:
            artist_scores.append(SequenceMatcher(None, an, haystack).ratio())
    artist_score = max(artist_scores) if artist_scores else 0.0
    return round(title_score * 0.72 + artist_score * 0.28, 4)


def _is_likely_official(candidate: dict[str, Any]) -> bool:
    uploader = str(candidate.get("uploader") or candidate.get("channel") or "")
    title = str(candidate.get("title") or "")
    lowered = f"{uploader} {title}".lower()
    return (
        uploader.endswith(" - Topic")
        or "official" in lowered
        or "provided to youtube by" in str(candidate.get("description") or "").lower()
    )


def _run_ytsearch(query: str, limit: int) -> list[dict[str, Any]]:
    cmd = [
        "yt-dlp",
        "--quiet",
        "--no-warnings",
        "--skip-download",
        "--dump-single-json",
        "--flat-playlist",
        f"ytsearch{limit}:{query}",
    ]
    proc = subprocess.run(cmd, text=True, encoding="utf-8", capture_output=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or f"yt-dlp exited {proc.returncode}")
    data = json.loads(proc.stdout)
    return [entry for entry in data.get("entries", []) if isinstance(entry, dict)]


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe yt-dlp candidate URLs without downloading.")
    parser.add_argument("--wishlist", required=True)
    parser.add_argument("--indexes", required=True, help="Comma-separated wishlist indexes/ranges.")
    parser.add_argument("--limit", type=int, default=5)
    args = parser.parse_args()

    selected = _parse_indexes(args.indexes)
    rows = [row for row in parse_wishlist(Path(args.wishlist)) if row["index"] in selected]
    results = []
    for i, row in enumerate(rows, 1):
        query = f"{row['title']} {row['artist_text']}"
        print(f"[{i}/{len(rows)}] probe: {query}")
        try:
            entries = _run_ytsearch(query, args.limit)
            candidates = []
            for entry in entries:
                candidates.append(
                    {
                        "url": entry.get("url"),
                        "title": entry.get("title"),
                        "uploader": entry.get("uploader") or entry.get("channel"),
                        "duration": entry.get("duration"),
                        "view_count": entry.get("view_count"),
                        "thumbnail": (entry.get("thumbnails") or [{}])[-1].get("url"),
                        "score": _score(row, entry),
                        "likely_official": _is_likely_official(entry),
                        "ingestable_without_review": False,
                    }
                )
            candidates.sort(key=lambda c: (c["likely_official"], c["score"]), reverse=True)
            status = "candidate_found" if candidates else "no_candidate"
            results.append({**row, "status": status, "candidates": candidates})
        except Exception as exc:
            results.append({**row, "status": "error", "error": str(exc), "candidates": []})

    TAG_DIR.mkdir(parents=True, exist_ok=True)
    out = TAG_DIR / f"yt_dlp_url_probe_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Report: {out}")


if __name__ == "__main__":
    main()
