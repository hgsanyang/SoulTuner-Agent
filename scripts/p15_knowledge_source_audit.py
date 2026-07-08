"""Audit local music knowledge-card source quality.

This script is read-only and offline. It inspects the SQLite knowledge store
and summarizes evidence sources so enrichment quality can be tracked without
calling any web API.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import json
from pathlib import Path
import sqlite3
import sys
from urllib.parse import urlparse

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from services.music_knowledge_store import resolve_store_path  # noqa: E402


HIGH_QUALITY_HOSTS = {
    "music.apple.com",
    "musicbrainz.org",
    "discogs.com",
    "www.discogs.com",
    "wikipedia.org",
    "en.wikipedia.org",
    "zh.wikipedia.org",
    "bandcamp.com",
    "soundcloud.com",
    "youtube.com",
    "www.youtube.com",
}
MEDIUM_QUALITY_HOSTS = {
    "shazam.com",
    "www.shazam.com",
    "genius.com",
    "baike.baidu.com",
    "namu.wiki",
    "kkbox.com",
    "www.kkbox.com",
    "streetvoice.cn",
    "dashi.streetvoice.cn",
}
LOW_QUALITY_HOSTS = {
    "open.spotify.com",
    "kugeci.com",
    "www.kugeci.com",
}


def _host(url: str) -> str:
    try:
        return urlparse(str(url or "")).netloc.casefold()
    except ValueError:
        return ""


def _quality_for_host(host: str) -> str:
    if not host:
        return "missing"
    if host in HIGH_QUALITY_HOSTS or any(host.endswith("." + item) for item in HIGH_QUALITY_HOSTS):
        return "high"
    if host in MEDIUM_QUALITY_HOSTS or any(host.endswith("." + item) for item in MEDIUM_QUALITY_HOSTS):
        return "medium"
    if host in LOW_QUALITY_HOSTS or any(host.endswith("." + item) for item in LOW_QUALITY_HOSTS):
        return "low"
    return "unknown"


def _rows(conn: sqlite3.Connection, table: str) -> list[sqlite3.Row]:
    return conn.execute(
        f"""
        SELECT '{table}' AS table_name, c.*, s.url AS source_url, s.provider AS source_provider
        FROM {table} c
        LEFT JOIN sources s ON s.id = c.source_id
        """
    ).fetchall()


def build_report(path: str | Path | None = None) -> dict:
    store_path = resolve_store_path(path)
    conn = sqlite3.connect(store_path)
    conn.row_factory = sqlite3.Row
    try:
        rows = [*_rows(conn, "song_cards"), *_rows(conn, "artist_cards")]
    finally:
        conn.close()

    host_counter: Counter[str] = Counter()
    quality_counter: Counter[str] = Counter()
    provider_counter: Counter[str] = Counter()
    by_kind: dict[str, Counter[str]] = defaultdict(Counter)
    release_year_count = 0
    confidence_sum = 0.0
    low_confidence_examples = []
    missing_source_examples = []

    for row in rows:
        kind = "song" if row["table_name"] == "song_cards" else "artist"
        host = _host(row["source_url"])
        quality = _quality_for_host(host)
        provider = str(row["source_provider"] or "unknown")
        host_counter[host or "missing"] += 1
        quality_counter[quality] += 1
        provider_counter[provider] += 1
        by_kind[kind][quality] += 1
        try:
            confidence = float(row["confidence"] or 0)
        except (TypeError, ValueError):
            confidence = 0.0
        confidence_sum += confidence
        if row["table_name"] == "song_cards" and row["release_year"]:
            release_year_count += 1
        if confidence < 0.65 and len(low_confidence_examples) < 10:
            low_confidence_examples.append(
                {
                    "kind": kind,
                    "title": row["title"] if "title" in row.keys() else "",
                    "artist": row["artist"] if "artist" in row.keys() else "",
                    "confidence": round(confidence, 3),
                    "source_url": row["source_url"],
                }
            )
        if not host and len(missing_source_examples) < 10:
            missing_source_examples.append(
                {
                    "kind": kind,
                    "title": row["title"] if "title" in row.keys() else "",
                    "artist": row["artist"] if "artist" in row.keys() else "",
                }
            )

    total = len(rows)
    return {
        "store_path": str(store_path),
        "total_cards": total,
        "song_cards": sum(1 for row in rows if row["table_name"] == "song_cards"),
        "artist_cards": sum(1 for row in rows if row["table_name"] == "artist_cards"),
        "song_release_year_cards": release_year_count,
        "mean_confidence": round(confidence_sum / total, 3) if total else 0.0,
        "quality_breakdown": dict(quality_counter.most_common()),
        "quality_by_kind": {kind: dict(counter.most_common()) for kind, counter in by_kind.items()},
        "top_hosts": [{"host": host, "n": n} for host, n in host_counter.most_common(30)],
        "providers": dict(provider_counter.most_common()),
        "low_confidence_examples": low_confidence_examples,
        "missing_source_examples": missing_source_examples,
    }


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    report = build_report()
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
