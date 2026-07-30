#!/usr/bin/env python
"""Find Song nodes whose audio_url points at a file that is not there.

A dangling node is worse than a missing one: it is returned by recall, rendered
as playable, and fails only when the user presses play. It looks like a broken
player rather than absent data.

Two outcomes, never a third:

* another Song with the same normalised title+artist has a real file → repoint
  audio_url at it and mark both ``possible_duplicate``, because two entities
  sharing one recording is a merge question a human resolves, not something to
  settle by deleting one;
* no such asset anywhere → ``audio_status='missing'``, which drops it out of
  playable recall while keeping the knowledge entity, its tags and its vectors.

The Song node is never deleted. It carries genre/mood/scenario relationships and
embeddings that remain valid even when the audio does not exist.

    python scripts/repair_dangling_audio.py                  # dry run
    python scripts/repair_dangling_audio.py --apply
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from services.audio_format import detect_file, is_lossless, mime_for  # noqa: E402

#: URL prefix -> directory under the data root.
STATIC_ROOTS: dict[str, str] = {
    "/static/audio/": "processed_audio/audio",
    "/static/online_audio/": "online_acquired/audio",
    "/static/mtg/": "mtg_sample/audio",
}


@dataclass
class Dangling:
    eid: str
    title: str
    artist: str
    url: str
    donor_url: str = ""
    donor_path: str = ""
    action: str = "missing"


def resolve(url: str, data_root: Path) -> Path | None:
    """Map a /static/... URL to a path on disk, or None if we cannot."""
    for prefix, relative in STATIC_ROOTS.items():
        if url.startswith(prefix):
            return data_root / relative / url[len(prefix):]
    return None


def find_dangling(rows: list[dict], data_root: Path) -> tuple[list[dict], list[dict]]:
    """Split rows into (present, dangling). Unmappable URLs count as present.

    An unrecognised prefix means this script does not know where that lives —
    calling it missing on that basis would be a guess dressed as a finding.
    """
    present, missing = [], []
    for row in rows:
        url = str(row.get("url") or "")
        if not url:
            continue
        path = resolve(url, data_root)
        if path is None:
            present.append(row)
            continue
        (present if path.exists() else missing).append(row)
    return present, missing


def plan(rows: list[dict], data_root: Path) -> list[Dangling]:
    from services.negative_feedback import song_key

    present, missing = find_dangling(rows, data_root)

    donors: dict[str, dict] = {}
    for row in present:
        key = song_key(row.get("title"), row.get("artist"))
        path = resolve(str(row.get("url") or ""), data_root)
        if path is not None and path.exists():
            donors.setdefault(key, {"url": str(row["url"]), "path": str(path)})

    planned: list[Dangling] = []
    for row in missing:
        key = song_key(row.get("title"), row.get("artist"))
        donor = donors.get(key)
        item = Dangling(
            eid=str(row["eid"]),
            title=str(row.get("title") or ""),
            artist=str(row.get("artist") or ""),
            url=str(row.get("url") or ""),
        )
        if donor:
            item.donor_url = donor["url"]
            item.donor_path = donor["path"]
            item.action = "repoint"
        planned.append(item)
    return planned


def _query(client) -> list[dict]:
    return [
        dict(row)
        for row in client.execute_query(
            """
            MATCH (s:Song) WHERE s.audio_url IS NOT NULL AND s.audio_url <> ''
            OPTIONAL MATCH (s)-[:PERFORMED_BY]->(a:Artist)
            RETURN elementId(s) AS eid, s.title AS title,
                   coalesce(a.name, s.artist, '') AS artist,
                   s.audio_url AS url,
                   coalesce(s.audio_status, '') AS audio_status
            """,
            {},
        )
    ]


def apply(planned: list[Dangling], client) -> dict[str, int]:
    counts = {"repointed": 0, "marked_missing": 0}
    for item in planned:
        if item.action == "repoint":
            container = detect_file(item.donor_path)
            client.execute_query(
                """
                MATCH (s:Song) WHERE elementId(s) = $eid
                SET s.audio_url = $url,
                    s.format = $container,
                    s.codec = $container,
                    s.mime_type = $mime,
                    s.lossless = $lossless,
                    s.audio_status = 'shared_source',
                    s.possible_duplicate = true,
                    s.updated_at = timestamp()
                """,
                {
                    "eid": item.eid,
                    "url": item.donor_url,
                    "container": container,
                    "mime": mime_for(container),
                    "lossless": is_lossless(container),
                },
            )
            # The donor is equally part of the duplicate pair.
            client.execute_query(
                """
                MATCH (s:Song) WHERE s.audio_url = $url
                SET s.possible_duplicate = true, s.updated_at = timestamp()
                """,
                {"url": item.donor_url},
            )
            counts["repointed"] += 1
        else:
            client.execute_query(
                """
                MATCH (s:Song) WHERE elementId(s) = $eid
                SET s.audio_status = 'missing', s.updated_at = timestamp()
                """,
                {"eid": item.eid},
            )
            counts["marked_missing"] += 1
    return counts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default=os.environ.get("MUSIC_DATA_PATH", "D:/SoulTunerData"))
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    from retrieval.neo4j_client import get_neo4j_client

    client = get_neo4j_client()
    rows = _query(client)
    planned = plan(rows, Path(args.data_root))

    print(f"检查了 {len(rows)} 个带 audio_url 的节点，发现 {len(planned)} 个指向不存在的文件")
    for item in planned:
        print(f"  [{item.action}] {item.title} - {item.artist}")
        print(f"      当前: {item.url}")
        if item.action == "repoint":
            print(f"      改指: {item.donor_url}  (同名同歌手且文件真实存在)")
        else:
            print("      没有同名同歌手的真实音频 → audio_status=missing，退出可播放召回")

    if not planned:
        return 0
    if not args.apply:
        print("\n干跑。加 --apply 执行。Song 实体不会被删除。")
        return 0

    counts = apply(planned, client)
    print(f"\n完成: 改指 {counts['repointed']}，标记 missing {counts['marked_missing']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
