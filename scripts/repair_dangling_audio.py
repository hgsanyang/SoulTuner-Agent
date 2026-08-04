#!/usr/bin/env python
"""Find Song nodes whose audio_url points at a file that is not there.

A dangling node is worse than an absent one: recall returns it, the UI renders it
as playable, and it fails only when the user presses play — so it reads as a
broken player rather than as missing data. That is why these went unnoticed.

Two outcomes, decided by whether the recording exists anywhere:

* another Song with the same title **and the same full artist set** has a real
  file → repoint audio_url at it, and mark both ``possible_duplicate``. Two
  entities sharing one recording is a merge question a human resolves, not
  something to settle by deleting one of them.
* nothing anywhere → preserve the URL in ``previous_audio_url``, clear
  ``audio_url`` and set ``unplayable_stub``.

That last combination is what actually removes a node from playable recall.
``audio_status='missing'`` alone does not:
``retrieval/recall_sources.py::_playable_song_where`` filters on
``unplayable_stub`` and a non-empty ``audio_url`` and never reads
``audio_status``, so the label on its own changes nothing the user sees.

The Song node is never deleted. It carries genre/mood/scenario relationships and
embeddings that stay valid whether or not the audio exists.

Every ``--apply`` stamps a ``repair_run_id`` and records the three fields it
overwrote, so ``--rollback <run-id>`` restores exactly that batch and nothing
else.

    python scripts/repair_dangling_audio.py                    # dry run
    python scripts/repair_dangling_audio.py --apply
    python scripts/repair_dangling_audio.py --rollback <run-id>
    python scripts/repair_dangling_audio.py --list-runs
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import uuid
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from services.audio_format import detect_file, is_lossless, mime_for  # noqa: E402

#: URL prefix -> directory under the data root.
STATIC_ROOTS: dict[str, str] = {
    "/static/audio/": "processed_audio/audio",
    "/static/online_audio/": "online_acquired/audio",
    "/static/mtg/": "mtg_sample/audio",
}

_NORMALISE = re.compile(r"[\s\-_·・,，.。'\"“”‘’()（）\[\]【】!！?？&+/\\|]+")


@dataclass
class Dangling:
    eid: str
    title: str
    artists: list[str] = field(default_factory=list)
    url: str = ""
    donor_url: str = ""
    donor_path: str = ""
    action: str = "missing"

    @property
    def artist_display(self) -> str:
        return "、".join(self.artists) if self.artists else ""


def identity_key(title: str, artists: list[str] | None) -> str:
    """Normalised title + the *complete* artist set, order-independent.

    Using only the first artist would merge two different recordings that share
    a title and a lead performer but not their collaborators — a remix credited
    to "A" and one credited to "A feat. B" are not the same audio, and sharing a
    file between them silently serves the wrong track.
    """
    names = sorted(_NORMALISE.sub("", str(a or "")).casefold() for a in (artists or []) if str(a or "").strip())
    return f"{_NORMALISE.sub('', str(title or '')).casefold()}|{'|'.join(names)}"


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


def _artists_of(row: dict) -> list[str]:
    names = [str(a).strip() for a in (row.get("artists") or []) if str(a or "").strip()]
    if names:
        return names
    single = str(row.get("artist") or "").strip()
    return [single] if single else []


def plan(rows: list[dict], data_root: Path) -> list[Dangling]:
    present, missing = find_dangling(rows, data_root)

    donors: dict[str, dict] = {}
    for row in present:
        path = resolve(str(row.get("url") or ""), data_root)
        if path is None or not path.exists():
            continue
        key = identity_key(row.get("title", ""), _artists_of(row))
        donors.setdefault(key, {"url": str(row["url"]), "path": str(path)})

    planned: list[Dangling] = []
    seen: set[str] = set()
    for row in missing:
        eid = str(row["eid"])
        if eid in seen:          # the query aggregates, but a caller may not
            continue
        seen.add(eid)
        artists = _artists_of(row)
        item = Dangling(
            eid=eid,
            title=str(row.get("title") or ""),
            artists=artists,
            url=str(row.get("url") or ""),
        )
        donor = donors.get(identity_key(item.title, artists))
        if donor:
            item.donor_url = donor["url"]
            item.donor_path = donor["path"]
            item.action = "repoint"
        planned.append(item)
    return planned


def query_nodes(client) -> list[dict]:
    """One row per Song, not one per artist.

    PERFORMED_BY is many-to-one, so the obvious OPTIONAL MATCH turns a
    three-artist track into three rows. Aggregating first keeps "how many nodes"
    an answerable question, and keeps the full artist list for identity.
    """
    return [
        dict(row)
        for row in client.execute_query(
            """
            MATCH (s:Song) WHERE s.audio_url IS NOT NULL AND s.audio_url <> ''
            OPTIONAL MATCH (s)-[:PERFORMED_BY]->(a:Artist)
            WITH s, [n IN collect(DISTINCT a.name) WHERE n IS NOT NULL] AS artist_names
            RETURN elementId(s) AS eid,
                   s.title AS title,
                   artist_names AS artists,
                   coalesce(s.artist, '') AS artist,
                   s.audio_url AS url,
                   coalesce(toString(s.music_id), '') AS music_id
            """,
            {},
        )
    ]


def node_counts(client) -> dict[str, int]:
    """Node counts, reported as node counts."""
    def one(query: str) -> int:
        rows = client.execute_query(query, {})
        return int(rows[0]["n"]) if rows else 0

    return {
        "songs": one("MATCH (s:Song) RETURN count(s) AS n"),
        "distinct_music_id": one(
            "MATCH (s:Song) WHERE s.music_id IS NOT NULL "
            "RETURN count(DISTINCT toString(s.music_id)) AS n"
        ),
        "with_audio_url": one(
            "MATCH (s:Song) WHERE s.audio_url IS NOT NULL AND s.audio_url <> '' "
            "RETURN count(s) AS n"
        ),
        "expanded_rows": one(
            "MATCH (s:Song) WHERE s.audio_url IS NOT NULL AND s.audio_url <> '' "
            "OPTIONAL MATCH (s)-[:PERFORMED_BY]->(a:Artist) RETURN count(*) AS n"
        ),
        "still_recallable_missing": one(
            "MATCH (s:Song) WHERE s.audio_status = 'missing' "
            "AND coalesce(properties(s)['unplayable_stub'], false) <> true "
            "AND s.audio_url IS NOT NULL AND trim(toString(s.audio_url)) <> '' "
            "RETURN count(s) AS n"
        ),
    }


def apply(planned: list[Dangling], client, *, run_id: str) -> dict[str, int]:
    """Write the plan, recording enough to undo exactly this batch."""
    tally = {"repointed": 0, "marked_missing": 0}
    for item in planned:
        if item.action == "repoint":
            container = detect_file(item.donor_path)
            client.execute_query(
                """
                MATCH (s:Song) WHERE elementId(s) = $eid
                SET s.previous_audio_url = coalesce(s.previous_audio_url, s.audio_url),
                    s.previous_audio_status = coalesce(s.previous_audio_status, coalesce(s.audio_status, '')),
                    s.previous_unplayable_stub = coalesce(s.previous_unplayable_stub,
                                                          coalesce(properties(s)['unplayable_stub'], false)),
                    s.repair_run_id = $run_id,
                    s.audio_url = $url,
                    s.format = $container,
                    s.codec = $container,
                    s.mime_type = $mime,
                    s.lossless = $lossless,
                    s.audio_status = 'shared_source',
                    s.possible_duplicate = true,
                    s.updated_at = timestamp()
                """,
                {
                    "eid": item.eid, "run_id": run_id, "url": item.donor_url,
                    "container": container, "mime": mime_for(container),
                    "lossless": is_lossless(container),
                },
            )
            client.execute_query(
                """
                MATCH (s:Song) WHERE s.audio_url = $url
                SET s.possible_duplicate = true, s.updated_at = timestamp()
                """,
                {"url": item.donor_url},
            )
            tally["repointed"] += 1
        else:
            client.execute_query(
                """
                MATCH (s:Song) WHERE elementId(s) = $eid
                SET s.previous_audio_url = coalesce(s.previous_audio_url, s.audio_url),
                    s.previous_audio_status = coalesce(s.previous_audio_status, coalesce(s.audio_status, '')),
                    s.previous_unplayable_stub = coalesce(s.previous_unplayable_stub,
                                                          coalesce(properties(s)['unplayable_stub'], false)),
                    s.repair_run_id = $run_id,
                    s.audio_url = '',
                    s.audio_status = 'missing',
                    s.unplayable_stub = true,
                    s.updated_at = timestamp()
                """,
                {"eid": item.eid, "run_id": run_id},
            )
            tally["marked_missing"] += 1
    return tally


def rollback(client, run_id: str) -> int:
    """Restore one batch, using what that batch recorded. Idempotent.

    Scoped to a run_id on purpose. Restoring every node that merely *has* a
    saved URL would also reset nodes another run touched, and writing a fixed
    ``audio_status=''`` would erase a status that was already there before any
    repair ran.
    """
    rows = client.execute_query(
        """
        MATCH (s:Song) WHERE s.repair_run_id = $run_id
        SET s.audio_url = coalesce(s.previous_audio_url, s.audio_url),
            s.audio_status = coalesce(s.previous_audio_status, ''),
            s.unplayable_stub = coalesce(s.previous_unplayable_stub, false),
            s.previous_audio_url = null,
            s.previous_audio_status = null,
            s.previous_unplayable_stub = null,
            s.repair_run_id = null,
            s.updated_at = timestamp()
        RETURN count(s) AS n
        """,
        {"run_id": run_id},
    )
    return int((rows[0] if rows else {}).get("n") or 0)


def list_runs(client) -> list[dict]:
    return [
        dict(row)
        for row in client.execute_query(
            """
            MATCH (s:Song) WHERE s.repair_run_id IS NOT NULL
            RETURN s.repair_run_id AS run_id, count(s) AS nodes
            ORDER BY run_id
            """,
            {},
        )
    ]


def adopt_legacy(client, run_id: str) -> int:
    """Bring nodes from an earlier, un-stamped run under run_id bookkeeping.

    An earlier version cleared audio_url into missing_audio_url without a run id
    or previous_* fields, so those nodes were not rollback-addressable.
    """
    rows = client.execute_query(
        """
        MATCH (s:Song)
        WHERE s.missing_audio_url IS NOT NULL AND s.missing_audio_url <> ''
          AND s.repair_run_id IS NULL
        SET s.previous_audio_url = s.missing_audio_url,
            s.previous_audio_status = '',
            s.previous_unplayable_stub = false,
            s.repair_run_id = $run_id,
            s.missing_audio_url = null,
            s.audio_url = '',
            s.audio_status = 'missing',
            s.unplayable_stub = true,
            s.updated_at = timestamp()
        RETURN count(s) AS n
        """,
        {"run_id": run_id},
    )
    return int((rows[0] if rows else {}).get("n") or 0)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default=os.environ.get("MUSIC_DATA_PATH", "D:/SoulTunerData"))
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--rollback", metavar="RUN_ID")
    parser.add_argument("--list-runs", action="store_true")
    parser.add_argument("--adopt-legacy", action="store_true",
                        help="put pre-run_id repairs under a fresh run id")
    args = parser.parse_args()

    from retrieval.neo4j_client import get_neo4j_client

    client = get_neo4j_client()

    if args.list_runs:
        for row in list_runs(client):
            print(f"  {row['run_id']}  {row['nodes']} 个节点")
        return 0
    if args.rollback:
        print(f"回滚 run {args.rollback}: 恢复 {rollback(client, args.rollback)} 个节点")
        return 0
    if args.adopt_legacy:
        run_id = f"legacy-{uuid.uuid4().hex[:8]}"
        print(f"纳入 {adopt_legacy(client, run_id)} 个旧修复节点，run_id={run_id}")
        return 0

    stat = node_counts(client)
    print(f"Song 节点 {stat['songs']}  |  DISTINCT music_id {stat['distinct_music_id']}  |  "
          f"带 audio_url 的 Song {stat['with_audio_url']}")
    print(f"（同一查询按 PERFORMED_BY 展开是 {stat['expanded_rows']} 行 —— "
          f"多歌手歌曲一首多行，行数不是节点数）")
    print(f"audio_status=missing 但仍满足可播放条件: {stat['still_recallable_missing']}")

    rows = query_nodes(client)
    planned = plan(rows, Path(args.data_root))
    print(f"\n逐节点检查 {len(rows)} 个，发现 {len(planned)} 个指向不存在的文件")
    for item in planned:
        print(f"  [{item.action}] {item.title} - {item.artist_display}")
        print(f"      当前: {item.url}")
        if item.action == "repoint":
            print(f"      改指: {item.donor_url}  (标题与完整歌手集合一致)")
        else:
            print("      无同一录音 → 存 previous_audio_url、清空 audio_url、置 unplayable_stub")

    if not planned:
        return 0
    if not args.apply:
        print("\n干跑，未写入。加 --apply 执行。Song 实体不会被删除。")
        return 0

    run_id = uuid.uuid4().hex[:12]
    tally = apply(planned, client, run_id=run_id)
    print(f"\nrun_id={run_id}  改指 {tally['repointed']}，标记 missing {tally['marked_missing']}")
    print(f"回滚: python scripts/repair_dangling_audio.py --rollback {run_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
