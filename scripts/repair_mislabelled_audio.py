#!/usr/bin/env python
"""Rename audio files whose extension disagrees with their bytes, and fix the graph.

Two files in ``online_acquired/audio`` are named ``.mp3`` and are actually FLAC.
They are stored with ``format='mp3'`` and served as ``content-type: audio/mpeg``,
which Safari and others refuse outright — so they have been unplayable in some
browsers for as long as the lossless acquisition path has existed. Their MuQ
vectors are fine, because ffmpeg identifies by content.

Idempotent: a second run finds nothing to do. Dry-run by default. Writes a
rollback file before touching anything.

    python scripts/repair_mislabelled_audio.py                 # report only
    python scripts/repair_mislabelled_audio.py --apply
    python scripts/repair_mislabelled_audio.py --rollback out.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from services.audio_format import detect_file, is_lossless, mime_for, suffix_for  # noqa: E402

DEFAULT_ROOTS = ("online_acquired/audio", "processed_audio/audio")


@dataclass
class Repair:
    old_path: str
    new_path: str
    container: str
    sha256: str
    old_audio_url: str = ""
    new_audio_url: str = ""
    old_format: str = ""
    music_ids: tuple[str, ...] = ()


def sha256_of(path: Path, *, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def find_mismatches(roots: list[Path]) -> list[Repair]:
    """Files whose bytes disagree with their extension.

    Wrapped containers are skipped: their suffix is not supposed to describe the
    audio inside, so "mismatch" is not a meaningful verdict for them.
    """
    wrapped = {".ncm", ".uc", ".uc!"}
    found: list[Repair] = []
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if not path.is_file() or path.suffix.lower() in wrapped:
                continue
            container = detect_file(path)
            if not container:
                continue
            correct = suffix_for(container)
            if not correct or path.suffix.lower() == correct:
                continue
            found.append(
                Repair(
                    old_path=str(path),
                    new_path=str(path.with_suffix(correct)),
                    container=container,
                    sha256=sha256_of(path),
                )
            )
    return found


def _graph_rows(client, filename: str) -> list[dict]:
    return [
        dict(row)
        for row in client.execute_query(
            """
            MATCH (s:Song)
            WHERE s.audio_url IS NOT NULL AND s.audio_url ENDS WITH $filename
            RETURN coalesce(toString(s.music_id), '') AS music_id,
                   s.audio_url AS audio_url,
                   coalesce(s.format, '') AS format,
                   size(coalesce(s.muq_embedding, [])) AS muq
            """,
            {"filename": filename},
        )
    ]


def enrich_from_graph(repairs: list[Repair], client) -> list[Repair]:
    for repair in repairs:
        old = Path(repair.old_path)
        rows = _graph_rows(client, old.name)
        if not rows:
            continue
        repair.music_ids = tuple(r["music_id"] for r in rows if r["music_id"])
        repair.old_audio_url = str(rows[0]["audio_url"])
        repair.old_format = str(rows[0]["format"])
        repair.new_audio_url = repair.old_audio_url.rsplit(".", 1)[0] + suffix_for(repair.container)
    return repairs


def apply_repair(repair: Repair, client) -> dict:
    """Rename on disk, then update every graph row that referenced the old name.

    Rename first: if the graph write fails the file is still findable by the
    rollback record, whereas a graph pointing at a name that does not exist yet
    is a live 404.
    """
    old, new = Path(repair.old_path), Path(repair.new_path)
    if new.exists() and not old.exists():
        return {"renamed": False, "note": "already renamed", "updated": 0}
    if new.exists():
        raise FileExistsError(f"目标已存在且源也在，拒绝覆盖: {new}")
    os.replace(old, new)

    updated = client.execute_query(
        """
        MATCH (s:Song)
        WHERE s.audio_url IS NOT NULL AND s.audio_url ENDS WITH $old_name
        SET s.audio_url = replace(s.audio_url, $old_name, $new_name),
            s.format = $container,
            s.codec = $container,
            s.mime_type = $mime,
            s.lossless = $lossless,
            s.updated_at = timestamp()
        RETURN count(s) AS n
        """,
        {
            "old_name": old.name,
            "new_name": new.name,
            "container": repair.container,
            "mime": mime_for(repair.container),
            "lossless": is_lossless(repair.container),
        },
    )
    return {"renamed": True, "updated": int((updated[0] if updated else {}).get("n") or 0)}


def rollback(path: Path) -> int:
    """Undo a previous --apply from its rollback file."""
    from retrieval.neo4j_client import get_neo4j_client

    payload = json.loads(path.read_text(encoding="utf-8"))
    client = get_neo4j_client()
    restored = 0
    for item in payload.get("repairs", []):
        old, new = Path(item["old_path"]), Path(item["new_path"])
        if new.exists() and not old.exists():
            os.replace(new, old)
        client.execute_query(
            """
            MATCH (s:Song)
            WHERE s.audio_url IS NOT NULL AND s.audio_url ENDS WITH $new_name
            SET s.audio_url = replace(s.audio_url, $new_name, $old_name),
                s.format = $old_format,
                s.updated_at = timestamp()
            """,
            {"new_name": new.name, "old_name": old.name,
             "old_format": item.get("old_format") or "mp3"},
        )
        restored += 1
    return restored


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default=os.environ.get("MUSIC_DATA_PATH", "D:/SoulTunerData"))
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--rollback", help="undo using a rollback JSON written by --apply")
    parser.add_argument("--rollback-out", default="repair_rollback.json")
    args = parser.parse_args()

    if args.rollback:
        print(f"回滚 {rollback(Path(args.rollback))} 项")
        return 0

    root = Path(args.data_root)
    repairs = find_mismatches([root / part for part in DEFAULT_ROOTS])
    if not repairs:
        print("没有发现扩展名与内容不符的文件（幂等：修过之后再跑就是这个结果）")
        return 0

    from retrieval.neo4j_client import get_neo4j_client

    client = get_neo4j_client()
    repairs = enrich_from_graph(repairs, client)

    print(f"发现 {len(repairs)} 个不符：")
    for repair in repairs:
        print(f"  {Path(repair.old_path).name}")
        print(f"    实际={repair.container}  -> {Path(repair.new_path).name}")
        print(f"    sha256={repair.sha256[:16]}…  图节点={len(repair.music_ids)}  旧format={repair.old_format}")

    if not args.apply:
        print("\n干跑。加 --apply 执行。")
        return 0

    out = Path(args.rollback_out)
    out.write_text(
        json.dumps({"repairs": [asdict(r) for r in repairs]}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\n回滚文件已写入 {out}")

    for repair in repairs:
        result = apply_repair(repair, client)
        print(f"  {Path(repair.new_path).name}: {result}")
    print("\n完成。验证失败时用 --rollback 恢复。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
