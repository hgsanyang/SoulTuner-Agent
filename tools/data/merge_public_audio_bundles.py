"""Merge verified public-audio bundles without duplicating local audio bytes."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from collections import Counter
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _link_or_copy(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_file() and target.stat().st_size == source.stat().st_size:
        return
    temporary = target.with_suffix(target.suffix + ".part")
    if temporary.exists():
        temporary.unlink()
    try:
        os.link(source, temporary)
    except OSError:
        shutil.copy2(source, temporary)
    os.replace(temporary, target)


def _materialize_tree(source: Path, target: Path) -> int:
    copied = 0
    if not source.is_dir():
        return copied
    for path in source.rglob("*"):
        if path.is_file():
            _link_or_copy(path, target / path.relative_to(source))
            copied += 1
    return copied


def merge(base: Path, expansion: Path, output: Path) -> dict[str, Any]:
    roots = [path.expanduser().resolve() for path in (base, expansion, output)]
    base, expansion, output = roots
    if output in {base, expansion}:
        raise ValueError("output must differ from both input bundles")
    for root in (base, expansion):
        if not (root / "catalog.jsonl").is_file() or not (root / "audio").is_dir():
            raise FileNotFoundError(f"incomplete input bundle: {root}")
    output.mkdir(parents=True, exist_ok=True)
    audio_files = _materialize_tree(base / "audio", output / "audio")
    audio_files += _materialize_tree(expansion / "audio", output / "audio")
    cover_files = _materialize_tree(base / "covers", output / "covers")
    cover_files += _materialize_tree(expansion / "covers", output / "covers")

    merged: dict[str, dict[str, Any]] = {}
    for row in [*_rows(base / "catalog.jsonl"), *_rows(expansion / "catalog.jsonl")]:
        song_id = str(row.get("song_id") or "").strip()
        if not song_id:
            raise ValueError("catalogue row is missing song_id")
        normalized = dict(row)
        if song_id.startswith("sdd-"):
            normalized.setdefault("dataset", "song_describer_full")
        merged[song_id] = normalized

    catalog = output / "catalog.jsonl"
    with catalog.open("w", encoding="utf-8", newline="\n") as handle:
        for song_id in sorted(merged):
            handle.write(json.dumps(merged[song_id], ensure_ascii=False, sort_keys=True) + "\n")

    genres = Counter()
    datasets = Counter()
    vocal_priority = 0
    for row in merged.values():
        datasets[str(row.get("dataset") or "unknown")] += 1
        genres.update(str(tag) for tag in row.get("genres") or [])
        vocal_priority += int(bool(row.get("vocal_priority")))
        relpath = str(row.get("audio_relpath") or "")
        audio = (output / "audio" / relpath).resolve()
        audio.relative_to((output / "audio").resolve())
        if not audio.is_file() or _sha256(audio) != str(row.get("audio_sha256") or ""):
            raise ValueError(f"audio verification failed: {row.get('song_id')}")

    audit = {
        "tracks": len(merged),
        "audio_files": audio_files,
        "cover_files": cover_files,
        "datasets": dict(sorted(datasets.items())),
        "top_genres": dict(genres.most_common(20)),
        "vocal_metadata_tracks": vocal_priority,
        "catalog_sha256": _sha256(catalog),
    }
    (output / "bundle_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output / "README.md").write_text(
        "\n".join(
            (
                "# SoulTuner Open Audio Demo V2",
                "",
                f"- {len(merged)} playable tracks with per-track attribution and licence metadata.",
                "- Song Describer / Jamendo supplies captioned audio and original remote artwork.",
                "- FMA Small adds a vocal-prioritised, genre-balanced Rock / Hip-Hop / Pop / Folk / International / Electronic expansion.",
                "- Audio SHA-256 is verified before publication; local assembly uses hard links where supported.",
                "",
            )
        ),
        encoding="utf-8",
    )
    return audit


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--expansion", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    print(
        json.dumps(
            merge(args.base, args.expansion, args.output),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
