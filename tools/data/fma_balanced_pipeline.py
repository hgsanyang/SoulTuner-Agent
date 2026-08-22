"""Build a vocal-prioritised, genre-balanced FMA Small expansion.

The official archive is 7.2 GiB.  This tool reads its ZIP central directory by
HTTP range and downloads only selected 30-second MP3 entries.  Selection is
deterministic, excludes the Experimental and Instrumental top-level buckets,
and prefers tracks with language or lyricist metadata inside each genre.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import html
import json
import os
import re
import threading
import time
import zlib
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any


FMA_AUDIO_URL = "https://os.unil.cloud.switch.ch/fma/fma_small.zip"
FMA_METADATA_URL = "https://os.unil.cloud.switch.ch/fma/fma_metadata.zip"
FMA_SOURCE_URL = "https://github.com/mdeff/fma"
DEFAULT_TARGETS = {
    "Rock": 300,
    "Hip-Hop": 220,
    "Pop": 220,
    "Folk": 160,
    "International": 120,
    "Electronic": 80,
}
_THREAD_LOCAL = threading.local()


def _remote_zip(url: str):
    try:
        from remotezip import RemoteZip
    except ImportError as exc:  # pragma: no cover - exercised by CLI users.
        raise RuntimeError("Install remotezip>=0.12 before running this pipeline") from exc
    key = "audio" if url == FMA_AUDIO_URL else "metadata"
    archive = getattr(_THREAD_LOCAL, key, None)
    if archive is None:
        archive = RemoteZip(url, initial_buffer_size=2 * 1024 * 1024)
        setattr(_THREAD_LOCAL, key, archive)
    return archive


def _metadata_dir(root: Path) -> Path:
    return root / "metadata"


def fetch_metadata(root: Path) -> None:
    destination = _metadata_dir(root)
    destination.mkdir(parents=True, exist_ok=True)
    archive = _remote_zip(FMA_METADATA_URL)
    for name in ("tracks.csv", "genres.csv", "raw_albums.csv"):
        target = destination / name
        if target.is_file() and target.stat().st_size:
            continue
        payload = archive.read(f"fma_metadata/{name}")
        temporary = target.with_suffix(target.suffix + ".part")
        temporary.write_bytes(payload)
        os.replace(temporary, target)


def _album_metadata(path: Path) -> dict[int, dict[str, str]]:
    albums: dict[int, dict[str, str]] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            try:
                album_id = int(row.get("album_id") or 0)
            except ValueError:
                continue
            albums[album_id] = {
                "album_title": str(row.get("album_title") or "").strip(),
                "url": str(row.get("album_url") or "").strip(),
                "cover_url": str(row.get("album_image_file") or "").strip(),
            }
    return albums


def _parse_tags(value: str) -> list[str]:
    try:
        raw = ast.literal_eval(value or "[]")
    except (ValueError, SyntaxError):
        return []
    if not isinstance(raw, list):
        return []
    return list(dict.fromkeys(str(item).strip() for item in raw if str(item).strip()))[:12]


def _stream_small_tracks(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        families = next(reader)
        fields = next(reader)
        next(reader)
        keys = [f"{family}.{field}".strip(".") for family, field in zip(families, fields)]
        index = {key: keys.index(key) for key in keys}
        for values in reader:
            if not values or values[index["set.subset"]] != "small":
                continue
            licence = values[index["track.license"]].strip()
            if not licence:
                continue
            track_id = int(values[0])
            language = values[index["track.language_code"]].strip()
            lyricist = values[index["track.lyricist"]].strip()
            try:
                listens = int(values[index["track.listens"]] or 0)
            except ValueError:
                listens = 0
            rows.append(
                {
                    "track_id": track_id,
                    "album_id": int(values[index["album.id"]] or 0),
                    "title": values[index["track.title"]].strip() or f"FMA {track_id:06d}",
                    "artist": values[index["artist.name"]].strip() or "Unknown FMA artist",
                    "genre": values[index["track.genre_top"]].strip(),
                    "license": licence,
                    "language": language,
                    "lyricist": lyricist,
                    "listens": listens,
                    "tags": _parse_tags(values[index["track.tags"]]),
                    "release_date": values[index["track.date_recorded"]].strip()
                    or values[index["track.date_created"]].strip(),
                }
            )
    return rows


def select_tracks(root: Path, targets: dict[str, int] | None = None) -> list[dict[str, Any]]:
    targets = targets or DEFAULT_TARGETS
    albums = _album_metadata(_metadata_dir(root) / "raw_albums.csv")
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in _stream_small_tracks(_metadata_dir(root) / "tracks.csv"):
        if row["genre"] in targets:
            row.update(albums.get(row["album_id"], {}))
            # Prefer available vocal clues, then broad listener evidence, while
            # keeping track-id as a stable tie-breaker.
            row["vocal_priority"] = int(bool(row["language"] or row["lyricist"]))
            grouped[row["genre"]].append(row)

    selected: list[dict[str, Any]] = []
    for genre, count in targets.items():
        ranked = sorted(
            grouped.get(genre, []),
            key=lambda row: (-row["vocal_priority"], -row["listens"], row["track_id"]),
        )
        selected.extend(ranked[:count])
    return sorted(selected, key=lambda row: row["track_id"])


def _licence_url(label: str) -> str:
    lowered = label.casefold()
    if "public domain mark" in lowered:
        return "https://creativecommons.org/publicdomain/mark/1.0/"
    if "public domain" in lowered or "cc0" in lowered:
        return "https://creativecommons.org/publicdomain/zero/1.0/"
    if "art libre" in lowered:
        return "https://artlibre.org/licence/lal/en/"
    if "sampling plus" in lowered:
        code = "nc-sampling+" if "noncommercial" in lowered else "sampling+"
        return f"https://creativecommons.org/licenses/{code}/1.0/"
    code = "by"
    if "noncommercial" in lowered:
        code += "-nc"
    if "share alike" in lowered or "sharealike" in lowered:
        code += "-sa"
    elif "no derivative" in lowered or "noderivative" in lowered or "noderivs" in lowered:
        code += "-nd"
    match = re.search(r"\b([234]\.\d)\b", label)
    version = match.group(1) if match else "3.0"
    return f"https://creativecommons.org/licenses/{code}/{version}/"


def _archive_name(track_id: int) -> str:
    return f"fma_small/{track_id // 1000:03d}/{track_id:06d}.mp3"


def _audio_path(root: Path, track_id: int) -> Path:
    return root / "audio" / "fma" / f"{track_id // 1000:03d}" / f"{track_id:06d}.mp3"


def _file_crc32(path: Path) -> int:
    checksum = 0
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            checksum = zlib.crc32(block, checksum)
    return checksum & 0xFFFFFFFF


def _download_one(root: Path, row: dict[str, Any], retries: int = 5) -> dict[str, Any]:
    track_id = int(row["track_id"])
    target = _audio_path(root, track_id)
    target.parent.mkdir(parents=True, exist_ok=True)
    archive_name = _archive_name(track_id)
    error: Exception | None = None
    for attempt in range(retries):
        try:
            archive = _remote_zip(FMA_AUDIO_URL)
            info = archive.getinfo(archive_name)
            if target.is_file() and target.stat().st_size == info.file_size:
                if _file_crc32(target) == info.CRC:
                    return {**row, "audio_file": target}
            temporary = target.with_suffix(".mp3.part")
            checksum = 0
            with archive.open(info) as source, temporary.open("wb") as destination:
                for block in iter(lambda: source.read(1024 * 1024), b""):
                    destination.write(block)
                    checksum = zlib.crc32(block, checksum)
            if temporary.stat().st_size != info.file_size or checksum & 0xFFFFFFFF != info.CRC:
                raise ValueError(f"CRC or size mismatch for {archive_name}")
            os.replace(temporary, target)
            return {**row, "audio_file": target}
        except Exception as exc:  # remote object stores occasionally reset ranges.
            error = exc
            setattr(_THREAD_LOCAL, "audio", None)
            time.sleep(min(2**attempt, 12))
    raise RuntimeError(f"failed to fetch {archive_name}: {error}")


def download_selected(
    root: Path,
    rows: list[dict[str, Any]],
    *,
    workers: int = 6,
) -> list[dict[str, Any]]:
    completed: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, min(workers, 12))) as pool:
        futures = {pool.submit(_download_one, root, row): row for row in rows}
        for index, future in enumerate(as_completed(futures), start=1):
            completed.append(future.result())
            if index == 1 or index % 25 == 0 or index == len(rows):
                print(f"FMA range download: {index}/{len(rows)}", flush=True)
    return sorted(completed, key=lambda row: row["track_id"])


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_cover(root: Path, row: dict[str, Any]) -> str:
    track_id = int(row["track_id"])
    relative = Path("covers") / "fma" / f"{track_id:06d}.svg"
    target = root / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    title = html.escape(str(row["title"])[:42])
    artist = html.escape(str(row["artist"])[:42])
    genre = html.escape(str(row["genre"]))
    hue = track_id % 360
    target.write_text(
        "".join(
            (
                '<svg xmlns="http://www.w3.org/2000/svg" width="600" height="600">',
                f'<defs><linearGradient id="g" x2="1" y2="1"><stop stop-color="hsl({hue} 65% 32%)"/>',
                f'<stop offset="1" stop-color="hsl({(hue + 55) % 360} 72% 12%)"/></linearGradient></defs>',
                '<rect width="600" height="600" rx="42" fill="url(#g)"/>',
                '<circle cx="465" cy="155" r="105" fill="none" stroke="white" stroke-opacity=".18" stroke-width="26"/>',
                f'<text x="48" y="390" fill="white" font-family="sans-serif" font-size="30" font-weight="700">{title}</text>',
                f'<text x="48" y="435" fill="white" fill-opacity=".78" font-family="sans-serif" font-size="22">{artist}</text>',
                f'<text x="48" y="515" fill="white" fill-opacity=".58" font-family="sans-serif" font-size="18">FMA · {genre}</text>',
                "</svg>",
            )
        ),
        encoding="utf-8",
    )
    return relative.as_posix()


def write_catalog(root: Path, rows: list[dict[str, Any]]) -> Path:
    manifest = root / "catalog.jsonl"
    with manifest.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            audio_file = Path(row["audio_file"])
            cover_path = _write_cover(root, row)
            album_url = str(row.get("url") or "").strip()
            source_url = album_url if album_url.startswith("http") else FMA_SOURCE_URL
            tags = list(dict.fromkeys([row["genre"], *row.get("tags", [])]))[:12]
            catalog_row = {
                "song_id": f"fma-{int(row['track_id']):06d}",
                "source_id": str(row["track_id"]),
                "dataset": "fma_small_balanced",
                "title": row["title"],
                "artist": row["artist"],
                "album": row.get("album_title", ""),
                "genres": tags,
                "moods_themes": [],
                "scenarios": [],
                "instruments": [],
                "language": row.get("language") or None,
                "lyricist": row.get("lyricist") or None,
                "vocal_priority": bool(row.get("vocal_priority")),
                "release_date": row.get("release_date") or None,
                "audio_relpath": audio_file.relative_to(root / "audio").as_posix(),
                "audio_sha256": _sha256(audio_file),
                "license": row["license"],
                "license_url": _licence_url(str(row["license"])),
                "attribution": f"{row['title']} by {row['artist']} (Free Music Archive)",
                "source_url": source_url,
                # The 2017 FMA image CDN paths currently return 404.  Preserve
                # the historical URL as provenance and render the packaged SVG.
                "cover_url": "",
                "cover_original_url": str(row.get("cover_url") or ""),
                "cover_fallback_path": cover_path,
                "cover_attribution": f"FMA album artwork for {row['artist']}",
                "cover_source_page_url": source_url,
                "cover_provider": "Free Music Archive",
                "cover_status": "packaged_fallback_original_url_unavailable",
                "archive_url": FMA_AUDIO_URL,
            }
            handle.write(json.dumps(catalog_row, ensure_ascii=False, sort_keys=True) + "\n")
    summary = {
        "tracks": len(rows),
        "genres": {genre: sum(row["genre"] == genre for row in rows) for genre in DEFAULT_TARGETS},
        "vocal_metadata_tracks": sum(bool(row.get("vocal_priority")) for row in rows),
        "catalog_sha256": _sha256(manifest),
        "source": FMA_SOURCE_URL,
    }
    (root / "audit.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--metadata-only", action="store_true")
    parser.add_argument("--select-only", action="store_true")
    args = parser.parse_args(argv)
    root = args.root.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    fetch_metadata(root)
    rows = select_tracks(root)
    print(
        json.dumps(
            {
                "selected": len(rows),
                "genres": {genre: sum(row["genre"] == genre for row in rows) for genre in DEFAULT_TARGETS},
                "vocal_metadata": sum(bool(row["vocal_priority"]) for row in rows),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    if args.metadata_only or args.select_only:
        return 0
    completed = download_selected(root, rows, workers=args.workers)
    manifest = write_catalog(root, completed)
    print(f"FMA catalogue ready: {manifest}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
