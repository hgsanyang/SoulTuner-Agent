"""Audit Song Describer covers and build a ModelScope-ready public library.

Jamendo exposes deterministic album-image URLs, but its API terms do not make
the audio track's Creative Commons licence automatically apply to album art and
discourage offline content caches.  This module therefore separates two jobs:

* probe/cache official JPEGs outside Git for availability and runtime use;
* build the redistributable ModelScope library with remote cover URLs plus a
  project-generated SVG fallback, never copying Jamendo artwork into the bundle.

The 706 Song Describer MP3 files are distributed by the dataset itself with
per-track licences, so they may be materialised into the bundle after checksum
verification.  No audio bytes are transformed.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import html
import json
import os
import shutil
import struct
import sys
import tempfile
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Mapping, Sequence
from urllib.request import Request, urlopen

from tools.data.song_describer_pipeline import hash_file, write_jsonl

COVER_SCHEMA_VERSION = "soultuner.cover.v1"
BUNDLE_SCHEMA_VERSION = "soultuner.public_library.v1"
JAMENDO_IMAGE_WIDTH = 600
JAMENDO_PROVIDER_URL = "https://www.jamendo.com"
JAMENDO_API_TERMS_URL = "https://devportal.jamendo.com/api_terms_of_use"
JAMENDO_COVER_RIGHTS = (
    "Jamendo display asset; artwork rights are not asserted by the Song Describer audio licence. "
    "Remote display must follow Jamendo terms, credit the artist/Jamendo, and link back to the track page."
)
PLACEHOLDER_LICENSE = "CC-BY-4.0"
USER_AGENT = "SoulTuner-Public-Library-Cover-Audit/1.0"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {path}:{line_number}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"manifest row {line_number} must be an object")
            rows.append(row)
    return rows


def jamendo_cover_url(row: Mapping[str, Any], *, width: int = JAMENDO_IMAGE_WIDTH) -> str:
    album_id = str(row.get("album_id") or "").strip()
    track_id = str(row.get("track_id") or "").strip()
    if not album_id or not track_id:
        return ""
    return (
        "https://usercontent.jamendo.com/"
        f"?type=album&id={album_id}&width={int(width)}&trackid={track_id}"
    )


def download_bytes(url: str, *, attempts: int = 3, timeout: float = 45.0) -> tuple[bytes, str]:
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            request = Request(url, headers={"User-Agent": USER_AGENT})
            with urlopen(request, timeout=timeout) as response:  # noqa: S310 - deterministic official Jamendo URL
                content_type = str(response.headers.get("Content-Type") or "").split(";", 1)[0].lower()
                payload = response.read(5 * 1024 * 1024)
            if content_type not in {"image/jpeg", "image/jpg"}:
                raise ValueError(f"unexpected content type {content_type!r}")
            if len(payload) < 512:
                raise ValueError(f"cover response is too small ({len(payload)} bytes)")
            return payload, content_type
        except Exception as exc:
            last_error = exc
            if attempt + 1 < attempts:
                time.sleep(0.35 * (2**attempt))
    raise RuntimeError(str(last_error or "cover download failed"))


def jpeg_dimensions(payload: bytes) -> tuple[int, int]:
    """Read JPEG SOF dimensions without decoding or rewriting the artwork."""

    if len(payload) < 4 or payload[:2] != b"\xff\xd8":
        raise ValueError("not a JPEG stream")
    index = 2
    sof_markers = {0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF}
    while index + 4 <= len(payload):
        if payload[index] != 0xFF:
            index += 1
            continue
        while index < len(payload) and payload[index] == 0xFF:
            index += 1
        if index >= len(payload):
            break
        marker = payload[index]
        index += 1
        if marker in {0xD8, 0xD9}:
            continue
        if marker == 0xDA:
            break
        if index + 2 > len(payload):
            break
        segment_length = struct.unpack(">H", payload[index : index + 2])[0]
        if segment_length < 2 or index + segment_length > len(payload):
            raise ValueError("malformed JPEG segment")
        if marker in sof_markers:
            if segment_length < 7:
                raise ValueError("malformed JPEG SOF")
            height, width = struct.unpack(">HH", payload[index + 3 : index + 7])
            if width < 1 or height < 1:
                raise ValueError("invalid JPEG dimensions")
            return width, height
        index += segment_length
    raise ValueError("JPEG dimensions not found")


def placeholder_svg(row: Mapping[str, Any]) -> bytes:
    """Return a deterministic, project-owned fallback cover for one track."""

    song_id = str(row.get("song_id") or row.get("track_id") or "unknown")
    title = html.escape(str(row.get("title") or "Untitled")[:42])
    artist = html.escape(str(row.get("artist") or "Unknown artist")[:42])
    digest = hashlib.sha256(song_id.encode("utf-8")).hexdigest()
    hue_a = int(digest[:4], 16) % 360
    hue_b = (hue_a + 45 + int(digest[4:8], 16) % 90) % 360
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="600" height="600" viewBox="0 0 600 600">
<defs><linearGradient id="g" x1="0" y1="0" x2="1" y2="1"><stop stop-color="hsl({hue_a} 58% 22%)"/><stop offset="1" stop-color="hsl({hue_b} 66% 42%)"/></linearGradient></defs>
<rect width="600" height="600" rx="42" fill="url(#g)"/>
<circle cx="300" cy="244" r="126" fill="none" stroke="white" stroke-opacity=".24" stroke-width="18"/>
<circle cx="300" cy="244" r="32" fill="white" fill-opacity=".72"/>
<path d="M382 116v214c0 39-32 70-71 70-34 0-61-23-61-52s27-52 61-52c13 0 25 3 35 9V139z" fill="white" fill-opacity=".88"/>
<text x="52" y="496" fill="white" font-family="Arial, sans-serif" font-size="31" font-weight="700">{title}</text>
<text x="52" y="542" fill="white" fill-opacity=".78" font-family="Arial, sans-serif" font-size="23">{artist}</text>
<text x="548" y="70" text-anchor="end" fill="white" fill-opacity=".58" font-family="Arial, sans-serif" font-size="19">SoulTuner</text>
</svg>\n"""
    return svg.encode("utf-8")


def atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=path.name, suffix=".tmp", delete=False) as handle:
        handle.write(payload)
        temporary = Path(handle.name)
    temporary.replace(path)


def _probe_one(
    row: Mapping[str, Any],
    *,
    cover_root: Path,
    default_cover_sha256: str,
    refresh: bool,
    fetcher: Callable[..., tuple[bytes, str]],
) -> dict[str, Any]:
    song_id = str(row.get("song_id") or "")
    if not song_id:
        raise ValueError("audio manifest row is missing song_id")
    source_url = jamendo_cover_url(row)
    original_path = cover_root / "runtime_cache" / f"{song_id}.jpg"
    placeholder_path = cover_root / "placeholders" / f"{song_id}.svg"
    placeholder = placeholder_svg(row)
    atomic_write(placeholder_path, placeholder)
    fallback_relpath = str(PurePosixPath("covers") / "placeholders" / placeholder_path.name)
    result: dict[str, Any] = {
        "schema_version": COVER_SCHEMA_VERSION,
        "song_id": song_id,
        "track_id": str(row.get("track_id") or ""),
        "album_id": str(row.get("album_id") or ""),
        "title": str(row.get("title") or ""),
        "artist": str(row.get("artist") or ""),
        "source_page_url": str(row.get("source_url") or ""),
        "cover_source_url": source_url,
        "cover_provider": "Jamendo",
        "cover_provider_url": JAMENDO_PROVIDER_URL,
        "cover_provider_terms_url": JAMENDO_API_TERMS_URL,
        "cover_rights_note": JAMENDO_COVER_RIGHTS,
        "cover_attribution": f"{row.get('artist') or 'Unknown artist'} — {row.get('album') or row.get('title') or 'Release'}; provided by Jamendo",
        "official_cover_redistributable_in_bundle": False,
        "fallback_cover_relpath": fallback_relpath,
        "fallback_cover_sha256": hashlib.sha256(placeholder).hexdigest(),
        "fallback_cover_license": PLACEHOLDER_LICENSE,
        "normalization": "Jamendo width=600 endpoint; validated JPEG; no local re-encode",
    }
    if not source_url:
        result.update(
            {
                "cover_status": "placeholder",
                "placeholder_reason": "missing_album_or_track_id",
                "display_cover_url": fallback_relpath,
            }
        )
        return result
    try:
        if original_path.is_file() and not refresh:
            payload = original_path.read_bytes()
            content_type = "image/jpeg"
        else:
            payload, content_type = fetcher(source_url)
        width, height = jpeg_dimensions(payload)
        payload_sha = hashlib.sha256(payload).hexdigest()
        if payload_sha == default_cover_sha256:
            result.update(
                {
                    "cover_status": "placeholder",
                    "placeholder_reason": "jamendo_default_image",
                    "display_cover_url": fallback_relpath,
                    "remote_default_sha256": payload_sha,
                }
            )
            return result
        if not original_path.is_file() or hash_file(original_path) != payload_sha:
            atomic_write(original_path, payload)
        result.update(
            {
                "cover_status": "official_remote",
                "placeholder_reason": "",
                "display_cover_url": source_url,
                "remote_cache_relpath": str(PurePosixPath("runtime_cache") / original_path.name),
                "remote_cover_sha256": payload_sha,
                "remote_cover_size_bytes": len(payload),
                "remote_cover_media_type": content_type,
                "remote_cover_width": width,
                "remote_cover_height": height,
            }
        )
    except Exception as exc:
        result.update(
            {
                "cover_status": "placeholder",
                "placeholder_reason": f"fetch_or_validation_error:{type(exc).__name__}",
                "cover_error": str(exc)[:240],
                "display_cover_url": fallback_relpath,
            }
        )
    return result


def prepare_covers(
    audio_manifest: Path,
    cover_root: Path,
    *,
    workers: int = 8,
    refresh: bool = False,
    fetcher: Callable[..., tuple[bytes, str]] = download_bytes,
) -> dict[str, Any]:
    rows = read_jsonl(audio_manifest)
    if not rows:
        raise ValueError("audio manifest is empty")
    cover_root = cover_root.resolve()
    cover_root.mkdir(parents=True, exist_ok=True)
    # A fully invalid album+track can return Jamendo's tiny GIF error asset.
    # Keep a real track id while using an impossible album id to obtain the
    # JPEG "missing album artwork" response that the actual cover endpoint
    # returns for stale/missing albums.
    sentinel_track_id = str(rows[0].get("track_id") or "1")
    sentinel_url = (
        "https://usercontent.jamendo.com/"
        f"?type=album&id=999999999&width={JAMENDO_IMAGE_WIDTH}&trackid={sentinel_track_id}"
    )
    try:
        sentinel_payload, _ = fetcher(sentinel_url)
        jpeg_dimensions(sentinel_payload)
        default_cover_sha256 = hashlib.sha256(sentinel_payload).hexdigest()
    except Exception:
        # Current Jamendo edge nodes sometimes return an image/gif error asset
        # for an invalid album. Actual per-track probes reject that MIME and
        # fall back safely, so a JPEG sentinel is an optimisation, not a gate.
        default_cover_sha256 = ""

    def run(row: Mapping[str, Any]) -> dict[str, Any]:
        return _probe_one(
            row,
            cover_root=cover_root,
            default_cover_sha256=default_cover_sha256,
            refresh=refresh,
            fetcher=fetcher,
        )

    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        cover_rows = list(executor.map(run, rows))
    cover_rows.sort(key=lambda row: int(str(row["track_id"]) or "0"))
    manifest_path = cover_root / "cover_manifest_full.jsonl"
    write_jsonl(manifest_path, cover_rows)
    status_counts = Counter(str(row["cover_status"]) for row in cover_rows)
    reason_counts = Counter(str(row.get("placeholder_reason") or "") for row in cover_rows if row["cover_status"] == "placeholder")
    audit = {
        "schema_version": COVER_SCHEMA_VERSION,
        "generated_at": utc_now(),
        "track_count": len(cover_rows),
        "official_remote_available": status_counts["official_remote"],
        "placeholder_count": status_counts["placeholder"],
        "display_cover_coverage": sum(bool(row.get("display_cover_url")) for row in cover_rows),
        "fallback_cover_coverage": sum(bool(row.get("fallback_cover_relpath")) for row in cover_rows),
        "source_page_coverage": sum(bool(row.get("source_page_url")) for row in cover_rows),
        "provider_attribution_coverage": sum(bool(row.get("cover_attribution")) for row in cover_rows),
        "placeholder_reasons": dict(sorted(reason_counts.items())),
        "jamendo_default_cover_sha256": default_cover_sha256,
        "official_cover_bytes_packaged": 0,
        "official_cover_redistribution_policy": "remote display only; runtime cache excluded from public bundle",
        "cover_manifest": manifest_path.name,
        "cover_manifest_sha256": hash_file(manifest_path),
    }
    audit_path = cover_root / "cover_audit_full.json"
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"manifest": str(manifest_path), "audit": str(audit_path), **audit}


def _resolve_audio(cache_dir: Path, relpath: str) -> Path:
    candidate = cache_dir / "audio" / PurePosixPath(relpath)
    if candidate.is_file():
        return candidate
    if relpath.lower().endswith(".mp3"):
        alternate = cache_dir / "audio" / PurePosixPath(relpath[:-4] + ".2min.mp3")
        if alternate.is_file():
            return alternate
    return candidate


def materialize_file(source: Path, destination: Path, *, mode: str, expected_sha256: str) -> str:
    if destination.is_file() and hash_file(destination) == expected_sha256:
        return "reused"
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    temporary.unlink(missing_ok=True)
    if mode == "hardlink":
        try:
            os.link(source, temporary)
            actual_mode = "hardlink"
        except OSError:
            shutil.copy2(source, temporary)
            actual_mode = "copy_fallback"
    elif mode == "copy":
        shutil.copy2(source, temporary)
        actual_mode = "copy"
    else:
        raise ValueError("mode must be hardlink or copy")
    if hash_file(temporary) != expected_sha256:
        temporary.unlink(missing_ok=True)
        raise ValueError(f"materialised file SHA-256 mismatch: {source}")
    temporary.replace(destination)
    return actual_mode


def build_public_bundle(
    audio_manifest: Path,
    cover_manifest: Path,
    cache_dir: Path,
    output_dir: Path,
    *,
    mode: str = "hardlink",
) -> dict[str, Any]:
    audio_rows = read_jsonl(audio_manifest)
    cover_rows = {str(row["song_id"]): row for row in read_jsonl(cover_manifest)}
    if len(cover_rows) != len(audio_rows):
        raise ValueError(f"cover/audio count mismatch: {len(cover_rows)} != {len(audio_rows)}")
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    bundle_rows: list[dict[str, Any]] = []
    modes = Counter()
    audio_bytes = 0
    placeholder_bytes = 0
    for row in audio_rows:
        song_id = str(row["song_id"])
        cover = cover_rows.get(song_id)
        if cover is None:
            raise ValueError(f"cover manifest missing {song_id}")
        relpath = str(PurePosixPath(str(row["audio_relpath"])))
        source_audio = _resolve_audio(cache_dir, relpath)
        expected_audio_sha = str(row.get("audio_sha256") or "")
        if not source_audio.is_file() or not expected_audio_sha:
            raise FileNotFoundError(f"verified audio missing for {song_id}: {source_audio}")
        destination_audio = output_dir / "audio" / PurePosixPath(relpath)
        modes[materialize_file(source_audio, destination_audio, mode=mode, expected_sha256=expected_audio_sha)] += 1
        audio_bytes += destination_audio.stat().st_size

        fallback_name = Path(str(cover["fallback_cover_relpath"])).name
        source_placeholder = cache_dir / "covers" / "placeholders" / fallback_name
        destination_placeholder = output_dir / "covers" / "placeholders" / fallback_name
        fallback_sha = str(cover["fallback_cover_sha256"])
        modes[materialize_file(source_placeholder, destination_placeholder, mode=mode, expected_sha256=fallback_sha)] += 1
        placeholder_bytes += destination_placeholder.stat().st_size

        bundled = dict(row)
        bundled.update(
            {
                "bundle_schema_version": BUNDLE_SCHEMA_VERSION,
                "audio_path": str(PurePosixPath("audio") / relpath),
                "cover_url": str(cover["display_cover_url"]),
                "cover_fallback_path": str(PurePosixPath("covers") / "placeholders" / fallback_name),
                "cover_status": str(cover["cover_status"]),
                "cover_attribution": str(cover["cover_attribution"]),
                "cover_source_page_url": str(cover["source_page_url"]),
                "cover_provider": "Jamendo" if cover["cover_status"] == "official_remote" else "SoulTuner",
                "cover_rights_note": str(cover["cover_rights_note"]),
                "official_cover_packaged": False,
            }
        )
        bundle_rows.append(bundled)

    catalog_path = output_dir / "catalog.jsonl"
    write_jsonl(catalog_path, bundle_rows)
    audit = {
        "schema_version": BUNDLE_SCHEMA_VERSION,
        "generated_at": utc_now(),
        "track_count": len(bundle_rows),
        "audio_count": len(bundle_rows),
        "audio_bytes": audio_bytes,
        "audio_gib": round(audio_bytes / (1024**3), 6),
        "cover_fallback_count": len(bundle_rows),
        "cover_fallback_bytes": placeholder_bytes,
        "remote_official_cover_count": sum(row["cover_status"] == "official_remote" for row in cover_rows.values()),
        "local_placeholder_display_count": sum(row["cover_status"] == "placeholder" for row in cover_rows.values()),
        "official_cover_bytes_packaged": 0,
        "materialization_modes": dict(sorted(modes.items())),
        "catalog_sha256": hash_file(catalog_path),
        "audio_sha_coverage": sum(bool(row.get("audio_sha256")) for row in bundle_rows),
        "audio_license_coverage": sum(bool(row.get("license_url")) for row in bundle_rows),
        "audio_attribution_coverage": sum(bool(row.get("attribution")) for row in bundle_rows),
        "cover_fallback_coverage": sum(bool(row.get("cover_fallback_path")) for row in bundle_rows),
    }
    (output_dir / "bundle_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    readme = f"""---
license: cc-by-sa-4.0
task_categories:
- audio-classification
- feature-extraction
tags:
- music-retrieval
- song-describer
---

# SoulTuner Open Audio Demo Library

This bundle contains {len(bundle_rows)} unmodified Song Describer Dataset v1.0.0 MP3 segments,
their captions/metadata, per-track licence evidence, and project-generated fallback covers.

The audio dataset metadata is CC-BY-SA-4.0, while each audio file retains the per-track licence
recorded in `catalog.jsonl`. NC/ND/SA conditions must be honoured individually. Audio bytes are
not transcoded, trimmed, normalised, or remuxed.

Jamendo album artwork is **not redistributed** in this bundle. `cover_url` may point to Jamendo's
official remote display endpoint and must be shown with `cover_attribution` plus the source-page
backlink. `cover_fallback_path` is always local and is project-generated under CC-BY-4.0.
See {JAMENDO_API_TERMS_URL} before enabling remote covers in another application.
"""
    (output_dir / "README.md").write_text(readme, encoding="utf-8", newline="\n")
    return {"output_dir": str(output_dir), "catalog": str(catalog_path), **audit}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare covers and a ModelScope-ready Song Describer library")
    subparsers = parser.add_subparsers(dest="command", required=True)
    covers = subparsers.add_parser("covers", help="Probe/cache remote covers and generate deterministic fallbacks")
    covers.add_argument("--audio-manifest", type=Path, required=True)
    covers.add_argument("--cover-root", type=Path, required=True)
    covers.add_argument("--workers", type=int, default=8)
    covers.add_argument("--refresh", action="store_true")

    bundle = subparsers.add_parser("bundle", help="Build a public library without redistributing Jamendo artwork")
    bundle.add_argument("--audio-manifest", type=Path, required=True)
    bundle.add_argument("--cover-manifest", type=Path, required=True)
    bundle.add_argument("--cache-dir", type=Path, required=True)
    bundle.add_argument("--output-dir", type=Path, required=True)
    bundle.add_argument("--mode", choices=("hardlink", "copy"), default="hardlink")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "covers":
        result = prepare_covers(
            args.audio_manifest,
            args.cover_root,
            workers=args.workers,
            refresh=args.refresh,
        )
    else:
        result = build_public_bundle(
            args.audio_manifest,
            args.cover_manifest,
            args.cache_dir,
            args.output_dir,
            mode=args.mode,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
