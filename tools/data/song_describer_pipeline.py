"""Prepare a license-audited Song Describer catalog for SoulTuner.

The pipeline deliberately keeps audio and generated manifests outside Git.  It
downloads the immutable Zenodo v1.0.0 metadata, verifies the checksums published
by Zenodo, joins captions/MTG-Jamendo metadata/per-track licences, and emits one
deterministic JSONL row per recording.  Optionally it downloads the 3.09 GiB
audio archive and extracts only the selected subset.

The dataset authors describe Song Describer as an *evaluation* dataset and
discourage using it for training.  This tool prepares it for retrieval,
playback, and evaluation only; it is not connected to the planner training
pipeline.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import sys
import tempfile
import time
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, Sequence
from urllib.error import HTTPError
from urllib.request import Request, urlopen

ZENODO_RECORD_ID = "10072001"
ZENODO_RECORD_API = f"https://zenodo.org/api/records/{ZENODO_RECORD_ID}"
DATASET_ID = "mulab-mir/song-describer-dataset"
DATASET_VERSION = "1.0.0"
DATASET_LICENSE = "CC-BY-SA-4.0"
DATASET_URL = f"https://zenodo.org/records/{ZENODO_RECORD_ID}"
SOURCE_REPOSITORY = "https://github.com/mulab-mir/song-describer-dataset"
MANIFEST_SCHEMA_VERSION = "soultuner.open_audio.v1"

METADATA_FILENAMES = (
    "song_describer.csv",
    "audio_metadata.tsv",
    "audio_licenses.txt",
    "song_describer_14_04_23.mtg-jamendo.tsv",
)
AUDIO_ARCHIVE_NAME = "audio.zip"
CHUNK_BYTES = 8 * 1024 * 1024
LICENSE_URL_RE = re.compile(
    r"https?://creativecommons\.org/licenses/"
    r"(?P<code>by(?:-nc)?(?:-nd|-sa)?)/(?P<version>[0-9.]+)/?(?P<jurisdiction>[a-z]{2})?/?",
    re.IGNORECASE,
)
ART_LIBRE_URL_RE = re.compile(r"https?://artlibre\.org/licence/lal/?", re.IGNORECASE)
OPEN_LICENSE_URL_RE = re.compile(
    rf"(?:{LICENSE_URL_RE.pattern})|(?:{ART_LIBRE_URL_RE.pattern})",
    re.IGNORECASE,
)
TRACK_ID_RE = re.compile(r"(?:[a-z]+_)?0*(\d+)$", re.IGNORECASE)


@dataclass(frozen=True)
class RemoteFile:
    """One immutable file advertised by the Zenodo record."""

    name: str
    size: int
    checksum_type: str
    checksum: str
    url: str


def default_cache_dir() -> Path:
    """Return a runtime cache that is outside the source checkout by default."""

    override = os.getenv("SOULTUNER_OPEN_AUDIO_CACHE")
    if override:
        return Path(override).expanduser()
    data_root = os.getenv("MUSIC_DATA_PATH") or os.getenv("MUSIC_DATA_ROOT")
    if data_root:
        return Path(data_root).expanduser() / "open_audio" / "song_describer"
    cache_root = os.getenv("LOCALAPPDATA") or os.getenv("XDG_CACHE_HOME")
    if cache_root:
        return Path(cache_root).expanduser() / "SoulTuner" / "open_audio" / "song_describer"
    return Path.home() / ".cache" / "soultuner" / "open_audio" / "song_describer"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def normalize_track_id(value: Any) -> str:
    text = str(value or "").strip()
    match = TRACK_ID_RE.fullmatch(text)
    if not match:
        raise ValueError(f"invalid track id: {value!r}")
    return match.group(1)


def hash_file(path: Path, algorithm: str = "sha256") -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch_record() -> tuple[dict[str, Any], dict[str, RemoteFile]]:
    """Read the pinned Zenodo record and return its file inventory."""

    with urlopen(ZENODO_RECORD_API, timeout=60) as response:  # noqa: S310 - pinned official API
        record = json.load(response)
    version = str(record.get("metadata", {}).get("version") or "")
    if version and version != DATASET_VERSION:
        raise RuntimeError(f"unexpected Zenodo version {version!r}; expected {DATASET_VERSION!r}")

    files: dict[str, RemoteFile] = {}
    for raw in record.get("files") or []:
        checksum_type, checksum = str(raw["checksum"]).split(":", 1)
        files[str(raw["key"])] = RemoteFile(
            name=str(raw["key"]),
            size=int(raw["size"]),
            checksum_type=checksum_type.lower(),
            checksum=checksum.lower(),
            url=str(raw["links"]["self"]),
        )
    required = set(METADATA_FILENAMES) | {AUDIO_ARCHIVE_NAME}
    missing = sorted(required - set(files))
    if missing:
        raise RuntimeError(f"Zenodo record is missing required files: {missing}")
    return record, files


def verify_remote_file(path: Path, remote: RemoteFile) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.stat().st_size != remote.size:
        raise ValueError(f"size mismatch for {path.name}: {path.stat().st_size} != {remote.size}")
    actual = hash_file(path, remote.checksum_type)
    if actual.lower() != remote.checksum.lower():
        raise ValueError(f"{remote.checksum_type} mismatch for {path.name}: {actual} != {remote.checksum}")


def download_file(remote: RemoteFile, destination: Path, *, refresh: bool = False) -> Path:
    """Download one file atomically, reusing a verified local copy when possible."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not refresh:
        try:
            verify_remote_file(destination, remote)
            return destination
        except (ValueError, FileNotFoundError):
            pass

    partial = destination.with_name(destination.name + ".part")
    if refresh:
        partial.unlink(missing_ok=True)
    max_retries = max(1, int(os.getenv("SOULTUNER_DOWNLOAD_MAX_RETRIES", "32")))
    failures = 0
    while True:
        offset = partial.stat().st_size if partial.exists() else 0
        if offset == remote.size:
            break
        if offset > remote.size:
            partial.unlink()
            offset = 0

        headers = {"User-Agent": "SoulTuner-Open-Audio-Pipeline/1.0"}
        if offset:
            headers["Range"] = f"bytes={offset}-"
        request = Request(remote.url, headers=headers)
        try:
            response = urlopen(request, timeout=120)  # noqa: S310 - URL comes from pinned Zenodo API
        except HTTPError as exc:
            if exc.code == 416 and partial.exists() and partial.stat().st_size == remote.size:
                break
            failures += 1
            if failures >= max_retries:
                raise
            time.sleep(min(2**failures, 30))
            continue
        except OSError:
            failures += 1
            if failures >= max_retries:
                raise
            time.sleep(min(2**failures, 30))
            continue

        status = getattr(response, "status", response.getcode())
        if offset and status != 206:
            response.close()
            partial.unlink(missing_ok=True)
            failures += 1
            if failures >= max_retries:
                raise RuntimeError(f"server ignored Range requests for {remote.name}")
            time.sleep(min(2**failures, 30))
            continue
        with response, partial.open("ab" if offset else "wb") as handle:
            shutil.copyfileobj(response, handle, length=CHUNK_BYTES)

        current_size = partial.stat().st_size
        if current_size < remote.size:
            # Zenodo/CDN connections can finish cleanly before Content-Length.
            # Treat that as a resumable short read, not a corrupted final file.
            failures += 1
            if failures >= max_retries:
                raise RuntimeError(
                    f"download ended early after {failures} attempts: "
                    f"{current_size} != {remote.size} for {remote.name}"
                )
            time.sleep(min(2**failures, 30))
            continue
        if current_size > remote.size:
            raise ValueError(
                f"download exceeded advertised size: {current_size} > {remote.size} for {remote.name}"
            )
        break

    verify_remote_file(partial, remote)
    partial.replace(destination)
    return destination


def read_captions(path: Path, subset: str) -> dict[str, list[dict[str, Any]]]:
    captions: dict[str, list[dict[str, Any]]] = defaultdict(list)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            is_valid = str(row.get("is_valid_subset") or "").casefold() == "true"
            if subset == "validated" and not is_valid:
                continue
            track_id = normalize_track_id(row.get("track_id"))
            captions[track_id].append(
                {
                    "caption_id": str(row.get("caption_id") or "").strip(),
                    "text": " ".join(str(row.get("caption") or "").split()),
                    "validated": is_valid,
                    "familiarity": int(row["familiarity"]) if str(row.get("familiarity") or "").strip() else None,
                }
            )
    for rows in captions.values():
        rows.sort(key=lambda item: int(item["caption_id"]) if item["caption_id"].isdigit() else item["caption_id"])
    return dict(captions)


def read_audio_metadata(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for parts in csv.reader(handle, delimiter="\t"):
            if len(parts) < 8:
                raise ValueError(f"malformed audio metadata row: {parts!r}")
            track_id = normalize_track_id(parts[0])
            rows[track_id] = {
                "artist_id": normalize_track_id(parts[1]),
                "album_id": normalize_track_id(parts[2]),
                "title": parts[3].strip(),
                "artist": parts[4].strip(),
                "album": parts[5].strip(),
                "release_date": parts[6].strip(),
                "source_url": parts[7].strip(),
            }
    return rows


def read_jamendo_metadata(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for parts in csv.reader(handle, delimiter="\t"):
            if len(parts) < 5:
                raise ValueError(f"malformed MTG-Jamendo row: {parts!r}")
            track_id = normalize_track_id(parts[0])
            tag_groups: dict[str, list[str]] = defaultdict(list)
            for raw_tag in parts[5:]:
                namespace, separator, value = raw_tag.partition("---")
                if separator and value:
                    tag_groups[namespace].append(value)
            rows[track_id] = {
                "artist_id": normalize_track_id(parts[1]),
                "album_id": normalize_track_id(parts[2]),
                "audio_relpath": str(PurePosixPath(parts[3].strip())),
                "duration_seconds": float(parts[4]),
                "genres": sorted(set(tag_groups.get("genre", []))),
                "instruments": sorted(set(tag_groups.get("instrument", []))),
                "moods_themes": sorted(set(tag_groups.get("mood/theme", []))),
            }
    return rows


def parse_license_url(url: str) -> dict[str, Any]:
    match = LICENSE_URL_RE.search(url.strip())
    if not match:
        if ART_LIBRE_URL_RE.search(url.strip()):
            return {
                "id": "LICENCE-ART-LIBRE",
                "url": url.strip(),
                "code": "lal",
                "version": None,
                "jurisdiction": "france",
                "attribution_required": True,
                "noncommercial_only": False,
                "no_derivatives": False,
                "share_alike": True,
                "public_playback_allowed_with_attribution": True,
                "commercial_demo_allowed": True,
                "transformations_allowed": True,
            }
        raise ValueError(f"unrecognised open-content licence URL: {url!r}")
    code = match.group("code").lower()
    version = match.group("version").rstrip(".")
    jurisdiction = (match.group("jurisdiction") or "international").lower()
    licence_id = f"CC-{code.upper()}-{version}"
    if jurisdiction != "international":
        licence_id += f"-{jurisdiction.upper()}"
    return {
        "id": licence_id,
        "url": url.strip(),
        "code": code,
        "version": version,
        "jurisdiction": jurisdiction,
        "attribution_required": True,
        "noncommercial_only": "nc" in code.split("-"),
        "no_derivatives": "nd" in code.split("-"),
        "share_alike": "sa" in code.split("-"),
        "public_playback_allowed_with_attribution": True,
        "commercial_demo_allowed": "nc" not in code.split("-"),
        "transformations_allowed": "nd" not in code.split("-"),
    }


def read_licenses(path: Path) -> dict[str, dict[str, Any]]:
    """Parse four-line records while preserving the upstream attribution evidence."""

    blocks = [block.strip() for block in re.split(r"(?m)^--\s*$", path.read_text(encoding="utf-8")) if block.strip()]
    rows: dict[str, dict[str, Any]] = {}
    for block in blocks:
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        if len(lines) < 3:
            raise ValueError(f"malformed licence block: {block!r}")
        audio_relpath = str(PurePosixPath(lines[0]))
        track_id = normalize_track_id(Path(audio_relpath).stem)
        licence_match = OPEN_LICENSE_URL_RE.search(lines[2])
        if not licence_match:
            raise ValueError(f"licence URL missing for track {track_id}")
        licence = parse_license_url(licence_match.group(0))
        licence.update(
            {
                "audio_relpath": audio_relpath,
                "attribution_text": lines[1],
                "license_statement": lines[2],
            }
        )
        if track_id in rows:
            raise ValueError(f"duplicate licence block for track {track_id}")
        rows[track_id] = licence
    return rows


def _resolve_audio_path(audio_root: Path, relpath: str) -> Path:
    candidates = [audio_root / PurePosixPath(relpath)]
    if relpath.lower().endswith(".mp3"):
        candidates.append(audio_root / PurePosixPath(relpath[:-4] + ".2min.mp3"))
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return candidates[0]


def build_manifest_rows(
    metadata_dir: Path,
    *,
    audio_root: Path,
    subset: str = "validated",
    max_tracks: int | None = None,
) -> list[dict[str, Any]]:
    """Join all source tables and return deterministic track-level rows."""

    if subset not in {"validated", "full"}:
        raise ValueError("subset must be 'validated' or 'full'")
    captions = read_captions(metadata_dir / "song_describer.csv", subset)
    audio_metadata = read_audio_metadata(metadata_dir / "audio_metadata.tsv")
    jamendo = read_jamendo_metadata(metadata_dir / "song_describer_14_04_23.mtg-jamendo.tsv")
    licences = read_licenses(metadata_dir / "audio_licenses.txt")
    track_ids = sorted(captions, key=int)
    if max_tracks is not None:
        track_ids = track_ids[: max(0, max_tracks)]

    rows: list[dict[str, Any]] = []
    for track_id in track_ids:
        missing = [
            name
            for name, table in (("audio_metadata", audio_metadata), ("jamendo", jamendo), ("licence", licences))
            if track_id not in table
        ]
        if missing:
            raise ValueError(f"track {track_id} missing joins: {', '.join(missing)}")
        meta = audio_metadata[track_id]
        tag_meta = jamendo[track_id]
        licence = licences[track_id]
        if tag_meta["audio_relpath"] != licence["audio_relpath"]:
            raise ValueError(
                f"track {track_id} audio path disagreement: {tag_meta['audio_relpath']} != {licence['audio_relpath']}"
            )
        audio_path = _resolve_audio_path(audio_root, tag_meta["audio_relpath"])
        audio_available = audio_path.is_file()
        rows.append(
            {
                "schema_version": MANIFEST_SCHEMA_VERSION,
                "dataset_id": DATASET_ID,
                "dataset_version": DATASET_VERSION,
                "dataset_subset": subset,
                "dataset_license": DATASET_LICENSE,
                "dataset_url": DATASET_URL,
                "source_repository": SOURCE_REPOSITORY,
                "song_id": f"sdd-{track_id}",
                "track_id": track_id,
                "title": meta["title"],
                "artist": meta["artist"],
                "album": meta["album"],
                "artist_id": meta["artist_id"],
                "album_id": meta["album_id"],
                "release_date": meta["release_date"],
                "source_url": meta["source_url"],
                "attribution": licence["attribution_text"],
                "license_id": licence["id"],
                "license_url": licence["url"],
                "duration_seconds": tag_meta["duration_seconds"],
                "genres": tag_meta["genres"],
                "instruments": tag_meta["instruments"],
                "moods_themes": tag_meta["moods_themes"],
                "captions": captions[track_id],
                "caption_count": len(captions[track_id]),
                "audio_relpath": tag_meta["audio_relpath"],
                "audio_available": audio_available,
                "audio_size_bytes": audio_path.stat().st_size if audio_available else None,
                "audio_sha256": hash_file(audio_path) if audio_available else None,
                "audio_license": licence,
                "usage_policy": {
                    "purpose": ["retrieval", "evaluation", "public_demo_playback"],
                    "planner_training_allowed": False,
                    "planner_training_reason": "The dataset authors discourage training use; keep SDD out of planner SFT.",
                    "requires_visible_attribution": True,
                    "noncommercial_only": licence["noncommercial_only"],
                    "no_derivatives": licence["no_derivatives"],
                    "share_alike": licence["share_alike"],
                    "audio_delivery_mode": "original_archive_bytes_only",
                    "audio_transforms_allowed": not licence["no_derivatives"],
                    "muq_feature_generation_only": True,
                    "muq_rewrites_audio": False,
                },
            }
        )
    return rows


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")
    temporary.replace(path)


def build_audit(rows: Sequence[Mapping[str, Any]], *, manifest_path: Path) -> dict[str, Any]:
    licence_counts = Counter(str(row["audio_license"]["id"]) for row in rows)
    provenance_coverage = {
        "attribution": sum(bool(str(row.get("attribution") or "").strip()) for row in rows),
        "license_url": sum(bool(str(row.get("license_url") or "").strip()) for row in rows),
        "source_url": sum(bool(str(row.get("source_url") or "").strip()) for row in rows),
    }
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "generated_at": utc_now(),
        "dataset_id": DATASET_ID,
        "dataset_version": DATASET_VERSION,
        "zenodo_record_id": ZENODO_RECORD_ID,
        "subset": rows[0]["dataset_subset"] if rows else "unknown",
        "track_count": len(rows),
        "caption_count": sum(int(row["caption_count"]) for row in rows),
        "audio_available": sum(bool(row["audio_available"]) for row in rows),
        "audio_missing": sum(not bool(row["audio_available"]) for row in rows),
        "noncommercial_only_tracks": sum(bool(row["audio_license"]["noncommercial_only"]) for row in rows),
        "no_derivatives_tracks": sum(bool(row["audio_license"]["no_derivatives"]) for row in rows),
        "share_alike_tracks": sum(bool(row["audio_license"]["share_alike"]) for row in rows),
        "commercial_demo_allowed_tracks": sum(bool(row["audio_license"]["commercial_demo_allowed"]) for row in rows),
        "transformations_allowed_tracks": sum(bool(row["audio_license"]["transformations_allowed"]) for row in rows),
        "license_counts": dict(sorted(licence_counts.items())),
        "unknown_or_missing_licenses": 0,
        "provenance_coverage": provenance_coverage,
        "manifest_file": manifest_path.name,
        "manifest_sha256": hash_file(manifest_path),
        "policy_notes": [
            "Song Describer is prepared for retrieval, evaluation, and attributed public-demo playback, not planner training.",
            "Per-track Creative Commons conditions prevail over the dataset-level CC-BY-SA-4.0 metadata licence.",
            "NC tracks require a non-commercial deployment; ND tracks must be served unmodified and need review before transformations.",
            "ND playback serves original archive bytes only: no transcoding, trimming, gain changes, or remuxing.",
            "MuQ/M2D may read decoded samples in memory to produce non-audio feature vectors; they never rewrite the source MP3.",
            "This machine-readable audit is provenance support, not legal advice.",
        ],
    }


def safe_archive_members(archive: zipfile.ZipFile) -> list[zipfile.ZipInfo]:
    members: list[zipfile.ZipInfo] = []
    for info in archive.infolist():
        normalized = PurePosixPath(info.filename.replace("\\", "/"))
        if normalized.is_absolute() or ".." in normalized.parts:
            raise ValueError(f"unsafe path in audio archive: {info.filename!r}")
        if not info.is_dir():
            members.append(info)
    return members


def extract_selected_audio(archive_path: Path, audio_root: Path, relpaths: Iterable[str]) -> int:
    """Extract selected tracks without trusting archive paths or materialising all 706 tracks."""

    with zipfile.ZipFile(archive_path) as archive:
        return _extract_selected_from_archive(archive, audio_root, relpaths)


def extract_selected_audio_from_remote(remote: RemoteFile, audio_root: Path, relpaths: Iterable[str]) -> int:
    """Range-read selected members without first downloading the 3.09 GiB ZIP.

    This is the fast smoke-test lane.  The pinned archive identity still comes
    from Zenodo, while every extracted MP3 receives its own SHA-256 in the
    manifest.  A production/full-catalog materialisation should use
    ``--download-audio --extract-audio`` so the complete ZIP MD5 is also checked.
    """

    try:
        import fsspec
    except ImportError as exc:  # pragma: no cover - environment-specific dependency gate
        raise RuntimeError(
            "--stream-audio needs fsspec (already installed with the ML runtime); "
            "install fsspec or use --download-audio --extract-audio"
        ) from exc

    with fsspec.open(
        remote.url,
        mode="rb",
        block_size=8 * 1024 * 1024,
        cache_type="readahead",
        headers={"User-Agent": "SoulTuner-Open-Audio-Pipeline/1.0"},
    ) as remote_handle:
        with zipfile.ZipFile(remote_handle) as archive:
            return _extract_selected_from_archive(archive, audio_root, relpaths)


def _extract_selected_from_archive(
    archive: zipfile.ZipFile,
    audio_root: Path,
    relpaths: Iterable[str],
) -> int:
    wanted = {str(PurePosixPath(path)) for path in relpaths}
    extracted = 0
    audio_root.mkdir(parents=True, exist_ok=True)
    members = safe_archive_members(archive)
    by_suffix: dict[str, zipfile.ZipInfo] = {}
    for info in members:
        normalized = str(PurePosixPath(info.filename.replace("\\", "/")))
        by_suffix[normalized] = info
        by_suffix.setdefault(normalized.removeprefix("audio/"), info)
    for relpath in sorted(wanted):
        info = by_suffix.get(relpath) or by_suffix.get(f"audio/{relpath}")
        if info is None and relpath.lower().endswith(".mp3"):
            stemmed = relpath[:-4] + ".2min.mp3"
            info = by_suffix.get(stemmed) or by_suffix.get(f"audio/{stemmed}")
        if info is None:
            raise FileNotFoundError(f"audio archive does not contain {relpath}")
        destination = _resolve_audio_path(audio_root, relpath)
        if destination.is_file():
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        with archive.open(info) as source, tempfile.NamedTemporaryFile(
            dir=destination.parent, prefix=destination.name, suffix=".tmp", delete=False
        ) as target:
            shutil.copyfileobj(source, target, length=CHUNK_BYTES)
            temporary = Path(target.name)
        temporary.replace(destination)
        extracted += 1
    return extracted


def source_inventory(record: Mapping[str, Any], files: Mapping[str, RemoteFile]) -> dict[str, Any]:
    metadata = record.get("metadata") or {}
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "record_id": ZENODO_RECORD_ID,
        "record_url": DATASET_URL,
        "doi": metadata.get("doi") or record.get("doi") or "10.5281/zenodo.10072001",
        "title": metadata.get("title") or "Song Describer Dataset",
        "version": metadata.get("version") or DATASET_VERSION,
        "dataset_license": metadata.get("license") or {"id": "cc-by-sa-4.0"},
        "files": {
            name: {
                "size": remote.size,
                "checksum": f"{remote.checksum_type}:{remote.checksum}",
                "url": remote.url,
            }
            for name, remote in sorted(files.items())
        },
    }


def prepare(
    cache_dir: Path,
    *,
    subset: str = "validated",
    download_audio: bool = False,
    extract_audio: bool = False,
    stream_audio: bool = False,
    refresh: bool = False,
    max_tracks: int | None = None,
) -> dict[str, Any]:
    """Run the complete metadata/audit pipeline and optionally materialise audio."""

    if stream_audio and (download_audio or extract_audio):
        raise ValueError("--stream-audio cannot be combined with --download-audio/--extract-audio")
    if extract_audio and not download_audio:
        download_audio = True
    cache_dir = cache_dir.expanduser().resolve()
    metadata_dir = cache_dir / "metadata"
    audio_root = cache_dir / "audio"
    artifact_dir = cache_dir / "artifacts"
    record, remotes = fetch_record()
    metadata_dir.mkdir(parents=True, exist_ok=True)
    for name in METADATA_FILENAMES:
        download_file(remotes[name], metadata_dir / name, refresh=refresh)

    rows = build_manifest_rows(
        metadata_dir,
        audio_root=audio_root,
        subset=subset,
        max_tracks=max_tracks,
    )
    if download_audio:
        archive_path = download_file(remotes[AUDIO_ARCHIVE_NAME], cache_dir / AUDIO_ARCHIVE_NAME, refresh=refresh)
        if extract_audio:
            extract_selected_audio(archive_path, audio_root, (str(row["audio_relpath"]) for row in rows))
            rows = build_manifest_rows(
                metadata_dir,
                audio_root=audio_root,
                subset=subset,
                max_tracks=max_tracks,
            )
    elif stream_audio:
        extract_selected_audio_from_remote(
            remotes[AUDIO_ARCHIVE_NAME],
            audio_root,
            (str(row["audio_relpath"]) for row in rows),
        )
        rows = build_manifest_rows(
            metadata_dir,
            audio_root=audio_root,
            subset=subset,
            max_tracks=max_tracks,
        )

    manifest_path = artifact_dir / f"song_describer_{subset}.jsonl"
    write_jsonl(manifest_path, rows)
    audit = build_audit(rows, manifest_path=manifest_path)
    inventory_path = artifact_dir / "source_inventory.json"
    audit_path = artifact_dir / f"license_audit_{subset}.json"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    inventory_path.write_text(
        json.dumps(source_inventory(record, remotes), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {
        "cache_dir": str(cache_dir),
        "manifest": str(manifest_path),
        "source_inventory": str(inventory_path),
        "license_audit": str(audit_path),
        **{key: audit[key] for key in ("track_count", "caption_count", "audio_available", "audio_missing")},
    }


def verify(cache_dir: Path, *, subset: str = "validated") -> dict[str, Any]:
    """Verify downloaded metadata, manifest identity, licences, and local audio hashes."""

    cache_dir = cache_dir.expanduser().resolve()
    _, remotes = fetch_record()
    metadata_dir = cache_dir / "metadata"
    for name in METADATA_FILENAMES:
        verify_remote_file(metadata_dir / name, remotes[name])
    manifest_path = cache_dir / "artifacts" / f"song_describer_{subset}.jsonl"
    audit_path = cache_dir / "artifacts" / f"license_audit_{subset}.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    actual_manifest_sha = hash_file(manifest_path)
    if audit.get("manifest_sha256") != actual_manifest_sha:
        raise ValueError("manifest SHA-256 does not match licence audit")

    checked_audio = 0
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            row = json.loads(line)
            parse_license_url(row["audio_license"]["url"])
            if row.get("audio_available"):
                audio_path = _resolve_audio_path(cache_dir / "audio", row["audio_relpath"])
                if not audio_path.is_file():
                    raise FileNotFoundError(f"manifest line {line_number} audio is missing: {audio_path}")
                if hash_file(audio_path) != row.get("audio_sha256"):
                    raise ValueError(f"manifest line {line_number} audio SHA-256 mismatch")
                checked_audio += 1
    return {
        "verified": True,
        "manifest": str(manifest_path),
        "manifest_sha256": actual_manifest_sha,
        "track_count": int(audit["track_count"]),
        "checked_audio": checked_audio,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare the license-audited Song Describer open-audio catalog")
    parser.add_argument("command", choices=("prepare", "verify"), nargs="?", default="prepare")
    parser.add_argument("--cache-dir", type=Path, default=default_cache_dir())
    parser.add_argument("--subset", choices=("validated", "full"), default="validated")
    parser.add_argument("--download-audio", action="store_true", help="Download and verify the 3.09 GiB audio.zip")
    parser.add_argument("--extract-audio", action="store_true", help="Extract only selected tracks (implies download)")
    parser.add_argument(
        "--stream-audio",
        action="store_true",
        help="Range-read only selected tracks from Zenodo; ideal with --max-tracks for a smoke test",
    )
    parser.add_argument("--refresh", action="store_true", help="Redownload even if local checksum is valid")
    parser.add_argument("--max-tracks", type=int, default=None, help="Deterministic smoke-test prefix; omit for all tracks")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    started = time.monotonic()
    if args.command == "verify":
        result = verify(args.cache_dir, subset=args.subset)
    else:
        result = prepare(
            args.cache_dir,
            subset=args.subset,
            download_audio=args.download_audio,
            extract_audio=args.extract_audio,
            stream_audio=args.stream_audio,
            refresh=args.refresh,
            max_tracks=args.max_tracks,
        )
    result["elapsed_seconds"] = round(time.monotonic() - started, 3)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
