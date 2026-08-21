"""Materialise and verify the small licensed-audio catalogue for the Space."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path, PurePosixPath
from typing import Any


DEFAULT_DATASET_ID = "hgsanyang/SoulTuner-Open-Audio-Demo"


def _default_root() -> Path:
    workspace = Path("/mnt/workspace")
    if workspace.is_dir() and os.access(workspace, os.W_OK):
        return workspace / "soultuner" / "open_audio"
    return Path(__file__).resolve().parent / "open_audio"


def _configure_paths() -> tuple[Path, Path, Path]:
    root = Path(os.getenv("SOULTUNER_OPEN_AUDIO_DIR", "") or _default_root()).resolve()
    catalog = Path(
        os.getenv("SOULTUNER_CATALOG_PATH", "") or root / "catalog.jsonl"
    ).resolve()
    audio_root = Path(
        os.getenv("SOULTUNER_AUDIO_ROOT", "") or root / "audio"
    ).resolve()
    os.environ["SOULTUNER_OPEN_AUDIO_DIR"] = str(root)
    os.environ["SOULTUNER_CATALOG_PATH"] = str(catalog)
    os.environ["SOULTUNER_AUDIO_ROOT"] = str(audio_root)
    return root, catalog, audio_root


def verify_open_audio(catalog: Path, audio_root: Path) -> int:
    """Verify paths, provenance, licences and original-file checksums."""

    if not catalog.is_file():
        raise FileNotFoundError(f"open-audio catalog is missing: {catalog}")
    rows = [
        json.loads(line)
        for line in catalog.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise ValueError("open-audio catalog is empty")
    root = audio_root.resolve()
    for row in rows:
        relpath = str(PurePosixPath(str(row.get("audio_relpath") or "")))
        expected = str(row.get("audio_sha256") or "").casefold()
        licence = str(row.get("license_url") or "")
        source = str(row.get("source_url") or "")
        attribution = str(row.get("attribution") or "")
        candidate = (root / relpath).resolve()
        candidate.relative_to(root)
        if not all((expected, licence.startswith("http"), source.startswith("http"), attribution)):
            raise ValueError(f"incomplete open-audio provenance: {row.get('song_id')}")
        if not candidate.is_file():
            raise FileNotFoundError(f"open-audio file is missing: {relpath}")
        actual = hashlib.sha256(candidate.read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"open-audio SHA-256 mismatch: {row.get('song_id')}")
    return len(rows)


def _fallback_to_bundled_catalog() -> dict[str, Any]:
    bundled = Path(__file__).resolve().parent / "data"
    os.environ["SOULTUNER_CATALOG_PATH"] = str((bundled / "catalog.jsonl").resolve())
    os.environ["SOULTUNER_AUDIO_ROOT"] = str((bundled / "audio").resolve())
    return {"state": "fallback", "tracks": 0}


def materialize_open_audio() -> dict[str, Any]:
    """Refresh the public library manifest, then reuse persistent audio blobs.

    ``modelscope download`` is content-addressed and resumes into the durable
    directory, so checking the remote revision on every fresh process does not
    download unchanged audio again.  This matters when a small bootstrap bundle
    has already verified successfully but the published dataset later expands.
    A failed refresh reuses a valid persistent copy before falling back to the
    synthetic catalogue.
    """

    if os.getenv("SOULTUNER_ENABLE_OPEN_AUDIO", "1").strip() != "1":
        return _fallback_to_bundled_catalog()

    root, catalog, audio_root = _configure_paths()
    if os.getenv("SOULTUNER_OPEN_AUDIO_ALREADY_VERIFIED", "0").strip() == "1":
        rows = [
            json.loads(line)
            for line in catalog.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if not rows:
            raise ValueError("open-audio catalog is empty after startup verification")
        print(
            f"SoulTuner open-audio catalog reused after startup verification: {len(rows)} tracks",
            flush=True,
        )
        return {"state": "ready", "tracks": len(rows), "root": str(root)}
    dataset_id = os.getenv("SOULTUNER_OPEN_AUDIO_DATASET_ID", DEFAULT_DATASET_ID)
    revision = os.getenv("SOULTUNER_OPEN_AUDIO_REVISION", "master")
    root.mkdir(parents=True, exist_ok=True)
    command = [
        "modelscope",
        "download",
        dataset_id,
        "--repo-type",
        "dataset",
        "--revision",
        revision,
        "--local-dir",
        str(root),
        "--max-workers",
        "4",
    ]
    try:
        timeout = int(os.getenv("SOULTUNER_OPEN_AUDIO_DOWNLOAD_TIMEOUT", "1800"))
        subprocess.run(command, check=True, timeout=timeout)
        tracks = verify_open_audio(catalog, audio_root)
    except (OSError, subprocess.SubprocessError, FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
        try:
            tracks = verify_open_audio(catalog, audio_root)
        except (FileNotFoundError, ValueError, json.JSONDecodeError):
            print(
                f"SoulTuner open-audio startup failed; using bundled catalog: {type(exc).__name__}: {exc}",
                flush=True,
            )
            return _fallback_to_bundled_catalog()
        print(
            f"SoulTuner open-audio refresh failed; reusing {tracks} cached tracks: {type(exc).__name__}",
            flush=True,
        )
        return {"state": "ready", "tracks": tracks, "root": str(root)}

    print(f"SoulTuner open-audio catalog downloaded and verified: {tracks} tracks", flush=True)
    return {"state": "ready", "tracks": tracks, "root": str(root)}


def startup_markdown(status: dict[str, Any]) -> str:
    if status.get("state") == "ready":
        return f"公开音频：`{status.get('tracks', 0)} 首已校验并可试听` · 含许可证、署名、来源与 SHA-256。"
    return "公开音频：`暂用合成目录` · 授权音频下载失败，详情见运行日志。"
