"""Persistent MI308X audio-vector backfill for the public Space catalogue.

The online Gradio process stays responsive while a separate worker fills MuQ
and OMAR vectors in Aura. M2D can be enabled for a CPU compatibility corpus, but is not
part of the default readiness gate. The worker is deliberately
idempotent: every restart queries Aura for missing vectors and resumes from the
first incomplete song, while model/audio caches remain under ``/mnt/workspace``.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import threading
import time
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Iterable


EXPECTED_DIMS = {"muq_embedding": 512, "m2d2_embedding": 768, "omar_embedding": 1024}
M2D_URL = "https://github.com/nttcslab/m2d/releases/download/v0.5.0/m2d_clap_vit_base-80x1001p16x16p16kpBpTI-2025.zip"
M2D_DIRNAME = "m2d_clap_vit_base-80x1001p16x16p16kpBpTI-2025"
M2D_SHA256 = "238521603c04862ab151cdd80980b591cb36ebe844d43203992fac9ef085c8a1"
BERT_REPO_ID = "AI-ModelScope/bert-base-uncased"
BERT_FILES = (
    "config.json",
    "model.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.txt",
)
_PACKAGE_GROUPS = {
    "muq_embedding": ("muq==0.1.0", "librosa>=0.11.0"),
    "omar_embedding": ("omar-rq==0.2.1", "librosa>=0.11.0"),
    "m2d2_embedding": ("timm==1.0.28", "librosa>=0.11.0", "nnAudio>=0.3.4"),
}
_MODULE_GROUPS = {
    "muq_embedding": ("muq", "librosa"),
    "omar_embedding": ("omar_rq", "librosa"),
    "m2d2_embedding": ("timm", "librosa", "nnAudio"),
}
_worker_process: subprocess.Popen[bytes] | None = None
_worker_log_thread: threading.Thread | None = None


def _workspace_root() -> Path:
    configured = os.getenv("SOULTUNER_WORKSPACE_ROOT", "").strip()
    if configured:
        return Path(configured).expanduser().resolve()
    workspace = Path("/mnt/workspace")
    if workspace.is_dir() and os.access(workspace, os.W_OK):
        return workspace / "soultuner"
    return Path(__file__).resolve().parent / ".runtime"


def _status_path() -> Path:
    return _workspace_root() / "enrichment" / "status.json"


def _write_status(state: str, **details: Any) -> None:
    path = _status_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"state": state, "updated_at": int(time.time()), **details}
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)
    print(f"SoulTuner enrichment: {state} {details}", flush=True)


def enrichment_status() -> dict[str, Any]:
    path = _status_path()
    if not path.is_file():
        return {"state": "not-started"}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        return {"state": "unknown"}
    return payload if isinstance(payload, dict) else {"state": "unknown"}


def status_markdown() -> str:
    status = enrichment_status()
    state = str(status.get("state") or "not-started")
    completed = int(status.get("completed") or 0)
    total = int(status.get("total") or 0)
    family = str(status.get("family") or "")
    if state == "ready":
        return f"音频向量：`MuQ + OMAR 已就绪` · **{completed or total}** 首完成 · M2D 仅用于纯 CPU 档。"
    if state in {"waiting-endpoint", "installing", "downloading-models", "running"}:
        label = family or state
        return f"音频向量：`后台补齐中` · {label} · **{completed}/{total or '?'}**。"
    if state == "disabled":
        return "音频向量：`未启用`。"
    if state == "failed":
        return "音频向量：`后台任务异常` · 已保留进度，下次启动会从缺失项继续。"
    return "音频向量：`等待后台任务`。"


def _has_aura_credentials() -> bool:
    username = os.getenv("NEO4J_USERNAME", "").strip() or os.getenv("NEO4J_USER", "").strip()
    return all((os.getenv("NEO4J_URI", "").strip(), username, os.getenv("NEO4J_PASSWORD", "").strip()))


def _mirror_output(stream: Any, log_path: Path) -> None:
    try:
        with log_path.open("ab", buffering=0) as log_file:
            for raw_line in iter(stream.readline, b""):
                log_file.write(raw_line)
                line = raw_line.decode("utf-8", errors="replace").rstrip()
                if line:
                    print(f"[SoulTuner Enrichment] {line}", flush=True)
    finally:
        stream.close()


def launch_enrichment_if_requested(track_count: int) -> dict[str, Any]:
    """Launch one resumable worker without blocking Gradio startup."""

    global _worker_process, _worker_log_thread
    if os.getenv("SOULTUNER_ENABLE_AUDIO_ENRICHMENT", "1").strip() != "1":
        _write_status("disabled")
        return {"state": "disabled"}
    minimum = int(os.getenv("SOULTUNER_ENRICHMENT_MIN_TRACKS", "700"))
    if track_count < minimum:
        return {"state": "waiting-catalog", "tracks": track_count, "minimum": minimum}
    if not _has_aura_credentials():
        return {"state": "waiting-aura"}
    if _worker_process is not None and _worker_process.poll() is None:
        return {"state": "running", "pid": _worker_process.pid}

    root = Path(__file__).resolve().parent
    log_path = _workspace_root() / "logs" / "audio-enrichment.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, str(Path(__file__).resolve()), "--worker"]
    _worker_process = subprocess.Popen(
        command,
        cwd=root,
        env=os.environ.copy(),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    if _worker_process.stdout is None:  # pragma: no cover
        raise RuntimeError("enrichment stdout pipe was not created")
    _worker_log_thread = threading.Thread(
        target=_mirror_output,
        args=(_worker_process.stdout, log_path),
        name="soultuner-enrichment-log",
        daemon=True,
    )
    _worker_log_thread.start()
    return {"state": "running", "pid": _worker_process.pid}


def _wait_for_endpoint(timeout_seconds: float = 1800.0) -> None:
    base_url = os.getenv("SOULTUNER_PLANNER_BASE_URL", "http://127.0.0.1:8000/v1").rstrip("/")
    required = {
        os.getenv("SOULTUNER_CHAT_MODEL", "qwen3.6-35b-a3b"),
        os.getenv("SOULTUNER_PLANNER_MODEL", "soultuner-v4.2-35b"),
    }
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        try:
            request = urllib.request.Request(f"{base_url}/models", headers={"Accept": "application/json"})
            api_key = (
                os.getenv("SOULTUNER_PLANNER_API_KEY", "").strip() or os.getenv("SOULTUNER_SERVE_API_KEY", "").strip()
            )
            if api_key:
                request.add_header("Authorization", f"Bearer {api_key}")
            with urllib.request.urlopen(request, timeout=5) as response:
                body = json.loads(response.read().decode("utf-8"))
            available = {
                str(item.get("id")) for item in body.get("data", []) if isinstance(item, dict) and item.get("id")
            }
            if required <= available:
                return
        except Exception:
            pass
        _write_status("waiting-endpoint")
        time.sleep(10)
    raise TimeoutError("35B dual-role endpoint did not become ready")


def _required_families() -> tuple[str, ...]:
    configured = os.getenv(
        "SOULTUNER_EMBEDDING_FAMILIES",
        "muq_embedding,omar_embedding",
    )
    families = tuple(
        dict.fromkeys(part.strip() for part in configured.split(",") if part.strip() in EXPECTED_DIMS)
    )
    if os.getenv("SOULTUNER_BACKFILL_M2D_FALLBACK", "0").strip() == "1" and "m2d2_embedding" not in families:
        families += ("m2d2_embedding",)
    return families or ("muq_embedding", "omar_embedding")


def _ensure_packages(families: Iterable[str]) -> None:
    selected = tuple(families)
    modules = tuple(dict.fromkeys(module for family in selected for module in _MODULE_GROUPS[family]))
    if all(importlib.util.find_spec(name) is not None for name in modules):
        return
    packages = tuple(dict.fromkeys(package for family in selected for package in _PACKAGE_GROUPS[family]))
    _write_status("installing")
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "--disable-pip-version-check", *packages],
        check=True,
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _ensure_m2d_checkpoint() -> Path:
    cache = _workspace_root() / "model_cache" / "m2d_clap"
    checkpoint = cache / M2D_DIRNAME / "checkpoint-30.pth"
    if checkpoint.is_file() and _sha256(checkpoint) == M2D_SHA256:
        return checkpoint

    _write_status("downloading-models", family="m2d2")
    cache.mkdir(parents=True, exist_ok=True)
    archive = cache / "m2d_clap_2025.zip"
    partial = archive.with_suffix(".zip.part")
    curl = [
        "curl",
        "--fail",
        "--location",
        "--retry",
        "8",
        "--retry-delay",
        "5",
        "--continue-at",
        "-",
        "--output",
        str(partial),
        M2D_URL,
    ]
    try:
        subprocess.run(curl, check=True)
    except (OSError, subprocess.CalledProcessError):
        urllib.request.urlretrieve(M2D_URL, partial)
    os.replace(partial, archive)
    with zipfile.ZipFile(archive) as bundle:
        if bundle.testzip() is not None:
            raise RuntimeError("M2D archive integrity check failed")
        bundle.extractall(cache)
    if not checkpoint.is_file() or _sha256(checkpoint) != M2D_SHA256:
        raise RuntimeError("M2D checkpoint SHA-256 mismatch")
    return checkpoint


def _ensure_bert_snapshot() -> Path:
    """Persist the M2D text tower on ModelScope's reachable model hub."""

    target = _workspace_root() / "model_cache" / "bert-base-uncased"
    if all((target / name).is_file() for name in BERT_FILES):
        return target
    _write_status("downloading-models", family="bert-text")
    target.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "modelscope",
            "download",
            BERT_REPO_ID,
            *BERT_FILES,
            "--local-dir",
            str(target),
            "--max-workers",
            "4",
        ],
        check=True,
    )
    missing = [name for name in BERT_FILES if not (target / name).is_file()]
    if missing:
        raise RuntimeError("BERT snapshot is incomplete: " + ", ".join(missing))
    return target


def _catalog_paths() -> tuple[Path, Path]:
    root = Path(os.getenv("SOULTUNER_OPEN_AUDIO_DIR", "") or _workspace_root() / "open_audio").resolve()
    catalog = Path(os.getenv("SOULTUNER_CATALOG_PATH", "") or root / "catalog.jsonl").resolve()
    audio_root = Path(os.getenv("SOULTUNER_AUDIO_ROOT", "") or root / "audio").resolve()
    return catalog, audio_root


def _catalog_rows() -> list[dict[str, Any]]:
    catalog, audio_root = _catalog_paths()
    rows: list[dict[str, Any]] = []
    for line in catalog.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        relpath = PurePosixPath(str(row.get("audio_relpath") or ""))
        path = (audio_root / relpath).resolve()
        path.relative_to(audio_root)
        if not path.is_file():
            raise FileNotFoundError(path)
        rows.append({**row, "audio_file": path})
    if len(rows) < int(os.getenv("SOULTUNER_ENRICHMENT_MIN_TRACKS", "700")):
        raise RuntimeError(f"full public catalogue is not ready: {len(rows)} tracks")
    return rows


def _driver():
    from neo4j import GraphDatabase

    username = os.getenv("NEO4J_USERNAME", "").strip() or os.getenv("NEO4J_USER", "").strip()
    return GraphDatabase.driver(
        os.environ["NEO4J_URI"],
        auth=(username, os.environ["NEO4J_PASSWORD"]),
    )


def _missing_ids(
    driver: Any,
    family: str,
    expected: int,
    song_ids: set[str],
) -> set[str]:
    database = os.getenv("NEO4J_DATABASE", "").strip() or None
    query = f"""
    MATCH (s:Song)
    WHERE toString(s.music_id) IN $song_ids
      AND coalesce(size(s.{family}), 0) <> $expected
    RETURN toString(s.music_id) AS music_id
    """
    records, _, _ = driver.execute_query(
        query,
        expected=expected,
        song_ids=sorted(song_ids),
        database_=database,
    )
    return {str(record["music_id"]) for record in records if record.get("music_id")}


def _upsert_catalog_rows(driver: Any, rows: list[dict[str, Any]]) -> None:
    """Make every playable catalogue row addressable by Aura vector indexes."""

    database = os.getenv("NEO4J_DATABASE", "").strip() or None
    query = """
    UNWIND $rows AS row
    MERGE (s:Song {music_id: row.song_id})
    SET s.source_id = row.source_id,
        s.dataset = row.dataset,
        s.title = row.title,
        s.artist = row.artist,
        s.genres = row.genres,
        s.language = row.language,
        s.audio_relpath = row.audio_relpath,
        s.cover_url = row.cover_url,
        s.license = row.license,
        s.license_url = row.license_url,
        s.source_url = row.source_url,
        s.updated_at = timestamp()
    MERGE (a:Artist {name: row.artist})
    MERGE (s)-[:PERFORMED_BY]->(a)
    """
    batch_size = 250
    for offset in range(0, len(rows), batch_size):
        payload = []
        for row in rows[offset : offset + batch_size]:
            song_id = str(row["song_id"])
            default_dataset = "song_describer_full" if song_id.startswith("sdd-") else "public_open_audio"
            payload.append(
                {
                    "song_id": song_id,
                    "source_id": str(row.get("source_id") or song_id),
                    "dataset": str(row.get("dataset") or default_dataset),
                    "title": str(row.get("title") or ""),
                    "artist": str(row.get("artist") or "未知艺人"),
                    "genres": list(row.get("genres") or row.get("tags") or [])[:16],
                    "language": row.get("language"),
                    "audio_relpath": str(row.get("audio_relpath") or ""),
                    "cover_url": str(row.get("cover_url") or ""),
                    "license": str(row.get("license") or row.get("license_id") or ""),
                    "license_url": str(row.get("license_url") or ""),
                    "source_url": str(row.get("source_url") or ""),
                }
            )
        driver.execute_query(query, rows=payload, database_=database)


def _update_vector(
    driver: Any,
    song_id: str,
    family: str,
    vector: list[float],
    required_families: Iterable[str] | None = None,
) -> None:
    expected = EXPECTED_DIMS[family]
    if len(vector) != expected:
        raise ValueError(f"{family} dimension mismatch: {len(vector)} != {expected}")
    database = os.getenv("NEO4J_DATABASE", "").strip() or None
    required = tuple(required_families or _required_families())
    ready_checks = " AND ".join(
        f"size(coalesce(s.{name}, [])) = {EXPECTED_DIMS[name]}" for name in required
    )
    query = f"""
    MATCH (s:Song {{music_id: $song_id}})
    SET s.{family} = $vector
    WITH s
    SET s.enrichment_status = CASE
          WHEN {ready_checks}
          THEN 'ready' ELSE 'processing' END,
        s.enrichment_error = '',
        s.updated_at = timestamp()
    RETURN elementId(s) AS eid
    """
    records, _, _ = driver.execute_query(
        query,
        song_id=song_id,
        vector=vector,
        database_=database,
    )
    if not records:
        raise RuntimeError(f"Aura Song not found: {song_id}")


def _audio_segment(path: Path, sample_rate: int) -> Any:
    import librosa
    import numpy as np

    duration = float(librosa.get_duration(path=str(path)))
    seconds = float(os.getenv("SOULTUNER_EMBED_SEGMENT_SECONDS", "10"))
    offset = max(0.0, (duration - seconds) * 0.35)
    audio, _ = librosa.load(str(path), sr=sample_rate, mono=True, offset=offset, duration=seconds)
    target = max(1, int(sample_rate * seconds))
    if audio.shape[0] < target:
        audio = np.pad(audio, (0, target - audio.shape[0]))
    return audio[:target].astype("float32", copy=False)


def _release_model(module: Any, attribute: str) -> None:
    setattr(module, attribute, None)
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _family_extractors(
    families: Iterable[str] | None = None,
) -> Iterable[tuple[str, int, Callable[[Path], list[float]], Callable[[], None]]]:
    from retrieval import audio_embedder, muq_embedder

    selected = tuple(families or _required_families())
    if "muq_embedding" in selected:
        yield (
            "muq_embedding",
            512,
            lambda path: muq_embedder.encode_audio_to_muq(_audio_segment(path, 24000), sample_rate=24000),
            lambda: _release_model(muq_embedder, "_MUQ_MODEL"),
        )
    if "omar_embedding" in selected:
        yield (
            "omar_embedding",
            1024,
            lambda path: audio_embedder.extract_audio_representation(_audio_segment(path, 16000), sample_rate=16000),
            lambda: _release_model(audio_embedder, "_OMAR_MODEL"),
        )
    if "m2d2_embedding" in selected:
        yield (
            "m2d2_embedding",
            768,
            lambda path: audio_embedder.encode_audio_to_embedding(_audio_segment(path, 16000), sample_rate=16000),
            lambda: _release_model(audio_embedder, "_M2D2_MODEL"),
        )


def _configure_model_caches(checkpoint: Path | None = None) -> None:
    workspace = _workspace_root()
    hf_home = workspace / "hf_cache"
    torch_home = workspace / "torch_cache"
    hf_home.mkdir(parents=True, exist_ok=True)
    torch_home.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(hf_home)
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(hf_home / "hub")
    os.environ["TORCH_HOME"] = str(torch_home)
    if checkpoint is not None:
        os.environ["M2D_CLAP_WEIGHT_DIR"] = str(checkpoint.parent)
        os.environ["M2D_CLAP_CHECKPOINT"] = str(checkpoint)
    os.environ["MUQ_MULAN_LOCAL_FILES_ONLY"] = "0"
    os.environ["HF_HUB_OFFLINE"] = "0"
    os.environ["TRANSFORMERS_OFFLINE"] = "0"
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")


def _ensure_project_source() -> Path:
    """Expose the small extractor source tree without bloating the Studio repo."""

    module_dir = Path(__file__).resolve().parent
    candidates = (module_dir, module_dir.parent.parent, _workspace_root() / "source")
    for candidate in candidates:
        if (candidate / "retrieval" / "audio_embedder.py").is_file():
            if str(candidate) not in sys.path:
                sys.path.insert(0, str(candidate))
            return candidate

    destination = _workspace_root() / "source"
    destination.parent.mkdir(parents=True, exist_ok=True)
    repository = os.getenv(
        "SOULTUNER_SOURCE_REPOSITORY",
        "https://github.com/hgsanyang/SoulTuner-Agent.git",
    )
    revision = os.getenv("SOULTUNER_SOURCE_REVISION", "main")
    subprocess.run(
        [
            "git",
            "clone",
            "--depth",
            "1",
            "--branch",
            revision,
            "--filter=blob:none",
            repository,
            str(destination),
        ],
        check=True,
    )
    if not (destination / "retrieval" / "audio_embedder.py").is_file():
        raise RuntimeError("SoulTuner extractor source checkout is incomplete")
    sys.path.insert(0, str(destination))
    return destination


def run_worker() -> None:
    try:
        _wait_for_endpoint(float(os.getenv("SOULTUNER_ENDPOINT_START_TIMEOUT", "1800")))
        rows = _catalog_rows()
        total = len(rows)
        required_families = _required_families()
        driver = _driver()
        try:
            _upsert_catalog_rows(driver, rows)
            song_ids = {str(row["song_id"]) for row in rows}
            initial_missing = {
                family: _missing_ids(driver, family, EXPECTED_DIMS[family], song_ids)
                for family in required_families
            }
            # MuQ text encoding is needed even when Aura already contains all
            # audio vectors. Prepare its small Python dependency and source
            # before declaring the deployment ready; model weights are then
            # materialised by the non-blocking dense warmup.
            _ensure_packages(("muq_embedding",))
            _configure_model_caches()
            _ensure_project_source()
            if not any(initial_missing.values()):
                _write_status(
                    "ready",
                    completed=total,
                    total=total,
                    updates=0,
                    families=list(required_families),
                )
                return

            # Download and import only what this deployment actually needs.
            # In particular, a healthy MuQ+OMAR catalogue no longer downloads
            # the optional M2D checkpoint on every fresh image.
            _ensure_packages(required_families)
            checkpoint = None
            if "m2d2_embedding" in required_families:
                checkpoint = _ensure_m2d_checkpoint()
                _ensure_bert_snapshot()
            _configure_model_caches(checkpoint)
            _ensure_project_source()
            completed_updates = 0
            for family, expected, extractor, release in _family_extractors(required_families):
                missing = _missing_ids(driver, family, expected, song_ids)
                selected = [row for row in rows if str(row.get("song_id")) in missing]
                _write_status("running", family=family, completed=0, total=len(selected))
                for index, row in enumerate(selected, start=1):
                    song_id = str(row["song_id"])
                    try:
                        vector = list(extractor(Path(row["audio_file"])))
                        _update_vector(driver, song_id, family, vector, required_families)
                    except Exception as exc:
                        _write_status(
                            "running",
                            family=family,
                            completed=index - 1,
                            total=len(selected),
                            last_error=f"{song_id}: {type(exc).__name__}: {exc}"[:500],
                        )
                        raise
                    completed_updates += 1
                    if index == 1 or index % 10 == 0 or index == len(selected):
                        _write_status("running", family=family, completed=index, total=len(selected))
                release()
            ready = total - len(_missing_ids(driver, required_families[0], EXPECTED_DIMS[required_families[0]], song_ids))
            still_missing = {
                family: len(_missing_ids(driver, family, expected, song_ids))
                for family, expected in EXPECTED_DIMS.items()
                if family in required_families
            }
            if any(still_missing.values()):
                raise RuntimeError(f"vector backfill incomplete: {still_missing}")
            _write_status(
                "ready",
                completed=ready,
                total=total,
                updates=completed_updates,
                families=list(required_families),
            )
        finally:
            driver.close()
    except Exception as exc:
        previous = enrichment_status()
        _write_status(
            "failed",
            completed=int(previous.get("completed") or 0),
            total=int(previous.get("total") or 0),
            error=f"{type(exc).__name__}: {exc}"[:1000],
        )
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true")
    args = parser.parse_args(argv)
    if not args.worker:
        parser.error("--worker is required")
    run_worker()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
