"""Deterministic fingerprints for proving that outcome eval is read-only."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


_VECTOR_KEY_MARKERS = ("embedding", "vector")


def _canonical(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, dict):
        return {str(key): _canonical(value[key]) for key in sorted(value, key=str)}
    if isinstance(value, (list, tuple)):
        return [_canonical(item) for item in value]
    if isinstance(value, set):
        return sorted((_canonical(item) for item in value), key=lambda item: repr(item))
    isoformat = getattr(value, "isoformat", None)
    if callable(isoformat):
        try:
            return isoformat()
        except Exception:
            pass
    return str(value)


def digest_rows(rows: Iterable[dict[str, Any]]) -> str:
    canonical_rows = [_canonical(dict(row)) for row in rows]
    canonical_rows.sort(key=lambda row: json.dumps(row, ensure_ascii=False, sort_keys=True, default=str))
    payload = json.dumps(canonical_rows, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _is_vector_key(key: Any) -> bool:
    text = str(key or "").lower()
    return any(marker in text for marker in _VECTOR_KEY_MARKERS)


def neo4j_fingerprint(client: Any) -> dict[str, Any]:
    """Hash graph structure and every non-vector property."""

    nodes = client.execute_query(
        "MATCH (n) RETURN elementId(n) AS id, labels(n) AS labels ORDER BY id"
    )
    node_properties = client.execute_query(
        "MATCH (n) UNWIND keys(n) AS key "
        "RETURN elementId(n) AS id, key, n[key] AS value ORDER BY id, key"
    )
    node_properties = [row for row in node_properties if not _is_vector_key(row.get("key"))]
    relationships = client.execute_query(
        "MATCH (a)-[r]->(b) "
        "RETURN elementId(r) AS id, elementId(a) AS source, elementId(b) AS target, "
        "type(r) AS type, properties(r) AS properties ORDER BY id"
    )
    return {
        "nodes": len(nodes),
        "node_properties": len(node_properties),
        "relationships": len(relationships),
        "node_digest": digest_rows(nodes),
        "node_property_digest": digest_rows(node_properties),
        "relationship_digest": digest_rows(relationships),
    }


def file_tree_fingerprint(paths: Iterable[Path]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for root in paths:
        root = Path(root)
        if not root.exists():
            continue
        candidates = [root] if root.is_file() else sorted(path for path in root.rglob("*") if path.is_file())
        for path in candidates:
            stat = path.stat()
            rows.append(
                {
                    "path": str(path.resolve()),
                    "size": stat.st_size,
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
            )
    return {"files": len(rows), "digest": digest_rows(rows)}


def capture_runtime_fingerprint(project_root: Path | None = None) -> dict[str, Any]:
    from retrieval.neo4j_client import get_neo4j_client

    root = Path(project_root or Path(__file__).resolve().parents[2])
    return {
        "neo4j": neo4j_fingerprint(get_neo4j_client()),
        "local_side_effect_files": file_tree_fingerprint(
            [root / "data" / "feedback", root / "data" / "teacher"]
        ),
    }


def fingerprint_changes(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    return {
        key: {"before": before.get(key), "after": after.get(key)}
        for key in sorted(set(before) | set(after))
        if before.get(key) != after.get(key)
    }

