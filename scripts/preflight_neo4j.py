#!/usr/bin/env python
"""Refuse to start Neo4j Community on an Enterprise `block` volume.

Volumes created by Neo4j Enterprise use its proprietary `block` store format;
Community only opens `aligned`. Handing a block volume to Community does not
"downgrade" it - the server refuses to start, and every retry looks like a
generic startup failure. Converting is an offline migration
(see docs/NEO4J_MIGRATION.md), never a tag change.

The repo default is Community (right for a fresh, empty volume). This check is
what stops that default from silently pointing at an existing Enterprise volume.

Pure standard library and ASCII-only output on purpose: it has to run before the
project's dependencies exist, and `conda run` on Windows re-encodes stdout as GBK
and dies on non-ASCII.

    python scripts/preflight_neo4j.py          # exit 1 blocks startup

Evidence, best first:
  1. Neo4j reachable   -> ask it: SHOW DATABASES YIELD name, store  (definitive)
  2. Neo4j down        -> which image last owned the container (docker ps -a)
  3. No volume at all  -> fresh install, Community is correct
"""

from __future__ import annotations

import base64
import json
import os
import re
import subprocess
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CONTAINER = "soultuner-neo4j"
DATA_VOLUME = "soultuner-agent_neo4j_data"
RUNBOOK = "docs/NEO4J_MIGRATION.md"


def _read_env(path: Path) -> dict:
    out: dict = {}
    if not path.exists():
        return out
    for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        raw = raw.strip()
        if not raw or raw.startswith("#") or "=" not in raw:
            continue
        key, value = raw.split("=", 1)
        out[key.strip()] = value.strip().strip('"').strip("'")
    return out


def _compose_default_image() -> str:
    """The image compose would use when .env sets no NEO4J_IMAGE."""
    compose = ROOT / "docker-compose.yml"
    if not compose.exists():
        return ""
    match = re.search(r"image:\s*\$\{NEO4J_IMAGE:-([^}]+)\}",
                      compose.read_text(encoding="utf-8", errors="ignore"))
    return match.group(1).strip() if match else ""


def _edition(image: str) -> str:
    low = image.lower()
    if "community" in low:
        return "community"
    if "enterprise" in low:
        return "enterprise"
    # Neo4j's untagged/plain tags are Community builds.
    return "community" if low.startswith("neo4j:") else "unknown"


def _docker(*args: str) -> str:
    try:
        done = subprocess.run(["docker", *args], capture_output=True, text=True, timeout=15)
    except (OSError, subprocess.SubprocessError):
        return ""
    return done.stdout.strip() if done.returncode == 0 else ""


def _store_format_via_http(env: dict) -> str:
    """Ask a RUNNING Neo4j for its store format. Empty string if unreachable."""
    password = env.get("NEO4J_PASSWORD") or os.getenv("NEO4J_PASSWORD") or ""
    user = env.get("NEO4J_USER") or "neo4j"
    if not password:
        return ""
    token = base64.b64encode(f"{user}:{password}".encode()).decode()
    request = urllib.request.Request(
        "http://127.0.0.1:7474/db/system/tx/commit",
        data=json.dumps({"statements": [{"statement": "SHOW DATABASES YIELD name, store"}]}).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Basic {token}"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=3) as response:
            body = json.loads(response.read().decode("utf-8", "replace"))
    except (urllib.error.URLError, OSError, ValueError):
        return ""
    for result in body.get("results") or []:
        for entry in result.get("data") or []:
            row = entry.get("row") or []
            if len(row) >= 2 and str(row[0]) == "neo4j":
                return str(row[1])
    return ""


def _previous_container_image() -> str:
    """Image of the existing soultuner-neo4j container - it owns the volume."""
    lines = _docker("ps", "-a", "--filter", f"name=^{CONTAINER}$", "--format", "{{.Image}}").splitlines()
    return lines[0].strip() if lines else ""


def _volume_exists() -> bool:
    names = _docker("volume", "ls", "--format", "{{.Name}}").splitlines()
    return DATA_VOLUME in {n.strip() for n in names}


def check() -> tuple[bool, list[str]]:
    """Return (ok, lines-to-print)."""
    env = _read_env(ROOT / ".env")
    image = (env.get("NEO4J_IMAGE") or os.getenv("NEO4J_IMAGE") or _compose_default_image()).strip()
    licence = (env.get("NEO4J_ACCEPT_LICENSE_AGREEMENT")
               or os.getenv("NEO4J_ACCEPT_LICENSE_AGREEMENT") or "").strip().lower()
    edition = _edition(image)
    out = [f"  image      : {image or '(unresolved)'}  -> {edition}"]

    store = _store_format_via_http(env)
    previous = _previous_container_image()
    volume = _volume_exists()
    if store:
        out.append(f"  store      : {store}  (live query)")
    elif previous:
        out.append(f"  store      : unknown - Neo4j is down; last container ran {previous}")
    else:
        out.append(f"  store      : unknown - no {CONTAINER} container; volume "
                   f"{'exists' if volume else 'not created yet'}")

    # Enterprise without an explicit licence choice: the container prints the
    # licence and exits, so say it here instead of letting `up` look broken.
    if edition == "enterprise" and licence not in {"yes", "eval"}:
        return False, out + [
            "",
            "REFUSING TO START: Enterprise image with no licence acceptance.",
            "  Neo4j accepts exactly two values, and this repo will not pick for you:",
            "    NEO4J_ACCEPT_LICENSE_AGREEMENT=yes   you hold a commercial licence",
            "    NEO4J_ACCEPT_LICENSE_AGREEMENT=eval  Neo4j Evaluation Licence "
            "(non-production, time-limited)",
            "  Set the true one in your local .env (gitignored).",
        ]

    if edition != "community":
        return True, out + ["  verdict    : OK (Enterprise opens both block and aligned)"]

    if store.startswith("block"):
        return False, out + [
            "",
            "REFUSING TO START: Community cannot open this volume.",
            f"  The `neo4j` database is {store}, an Enterprise-only format; Community",
            "  only opens `aligned`. Community would fail to start, and the failure",
            "  reads like a generic crash.",
            "  Fix (either one):",
            "    a) keep Enterprise - set in your local .env (gitignored):",
            "         NEO4J_IMAGE=neo4j:2026-enterprise",
            "         NEO4J_ACCEPT_LICENSE_AGREEMENT=eval   # or yes",
            f"    b) convert block -> aligned offline, on a CLONE: see {RUNBOOK}",
            "  Never point Community at the live volume to 'see what happens'.",
        ]

    if not store and _edition(previous) == "enterprise":
        return False, out + [
            "",
            "REFUSING TO START: cannot prove this volume is Community-compatible.",
            f"  Neo4j is not answering, and the existing {CONTAINER} container ran",
            f"  {previous} - so the volume was almost certainly created by Enterprise",
            "  and is in `block` format, which Community cannot open.",
            "  Start Enterprise once and re-run this check, or follow",
            f"  {RUNBOOK}. Do not just switch the tag.",
        ]

    if store:
        return True, out + ["  verdict    : OK (aligned volume, Community can open it)"]
    if not volume and not previous:
        return True, out + ["  verdict    : OK (fresh install, no volume yet)"]
    return True, out + ["  verdict    : OK (no evidence of an Enterprise volume)"]


def main() -> int:
    print("Neo4j preflight (store format vs configured edition)")
    ok, lines = check()
    for line in lines:
        print(line)
    if not ok:
        print("")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
