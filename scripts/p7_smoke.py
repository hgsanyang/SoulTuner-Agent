"""P7 lightweight smoke checks for quality readiness.

This script intentionally avoids LLM calls and full outcome eval.  It checks the
small pieces that tend to regress before a feedback-learning iteration:
shared-environment guards, ranking-policy readiness, alignment switches, and
optional backend diagnostic endpoints.

Examples:
    python scripts/p7_smoke.py
    python scripts/p7_smoke.py --api-base http://localhost:8501
    python scripts/p7_smoke.py --api-base http://localhost:8501 --require-api --json
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any
from urllib.error import URLError
from urllib.request import Request, urlopen

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from api.security import HTTPException, reject_public_demo_action, safe_resolve_child  # noqa: E402
from config.settings import settings  # noqa: E402
from services.feedback_logger import load_jsonl  # noqa: E402
from services.ranking_policy import feedback_dir, summarize_policy_readiness  # noqa: E402


def _ok(name: str, detail: str = "") -> dict[str, Any]:
    return {"name": name, "status": "pass", "detail": detail}


def _warn(name: str, detail: str) -> dict[str, Any]:
    return {"name": name, "status": "warn", "detail": detail}


def _fail(name: str, detail: str) -> dict[str, Any]:
    return {"name": name, "status": "fail", "detail": detail}


def _check_shared_environment_guards() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    previous = getattr(settings, "public_demo_mode", False)
    try:
        setattr(settings, "public_demo_mode", True)
        try:
            reject_public_demo_action("delete song")
            rows.append(_fail("shared_write_guard", "destructive action was not blocked"))
        except HTTPException as exc:
            rows.append(
                _ok("shared_write_guard", f"blocked with status={getattr(exc, 'status_code', '')}")
            )
    finally:
        setattr(settings, "public_demo_mode", previous)

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        try:
            safe_resolve_child(root, "../secret.txt")
            rows.append(_fail("path_traversal_guard", "escape path was accepted"))
        except ValueError:
            rows.append(_ok("path_traversal_guard", "escape path rejected"))
    return rows


def _load_policy_summary(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        global_model = payload.get("global") or {}
        return {
            "status": payload.get("status"),
            "gate_passed": payload.get("gate_passed"),
            "global_status": global_model.get("status"),
        }
    except Exception:
        return {"status": "invalid"}


def _check_ranking_readiness() -> dict[str, Any]:
    root = feedback_dir()
    exposures = load_jsonl(root / "exposures.jsonl")
    events = load_jsonl(root / "events.jsonl")
    slate = load_jsonl(root / "slate_feedback.jsonl")
    active = _load_policy_summary(root / "ranking_policy.json")
    candidate = _load_policy_summary(root / "ranking_policy.candidate.json")
    readiness = summarize_policy_readiness(
        num_exposures=len(exposures),
        num_events=len(events),
        num_slate_feedback=len(slate),
        active=active,
        candidate=candidate,
    )
    status = "pass" if readiness["stage"] != "collect_feedback" else "warn"
    return {
        "name": "ranking_policy_readiness",
        "status": status,
        "detail": readiness["next_action"],
        "data": readiness,
    }


def _check_alignment_switches() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    backend = str(getattr(settings, "dense_text_audio_backend", "") or "").lower()
    variant_mode = str(getattr(settings, "dense_query_variant_mode", "") or "").lower()
    if backend in {"muq", "m2d", "both"}:
        rows.append(_ok("dense_backend_config", backend))
    else:
        rows.append(_fail("dense_backend_config", f"unexpected backend={backend!r}"))
    if variant_mode in {"auto", "on", "off"}:
        rows.append(_ok("dense_query_variants", variant_mode))
    else:
        rows.append(_fail("dense_query_variants", f"unexpected mode={variant_mode!r}"))

    calibration_path = os.getenv("MUSIC_ALIGNMENT_CALIBRATION_PATH", "").strip()
    if calibration_path and not Path(calibration_path).exists():
        rows.append(_warn("alignment_calibration_path", f"configured file missing: {calibration_path}"))
    elif calibration_path:
        rows.append(_ok("alignment_calibration_path", calibration_path))
    else:
        rows.append(_ok("alignment_calibration_path", "unset; runtime is safe no-op"))

    adapter_path = os.getenv("MUSIC_ALIGNMENT_ADAPTER_PATH", "").strip()
    if adapter_path and not Path(adapter_path).exists():
        rows.append(_warn("alignment_adapter_path", f"configured file missing: {adapter_path}"))
    elif adapter_path:
        rows.append(_ok("alignment_adapter_path", adapter_path))
    else:
        rows.append(_ok("alignment_adapter_path", "unset; runtime is safe no-op"))
    return rows


def _get_json(api_base: str, path: str, timeout: float) -> tuple[bool, dict[str, Any] | str]:
    url = api_base.rstrip("/") + path
    try:
        req = Request(url, headers={"Accept": "application/json"})
        with urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
        return True, json.loads(raw)
    except (URLError, TimeoutError, json.JSONDecodeError, OSError) as exc:
        return False, str(exc)


def _check_api(api_base: str, timeout: float, require_api: bool) -> list[dict[str, Any]]:
    endpoints = [
        ("ranking_policy_status", "/api/ranking-policy/status"),
        ("memory_profile", "/api/memory/profile"),
        ("ingest_jobs", "/api/ingest-jobs?limit=5"),
        ("catalog_diagnostics", "/api/catalog-diagnostics?limit=5"),
    ]
    rows: list[dict[str, Any]] = []
    for name, path in endpoints:
        ok, payload = _get_json(api_base, path, timeout)
        if ok:
            success = payload.get("success", True) if isinstance(payload, dict) else True
            if success:
                rows.append(_ok(name, path))
            else:
                rows.append(_warn(name, str(payload)[:240]))
        else:
            rows.append((_fail if require_api else _warn)(name, str(payload)[:240]))
    return rows


def run_checks(api_base: str = "", require_api: bool = False, timeout: float = 3.0) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    checks.extend(_check_shared_environment_guards())
    checks.extend(_check_alignment_switches())
    checks.append(_check_ranking_readiness())
    if api_base:
        checks.extend(_check_api(api_base, timeout, require_api))
    return {
        "checks": checks,
        "summary": {
            "passed": sum(1 for row in checks if row["status"] == "pass"),
            "warnings": sum(1 for row in checks if row["status"] == "warn"),
            "failed": sum(1 for row in checks if row["status"] == "fail"),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run P7 lightweight smoke checks")
    parser.add_argument("--api-base", default="", help="Optional backend URL, e.g. http://localhost:8501")
    parser.add_argument("--require-api", action="store_true", help="Fail if API endpoint checks cannot connect")
    parser.add_argument("--timeout", type=float, default=3.0)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    report = run_checks(args.api_base, args.require_api, args.timeout)
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        for row in report["checks"]:
            marker = {"pass": "PASS", "warn": "WARN", "fail": "FAIL"}[row["status"]]
            print(f"{marker} {row['name']}: {row.get('detail', '')}")
        print("summary:", json.dumps(report["summary"], ensure_ascii=False))
    return 1 if report["summary"]["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
