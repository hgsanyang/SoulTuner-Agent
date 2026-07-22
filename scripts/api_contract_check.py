"""API contract + cold-start check against a running backend.

Read-only by design: only GET endpoints with no side effects are exercised,
so this is safe to run against a live instance at any time. Intended for
release gates (v1.0-local) and after Docker image rebuilds.

Usage:
    python scripts/api_contract_check.py                 # assumes backend already up
    python scripts/api_contract_check.py --wait 180      # cold start: poll /health first
    python scripts/api_contract_check.py --base-url http://127.0.0.1:8501
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request

DEFAULT_BASE_URL = "http://127.0.0.1:8501"


def _get(base_url: str, path: str, timeout: float = 15.0) -> tuple[int, dict | list | None]:
    request = urllib.request.Request(base_url + path, headers={"Accept": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8", errors="replace")
            try:
                return response.status, json.loads(body)
            except json.JSONDecodeError:
                return response.status, None
    except urllib.error.HTTPError as exc:
        return exc.code, None
    except Exception:
        return 0, None


def wait_for_health(base_url: str, wait_seconds: float) -> float | None:
    """Poll /health until it answers 200; return cold-start latency in seconds."""
    started = time.monotonic()
    deadline = started + wait_seconds
    while time.monotonic() < deadline:
        status, _ = _get(base_url, "/health", timeout=5.0)
        if status == 200:
            return time.monotonic() - started
        time.sleep(2.0)
    return None


CHECKS: list[tuple[str, str, callable]] = [
    (
        "health",
        "/health",
        lambda status, data: status == 200,
    ),
    (
        "memory profile (editable memory + views)",
        "/api/memory/profile",
        lambda status, data: status == 200
        and isinstance(data, dict)
        and data.get("success") is True
        and isinstance(data.get("memory", {}).get("records"), list)
        and isinstance(data.get("memory", {}).get("profile_views"), dict),
    ),
    (
        "memory profile-views (scope-grouped)",
        "/api/memory/profile-views",
        lambda status, data: status == 200
        and isinstance(data, dict)
        and data.get("success") is True
        and isinstance(data.get("views"), list),
    ),
    (
        "catalog diagnostics",
        "/api/catalog-diagnostics?limit=5",
        lambda status, data: status == 200
        and isinstance(data, dict)
        and data.get("success") is True,
    ),
    (
        "ranking policy status",
        "/api/ranking-policy/status",
        lambda status, data: status == 200 and isinstance(data, dict),
    ),
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument(
        "--wait",
        type=float,
        default=0.0,
        help="cold-start mode: poll /health for up to N seconds before checking",
    )
    args = parser.parse_args()

    if args.wait > 0:
        latency = wait_for_health(args.base_url, args.wait)
        if latency is None:
            print(f"FAIL cold start: /health not healthy within {args.wait:.0f}s")
            return 1
        print(f"cold start: healthy after {latency:.1f}s")

    failures = 0
    for name, path, validator in CHECKS:
        status, data = _get(args.base_url, path)
        ok = False
        try:
            ok = bool(validator(status, data))
        except Exception:
            ok = False
        print(f"{'PASS' if ok else 'FAIL'}  {name}  [{path}] -> HTTP {status or 'no response'}")
        failures += 0 if ok else 1

    if failures:
        print(f"\n{failures} contract check(s) failed")
        return 1
    print("\nall contract checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
