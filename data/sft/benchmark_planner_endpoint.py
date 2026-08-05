"""Benchmark an OpenAI-compatible local planner endpoint on a frozen split.

The JSON report contains sample identifiers and timings, never user text or
credentials. It is a deployment gate, not part of checkpoint selection.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import statistics
import time
from typing import Any

import httpx

from schemas.planner_decision_v3 import PlannerDecisionV3


def percentile(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * quantile)))
    return ordered[index]


def summarise(records: list[dict[str, Any]], *, max_p50_ms: float) -> dict[str, Any]:
    measured = [float(record["latency_ms"]) for record in records if record.get("ok")]
    valid = sum(bool(record.get("schema_valid")) for record in records)
    total = len(records)
    report = {
        "rows": total,
        "successful_requests": len(measured),
        "schema_valid": round(valid / total, 4) if total else 0.0,
        "latency_ms": {
            "p50": round(percentile(measured, 0.50), 1),
            "p95": round(percentile(measured, 0.95), 1),
            "mean": round(statistics.fmean(measured), 1) if measured else 0.0,
        },
        "failures": [
            {key: record.get(key) for key in ("sample_id", "error")}
            for record in records
            if not record.get("schema_valid")
        ][:20],
    }
    findings = []
    if total == 0:
        findings.append("no benchmark rows")
    if len(measured) != total:
        findings.append("one or more endpoint requests failed")
    if report["schema_valid"] < 1.0:
        findings.append("schema_valid must be 1.0")
    if report["latency_ms"]["p50"] > max_p50_ms:
        findings.append(
            f"p50 {report['latency_ms']['p50']:.1f}ms exceeds {max_p50_ms:.1f}ms"
        )
    report["gate"] = {"passed": not findings, "max_p50_ms": max_p50_ms, "findings": findings}
    return report


def _sample_id(row: dict[str, Any], index: int) -> str:
    meta = row.get("meta") or {}
    return f"{meta.get('episode_id', index)}#{meta.get('turn_id', 0)}"


def benchmark(
    rows: list[dict[str, Any]],
    *,
    endpoint: str,
    model: str,
    api_key: str,
    timeout: float,
) -> list[dict[str, Any]]:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    records = []
    with httpx.Client(timeout=timeout, trust_env=False) as client:
        for index, row in enumerate(rows):
            sample_id = _sample_id(row, index)
            request_messages = [
                {"role": message["role"], "content": message.get("content", "")}
                for message in (row.get("messages") or [])
                if message.get("role") in {"system", "user", "tool"}
            ]
            started = time.perf_counter()
            try:
                response = client.post(
                    endpoint,
                    headers=headers,
                    json={
                        "model": model,
                        "messages": request_messages,
                        "temperature": 0.0,
                        "max_tokens": 1024,
                        "enable_thinking": False,
                    },
                )
                response.raise_for_status()
                content = response.json()["choices"][0]["message"]["content"]
                PlannerDecisionV3.model_validate_json(content)
                records.append(
                    {
                        "sample_id": sample_id,
                        "ok": True,
                        "schema_valid": True,
                        "latency_ms": round((time.perf_counter() - started) * 1000, 1),
                    }
                )
            except Exception as exc:
                records.append(
                    {
                        "sample_id": sample_id,
                        "ok": False,
                        "schema_valid": False,
                        "latency_ms": round((time.perf_counter() - started) * 1000, 1),
                        "error": type(exc).__name__,
                    }
                )
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--endpoint", default="http://127.0.0.1:8000/v1/chat/completions")
    parser.add_argument("--model", required=True)
    parser.add_argument("--api-key-env", default="LOCAL_MODEL_API_KEY")
    parser.add_argument("--limit", type=int, default=120)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--max-p50-ms", type=float, default=2000.0)
    parser.add_argument("--json", dest="json_out", type=Path, required=True)
    args = parser.parse_args()
    rows = [
        json.loads(line)
        for line in args.dataset.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ][: max(1, args.limit)]
    records = benchmark(
        rows,
        endpoint=args.endpoint,
        model=args.model,
        api_key=str(os.getenv(args.api_key_env) or ""),
        timeout=args.timeout,
    )
    report = summarise(records, max_p50_ms=args.max_p50_ms)
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["gate"]["passed"] else 13


if __name__ == "__main__":
    raise SystemExit(main())
