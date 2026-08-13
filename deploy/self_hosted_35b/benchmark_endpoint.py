"""Benchmark a SoulTuner Planner endpoint with public representative prompts.

The report deliberately excludes prompt and response text.  It measures the
same frozen prompt, parser, and deterministic guard that the self-hosted demo
uses, without reading private regression or sealed evaluation rows.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import statistics
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from .planner_guard import guard_candidate, parse_candidate_content
    from .prompt_v42 import STUDENT_SYSTEM_PROMPT_V4_2, format_student_user_message
except ImportError:  # Support direct execution from the deployment directory.
    from planner_guard import guard_candidate, parse_candidate_content
    from prompt_v42 import STUDENT_SYSTEM_PROMPT_V4_2, format_student_user_message


@dataclass(frozen=True)
class PublicCase:
    case_id: str
    query: str
    context: dict[str, str]


PUBLIC_CASES = (
    PublicCase("mood", "我今天心情有点差，想听温暖治愈、但不要太吵的歌", {}),
    PublicCase("acoustics", "想要低音更重、鼓点清晰，适合夜跑的音乐", {}),
    PublicCase("hybrid", "给我一些 90 年代英文摇滚，整体不要太沉重", {}),
    PublicCase(
        "reference",
        "刚才那种氛围很好，再来一组更安静、更有空间感的",
        {"reference_title": "公开参考曲", "reference_artist": "公开演示艺人"},
    ),
    PublicCase("catalog", "周末小聚想听轻松明亮的中文流行", {}),
)


def percentile(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * quantile)))
    return ordered[index]


def _request_payload(model: str, case: PublicCase) -> bytes:
    return json.dumps(
        {
            "model": model,
            "messages": [
                {"role": "system", "content": STUDENT_SYSTEM_PROMPT_V4_2},
                {
                    "role": "user",
                    "content": format_student_user_message(
                        case.query,
                        reference_title=case.context.get("reference_title", ""),
                        reference_artist=case.context.get("reference_artist", ""),
                    ),
                },
            ],
            "temperature": 0,
            "max_tokens": 1024,
            "stream": False,
            "enable_thinking": False,
            "response_format": {"type": "json_object"},
            "chat_template_kwargs": {"enable_thinking": False},
        },
        ensure_ascii=False,
    ).encode("utf-8")


def request_once(
    *,
    endpoint: str,
    model: str,
    api_key: str,
    case: PublicCase,
    timeout: float,
) -> dict[str, Any]:
    request = urllib.request.Request(
        endpoint,
        data=_request_payload(model, case),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    if api_key:
        request.add_header("Authorization", f"Bearer {api_key}")
    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = json.loads(response.read().decode("utf-8"))
    latency_ms = (time.perf_counter() - started) * 1000
    content = body["choices"][0]["message"]["content"]
    candidate = parse_candidate_content(str(content))
    guarded, findings = guard_candidate(case.query, candidate, case.context)
    usage = body.get("usage") if isinstance(body, dict) else None
    usage = usage if isinstance(usage, dict) else {}
    return {
        "latency_ms": round(latency_ms, 1),
        "schema_valid": True,
        "safe_plan_available": isinstance(guarded, dict) and bool(guarded),
        "guard_accepted": guarded.get("source") == "model_candidate_guarded",
        "guard_finding_count": 0 if guarded.get("source") == "model_candidate_guarded" else len(findings),
        "guard_findings": (
            []
            if guarded.get("source") == "model_candidate_guarded"
            else [
                finding
                for finding in findings
                if finding != "候选被拒绝，已回退到确定性安全计划"
            ]
        ),
        "prompt_tokens": int(usage.get("prompt_tokens") or 0),
        "completion_tokens": int(usage.get("completion_tokens") or 0),
    }


def _failed_record(case_id: str, repeat: int, exc: BaseException) -> dict[str, Any]:
    return {
        "case_id": case_id,
        "repeat": repeat,
        "ok": False,
        "schema_valid": False,
        "guard_accepted": False,
        "latency_ms": 0.0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "error": type(exc).__name__,
    }


def summarise(
    records: list[dict[str, Any]],
    *,
    concurrency: int = 1,
    wall_seconds: float | None = None,
) -> dict[str, Any]:
    latencies = [float(row["latency_ms"]) for row in records if row["ok"]]
    total = len(records)
    successful = len(latencies)
    schema_valid = sum(bool(row.get("schema_valid")) for row in records)
    guard_accepted = sum(bool(row.get("guard_accepted")) for row in records)
    safe_plan_available = sum(bool(row.get("safe_plan_available")) for row in records)
    prompt_tokens = sum(int(row.get("prompt_tokens") or 0) for row in records)
    completion_tokens = sum(int(row.get("completion_tokens") or 0) for row in records)
    guard_finding_counts = collections.Counter(
        finding
        for row in records
        for finding in row.get("guard_findings", [])
        if isinstance(finding, str)
    )
    case_metrics: dict[str, dict[str, Any]] = {}
    for case_id in sorted({str(row["case_id"]) for row in records}):
        case_rows = [row for row in records if row["case_id"] == case_id]
        case_latencies = [float(row["latency_ms"]) for row in case_rows if row["ok"]]
        case_total = len(case_rows)
        case_metrics[case_id] = {
            "requests": case_total,
            "successful_requests": len(case_latencies),
            "schema_valid_rate": round(
                sum(bool(row.get("schema_valid")) for row in case_rows) / case_total,
                4,
            ),
            "guard_accept_rate": round(
                sum(bool(row.get("guard_accepted")) for row in case_rows) / case_total,
                4,
            ),
            "safe_plan_rate": round(
                sum(bool(row.get("safe_plan_available")) for row in case_rows) / case_total,
                4,
            ),
            "latency_ms": {
                "p50": round(percentile(case_latencies, 0.50), 1),
                "p95": round(percentile(case_latencies, 0.95), 1),
            },
        }
    elapsed = max(0.0, float(wall_seconds or 0.0))
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "scope": "public representative prompts; no private regression or sealed rows",
        "concurrency": max(1, concurrency),
        "requests": total,
        "successful_requests": successful,
        "schema_valid_rate": round(schema_valid / total, 4) if total else 0.0,
        "guard_accept_rate": round(guard_accepted / total, 4) if total else 0.0,
        "safe_plan_rate": round(safe_plan_available / total, 4) if total else 0.0,
        "latency_ms": {
            "min": round(min(latencies), 1) if latencies else 0.0,
            "mean": round(statistics.fmean(latencies), 1) if latencies else 0.0,
            "p50": round(percentile(latencies, 0.50), 1),
            "p95": round(percentile(latencies, 0.95), 1),
            "max": round(max(latencies), 1) if latencies else 0.0,
        },
        "throughput": {
            "wall_seconds": round(elapsed, 3),
            "requests_per_second": round(successful / elapsed, 3) if elapsed else 0.0,
            "completion_tokens_per_second": (
                round(completion_tokens / elapsed, 3) if elapsed else 0.0
            ),
        },
        "tokens": {
            "prompt": prompt_tokens,
            "completion": completion_tokens,
        },
        "failures": [
            {"case_id": row["case_id"], "error": row["error"]}
            for row in records
            if not row["ok"]
        ],
        "guard_findings": dict(sorted(guard_finding_counts.items())),
        "case_metrics": case_metrics,
        "measurement_note": "Non-streaming end-to-end latency; TTFT is not measured.",
    }


def _run_one(
    *,
    endpoint: str,
    model: str,
    api_key: str,
    case: PublicCase,
    repeat: int,
    timeout: float,
) -> dict[str, Any]:
    try:
        result = request_once(
            endpoint=endpoint,
            model=model,
            api_key=api_key,
            case=case,
            timeout=timeout,
        )
        return {"case_id": case.case_id, "repeat": repeat, "ok": True, **result}
    except (
        OSError,
        TimeoutError,
        ValueError,
        KeyError,
        IndexError,
        json.JSONDecodeError,
        urllib.error.URLError,
    ) as exc:
        return _failed_record(case.case_id, repeat, exc)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", default="http://127.0.0.1:8000/v1/chat/completions")
    parser.add_argument("--model", default="soultuner-planner-v4.2-35b")
    parser.add_argument("--api-key-env", default="SOULTUNER_PLANNER_API_KEY")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--json", dest="json_out", type=Path, required=True)
    args = parser.parse_args()
    api_key = str(os.getenv(args.api_key_env) or os.getenv("SOULTUNER_SERVE_API_KEY") or "")

    for index in range(max(0, args.warmup)):
        case = PUBLIC_CASES[index % len(PUBLIC_CASES)]
        _run_one(
            endpoint=args.endpoint,
            model=args.model,
            api_key=api_key,
            case=case,
            repeat=-1,
            timeout=args.timeout,
        )

    jobs = [
        (repeat, case)
        for repeat in range(max(1, args.repeat))
        for case in PUBLIC_CASES
    ]
    started = time.perf_counter()
    records: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as executor:
        futures = [
            executor.submit(
                _run_one,
                endpoint=args.endpoint,
                model=args.model,
                api_key=api_key,
                case=case,
                repeat=repeat,
                timeout=args.timeout,
            )
            for repeat, case in jobs
        ]
        for future in as_completed(futures):
            records.append(future.result())
    wall_seconds = time.perf_counter() - started
    records.sort(key=lambda row: (int(row["repeat"]), str(row["case_id"])))

    report = summarise(
        records,
        concurrency=max(1, args.concurrency),
        wall_seconds=wall_seconds,
    )
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["successful_requests"] == report["requests"] else 7


if __name__ == "__main__":
    raise SystemExit(main())
