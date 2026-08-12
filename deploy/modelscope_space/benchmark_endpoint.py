"""Benchmark the public SoulTuner Planner endpoint without private eval rows."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PUBLIC_CASES = (
    ("mood", "我今天心情有点差，想听温暖治愈、但不要太吵的歌"),
    ("acoustics", "想要低音更重、鼓点清晰，适合夜跑的音乐"),
    ("hybrid", "给我一些 90 年代英文摇滚，整体不要太沉重"),
    ("reference", "刚才那种氛围很好，再来一组更安静、更有空间感的"),
    ("catalog", "周末小聚想听轻松明亮的中文流行"),
)
REQUIRED_KEYS = {
    "task_mode",
    "evidence",
    "lane_policy",
    "hard",
    "soft",
    "hints",
    "metadata",
}


def percentile(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * quantile)))
    return ordered[index]


def _extract_json(text: str) -> dict[str, Any]:
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("response does not contain a JSON object")
    value = json.loads(text[start : end + 1])
    if not isinstance(value, dict) or not REQUIRED_KEYS.issubset(value):
        raise ValueError("response does not satisfy the public Planner contract")
    return value


def request_once(*, endpoint: str, model: str, api_key: str, query: str, timeout: float) -> tuple[float, bool]:
    payload = json.dumps(
        {
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "你是 SoulTuner 音乐检索规划器。只输出一个 JSON 对象，"
                        "必须包含 task_mode、evidence、lane_policy、hard、soft、hints、metadata。"
                    ),
                },
                {"role": "user", "content": query},
            ],
            "temperature": 0,
            "max_tokens": 1024,
            "enable_thinking": False,
        },
        ensure_ascii=False,
    ).encode("utf-8")
    request = urllib.request.Request(
        endpoint,
        data=payload,
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
    _extract_json(content)
    return latency_ms, True


def summarise(records: list[dict[str, Any]]) -> dict[str, Any]:
    latencies = [float(row["latency_ms"]) for row in records if row["ok"]]
    total = len(records)
    successful = len(latencies)
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "scope": "public representative prompts; no private regression or sealed rows",
        "requests": total,
        "successful_requests": successful,
        "contract_valid_rate": round(successful / total, 4) if total else 0.0,
        "latency_ms": {
            "min": round(min(latencies), 1) if latencies else 0.0,
            "mean": round(statistics.fmean(latencies), 1) if latencies else 0.0,
            "p50": round(percentile(latencies, 0.50), 1),
            "p95": round(percentile(latencies, 0.95), 1),
            "max": round(max(latencies), 1) if latencies else 0.0,
        },
        "failures": [{"case_id": row["case_id"], "error": row["error"]} for row in records if not row["ok"]],
        "measurement_note": "Non-streaming end-to-end latency; TTFT is not measured.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", default="http://127.0.0.1:8000/v1/chat/completions")
    parser.add_argument("--model", default="soultuner-v4.2-35b")
    parser.add_argument("--api-key-env", default="SOULTUNER_PLANNER_API_KEY")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--json", dest="json_out", type=Path, required=True)
    args = parser.parse_args()
    api_key = str(os.getenv(args.api_key_env) or os.getenv("SOULTUNER_SERVE_API_KEY") or "")

    for index in range(max(0, args.warmup)):
        _, query = PUBLIC_CASES[index % len(PUBLIC_CASES)]
        try:
            request_once(
                endpoint=args.endpoint,
                model=args.model,
                api_key=api_key,
                query=query,
                timeout=args.timeout,
            )
        except (OSError, TimeoutError, ValueError, KeyError, urllib.error.URLError):
            pass

    records: list[dict[str, Any]] = []
    for repeat in range(max(1, args.repeat)):
        for case_id, query in PUBLIC_CASES:
            try:
                latency_ms, valid = request_once(
                    endpoint=args.endpoint,
                    model=args.model,
                    api_key=api_key,
                    query=query,
                    timeout=args.timeout,
                )
                records.append(
                    {
                        "case_id": case_id,
                        "repeat": repeat,
                        "ok": valid,
                        "latency_ms": round(latency_ms, 1),
                    }
                )
            except (OSError, TimeoutError, ValueError, KeyError, urllib.error.URLError) as exc:
                records.append(
                    {
                        "case_id": case_id,
                        "repeat": repeat,
                        "ok": False,
                        "latency_ms": 0.0,
                        "error": type(exc).__name__,
                    }
                )

    report = summarise(records)
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["successful_requests"] == report["requests"] else 7


if __name__ == "__main__":
    raise SystemExit(main())
