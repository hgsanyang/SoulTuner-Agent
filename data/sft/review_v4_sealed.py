"""Blindly review balanced V4 candidates and freeze 100 rows per request kind."""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import copy
import json
import os
from pathlib import Path
import time
from typing import Any

import httpx

from data.sft.build_v4_sealed_seeds import load_jsonl
from data.sft.v4_contract import row_contract_errors
from schemas.planner_decision_v3 import PlannerDecisionV3


URL = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
KINDS = ("recommendation", "information", "acquisition", "library", "conversation")


def _api_key() -> str:
    return str(os.getenv("DASHSCOPE_API_KEY") or "").strip()


def _review_input(row: dict[str, Any]) -> dict[str, Any]:
    messages = row.get("messages") or []
    user = next(str(message.get("content") or "") for message in messages if message.get("role") == "user")
    assistant = next(str(message.get("content") or "") for message in reversed(messages) if message.get("role") == "assistant")
    return {"sample_id": row["meta"]["episode_id"], "user_context": user, "candidate_decision": json.loads(assistant)}


def _payload(rows: list[dict[str, Any]], model: str) -> dict[str, Any]:
    system = (
        "你是独立的音乐 Agent Planner 评审。你看不到教师身份，只按用户上下文和"
        "PlannerDecisionV3 候选判断。核对 request_kind、是否应该澄清、工具通道、"
        "当前请求是否压过冲突的长期偏好，以及英文 acoustic_queries 是否贴合听感。"
        "只有选择 dense 通道时才能填写 acoustic_queries，选择 dense 时必须填写。"
        "library 只读个人曲库；ingest 只用于 acquisition；conversation 不调用工具。"
        "不要按关键词机械裁决。输出 JSON 对象 {reviews:[{sample_id,verdict,"
        "reason_codes,corrected_decision}]}。verdict 只能 accept/accept_with_edit/reject；"
        "只有 accept_with_edit 才填写完整 corrected_decision。不要输出解释或思维过程。"
    )
    return {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps({"samples": [_review_input(row) for row in rows]}, ensure_ascii=False)},
        ],
        "temperature": 0.0,
        "max_tokens": max(1800, 1000 * len(rows)),
        "enable_thinking": False,
        "response_format": {"type": "json_object"},
    }


def _parse(content: Any, rows: list[dict[str, Any]], model: str) -> tuple[list[dict[str, Any]], Counter]:
    failures: Counter[str] = Counter()
    try:
        payload = json.loads(str(content or ""))
        reviews = payload.get("reviews")
    except (json.JSONDecodeError, AttributeError):
        return [], Counter({"invalid_json": len(rows)})
    if not isinstance(reviews, list):
        return [], Counter({"missing_reviews": len(rows)})
    by_id = {str(item.get("sample_id")): item for item in reviews if isinstance(item, dict)}
    accepted = []
    for source in rows:
        sample_id = str(source["meta"]["episode_id"])
        review = by_id.get(sample_id)
        if review is None:
            failures["missing_sample"] += 1
            continue
        verdict = str(review.get("verdict") or "")
        if verdict == "reject":
            failures["rejected"] += 1
            continue
        row = copy.deepcopy(source)
        if verdict == "accept_with_edit":
            try:
                decision = PlannerDecisionV3.model_validate(review.get("corrected_decision"))
            except Exception:
                failures["invalid_edit"] += 1
                continue
            for message in reversed(row["messages"]):
                if message.get("role") == "assistant":
                    message["content"] = decision.model_dump_json(exclude_none=True)
                    break
        elif verdict != "accept":
            failures["invalid_verdict"] += 1
            continue
        row["meta"]["reviewer"] = {"model": model, "version": "2026-08", "vendor": "dashscope"}
        row["meta"]["reviewer_verdict"] = verdict
        row.setdefault("lineage", {})["review_reason_codes"] = [str(value)[:80] for value in (review.get("reason_codes") or [])[:5]]
        errors = row_contract_errors(row)
        if errors:
            failures["post_review_contract_invalid"] += 1
            continue
        accepted.append(row)
    return accepted, failures


def _call(rows: list[dict[str, Any]], api_key: str, model: str, timeout: float) -> tuple[list[dict[str, Any]], Counter, dict[str, Any]]:
    started = time.perf_counter()
    with httpx.Client(timeout=timeout, trust_env=False) as client:
        response = client.post(URL, headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}, json=_payload(rows, model))
        response.raise_for_status()
        body = response.json()
    accepted, failures = _parse(body["choices"][0]["message"]["content"], rows, model)
    return accepted, failures, {"latency_ms": round((time.perf_counter() - started) * 1000, 1), "usage": body.get("usage") or {}}


def review(
    rows: list[dict[str, Any]],
    *,
    api_key: str,
    model: str,
    per_kind: int = 100,
    batch_size: int = 5,
    workers: int = 4,
    timeout: float = 180.0,
    kinds: tuple[str, ...] = KINDS,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not api_key:
        raise RuntimeError("DASHSCOPE_API_KEY is not configured")
    batches = [rows[index:index + batch_size] for index in range(0, len(rows), batch_size)]
    accepted = []
    failures: Counter[str] = Counter()
    usage: Counter[str] = Counter()
    latencies = []
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = {pool.submit(_call, batch, api_key, model, timeout): batch for batch in batches}
        for future in as_completed(futures):
            batch = futures[future]
            try:
                rows_out, failed, diagnostics = future.result()
            except Exception as exc:
                failures[f"request_{type(exc).__name__}"] += len(batch)
                continue
            accepted.extend(rows_out)
            failures.update(failed)
            latencies.append(diagnostics["latency_ms"])
            for key, value in diagnostics["usage"].items():
                if isinstance(value, int):
                    usage[key] += value
    selected = []
    by_kind = Counter()
    for row in sorted(accepted, key=lambda item: str(item["meta"]["episode_id"])):
        kind = str(row["meta"]["request_kind"])
        if kind in kinds and by_kind[kind] < per_kind:
            selected.append(row)
            by_kind[kind] += 1
    missing = {kind: per_kind - by_kind[kind] for kind in kinds if by_kind[kind] < per_kind}
    report = {"candidates": len(rows), "accepted_before_cap": len(accepted), "selected": len(selected), "counts": dict(by_kind), "missing": missing, "failures": dict(failures), "usage": dict(usage), "latency_ms_max": max(latencies, default=0)}
    return selected, report


def replace_reviewed(
    base_rows: list[dict[str, Any]],
    replacements: list[dict[str, Any]],
    *,
    kinds: tuple[str, ...],
    per_kind: int,
) -> list[dict[str, Any]]:
    retained = [
        row
        for row in base_rows
        if str((row.get("meta") or {}).get("request_kind")) not in kinds
    ]
    combined = [*retained, *replacements]
    final_counts = Counter(row["meta"]["request_kind"] for row in combined)
    if any(final_counts[kind] != per_kind for kind in KINDS):
        raise ValueError(f"replacement would unbalance sealed split: {dict(final_counts)}")
    return combined


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("data/teacher/private/v4/sealed_v4_balanced_candidates.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("data/teacher/private/v4/sealed_v4_reviewed.jsonl"))
    parser.add_argument("--report", type=Path, default=Path("data/teacher/private/v4/sealed_v4_review_report.json"))
    parser.add_argument("--model", default="qwen3.7-flash")
    parser.add_argument("--per-kind", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--kind", action="append", choices=KINDS)
    parser.add_argument(
        "--replace-in",
        type=Path,
        help="Keep reviewed rows for other request kinds and replace only --kind.",
    )
    args = parser.parse_args()
    kinds = tuple(args.kind or KINDS)
    candidates = [
        row
        for row in load_jsonl(args.input)
        if str((row.get("meta") or {}).get("request_kind")) in kinds
    ]
    rows, report = review(
        candidates,
        api_key=_api_key(),
        model=args.model,
        per_kind=args.per_kind,
        batch_size=args.batch_size,
        workers=args.workers,
        kinds=kinds,
    )
    if args.replace_in:
        rows = replace_reviewed(
            load_jsonl(args.replace_in),
            rows,
            kinds=kinds,
            per_kind=args.per_kind,
        )
        final_counts = Counter(row["meta"]["request_kind"] for row in rows)
        report["final_counts"] = dict(final_counts)
    for path in (args.output, args.report):
        if "private" not in {part.casefold() for part in path.parts}:
            raise ValueError("sealed review outputs must stay private")
        path.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({**report, "output": str(args.output)}, ensure_ascii=False, indent=2))
    return 2 if report["missing"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
