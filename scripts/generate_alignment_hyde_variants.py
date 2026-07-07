"""Generate frozen LLM acoustic HyDE variants for the alignment attribute ruler.

The output belongs outside the repository (for example ``../data``).  The
attribute evaluator can then consume the frozen JSON without calling an LLM.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def _query_ids(limit: int = 0) -> list[dict[str, Any]]:
    from tests.eval.evaluate_alignment_attribute import FROZEN_ATTRIBUTE_QUERIES

    queries = list(FROZEN_ATTRIBUTE_QUERIES)
    return queries[:limit] if limit and limit > 0 else queries


def _prompt_for_query(query: dict[str, Any]) -> list[dict[str, str]]:
    system = (
        "You create MusicCaps-style acoustic retrieval descriptions for a music search system. "
        "Return only JSON. Do not mention specific artists or song titles. "
        "Each variant must describe audible musical properties: instrumentation, tempo, rhythm, "
        "vocal presence/language if relevant, timbre, dynamics, and listening context."
    )
    user = (
        "Generate 3 complementary English acoustic HyDE descriptions for this retrieval query. "
        "The first variant should be the strongest general description. The other two should "
        "look at different acoustic angles rather than repeating keywords. Keep each under 45 words.\n\n"
        f"Query id: {query['id']}\n"
        f"Original query language: {query['query_language']}\n"
        f"Original query text: {query['text']}\n"
        f"Target label description: {json.dumps(query['target'], ensure_ascii=False)}\n\n"
        'Return JSON exactly as: {"variants": ["...", "...", "..."]}'
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _parse_variants(content: str) -> list[str]:
    try:
        payload = json.loads(content)
    except json.JSONDecodeError:
        start = content.find("{")
        end = content.rfind("}")
        if start < 0 or end <= start:
            return []
        payload = json.loads(content[start : end + 1])
    variants = payload.get("variants", [])
    cleaned: list[str] = []
    for item in variants:
        text = str(item or "").strip()
        if text and text not in cleaned:
            cleaned.append(text)
    return cleaned[:4]


def _call_dashscope(
    *,
    api_key: str,
    model: str,
    base_url: str,
    query: dict[str, Any],
    timeout: float,
) -> list[str]:
    with httpx.Client(timeout=timeout, trust_env=False) as client:
        response = client.post(
            f"{base_url.rstrip('/')}/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={
                "model": model,
                "messages": _prompt_for_query(query),
                "temperature": 0.0,
                "max_tokens": 600,
                "enable_thinking": False,
                "response_format": {"type": "json_object"},
            },
        )
    response.raise_for_status()
    content = response.json()["choices"][0]["message"]["content"]
    return _parse_variants(content)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="Generate frozen LLM acoustic HyDE variants")
    parser.add_argument("--output", required=True, help="Output JSON path outside the repo")
    parser.add_argument("--model", default=os.getenv("DASHSCOPE_MODEL", "qwen3.7-plus"))
    parser.add_argument("--base-url", default=os.getenv("DASHSCOPE_BASE_URL", DEFAULT_BASE_URL))
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--sleep", type=float, default=0.2)
    parser.add_argument("--timeout", type=float, default=90)
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    _load_env_file(REPO_ROOT / ".env")
    api_key = os.getenv("DASHSCOPE_API_KEY", "").strip()
    if not api_key:
        raise SystemExit("DASHSCOPE_API_KEY is not configured")

    out = Path(args.output)
    payload: dict[str, Any] = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "model": args.model,
        "source": "dashscope_compatible_chat",
        "variants": {},
        "errors": {},
    }
    if args.skip_existing and out.exists():
        payload.update(json.loads(out.read_text(encoding="utf-8")))
        payload.setdefault("variants", {})
        payload.setdefault("errors", {})

    queries = _query_ids(args.limit)
    for index, query in enumerate(queries, start=1):
        query_id = str(query["id"])
        if args.skip_existing and payload["variants"].get(query_id):
            print(f"[{index}/{len(queries)}] skip {query_id}")
            continue
        try:
            variants = _call_dashscope(
                api_key=api_key,
                model=args.model,
                base_url=args.base_url,
                query=query,
                timeout=args.timeout,
            )
            if not variants:
                raise RuntimeError("empty variants")
            payload["variants"][query_id] = variants
            payload["errors"].pop(query_id, None)
            print(f"[{index}/{len(queries)}] {query_id}: {len(variants)} variants")
        except Exception as exc:
            payload["errors"][query_id] = f"{type(exc).__name__}: {exc}"
            print(f"[{index}/{len(queries)}] {query_id}: ERROR {type(exc).__name__}")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        if args.sleep > 0:
            time.sleep(args.sleep)

    print(f"wrote {out}")


if __name__ == "__main__":
    main()
