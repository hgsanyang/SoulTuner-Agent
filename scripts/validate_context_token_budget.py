"""Validate the shared memory context budget with the formal model tokenizer."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from retrieval.gssc_context_builder import build_context


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _cases() -> dict[str, dict[str, str]]:
    return {
        "long_chinese_session": {
            "explicit_profile": "用户明确设置偏好舒缓、低动态、女声。" * 300,
            "graphzep_facts": "夜间独处时更常收藏空间感明显的歌曲。" * 500,
            "chat_history": "\n".join([f"第{i}轮：请保持前一轮氛围并更安静一点。" for i in range(900)]),
            "retrieval_context": "候选歌曲及解释。" * 1000,
        },
        "mixed_language": {
            "explicit_profile": "Explicit preference: low dynamics; 中文女声。" * 300,
            "graphzep_facts": "Episodic memory: late-night driving, skip loud tracks. 夜间驾驶。" * 400,
            "chat_history": "\n".join([f"turn {i}: keep the ambience, 但不要太悲伤。" for i in range(800)]),
            "retrieval_context": "Graph + Dense candidate evidence. " * 1200,
        },
        "latest_correction": {
            "explicit_profile": "长期明确偏好摇滚。" * 500,
            "graphzep_facts": "过去经常播放高能量音乐。" * 700,
            "chat_history": "\n".join([f"旧对话{i}: 继续高能量。" for i in range(1000)]) + "\n最新纠正：今天不要摇滚，只要安静钢琴。",
            "retrieval_context": "候选证据。" * 1000,
        },
    }


async def _run(tokenizer_path: Path, budget: int) -> dict:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        str(tokenizer_path), local_files_only=True, trust_remote_code=True
    )

    def count(text: str) -> int:
        return len(tokenizer.encode(text or "", add_special_tokens=False))

    results = []
    for name, payload in _cases().items():
        bounded = await build_context(
            **payload,
            user_input="本轮请求不计入共享上下文预算",
            total_budget=budget,
            user_id="validation-user",
            conversation_id=f"validation-{name}",
            token_counter=count,
        )
        counts = {key: count(value) for key, value in bounded.items()}
        total = sum(counts.values())
        results.append(
            {
                "case": name,
                "field_tokens": counts,
                "total_tokens": total,
                "within_budget": total <= budget,
                "latest_correction_preserved": (
                    "最新纠正" in bounded.get("chat_history", "")
                    if name == "latest_correction"
                    else None
                ),
            }
        )
    tracked = {}
    for filename in ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt"):
        path = tokenizer_path / filename
        if path.exists():
            tracked[filename] = _sha256(path)
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "tokenizer_directory": tokenizer_path.name,
        "tokenizer_class": type(tokenizer).__name__,
        "vocab_size": len(tokenizer),
        "budget": budget,
        "files_sha256": tracked,
        "cases": results,
        "passed": all(item["within_budget"] and item["latest_correction_preserved"] is not False for item in results),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer-path", type=Path, required=True)
    parser.add_argument("--budget", type=int, default=8000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = asyncio.run(_run(args.tokenizer_path.resolve(), args.budget))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"passed": report["passed"], "output": str(args.output)}, ensure_ascii=False))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
