#!/usr/bin/env python
"""Export an applied cache-import run as web-LLM tag review bundles."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from services.offline_tag_review import build_review_prompt, make_task

DEFAULT_BATCH_SIZE = 50
MAX_BATCH_SIZE = 50


def export_runs(
    report_paths: list[str | Path], output_dir: str | Path, *, batch_size: int = DEFAULT_BATCH_SIZE
) -> dict:
    reports = [json.loads(Path(path).read_text(encoding="utf-8")) for path in report_paths]
    tasks_by_id = {}
    for report in reports:
        for item in report.get("published") or []:
            task = make_task(item.get("record") or {})
            tasks_by_id.setdefault(task["task_id"], task)
    return export_tasks(
        list(tasks_by_id.values()),
        output_dir,
        batch_size=batch_size,
        run_ids=[report.get("run_id") for report in reports],
    )


def export_tasks(
    tasks: list[dict],
    output_dir: str | Path,
    *,
    batch_size: int = DEFAULT_BATCH_SIZE,
    run_ids: list[str | None] | None = None,
) -> dict:
    """Write a self-describing prompt/result bundle from frozen task rows."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "results").mkdir(exist_ok=True)
    manifest = output / "tasks.jsonl"
    manifest.write_text(
        "\n".join(json.dumps(task, ensure_ascii=False) for task in tasks) + ("\n" if tasks else ""),
        encoding="utf-8",
    )
    prompt_paths = []
    expected_results = []
    size = min(MAX_BATCH_SIZE, max(1, int(batch_size)))
    for index, start in enumerate(range(0, len(tasks), size), 1):
        path = output / f"prompt-{index:03d}.md"
        result_relative = f"results/result-{index:03d}.jsonl"
        batch = tasks[start : start + size]
        path.write_text(
            build_review_prompt(batch, result_filename=result_relative),
            encoding="utf-8",
        )
        prompt_paths.append(str(path))
        expected_results.append({
            "prompt": path.name,
            "result": result_relative,
            "rows": len(batch),
        })
    readme = output / "RESULTS_README.md"
    readme.write_text(_results_readme(len(tasks), expected_results), encoding="utf-8")
    summary = {
        "run_ids": run_ids or [],
        "tasks": len(tasks),
        "batch_size": size,
        "prompt_files": len(prompt_paths),
        "manifest": str(manifest),
        "prompts": prompt_paths,
        "expected_results": expected_results,
        "results_readme": str(readme),
    }
    (output / "export-summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def _results_readme(task_count: int, batches: list[dict]) -> str:
    rows = [
        "# 网页 LLM 标签结果保存说明",
        "",
        f"共 {task_count} 首。每个 prompt 最多 {MAX_BATCH_SIZE} 首，可逐批校验和导入。",
        "",
        "- 将整个 `prompt-XXX.md` 发给开启联网搜索的网页模型。",
        "- 只保存模型返回的 JSONL，编码为 UTF-8；每首歌一行 JSON。",
        "- 不要保存为 JSON 数组，不要保留 Markdown 代码围栏或解释文字。",
        "- 不要修改 task_id、music_id、title、artist。",
        "- 结果文件放入本目录的 `results/`。",
        "",
        "| 输入文件 | 输出文件 | 应有行数 |",
        "|---|---|---:|",
    ]
    rows.extend(
        f"| `{batch['prompt']}` | `{batch['result']}` | {batch['rows']} |"
        for batch in batches
    )
    return "\n".join(rows) + "\n"


def export_run(
    report_path: str | Path,
    output_dir: str | Path,
    *,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> dict:
    """Backward-compatible single-run wrapper."""
    return export_runs([report_path], output_dir, batch_size=batch_size)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-report", required=True, action="append")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"songs per prompt (default and maximum: {MAX_BATCH_SIZE})",
    )
    args = parser.parse_args()
    print(json.dumps(
        export_runs(args.run_report, args.output_dir, batch_size=args.batch_size),
        ensure_ascii=False,
        indent=2,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
