#!/usr/bin/env python
"""Export an applied cache-import run as web-LLM tag review bundles."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from services.offline_tag_review import build_review_prompt, make_task


def export_runs(
    report_paths: list[str | Path], output_dir: str | Path, *, batch_size: int = 10
) -> dict:
    reports = [json.loads(Path(path).read_text(encoding="utf-8")) for path in report_paths]
    tasks_by_id = {}
    for report in reports:
        for item in report.get("published") or []:
            task = make_task(item.get("record") or {})
            tasks_by_id.setdefault(task["task_id"], task)
    tasks = list(tasks_by_id.values())
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    manifest = output / "tasks.jsonl"
    manifest.write_text(
        "\n".join(json.dumps(task, ensure_ascii=False) for task in tasks) + ("\n" if tasks else ""),
        encoding="utf-8",
    )
    prompt_paths = []
    size = max(1, int(batch_size))
    for index, start in enumerate(range(0, len(tasks), size), 1):
        path = output / f"prompt-{index:03d}.md"
        path.write_text(build_review_prompt(tasks[start : start + size]), encoding="utf-8")
        prompt_paths.append(str(path))
    summary = {
        "run_ids": [report.get("run_id") for report in reports],
        "tasks": len(tasks),
        "batch_size": size,
        "prompt_files": len(prompt_paths),
        "manifest": str(manifest),
        "prompts": prompt_paths,
    }
    (output / "export-summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def export_run(report_path: str | Path, output_dir: str | Path, *, batch_size: int = 10) -> dict:
    """Backward-compatible single-run wrapper."""
    return export_runs([report_path], output_dir, batch_size=batch_size)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-report", required=True, action="append")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=10)
    args = parser.parse_args()
    print(json.dumps(
        export_runs(args.run_report, args.output_dir, batch_size=args.batch_size),
        ensure_ascii=False,
        indent=2,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
