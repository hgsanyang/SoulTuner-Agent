#!/usr/bin/env python
"""Derive an eval input that asks the sealed questions under the deployed prompt.

The frozen V4 splits do not agree on the system prompt. train and regression
carry the 662-character ``STUDENT_SYSTEM_PROMPT_V3``; sealed carries a
77-character one. That is an accident of two code paths, not a
prompt-robustness design: ``build_v4_contract_curriculum`` inherits the prompt
from ``train_rows[0]`` while ``collect_v4_sealed_teacher`` hardcodes a short
one. Each side is internally uniform, which is what an accident looks like.

It still matters, because a sealed score is then a claim about the student under
a prompt it never saw in training, and that is not the number anyone will quote.
So this derives a second eval *input* whose system message is the one the model
was actually trained under. Gold answers, row order and every other field are
carried through untouched — this changes the question's framing, never its
answer. Scoring both files and reporting both numbers turns an unintended
difference into a measured one: the gap is the student's prompt sensitivity.

The canonical prompt is read from the frozen train split rather than imported
from ``build_sft_chatml``, because the constant in the code can drift from the
bytes that were frozen, and the bytes are what the model saw. The constant is
still compared against it and any divergence is reported.

The frozen files are never written to. Exit codes: 0 OK, 4 unusable input,
6 the derivation did not preserve what it promised to preserve.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections.abc import Iterator
from pathlib import Path

EXIT_OK = 0
EXIT_BAD_INPUT = 4
EXIT_MISMATCH = 6

SYSTEM = "system"


def _rows(path: Path) -> Iterator[dict]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as raw:
        for chunk in iter(lambda: raw.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _system_of(row: dict, index: int) -> str:
    messages = row.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError(f"row {index} has no messages")
    head = messages[0]
    if head.get("role") != SYSTEM:
        raise ValueError(
            f"row {index} starts with role={head.get('role')!r}, not {SYSTEM!r}; "
            "refusing to guess which message is the system prompt"
        )
    return str(head.get("content") or "")


def canonical_prompt(reference: Path) -> str:
    """The one system prompt the reference split uses.

    More than one means the reference has no single canonical prompt, and
    picking one of them would be a coin flip presented as a measurement.
    """
    seen: dict[str, int] = {}
    for index, row in enumerate(_rows(reference), start=1):
        prompt = _system_of(row, index)
        seen[prompt] = seen.get(prompt, 0) + 1
    if not seen:
        raise ValueError(f"{reference} has no rows")
    if len(seen) > 1:
        summary = ", ".join(
            f"{hashlib.sha256(p.encode()).hexdigest()[:8]}×{n}" for p, n in seen.items()
        )
        raise ValueError(
            f"{reference} carries {len(seen)} distinct system prompts ({summary}); "
            "there is no single canonical prompt to derive from"
        )
    return next(iter(seen))


def derive(source: Path, target: Path, prompt: str) -> dict:
    if target.resolve() == source.resolve():
        raise ValueError("target is the source; the frozen split is never rewritten")
    target.parent.mkdir(parents=True, exist_ok=True)

    rows = replaced = already = 0
    source_prompts: dict[str, int] = {}

    # Same atomic discipline as the ChatML projection: a half-written eval input
    # looks entirely normal on the next run, and would silently score a subset.
    staging = target.with_suffix(target.suffix + ".partial")
    try:
        with staging.open("w", encoding="utf-8") as out:
            for row in _rows(source):
                rows += 1
                current = _system_of(row, rows)
                key = hashlib.sha256(current.encode("utf-8")).hexdigest()[:12]
                source_prompts[key] = source_prompts.get(key, 0) + 1
                if current == prompt:
                    already += 1
                else:
                    replaced += 1
                # Rebuild only the system message. The gold assistant turn, the
                # user turn, meta and lineage are carried through by reference.
                row["messages"] = [
                    {**row["messages"][0], "content": prompt},
                    *row["messages"][1:],
                ]
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
            out.flush()
            os.fsync(out.fileno())
        os.replace(staging, target)
    except BaseException:
        staging.unlink(missing_ok=True)
        raise

    return {
        "source": str(source),
        "source_sha256": _sha256(source),
        "target": str(target),
        "target_sha256": _sha256(target),
        "rows": rows,
        "system_replaced": replaced,
        "system_already_canonical": already,
        "source_system_prompts": source_prompts,
        "canonical_prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "canonical_prompt_chars": len(prompt),
    }


def verify(source: Path, target: Path, prompt: str) -> list[str]:
    """Only the system message moved.

    The derived file is scored against the same gold, so anything else changing
    would silently compare the student to a different question. Both sides are
    walked as parsed rows in order, never as raw lines.
    """
    problems: list[str] = []
    src = _rows(source)
    dst = _rows(target)
    index = 0
    for src_row, dst_row in zip(src, dst):
        index += 1
        if dst_row["messages"][0].get("content") != prompt:
            problems.append(f"row {index}: system message is not the canonical prompt")
            break
        if dst_row["messages"][1:] != src_row["messages"][1:]:
            problems.append(f"row {index}: a non-system message changed")
            break
        if dst_row["messages"][0].get("role") != SYSTEM:
            problems.append(f"row {index}: system message lost its role")
            break
        rest_src = {k: v for k, v in src_row.items() if k != "messages"}
        rest_dst = {k: v for k, v in dst_row.items() if k != "messages"}
        if rest_src != rest_dst:
            # meta.episode_id/turn_id is how score_student aligns gold to
            # predictions; losing it silently scores nothing against nothing.
            problems.append(f"row {index}: fields outside 'messages' changed")
            break

    extra_src = sum(1 for _ in src)
    extra_dst = sum(1 for _ in dst)
    if extra_src or extra_dst:
        problems.append(
            f"row counts diverge after {index} matched rows: "
            f"{extra_src} extra source rows, {extra_dst} extra derived rows"
        )
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True,
                        help="frozen eval split to reframe (e.g. sealed_v4_chatml.jsonl)")
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True,
                        help="split whose system prompt is canonical (the train split)")
    parser.add_argument("--json", dest="json_out", type=Path)
    args = parser.parse_args(argv)

    for path in (args.source, args.reference):
        if not path.is_file():
            print(f"DERIVE FAIL: no such file: {path}")
            return EXIT_BAD_INPUT

    try:
        prompt = canonical_prompt(args.reference)
        report = derive(args.source, args.target, prompt)
    except (ValueError, KeyError, json.JSONDecodeError) as exc:
        print(f"DERIVE FAIL: {exc}")
        return EXIT_BAD_INPUT

    report["reference"] = str(args.reference)
    # Report, never enforce: the frozen bytes win, and a drifted constant is a
    # finding about the code rather than a reason to refuse the derivation.
    try:
        from data.sft.build_sft_chatml import STUDENT_SYSTEM_PROMPT_V3

        report["matches_code_constant"] = STUDENT_SYSTEM_PROMPT_V3 == prompt
    except ImportError:
        report["matches_code_constant"] = None

    problems = verify(args.source, args.target, prompt)
    report["problems"] = problems

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )

    if problems:
        print("DERIVE FAIL:")
        for problem in problems:
            print(f"  - {problem}")
        return EXIT_MISMATCH

    if report["matches_code_constant"] is False:
        print(
            "NOTE: the frozen train prompt differs from STUDENT_SYSTEM_PROMPT_V3 in "
            "the code; the frozen bytes were used, since those are what the model saw."
        )
    print(
        f"DERIVE OK: {report['rows']} rows -> {args.target.name} "
        f"({report['system_replaced']} reframed, "
        f"{report['system_already_canonical']} already canonical, "
        f"prompt {report['canonical_prompt_chars']} chars)"
    )
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
