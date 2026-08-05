#!/usr/bin/env python
"""Project a frozen ChatML split down to the one column training reads.

Why this exists: the frozen V4 train split carries provenance alongside the
conversation, and ``lineage`` is not the same shape on every row — 5700 rows
have ``{builder, builder_version}`` and 411 also carry ``clarification_trope``.
Arrow structs are strongly typed, so ``datasets`` infers the two-key struct from
the head of the file and then fails to cast the three-key rows:

    TypeError: Couldn't cast array of type
    struct<builder: string, builder_version: string, clarification_trope: string>
    to {'builder': Value('string'), 'builder_version': Value('string')}

The training run never reads those columns; ms-swift consumes ``messages``.
Rewriting the frozen files to make the struct uniform would change their
SHA-256 and invalidate the manifest that pins the dataset identity, so the
frozen bytes stay exactly as audited and this writes a derived copy instead.

The projection is checked, not assumed: row count must match and every row's
``messages`` must be byte-identical to the source once re-serialised the same
way. A projection that quietly dropped or reordered a conversation would be far
worse than the load error it replaces, so it fails closed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

EXIT_OK = 0
EXIT_BAD_INPUT = 4
EXIT_MISMATCH = 6

#: The only key ms-swift reads for a ChatML SFT dataset. Everything else in the
#: frozen rows is provenance for offline audit.
KEPT = "messages"


def project(source: Path, target: Path) -> dict:
    target.parent.mkdir(parents=True, exist_ok=True)
    rows = kept = 0
    digest_src = hashlib.sha256()
    with source.open("rb") as raw:
        for chunk in iter(lambda: raw.read(1 << 20), b""):
            digest_src.update(chunk)

    with source.open(encoding="utf-8") as fin, target.open("w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            rows += 1
            record = json.loads(line)
            if KEPT not in record:
                raise ValueError(f"row {rows} has no '{KEPT}' key; refusing to project")
            fout.write(json.dumps({KEPT: record[KEPT]}, ensure_ascii=False) + "\n")
            kept += 1

    digest_dst = hashlib.sha256(target.read_bytes()).hexdigest()
    return {
        "source": str(source),
        "source_sha256": digest_src.hexdigest(),
        "target": str(target),
        "target_sha256": digest_dst,
        "rows": rows,
        "kept": kept,
    }


def verify(source: Path, target: Path) -> list[str]:
    """Every conversation survived, in order, unchanged."""
    problems: list[str] = []
    with source.open(encoding="utf-8") as fsrc, target.open(encoding="utf-8") as fdst:
        index = 0
        for src_line, dst_line in zip(fsrc, fdst):
            if not src_line.strip():
                continue
            index += 1
            src_messages = json.loads(src_line).get(KEPT)
            dst_record = json.loads(dst_line)
            if set(dst_record) != {KEPT}:
                problems.append(f"row {index}: projection carries extra keys {sorted(dst_record)}")
                break
            if dst_record[KEPT] != src_messages:
                problems.append(f"row {index}: '{KEPT}' differs from the source")
                break
        remaining_src = sum(1 for line in fsrc if line.strip())
        remaining_dst = sum(1 for _ in fdst)
        if remaining_src or remaining_dst:
            problems.append(
                f"row counts diverge: {remaining_src} extra source rows, "
                f"{remaining_dst} extra projected rows"
            )
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--json", dest="json_out", type=Path)
    args = parser.parse_args(argv)

    if not args.source.is_file():
        print(f"PROJECTION FAIL: no such file: {args.source}")
        return EXIT_BAD_INPUT

    try:
        report = project(args.source, args.target)
    except (ValueError, json.JSONDecodeError) as exc:
        print(f"PROJECTION FAIL: {exc}")
        return EXIT_BAD_INPUT

    problems = verify(args.source, args.target)
    report["problems"] = problems
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )

    if problems:
        print("PROJECTION FAIL:")
        for problem in problems:
            print(f"  - {problem}")
        return EXIT_MISMATCH

    print(
        f"PROJECTION OK: {report['rows']} rows -> {args.target.name} "
        f"(source sha256={report['source_sha256'][:16]})"
    )
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
