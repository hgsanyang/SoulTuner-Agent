#!/usr/bin/env python
"""Verify a frozen V4 manifest before any GPU time is spent on it.

``MANIFEST.schema.json`` has existed since the V4 interface was drafted, but
nothing ever validated a document against it, and nothing ever checked that the
``sha256`` a manifest records for a split is the sha256 of the file on disk. A
manifest that is never checked is a comment, and a recorded digest that is never
compared is worse than no digest — it looks like provenance.

This is the check the training script runs before ``swift sft``:

* the manifest validates against ``data/sft/v4/MANIFEST.schema.json``;
* every split path exists and its actual sha256 equals the recorded one;
* every split's actual row count equals the recorded one;
* ``counts_by_request_kind`` sums to ``rows``;
* the sealed policy's measured disjointness is all-zero (the schema enforces
  this too, but the message here names the offending field);
* optionally, that ``--train``/``--val`` handed to the trainer are the very
  files the manifest describes, so a manifest cannot be waved at a run that
  trains on something else.

Exit codes: 0 ok, 4 unusable input, 6 fingerprint or contract mismatch.

    python -m data.sft.verify_frozen_manifest --manifest data/sft/v4/MANIFEST.json
    python -m data.sft.verify_frozen_manifest --manifest ... \
        --expect-train <path> --expect-val <path> --json run/manifest_check.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from data.sft.v4_contract import row_contract_errors

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_PATH = PROJECT_ROOT / "data" / "sft" / "v4" / "MANIFEST.schema.json"

EXIT_OK = 0
EXIT_UNUSABLE = 4
EXIT_MISMATCH = 6


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def count_rows(path: Path) -> int:
    with path.open(encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def validate_jsonl_rows(path: Path) -> tuple[int, list[dict[str, Any]]]:
    invalid = []
    total = 0
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            total += 1
            try:
                row = json.loads(line)
                errors = row_contract_errors(row)
            except Exception as exc:
                errors = [f"row: {type(exc).__name__}: {exc}"]
            if errors:
                invalid.append({"line": line_number, "errors": errors})
    return total, invalid


def schema_errors(manifest: dict[str, Any], schema: dict[str, Any]) -> list[str]:
    try:
        from jsonschema import Draft202012Validator
    except ImportError:  # pragma: no cover - exercised only without jsonschema
        return ["jsonschema is not installed, so the manifest cannot be validated"]
    validator = Draft202012Validator(schema)
    return [
        f"{'/'.join(str(part) for part in error.path) or '(root)'}: {error.message}"
        for error in sorted(validator.iter_errors(manifest), key=lambda e: list(e.path))
    ]


def check_manifest(
    manifest_path: Path,
    *,
    schema_path: Path = SCHEMA_PATH,
    root: Path = PROJECT_ROOT,
    expect_train: Path | None = None,
    expect_val: Path | None = None,
    expect_sealed: Path | None = None,
) -> tuple[int, dict[str, Any]]:
    """Return ``(exit_code, report)``. Never raises on bad input."""

    report: dict[str, Any] = {
        "manifest": str(manifest_path),
        "manifest_sha256": None,
        "schema_errors": [],
        "splits": {},
        "problems": [],
    }
    if not manifest_path.is_file():
        report["problems"].append(f"manifest not found: {manifest_path}")
        return EXIT_UNUSABLE, report
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        report["problems"].append(f"manifest is not readable JSON: {exc}")
        return EXIT_UNUSABLE, report
    if not isinstance(manifest, dict):
        report["problems"].append("manifest must be a JSON object")
        return EXIT_UNUSABLE, report

    report["manifest_sha256"] = sha256_file(manifest_path)
    report["dataset_version"] = manifest.get("dataset_version")
    report["generator_commit"] = manifest.get("generator_commit")

    try:
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        report["problems"].append(f"schema is not readable JSON: {exc}")
        return EXIT_UNUSABLE, report

    report["schema_errors"] = schema_errors(manifest, schema)
    if report["schema_errors"]:
        report["problems"].append(
            f"manifest fails MANIFEST.schema.json ({len(report['schema_errors'])} error(s))"
        )

    for name, split in (manifest.get("splits") or {}).items():
        if not isinstance(split, dict):
            report["problems"].append(f"splits/{name} is not an object")
            continue
        entry: dict[str, Any] = {
            "path": split.get("path"),
            "recorded_sha256": split.get("sha256"),
            "recorded_rows": split.get("rows"),
        }
        report["splits"][name] = entry
        raw_path = str(split.get("path") or "")
        if not raw_path:
            report["problems"].append(f"splits/{name} has no path")
            continue
        path = Path(raw_path)
        if not path.is_absolute():
            path = root / path
        entry["resolved_path"] = str(path)
        if not path.is_file():
            report["problems"].append(f"splits/{name}: file not found at {path}")
            continue
        entry["actual_sha256"] = sha256_file(path)
        entry["actual_rows"], invalid_rows = validate_jsonl_rows(path)
        entry["invalid_contract_rows"] = len(invalid_rows)
        entry["contract_examples"] = invalid_rows[:5]
        if invalid_rows:
            report["problems"].append(
                f"splits/{name}: {len(invalid_rows)} row(s) fail the V4 provenance or execution contract"
            )
        if entry["actual_sha256"] != entry["recorded_sha256"]:
            report["problems"].append(
                f"splits/{name}: sha256 mismatch "
                f"(recorded {entry['recorded_sha256']}, actual {entry['actual_sha256']})"
            )
        if entry["recorded_rows"] != entry["actual_rows"]:
            report["problems"].append(
                f"splits/{name}: row count mismatch "
                f"(recorded {entry['recorded_rows']}, actual {entry['actual_rows']})"
            )
        counts = split.get("counts_by_request_kind") or {}
        if isinstance(counts, dict) and counts:
            total = sum(int(v) for v in counts.values())
            entry["counts_total"] = total
            if total != entry["recorded_rows"]:
                report["problems"].append(
                    f"splits/{name}: counts_by_request_kind sums to {total}, "
                    f"but rows is {entry['recorded_rows']}"
                )

    measured = ((manifest.get("sealed_policy") or {}).get("measured") or {})
    for field in ("shared_episodes", "shared_artists", "shared_songs", "shared_templates"):
        value = measured.get(field)
        if value not in (0, None):
            report["problems"].append(f"sealed_policy/measured/{field} = {value}, must be 0")

    validator = manifest.get("validator") or {}
    if validator.get("hard_findings") not in (0,):
        report["problems"].append(
            f"validator/hard_findings = {validator.get('hard_findings')!r}, must be 0"
        )

    for label, expected in (
        ("train", expect_train),
        ("val", expect_val),
        ("sealed", expect_sealed),
    ):
        if expected is None:
            continue
        split_name = {"train": "train", "val": "regression", "sealed": "sealed"}[label]
        entry = report["splits"].get(split_name) or {}
        resolved = entry.get("resolved_path")
        if resolved is None:
            report["problems"].append(
                f"--expect-{label} was given but the manifest has no usable '{split_name}' split"
            )
            continue
        if Path(resolved).resolve() != expected.resolve():
            report["problems"].append(
                f"--expect-{label} {expected} is not the manifest's '{split_name}' split "
                f"({resolved})"
            )

    if report["problems"]:
        return EXIT_MISMATCH, report
    return EXIT_OK, report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--schema", type=Path, default=SCHEMA_PATH)
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--expect-train", type=Path)
    parser.add_argument("--expect-val", type=Path)
    parser.add_argument("--expect-sealed", type=Path)
    parser.add_argument("--json", dest="json_out", type=Path)
    args = parser.parse_args(argv)

    code, report = check_manifest(
        args.manifest,
        schema_path=args.schema,
        root=args.root,
        expect_train=args.expect_train,
        expect_val=args.expect_val,
        expect_sealed=args.expect_sealed,
    )
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    if code == EXIT_OK:
        print(
            f"MANIFEST OK: {report.get('dataset_version')} "
            f"sha256={report['manifest_sha256'][:16]} splits={sorted(report['splits'])}"
        )
        return EXIT_OK
    print(f"MANIFEST FAIL ({len(report['problems'])} problem(s)):")
    for problem in report["problems"]:
        print(f"  - {problem}")
    for error in report["schema_errors"][:10]:
        print(f"  [schema] {error}")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
